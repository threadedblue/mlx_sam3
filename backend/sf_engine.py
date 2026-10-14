"""SF (Segmentation & Inpainting) engine — model v2 (see
DoubleNaught/doc/Clippings/sf-model-v2-design.md, which this implements).

Every selected object is a `MaskRecord` carrying two INDEPENDENT axes:

- `dataset_status` — `unassigned` or `keep` (captioned). There is no stored
  `discard`: an object that is held and scrubbed without ever reaching
  `keep` simply never got captioned, and is never exported.
- `held` — whether it is queued for the current pass's next scrub batch.

Selecting, holding and captioning are three separate actions, in any order.
Captioning never makes a record scrub-ineligible, and scrubbing never
touches a record's caption.

Passes form an unbounded chain: pass 0 is the original image; scrubbing
pass N unions the geometry of every record *held in pass N* (whatever its
dataset_status), inpaints pass N's image with it, and produces pass N+1.
Any pass can be scrubbed, any number of times.

AA design notes (see the associative-arrays skill / aa-spec.pdf for why):
- Mask row keys are minted as `f"{pass}:{uuid4()}"`, unique across every
  pass — d4m_juliacall_bridge.build_triples combines duplicate (row, col)
  pairs via D4M.jl's default operator, which would silently merge two
  distinct masks if IDs collided.
- `pass` is one typed column (numeric string per row), not one column per
  pass: with an unbounded pass count, per-pass columns would guarantee
  fully-empty columns — a no-empty-column invariant violation.
"""

from __future__ import annotations

import base64
import io
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np
from PIL import Image, ImageFilter

import d4m_juliacall_bridge as bridge

ORIGINAL_PASS = 0

# `MaskRecord.source` values — how a selection was made. SOURCE_BALLOON is
# mechanically identical to SOURCE_BOX (balloon detection goes through
# `add_box_selection` unchanged); it exists only so a scrub can tell that its
# batch came from balloon detection, which is what makes the resulting pass
# identifiable as the WB-scrub result rather than an ordinary S&S round.
SOURCE_TEXT_PROMPT = "text_prompt"
SOURCE_BOX = "box"
SOURCE_POINT = "point"
SOURCE_BALLOON = "balloon"

# Smallest accepted panel side, in page pixels (v4 §4, manual authoring).
# 16px is the floor, not a guess at a real panel size: the real pages this
# runs against are ~1740x2376, where the narrowest genuine panel measured
# during the panel-bounds validation was 241px wide (little-nemo_0000's
# tier-6 strip), so 16 is an order of magnitude below anything legitimate
# while still rejecting the actual failure mode — a click or a 2px twitch
# registering as a drag and leaving a sliver in the panel list. Deliberately
# an absolute pixel floor rather than a fraction of the image: what makes a
# panel unusable is being too small to see and click a delete control for,
# which does not scale with page size.
MIN_PANEL_SIDE = 16

# `panel_source` value on an exported mask row when NO panel covered it and
# the whole page was used instead. A sentinel string rather than an omitted
# column: see build_sf_payload for why the fallback has to stay visible.
PANEL_SOURCE_FULL_PAGE = "_full_page"


class DatasetStatus(str, Enum):
    UNASSIGNED = "unassigned"
    KEEP = "keep"


class UnknownMaskError(KeyError):
    """No MaskRecord with that mask_id exists in this session."""


class UnknownPanelError(KeyError):
    """No PanelRecord with that panel_id exists in this session."""


@dataclass(frozen=True)
class MaskRecord:
    mask_id: str
    pass_: int
    geometry: np.ndarray  # binary mask, shape (H, W)
    source: str           # "text_prompt" | "point" | "box"
    # Provenance: the SAM3 text prompt, or a box/point's optional
    # text_substitute. NOT the training caption — see `caption`.
    text_tag: Optional[str] = None
    score: Optional[float] = None  # SAM3's confidence for this mask, if reported
    dataset_status: DatasetStatus = DatasetStatus.UNASSIGNED
    held: bool = False
    # Set only by SFSession.attach_caption; present iff dataset_status is KEEP.
    caption: Optional[str] = None


@dataclass(frozen=True)
class PanelRecord:
    """A manually authored panel boundary — v4 §4's `panel_bounds`.

    A panel is a LAYOUT REGION, not an object: it is never segmented, never
    carries a caption of its own, and is deliberately NOT a `MaskRecord`.
    Nothing here goes near SAM3 — authoring one is pure geometry capture.

    Integer page-pixel coordinates, half-open `[x0, x1) x [y0, y1)` to match
    `_bbox_from_geometry`'s own convention elsewhere in this module.

    NO PASS NUMBER, deliberately — unlike `MaskRecord`. v4 §4 describes the
    record as carrying "which pass/layer it was captured against", but every
    pass of a session is the same image dimensions (a scrub inpaints pixels,
    it never crops or resizes), so one bbox is valid against every pass. A
    pass field here could only ever go stale or disagree with the mask it is
    being matched against; `assign_panel` takes the mask's pass instead, and
    `panel_crop_png_bytes` is handed whichever pass image the caller wants.
    """
    panel_id: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_box(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class ScrubRecord:
    """Audit trail for the most recent scrub — enough for LBSCard's "Last
    scrub: Pass N → Pass N+1" line, and for callers that need the batch's
    mask_ids after `held` has already been cleared on them."""
    from_pass: int
    to_pass: int
    mask_ids: tuple[str, ...]
    # What `wb_scrub_pass` was BEFORE this scrub ran, so undoing it can put
    # the Display Layers "WB-scrub result" layer back exactly. Defaulted so a
    # record rebuilt from storage (which never persisted it) still constructs.
    prev_wb_scrub_pass: Optional[int] = None


class CannotUndoScrubError(ValueError):
    """Undo was refused; the message says why and what to do about it."""


class ImmutableOriginal:
    """Pass 0 / $I_{orig}$. Never mutated; every op works on a copy."""

    def __init__(self, image_bytes: bytes):
        self._bytes = image_bytes
        with Image.open(io.BytesIO(image_bytes)) as probe:
            self.size = probe.size

    @property
    def bytes(self) -> bytes:
        return self._bytes

    def new_working_copy(self) -> "WorkingCopy":
        image = Image.open(io.BytesIO(self._bytes)).convert("RGBA")
        return WorkingCopy(image)


@dataclass
class WorkingCopy:
    """An isolated, mutable canvas buffer derived from a pass's image."""

    image: Image.Image


class InpaintingEngine(Protocol):
    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image: ...


# Pixels the mask is grown by before it reaches LaMa. SAM3's masks stop at
# the object's interior, leaving its own outline outside the mask: measured
# on the page this was root-caused against, 455 dark "ink" pixels sat within
# 1px outside the mask edge and 1288 within 3px. Inpainting an undilated
# mask therefore preserves that outline verbatim, so a scrubbed object still
# reads as a crisp, correctly-shaped ghost however good the fill is. Growing
# the mask first is standard for LaMa pipelines. 6px clears the contour at
# this page scale (~1800px wide) while growing the measured mask only ~23%,
# which matters in the other direction: an over-grown mask eats neighbouring
# art that was never selected.
_MASK_DILATION_PX = 6


def _dilate_mask(mask_image: Image.Image, radius: int) -> Image.Image:
    """Grow `mask_image`'s white region by `radius` pixels, scoped to its
    bounding box (+ `radius` margin) rather than filtering the whole page.

    Iterated 3x3 max-filtering is local: a pixel's dilated value depends
    only on pixels within `radius` of it, so cropping to the bbox expanded
    by exactly `radius` on each side (clamped to the image edges — the
    filter can't grow past a real edge either way, so clamping reproduces
    the same edge condition full-page filtering would hit) gives every
    dilated pixel access to everything it could possibly need, and nothing
    outside that crop can affect the result. Confirmed pixel-identical to
    full-page filtering on the real mask that found the ghost-artifact bug
    (TestMaskDilationIsBoundingBoxScoped) and measured meaningfully faster
    on it: filtering scales with the mask's size, not the page's, and a
    ~20k px mask on a 1791x2298 page was previously ~0.6s of a ~1.76s
    post-MPS scrub — a fraction of that page for a mask this size.
    """
    bbox = mask_image.getbbox()
    if bbox is None:
        return mask_image  # nothing set; nothing to grow
    width, height = mask_image.size
    x0, y0, x1, y1 = bbox
    cx0, cy0 = max(0, x0 - radius), max(0, y0 - radius)
    cx1, cy1 = min(width, x1 + radius), min(height, y1 + radius)
    crop = mask_image.crop((cx0, cy0, cx1, cy1))
    for _ in range(radius):
        crop = crop.filter(ImageFilter.MaxFilter(3))
    grown = Image.new("L", (width, height), 0)
    grown.paste(crop, (cx0, cy0))
    return grown


class LamaInpainter:
    """LaMa (Large Mask Inpainting via Fast Fourier Convolutions).

    Loads its checkpoint lazily on first construction rather than at module
    import — `simple-lama-inpainting` pulls a multi-hundred-MB torchscript
    checkpoint on first use, which importing this module shouldn't trigger
    on its own.
    """

    def __init__(self) -> None:
        import torch  # noqa: PLC0415
        from simple_lama_inpainting import SimpleLama  # noqa: PLC0415

        from lora_trainer import _device  # noqa: PLC0415

        # SimpleLama's own default only checks torch.cuda.is_available() —
        # no MPS branch at all (confirmed by reading its installed source)
        # — so on this machine it silently ran on CPU despite MPS being
        # available: 28.92s per scrub, measured on the same page/mask that
        # found the mask-dilation ghost bug above, vs 1.76s on MPS (16.5x),
        # with output numerically equivalent (max channel diff 1/255 across
        # 4.1M pixels). LaMa's Fourier convolutions need FFT ops that are
        # natively supported on MPS under this torch version — confirmed
        # with PYTORCH_ENABLE_MPS_FALLBACK=0, so any unsupported op would
        # have raised rather than silently falling back to CPU per-op.
        #
        # Device has to be supplied at construction, not set after:
        # SimpleLama stores it and uses it to place INPUT tensors in
        # prepare_img_and_mask, so a post-hoc `.to(device)` on the loaded
        # module would move the weights but leave inputs on whatever
        # device the default put them on — a device mismatch. Reusing
        # lora_trainer's own _device() (same cuda->mps->cpu precedence
        # already used for the training path) rather than writing new
        # device-selection logic here. _dtype() is NOT reused: the
        # checkpoint is float32, and _dtype()'s own comment notes fp16 is
        # unsafe on MPS regardless.
        self._lama = SimpleLama(device=torch.device(_device()))

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        # Grow the mask past the object's own edge — see _MASK_DILATION_PX
        # and _dilate_mask. Done here rather than in
        # `_union_mask`/`run_lama_pass` because it is a property of what
        # this inpainter needs, not of what the user selected: the
        # engine's union stays exactly the held set, which is what its own
        # tests pin (with fake inpainters that need no such margin).
        mask_image = _dilate_mask(mask_image, _MASK_DILATION_PX)
        result = self._lama(image.convert("RGB"), mask_image)
        # simple_lama_inpainting pads its input to a multiple of 8
        # (pad_img_to_modulo, padding added only at the bottom/right — see
        # its prepare_img_and_mask) and returns the padded output as-is,
        # with no crop-back step of its own. Confirmed live: a 1791x2298
        # source came back as 1792x2304. Left uncropped, this result
        # becomes the next pass's image via set_image, silently shifting
        # every subsequent pass's mask coordinates relative to the
        # original — the exact case a recursive peel (select in revealed
        # background -> hold -> scrub again) depends on being correct.
        # Padding is on the end, not the start, so a top-left crop back to
        # the pre-padding size recovers the original-registered pixels.
        result = result.crop((0, 0, image.width, image.height))
        return result.convert("RGBA")


class Sam3Like(Protocol):
    """Structural shape of `sam3.model.sam3_image_processor.Sam3Processor`.

    Kept as a Protocol so this module has no hard SAM3 import and stays
    unit-testable with a fake.
    """

    def set_image(self, image: Image.Image) -> dict: ...
    def set_text_prompt(self, prompt: str, state: dict) -> dict: ...
    def add_geometric_prompt(self, box: list[float], label: bool, state: dict) -> dict: ...
    def add_point_prompt(self, point: list[float], label: bool, state: dict) -> dict: ...


def _binary_mask(raw_mask) -> np.ndarray:
    arr = np.array(raw_mask)
    binary = (arr > 0.5).astype(np.uint8)
    if binary.ndim == 3:
        binary = binary[0]
    return binary


def _union_mask(masks: Sequence[np.ndarray]) -> np.ndarray:
    return np.logical_or.reduce(list(masks)).astype(np.uint8)


# Minimum IoU for a post-call mask to count as the "same" object as a
# pre-call MaskRecord. Below this, it's treated as a distinct selection.
_MASK_MATCH_IOU_THRESHOLD = 0.5


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return np.logical_and(a, b).sum() / union


def _best_match(
    geometry: np.ndarray, candidates: Sequence[MaskRecord], taken: set[str],
) -> Optional[MaskRecord]:
    """Highest-IoU untaken candidate for *geometry*, if it clears the threshold.

    Greedy, not a globally optimal assignment — acceptable here since a
    single SF selection call produces at most a handful of new/changed
    masks, not a dense many-to-many matching problem.
    """
    best: Optional[MaskRecord] = None
    best_score = 0.0
    for candidate in candidates:
        if candidate.mask_id in taken:
            continue
        score = _iou(geometry, candidate.geometry)
        if score > best_score:
            best_score = score
            best = candidate
    return best if best_score >= _MASK_MATCH_IOU_THRESHOLD else None


def _carries_user_state(record: MaskRecord) -> bool:
    """Whether the user has attached anything to this record that SAM3's
    re-grounding knows nothing about and must not be able to discard."""
    return record.dataset_status == DatasetStatus.KEEP or record.held


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _crop_png_bytes(original_image: Image.Image, mask: np.ndarray) -> bytes:
    """The object as it actually appears — `original_image` cut out via
    `mask` used as an alpha channel — RGBA PNG bytes.

    A faithful port of `aa_persistence.py`'s function of the same name, not
    a shared import: `aa_persistence` already imports this module at
    module level, so the reverse import would be circular. Keep this in
    sync with that copy if either ever changes — same logic, same
    `_binary_mask`-equivalent normalisation (this module's own
    `_binary_mask`, already used identically elsewhere in this file).
    """
    mask_image = Image.fromarray(_binary_mask(mask) * 255, mode="L")
    segment_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    segment_image.paste(original_image, (0, 0), mask_image)
    return _png_bytes(segment_image)


def _norm_cxcywh_to_pixel_xyxy(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, w, h = box
    return ((cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height)


def _geometry_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight `[x0, y0, x1, y1)` bounds of a binary mask, in page pixels.

    Same convention and same arithmetic as `services._bbox_from_geometry`
    and `aa_persistence._bbox_from_geometry`; kept local rather than
    imported because both of those modules import THIS one, so either
    direction of sharing would be circular (the same reason
    `_crop_png_bytes` is duplicated here — see its docstring). An all-zero
    mask returns a zero-area box at the origin, which `assign_panel` then
    treats as degenerate.
    """
    ys, xs = np.nonzero(_binary_mask(mask))
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _argmax_score(scores) -> Optional[int]:
    if len(scores) == 0:
        return None
    return int(np.argmax(np.array(scores)))


def _pick_for_box(
    drawn_box: Sequence[float], raw_boxes, scores, width: int, height: int,
) -> Optional[int]:
    """Index of the returned instance corresponding to the box the user drew.

    SAM3 treats a geometric prompt as a *concept exemplar*: `_call_grounding`
    returns every instance above `confidence_threshold`, so one box over one
    object can legitimately come back as many masks spread across the image
    (9, on the page this was root-caused against). The instance the user
    actually drew around is the one whose predicted bbox overlaps the drawn
    box; the generalisations sit elsewhere with ~zero overlap. Matching on
    that overlap is a geometric fact about the prompt, unlike a global
    confidence cut-off, which is a per-image magic number.
    """
    if len(raw_boxes) == 0:
        return _argmax_score(scores)
    drawn = _norm_cxcywh_to_pixel_xyxy(drawn_box, width, height)
    boxes_np = np.array(raw_boxes)
    ious = [_bbox_iou(drawn, boxes_np[i]) for i in range(len(boxes_np))]
    best = int(np.argmax(ious))
    # No overlap at all means the drawn region produced no instance of its
    # own (e.g. the user boxed empty background); fall back to confidence
    # rather than silently returning an unrelated far-away instance.
    return best if ious[best] > 0.0 else _argmax_score(scores)


def _pick_for_point(
    point: Sequence[float], geometries: Sequence[np.ndarray], scores, width: int, height: int,
) -> Optional[int]:
    """Index of the returned instance the user actually clicked inside.

    No top-score fallback (v3 spec §3 Fix 2): unlike `_pick_for_box`, a
    point has no drawn REGION to fall back on when nothing contains it —
    "no returned instance was under this click" and "no returned instance
    was under this NxN box" are not the same confidence of "the user
    probably meant one of these anyway". Confirmed live that falling back
    to the highest-confidence instance ANYWHERE on the page turned a
    missed click into a silent selection of an unrelated object, up to
    1,500px away — worse than doing nothing. A click that hits nothing
    records nothing; the caller (`_select_single`) leaves focus untouched
    rather than silently reassigning it.
    """
    if not geometries:
        return None
    x, y = point
    px = min(max(int(round(x * width)), 0), width - 1)
    py = min(max(int(round(y * height)), 0), height - 1)
    hits = [i for i, g in enumerate(geometries) if g.shape[0] > py and g.shape[1] > px and g[py, px]]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    scores_np = np.array(scores) if len(scores) else None
    if scores_np is None:
        return hits[0]
    return max(hits, key=lambda i: float(scores_np[i]) if i < len(scores_np) else 0.0)


class SFSession:
    """Selection/hold/caption/scrub state machine for one SF session (one
    image, any number of passes).

    `state` is the SAM3 processor state already produced by the caller's own
    `segmentor.set_image(...)` — image loading isn't this module's concern,
    only what happens to selections made against it.

    Pass comparisons use `==`, never `is`. `is` was only ever correct against
    the old `Pass` enum's singleton members; on ints it holds for CPython's
    cached -5..256 and silently fails above that (and for any pass value
    rebuilt from storage, e.g. `int("300")`), which would break pass scoping
    at depth with no error.
    """

    def __init__(self, session_id: str, original: ImmutableOriginal, segmentor: Sam3Like, state: dict):
        self.session_id = session_id
        self.original = original
        self.segmentor = segmentor
        self.state = state
        self.masks: list[MaskRecord] = []
        # Manually authored panel boundaries (v4 §4). Layout regions, NOT
        # objects: nothing here is a MaskRecord, nothing here has ever been
        # near SAM3, and nothing here is pass-scoped (see PanelRecord).
        # Kept in authoring order — the Panels card lists them as drawn.
        self.panels: list[PanelRecord] = []
        self._pass = ORIGINAL_PASS
        # Scrubbed image per pass >= 1. Pass 0's image is `original`. Every
        # pass is kept, not just the latest: a record's crop has to come from
        # the image it was traced against, and records from any earlier pass
        # (e.g. keepers captioned in pass 1 of a session now at pass 4) are
        # still exported.
        self._pass_images: dict[int, Image.Image] = {}
        self.last_scrub: Optional[ScrubRecord] = None
        # Every scrub this session has done, oldest first, so scrubs can be
        # undone one after another down to the page as first loaded. In
        # memory only: a reloaded session knows just its `last_scrub`, so it
        # can undo one level (see `restore`).
        self.scrub_history: list[ScrubRecord] = []
        # Which pass a WB (word-balloon) scrub produced, or None if one has
        # never run. v4 §2's layer table calls this layer "WB-scrub result"
        # and expects it at pass 1 — but pass 1 only MEANS that if Balloon
        # Scrub happened to run before any ordinary S&S scrub, which nothing
        # enforces (an S&S scrub first takes pass 1, pushing WB-scrub to 2+).
        # So the layer is identified by what produced it, recorded here, not
        # by its pass number. Stays None for a page with no balloons, so no
        # phantom WB layer appears for a scrub that never happened.
        self.wb_scrub_pass: Optional[int] = None
        # The mask id a box/point call most recently matched-or-created —
        # set by `_select_single` only (text prompts are multi-instance, so
        # there's no single "the" mask to point at). `serialize_sf_masks`'s
        # response is the full current-pass list, not a diff, so this is
        # the caller's only way to know which one THIS call was about —
        # needed to route the Prompt card's caption-mode call at the right
        # mask_id after a box/point selection.
        self.last_touched_mask_id: Optional[str] = None
        # mask_ids whose GEOMETRY was refreshed in place by the most recent
        # `_select`/`_select_single` call (reground reconciliation or an
        # Avoid refinement) — set fresh on every call, empty if none were.
        # A crop cache keyed by mask_id (services.py) drains this after each
        # mutation to know which cached crops are now stale; nothing else
        # (caption/hold/dataset_status) touches geometry, so nothing else
        # needs to appear here.
        self.last_refreshed_mask_ids: set[str] = set()

    @property
    def pass_(self) -> int:
        return self._pass

    def image_for_pass(self, pass_: int) -> Image.Image:
        """A copy of the image pass *pass_* was traced against (RGBA).

        A copy, like `ImmutableOriginal.new_working_copy`, so no caller can
        mutate a pass's history in place.
        """
        if pass_ == ORIGINAL_PASS:
            return self.original.new_working_copy().image
        if pass_ not in self._pass_images:
            raise ValueError(f"no image for pass {pass_}; this session is at pass {self._pass}")
        return self._pass_images[pass_].copy()

    def restore(
        self,
        pass_: int,
        pass_images: dict[int, Image.Image],
        masks: list[MaskRecord],
        last_scrub: Optional[ScrubRecord] = None,
        wb_scrub_pass: Optional[int] = None,
        panels: Optional[list[PanelRecord]] = None,
    ) -> None:
        """Reconstruct state from persistence in one call, rather than the
        caller reaching into `_pass`/`_pass_images`/`masks` directly.

        `wb_scrub_pass` must be supplied by the caller rather than re-derived
        from `masks`: `MaskRecord.source` is deliberately not persisted (it
        comes back as "loaded" — see aa_persistence's module docstring), so a
        reloaded session cannot tell which of its masks were balloons. The
        pass number itself is what survives.

        `state` (the live SAM3 processor dict) is deliberately NOT part of
        this: the caller must still `segmentor.set_image(...)` on whichever
        image is appropriate (see services.py's `_reconstruct_sf_session`
        for why that isn't simply `pass_images[pass_]` when `pass_ > 0` and
        prompt history can't be replayed) and pass the resulting state into
        this session directly.
        """
        self._pass = pass_
        self._pass_images = dict(pass_images)
        self.masks = list(masks)
        self.last_scrub = last_scrub
        self.scrub_history = [last_scrub] if last_scrub is not None else []
        self.wb_scrub_pass = wb_scrub_pass
        # `None` means "this caller has no panels to restore", which for a
        # session saved before panels existed is the same as "none were ever
        # authored" — both correctly land on an empty list rather than
        # leaving whatever this object happened to hold.
        self.panels = list(panels) if panels else []

    @property
    def working_copy(self) -> Optional[WorkingCopy]:
        """The current pass's scrubbed image, or None at pass 0 (nothing has
        been scrubbed yet). Non-raising, unlike `background`."""
        if self._pass == ORIGINAL_PASS:
            return None
        return WorkingCopy(self.image_for_pass(self._pass))

    @property
    def background(self) -> WorkingCopy:
        if self._pass == ORIGINAL_PASS:
            raise RuntimeError("run_lama_pass() has not been called yet; no scrubbed background exists.")
        return WorkingCopy(self.image_for_pass(self._pass))

    # -- selection -----------------------------------------------------------

    def _select(
        self,
        text_tag: Optional[str],
        source: str,
        apply: Callable[[dict], dict],
    ) -> list[MaskRecord]:
        """Apply one SAM3 call and reconcile its resulting masks against this
        pass's existing MaskRecords by geometry (IoU), not position or count.

        Neither `set_text_prompt` (replaces the prior text prompt's masks)
        nor re-grounding guarantees append-only growth or stable ordering —
        see aa_persistence.py's module docstring. So a post-call mask that
        still overlaps a pre-call MaskRecord keeps that record (identity,
        caption, held, dataset_status all intact) with its geometry
        refreshed; one with no match is a new selection; and a pre-call
        record with no post-call match was dropped by SAM3's re-ground and
        is removed — UNLESS it carries user-attached state (keep or held).

        That exception matters under v2: `set_text_prompt` replaces, so a
        second text prompt in a pass reports none of the first prompt's
        objects. Evicting on that would silently delete captioned keepers
        and silently drop objects from the pending scrub batch — exactly the
        "captioned work never reaches the training set" failure the design
        doc (§6) treats as the one to prevent. A kept/held record's geometry
        is still a valid mask on its own; it just stops being refreshed.

        Other passes' records are untouched throughout: `prior` and the
        carried-over list are exact complements (`==` vs `!=` on pass_).
        """
        prior = [m for m in self.masks if m.pass_ == self._pass]
        self.state = apply(self.state)
        self.last_refreshed_mask_ids = set()
        # `is None`, not `or []` — state["masks"]/["scores"] are MLX arrays
        # for the real Sam3Processor, and `X or []` forces a bool() on X
        # first; mx.array's __bool__ raises "[convert] Only length-1 arrays
        # can be converted to Python scalars" for anything but a single
        # element (including the empty case). Same guard services.py's
        # serialize_state already uses for this exact reason.
        raw_masks = self.state.get("masks")
        raw_masks = [] if raw_masks is None else raw_masks
        raw_scores = self.state.get("scores")
        raw_scores = [] if raw_scores is None else raw_scores
        geometries = [_binary_mask(m) for m in raw_masks]

        taken: set[str] = set()
        reconciled: list[MaskRecord] = []
        new_records: list[MaskRecord] = []
        for i, geometry in enumerate(geometries):
            match = _best_match(geometry, prior, taken)
            if match is not None:
                taken.add(match.mask_id)
                reconciled.append(replace(match, geometry=geometry))
                self.last_refreshed_mask_ids.add(match.mask_id)
            else:
                record = MaskRecord(
                    mask_id=f"{self._pass}:{uuid.uuid4()}",
                    pass_=self._pass,
                    geometry=geometry,
                    source=source,
                    text_tag=text_tag,
                    # np.array(...) first, not float() directly on the raw
                    # element — state["scores"] is MLX-backed for the real
                    # Sam3Processor, and mx.array's own __float__ raises
                    # "[convert] Only length-1 arrays can be converted to
                    # Python scalars" on some shapes numpy's wrapper handles
                    # fine. Same conversion services.py's serialize_state
                    # already uses for exactly this reason.
                    score=float(np.array(raw_scores[i])) if i < len(raw_scores) else None,
                )
                reconciled.append(record)
                new_records.append(record)

        protected = [m for m in prior if m.mask_id not in taken and _carries_user_state(m)]
        self.masks = [m for m in self.masks if m.pass_ != self._pass] + reconciled + protected
        return new_records

    def _select_single(
        self,
        text_tag: Optional[str],
        source: str,
        label: bool,
        apply: Callable[[dict], dict],
        pick: Callable[[list[np.ndarray], Any, Any, int, int], Optional[int]],
    ) -> list[MaskRecord]:
        """Apply one *geometric* SAM3 call and record at most ONE mask — the
        instance corresponding to the box/point the user actually drew.

        Box/point prompts are concept exemplars to SAM3 (see `_pick_for_box`),
        so `state["masks"]` still holds every generalised instance after the
        call. `pick` resolves which one the drawn prompt refers to; only that
        one becomes a selection. Text prompts deliberately keep the
        multi-instance behaviour and stay on `_select`.

        Unlike `_select`, this never evicts: one drawn prompt says nothing
        about selections made earlier in this pass, so prior records survive
        untouched even though they're absent from this call's single pick.
        """
        self.state = apply(self.state)
        self.last_refreshed_mask_ids = set()

        raw_masks = self.state.get("masks")
        raw_masks = [] if raw_masks is None else raw_masks
        raw_boxes = self.state.get("boxes")
        raw_boxes = [] if raw_boxes is None else raw_boxes
        raw_scores = self.state.get("scores")
        raw_scores = [] if raw_scores is None else raw_scores

        geometries = [_binary_mask(m) for m in raw_masks]

        # No standalone "geometries is empty -> return early" guard here
        # (v3 spec §3 Fix 2, second hijack path): that used to short-circuit
        # before `pick` ever ran, on a POSITIVE label leaving
        # last_touched_mask_id at whatever an EARLIER, unrelated call set
        # it to — the same stale-echo hijack Fix 2 closes for the
        # non-empty-but-no-match case, just reachable from a genuinely
        # empty grounding result instead. `pick` (both `_pick_for_box` and
        # `_pick_for_point`) already returns None for empty input, so
        # letting execution reach the `idx is None` branch below handles
        # it correctly and uniformly. The reconciliation loop and the
        # `label` check immediately below are both no-ops/pass-through on
        # an empty `geometries` list, so nothing meaningful was actually
        # being skipped by returning early here.

        # Reconcile EVERY raw instance against this pass's existing records
        # by geometry (IoU) — regardless of `label`. `add_geometric_prompt`
        # re-runs grounding over the whole accumulated prompt set on every
        # call (aa_persistence.py's module docstring), so an Avoid prompt
        # can refine an already-recorded object's boundary (e.g.
        # suppressing a neighbour that previously bled into it) even though
        # it creates no new selection of its own. Returning before this ran
        # for label=False left the refined geometry stranded in raw SAM3
        # output forever, with the recorded MaskRecord frozen pre-refinement
        # — confirmed live: an Avoid point shrank an already-selected
        # figure's own raw mask (2317px -> 2297px, IoU 0.987) while the
        # recorded copy stayed byte-identical. Same matching this module's
        # `_select` already uses for text prompts, applied here too: only
        # the geometry changes on a match (`replace(match, geometry=...)`),
        # so mask_id/dataset_status/held/caption all survive untouched.
        # `matched_mask_id_by_index` records which prior mask_id (if any)
        # each geometry index resolved to, so the later `pick(...)` step can
        # tell whether the drawn box/point's own instance was one of them.
        prior = [m for m in self.masks if m.pass_ == self._pass]
        taken: set[str] = set()
        refreshed: dict[str, MaskRecord] = {}
        matched_mask_id_by_index: dict[int, str] = {}
        for i, geometry in enumerate(geometries):
            match = _best_match(geometry, prior, taken)
            if match is not None:
                taken.add(match.mask_id)
                refreshed[match.mask_id] = replace(match, geometry=geometry)
                self.last_refreshed_mask_ids.add(match.mask_id)
                matched_mask_id_by_index[i] = match.mask_id
        if refreshed:
            self.masks = [refreshed.get(m.mask_id, m) for m in self.masks]

        # A negative ("Avoid") prompt is a grounding hint, not a selection —
        # it tells SAM3 what to exclude from the concept. The reconciliation
        # above already applied any resulting geometry refinement to
        # existing records; nothing new is logged, and focus (below) is
        # left wherever it was — an Avoid call doesn't by itself say which
        # object the user now means to focus on.
        if not label:
            return []

        width = self.state.get("original_width") or self.original.size[0]
        height = self.state.get("original_height") or self.original.size[1]
        idx = pick(geometries, raw_boxes, raw_scores, width, height)
        if idx is None:
            # `pick` found nothing genuinely corresponding to the drawn
            # prompt (v3 spec §3 Fix 2 — reachable for point via
            # `_pick_for_point`'s empty-`hits` case now that it no longer
            # falls back to a top-score guess). Explicitly clear focus
            # rather than leaving `last_touched_mask_id` pointing at
            # whatever an EARLIER, unrelated call last touched: left
            # alone, the endpoint would echo that stale id as this call's
            # own `selected_mask_id`, making a failed click indistinguishable
            # from a successful one that re-touched the same object.
            self.last_touched_mask_id = None
            return []

        # The drawn box/point's own instance was already reconciled above
        # (it refined an existing record) — not a new selection.
        if idx in matched_mask_id_by_index:
            self.last_touched_mask_id = matched_mask_id_by_index[idx]
            return []

        geometry = geometries[idx]
        score = float(np.array(raw_scores[idx])) if idx < len(raw_scores) else None

        record = MaskRecord(
            mask_id=f"{self._pass}:{uuid.uuid4()}",
            pass_=self._pass,
            geometry=geometry,
            source=source,
            text_tag=text_tag,
            score=score,
        )
        self.masks.append(record)
        self.last_touched_mask_id = record.mask_id
        return [record]

    def add_text_selection(self, prompt: str) -> list[MaskRecord]:
        return self._select(
            prompt, "text_prompt",
            lambda s: self.segmentor.set_text_prompt(prompt, s),
        )

    def add_box_selection(
        self, box: list[float], label: bool, text_substitute: Optional[str] = None,
        source: str = SOURCE_BOX,
    ) -> list[MaskRecord]:
        """A box selection. `source` exists so balloon detection can mark its
        own selections ([SOURCE_BALLOON]) while going through this exact same
        path — a detected balloon IS an ordinary box selection mechanically,
        and must stay one; only its provenance differs. See
        `run_lama_pass`'s `wb_scrub_pass` for what that provenance buys."""
        return self._select_single(
            text_substitute, source, label,
            lambda s: self.segmentor.add_geometric_prompt(box, label, s),
            lambda geoms, boxes, scores, w, h: _pick_for_box(box, boxes, scores, w, h),
        )

    def add_point_selection(
        self, point: list[float], label: bool, text_substitute: Optional[str] = None,
    ) -> list[MaskRecord]:
        return self._select_single(
            text_substitute, "point", label,
            lambda s: self.segmentor.add_point_prompt(point, label, s),
            lambda geoms, boxes, scores, w, h: _pick_for_point(point, geoms, scores, w, h),
        )

    # -- holding and captioning (independent of selection and each other) ----

    def get_mask(self, mask_id: str) -> MaskRecord:
        for m in self.masks:
            if m.mask_id == mask_id:
                return m
        raise UnknownMaskError(f"no mask {mask_id!r} in session {self.session_id!r}")

    def _update(self, mask_id: str, **changes: Any) -> MaskRecord:
        self.get_mask(mask_id)
        updated: Optional[MaskRecord] = None
        new_masks: list[MaskRecord] = []
        for m in self.masks:
            if m.mask_id == mask_id:
                updated = replace(m, **changes)
                new_masks.append(updated)
            else:
                new_masks.append(m)
        self.masks = new_masks
        assert updated is not None
        return updated

    def attach_caption(self, mask_id: str, caption: str) -> MaskRecord:
        """Caption an existing selection, which marks it `keep`.

        Works on a record from any pass, held or not — captioning has no
        bearing on scrub eligibility. An empty caption is refused: a `keep`
        record without a caption is exactly what the compile step filters
        out, so allowing one would make "kept" silently mean "dropped".
        """
        if not caption or not caption.strip():
            raise ValueError("caption must be non-empty; a keep record with no caption never reaches the training set")
        return self._update(mask_id, dataset_status=DatasetStatus.KEEP, caption=caption)

    # -- panels (v4 §4: layout regions, never objects) ------------------------

    def add_panel(self, x0: int, y0: int, x1: int, y1: int) -> PanelRecord:
        """Record a manually drawn panel boundary.

        Pure geometry: this never calls the segmentor, never creates a
        MaskRecord, and never touches `state`/`masks`/the pass chain. A panel
        is a layout region, so nothing about selection or scrubbing applies.

        Coordinates are clamped to the image, then validated: a box that is
        inverted, degenerate, or smaller than `MIN_PANEL_SIDE` on either side
        is refused rather than silently fixed up, because the only way to get
        one is a stray click or an off-canvas drag, and recording it would put
        an undeletable sliver in the panel list.
        """
        width, height = self.original.size
        cx0, cx1 = sorted((int(x0), int(x1)))
        cy0, cy1 = sorted((int(y0), int(y1)))
        cx0 = max(0, min(cx0, width))
        cx1 = max(0, min(cx1, width))
        cy0 = max(0, min(cy0, height))
        cy1 = max(0, min(cy1, height))
        if cx1 - cx0 < MIN_PANEL_SIDE or cy1 - cy0 < MIN_PANEL_SIDE:
            raise ValueError(
                f"panel must be at least {MIN_PANEL_SIDE}x{MIN_PANEL_SIDE}px "
                f"after clamping to the {width}x{height} image; got "
                f"{cx1 - cx0}x{cy1 - cy0}"
            )
        panel = PanelRecord(panel_id=str(uuid.uuid4()), x0=cx0, y0=cy0, x1=cx1, y1=cy1)
        self.panels.append(panel)
        return panel

    def delete_panel(self, panel_id: str) -> PanelRecord:
        """Remove a panel by id. Raises `UnknownPanelError` if absent, rather
        than succeeding silently — the caller is a UI delete button, and a
        no-op that reports success would leave a row on screen."""
        for i, panel in enumerate(self.panels):
            if panel.panel_id == panel_id:
                return self.panels.pop(i)
        raise UnknownPanelError(f"no panel with id {panel_id!r} in this session")

    def set_held(self, mask_id: str, held: bool) -> MaskRecord:
        """Queue (or un-queue) a selection for the current pass's next scrub.

        Holding is refused for a record outside the current pass: only the
        current pass's held set feeds the next scrub, so an earlier pass's
        record would sit held forever and never be scrubbed. To scrub an
        object revealed earlier, select it again in the current pass.
        Un-holding is always allowed.
        """
        record = self.get_mask(mask_id)
        if held and record.pass_ != self._pass:
            raise ValueError(
                f"cannot hold {mask_id!r}: it belongs to pass {record.pass_}, "
                f"and only pass {self._pass} is eligible for the next scrub"
            )
        return self._update(mask_id, held=held)

    def remove_mask(self, mask_id: str) -> MaskRecord:
        """Delete a selection made in the CURRENT pass — the "I masked that
        by mistake" action. Returns the removed record.

        Current pass only: an earlier pass's record is already part of that
        pass's history (it may have been scrubbed or exported), so removing
        it is a different, riskier action and is refused rather than done
        quietly. A captioned record can be removed; removing it is the
        user's explicit act and it simply stops being a keeper.

        Pure bookkeeping on `masks`: nothing is inpainted and the pass chain
        is untouched. SAM3's own grounding state still contains the object,
        so selecting it again (or running a text prompt that finds it)
        legitimately brings it back as a fresh record.
        """
        record = self.get_mask(mask_id)
        if record.pass_ != self._pass:
            raise ValueError(
                f"cannot remove {mask_id!r}: it belongs to pass {record.pass_}, "
                f"and only pass {self._pass} selections can be removed"
            )
        self.masks = [m for m in self.masks if m.mask_id != mask_id]
        if self.last_touched_mask_id == mask_id:
            self.last_touched_mask_id = None
        return record

    def undo_last_scrub(self) -> ScrubRecord:
        """Step back one scrub: the pass chain, the held flags and the
        Display Layers WB layer return to how they were just before it.

        What comes back: the previous pass becomes current, every mask that
        was held for that scrub is held again, and `wb_scrub_pass` is what it
        was. What goes away: the scrubbed image, and any selections made in
        the pass the scrub created (they were traced against pixels that no
        longer exist).

        REFUSED, not forced, when a selection in the pass being discarded
        carries a caption: undoing would silently destroy captioned work.
        The caller is told how many; remove or finish those first.

        Every input is computed before the session is mutated, so if the
        segmentor's `set_image` raises, nothing has changed.
        """
        if not self.scrub_history:
            raise CannotUndoScrubError("nothing to undo: no scrub has been done in this session")
        rec = self.scrub_history[-1]
        if rec.to_pass != self._pass:
            raise CannotUndoScrubError(
                f"cannot undo: the last scrub led to pass {rec.to_pass}, but the session is at pass {self._pass}"
            )
        kept = [m for m in self.masks if m.pass_ == rec.to_pass and m.dataset_status == DatasetStatus.KEEP]
        if kept:
            raise CannotUndoScrubError(
                f"cannot undo: {len(kept)} captioned selection(s) were made after that scrub "
                f"and would be lost. Remove them first, or keep working from this pass."
            )

        previous_image = self.image_for_pass(rec.from_pass)
        new_state = self.segmentor.set_image(previous_image)

        held_again = set(rec.mask_ids)
        self.masks = [
            replace(m, held=True) if m.mask_id in held_again else m
            for m in self.masks
            if m.pass_ != rec.to_pass
        ]
        self._pass_images.pop(rec.to_pass, None)
        self.state = new_state
        self._pass = rec.from_pass
        self.scrub_history.pop()
        self.last_scrub = self.scrub_history[-1] if self.scrub_history else None
        self.wb_scrub_pass = rec.prev_wb_scrub_pass
        self.last_touched_mask_id = None
        return rec

    def discard_ungrounded_selections(self) -> None:
        """Reset-time cleanup for the CURRENT pass: discard selections that
        exist only because SAM3's now-cleared prompt state was backing
        them, while preserving anything carrying user state (kept or
        held) — the same rule `_select`'s own reconciliation uses to
        decide what a re-ground is allowed to evict (`_carries_user_state`).

        `/reset` clears SAM3's grounding, which invalidates any
        UNCOMMITTED selection — nothing else backs it. It must not also
        discard already-captioned/held work just because the live
        grounding state moved on: those two are different things, and
        conflating them made kept/held masks vanish from the canvas (and
        from `serialize_sf_masks`'s response) the moment a user reset
        prompts, even though the work itself was never touched. Other
        passes are untouched, matching `_select`'s own eviction scope.
        """
        self.masks = [
            m for m in self.masks
            if m.pass_ != self._pass or _carries_user_state(m)
        ]

    # -- scrubbing ------------------------------------------------------------

    def run_lama_pass(self, inpainter: InpaintingEngine) -> WorkingCopy:
        """Scrub the current pass's held set and advance to the next pass.

        Unions the geometry of every record held in the CURRENT pass —
        regardless of dataset_status, since captioning has no bearing on
        whether something is scrubbed — inpaints the CURRENT pass's image
        with it, and makes the result pass N+1. Held is cleared on every
        record in the batch; nothing else about them changes.

        Repeatable without limit. Each scrub builds on the previous pass's
        image, not the original, so peels accumulate.

        Every input is computed before anything on the session is mutated,
        so if inpainting or the segmentor's `set_image` raises, the session
        is left exactly as it was.

        An empty held set still advances the pass (LaMa isn't called; the
        new pass's image is a copy of the current one) — preserved from the
        pre-v2 behaviour rather than newly guarded.
        """
        from_pass = self._pass
        to_pass = from_pass + 1
        batch = [m for m in self.masks if m.pass_ == from_pass and m.held]

        base = self.image_for_pass(from_pass)
        scrubbed = inpainter.inpaint(base, _union_mask([m.geometry for m in batch])) if batch else base
        new_state = self.segmentor.set_image(scrubbed)

        batch_ids = {m.mask_id for m in batch}
        self._pass_images[to_pass] = scrubbed
        self.masks = [replace(m, held=False) if m.mask_id in batch_ids else m for m in self.masks]
        self.state = new_state
        self._pass = to_pass
        self.last_scrub = ScrubRecord(
            from_pass, to_pass, tuple(m.mask_id for m in batch),
            prev_wb_scrub_pass=self.wb_scrub_pass,
        )
        self.scrub_history.append(self.last_scrub)
        # Derived from what was actually scrubbed, not from a caller's claim
        # or from the pass number: a batch containing balloon-detected masks
        # makes the resulting pass the WB-scrub result, wherever it lands in
        # the chain. An empty batch (a zero-balloon page) contains no balloon
        # masks, so this correctly stays as it was.
        if any(m.source == SOURCE_BALLOON for m in batch):
            self.wb_scrub_pass = to_pass
        return WorkingCopy(self.image_for_pass(to_pass))


def assign_panel(
    mask_bbox: Sequence[float], panels: Sequence[PanelRecord]
) -> Optional[PanelRecord]:
    """Which panel a mask belongs to — the one holding the largest share of
    `mask_bbox`'s area. Pure; no session, no I/O.

    `mask_bbox` is `[x0, y0, x1, y1]` in page pixels, the same shape
    `_bbox_from_geometry` returns.

    Rules, in order:
      * No panel overlaps the bbox at all -> None. The caller then uses the
        full page; this function does NOT invent a fallback panel, so that
        "fell back" stays distinguishable from "genuinely assigned".
      * Otherwise the panel containing the largest absolute overlap area
        wins — a mask straddling two panels goes to whichever holds more of
        it, which is the only answer that does not need a tie-break policy
        for the common case.
      * NESTED panels are the exception that needs its own rule: an inset
        sitting inside a host panel overlaps the bbox by the same area the
        host does (the inset's whole area, when the mask is inside the
        inset), so "largest overlap" alone would tie, and resolving that tie
        by panel area would pick the HOST — exactly backwards. Among
        panels whose overlap is tied at the maximum, the SMALLEST panel
        wins, matching the innermost-frame rule the panel-bounds validation
        settled on.
      * A degenerate bbox (zero width or height) has no area to share out,
        so it cannot have a "largest share" — it falls back to containment:
        the smallest panel that contains it, else None.

    Ties beyond nesting (two equally-sized panels with equal overlap) resolve
    to the first in `panels`, i.e. authoring order — deterministic rather
    than arbitrary.
    """
    if not panels:
        return None
    mx0, my0, mx1, my1 = (float(v) for v in mask_bbox)
    if mx1 < mx0:
        mx0, mx1 = mx1, mx0
    if my1 < my0:
        my0, my1 = my1, my0

    degenerate = (mx1 - mx0) <= 0 or (my1 - my0) <= 0

    best: Optional[PanelRecord] = None
    best_overlap = 0.0
    for panel in panels:
        ox = max(0.0, min(mx1, panel.x1) - max(mx0, panel.x0))
        oy = max(0.0, min(my1, panel.y1) - max(my0, panel.y0))
        if degenerate:
            # No area to compare; containment is the only meaningful test.
            # Use >= so a point exactly on a panel's edge still counts as in.
            if not (panel.x0 <= mx0 <= panel.x1 and panel.y0 <= my0 <= panel.y1):
                continue
            if best is None or panel.area < best.area:
                best = panel
            continue
        overlap = ox * oy
        if overlap <= 0:
            continue
        if overlap > best_overlap:
            best, best_overlap = panel, overlap
        elif overlap == best_overlap and best is not None and panel.area < best.area:
            # The nesting case: same overlap, so the inner frame wins.
            best = panel
    return best


def panel_crop_png_bytes(image: Image.Image, box: Sequence[int]) -> bytes:
    """A panel's rectangular cutout of `image`, as RGBA PNG bytes.

    Pure; the caller chooses which pass image to hand in (see
    `build_sf_payload`, which uses each mask's OWN pass image). Unlike
    `_crop_png_bytes` there is no alpha-masking step: a panel is a
    rectangular layout region, not a segmented object, so the crop is the
    rectangle itself.

    The box is clamped to the image before cropping, so a panel recorded
    against one pass cannot raise on another; an empty intersection yields a
    1x1 transparent PNG rather than a zero-size image PIL refuses to encode.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = image.size
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return _png_bytes(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    return _png_bytes(image.convert("RGBA").crop((x0, y0, x1, y1)))


# ── LLM export (v1) ───────────────────────────────────────────────────────
#
# A read-only, separate export path alongside build_sf_payload — NOT a
# replacement for it. build_sf_payload's own output/tests are unchanged;
# this exists because an LLM vision submission has different needs than the
# training-set export: a shared macro-context page (Artifact 1), a panel
# crop cut from THAT page rather than the mask's own pass (Artifact 2), and
# a segment crop that is TIGHT (not full-canvas) with no alpha channel at
# all (Artifact 3) — see the three helpers and build_llm_export_payload
# below for why each of those differs from what build_sf_payload already
# does.

# "wb_scrub" when Artifact 1 is the pass a balloon scrub actually produced;
# "original" when no balloon scrub ever ran (wb_scrub_pass is None) and
# Artifact 1 falls back to pass 0. A legacy session saved before
# wb_scrub_pass existed at all also reads back None — this makes that case
# visible as "original" rather than silently indistinguishable from a page
# that genuinely has no balloons.
PAGE_RULE_WB_SCRUB = "wb_scrub"
PAGE_RULE_ORIGINAL = "original"

# JPEG quality for every LLM-export artifact that uses JPEG (the macro-
# context page and panel crops). 85 is the standard "visually lossless for
# this purpose" choice; measured on a real page during the investigation
# that designed this contract, JPEG 85 ran ~14x smaller than the equivalent
# PNG with no visible loss in a vision-API context.
LLM_EXPORT_JPEG_QUALITY = 85

# Artifact 1 is downscaled to this long side before being written to disk.
# Sending more buys no extra fidelity: a vision API downscales further on
# its own regardless of what is sent (OpenAI's own pipeline fits to a 2048
# square, then caps the shortest side to 768), so anything past this is
# pure upload cost.
ARTIFACT1_MAX_LONG_SIDE = 1536

# Composited background for a segment's tight crop (Artifact 3), wherever
# the mask does not cover a pixel inside its own bounding box. A measured
# choice, not a default: across 13 real segments from a real session, white
# collided with 21.4% of subject pixels (luminance > 210), black with 4.6%
# (luminance < 45) but is indistinguishable from the OTHER failure mode this
# guards against — silently dropping the alpha channel turns a transparent
# pixel black (`Image.convert("RGB")` on an RGBA image, confirmed) — while
# mid-grey collided with only 8.1% and cannot be mistaken for that bug.
_SEGMENT_CROP_BACKGROUND = (128, 128, 128)


def resolve_artifact1_pass(wb_scrub_pass: Optional[int]) -> tuple[int, str]:
    """Which pass is Artifact 1, and why — `(page_pass, page_rule)`.

    Pure arithmetic, independent of any session object, so the pass/rule
    pairing for None/1/2/... can be pinned in isolation from pixel work.
    """
    if wb_scrub_pass is None:
        return ORIGINAL_PASS, PAGE_RULE_ORIGINAL
    return wb_scrub_pass, PAGE_RULE_WB_SCRUB


def segment_crop_png_bytes(image: Image.Image, mask: np.ndarray) -> bytes:
    """A mask's TIGHT crop (Artifact 3), composited onto opaque mid-grey
    outside the mask. PNG, RGB — no alpha channel.

    Deliberately NOT `_crop_png_bytes` (full-canvas RGBA, transparent
    outside the mask) — that function and `build_sf_payload`'s use of it
    are unchanged; this is a new, separate helper for a different need:

      * TIGHT, not full-canvas: a vision API downscales the whole image it
        is sent to a fixed budget regardless of content, so a small object
        sent on a full page can arrive smaller than its own native pixels
        (measured: a 97x70 balloon arrived at 43x31 inside a 1740x2376 page
        sent through a vision API's own resize pipeline).
      * NO alpha, ever: JPEG cannot encode RGBA at all
        (`OSError: cannot write mode RGBA as JPEG`, confirmed) and a PNG's
        alpha is not safe to rely on downstream either — any later
        `.convert("RGB")` silently turns a transparent pixel BLACK, which a
        composited grey background can never be confused with. Still PNG
        rather than JPEG: unlike the already-huge macro-context page, the
        object itself should not lose fidelity to lossy compression.

    An all-zero mask (nothing set) has no bounding box to crop to and
    returns a 1x1 grey pixel — the same "nothing here" shape
    `panel_crop_png_bytes` returns for a zero-area box, not a crash.
    """
    binary = _binary_mask(mask)
    mask_image = Image.fromarray(binary * 255, mode="L")
    composed = Image.new("RGB", image.size, _SEGMENT_CROP_BACKGROUND)
    composed.paste(image.convert("RGB"), (0, 0), mask_image)
    x0, y0, x1, y1 = _geometry_bbox(mask)
    if x1 <= x0 or y1 <= y0:
        return _png_bytes(Image.new("RGB", (1, 1), _SEGMENT_CROP_BACKGROUND))
    return _png_bytes(composed.crop((x0, y0, x1, y1)))


def _jpeg_bytes(image: Image.Image, quality: int = LLM_EXPORT_JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _downscaled_to_long_side(image: Image.Image, max_long_side: int) -> Image.Image:
    """`image`, resized so its longer side is at most `max_long_side` —
    unchanged (same object, not a copy) if it already fits. Aspect ratio
    preserved."""
    w, h = image.size
    long_side = max(w, h)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / long_side
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _panel_crop_jpeg_bytes(
    image: Image.Image, box: Sequence[int], quality: int = LLM_EXPORT_JPEG_QUALITY,
) -> bytes:
    """A panel's rectangular JPEG cutout of `image` — the LLM-export sibling
    of `panel_crop_png_bytes`, differing in exactly the two ways this
    contract needs: JPEG/RGB (a vision-API message part, not an archival
    RGBA PNG) and always cut from Artifact 1's page — the caller passes
    THAT image in, never a mask's own pass image.

    Same clamp-before-crop defensiveness as `panel_crop_png_bytes`,
    duplicated rather than shared: the two encode differently (JPEG has no
    alpha to begin with, so there is no transparent-fallback case to share),
    and this module already duplicates small pass-image crop logic across
    its export paths rather than add a shared dependency neither existing
    path asked for (see `_crop_png_bytes`'s own docstring).
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = image.size
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return _jpeg_bytes(Image.new("RGB", (1, 1), _SEGMENT_CROP_BACKGROUND), quality=quality)
    return _jpeg_bytes(image.crop((x0, y0, x1, y1)), quality=quality)


def build_llm_export_payload(session: SFSession, include: str = "keep") -> dict:
    """Assemble the read-only LLM-submission payload (v1) for `session` —
    pure, in-memory; no file is written here (the caller, services.py's
    `SegmentationService.export_llm_payload`, writes Artifact 1 to its
    stable path and base64-encodes everything for the wire).

    `include` selects which masks are rows at all:
      * `"keep"` (default) — `dataset_status == KEEP`, the exact filter
        `build_sf_payload` already uses; this export's default is meant to
        agree with the training-set export, not introduce a second notion
        of "kept".
      * `"held_or_keep"` — also includes anything queued for the next
        scrub but not yet captioned.
    Any other value raises `ValueError` (the caller turns that into a 422,
    same convention as `SFSession.add_panel`'s own validation).

    Returns a dict of RAW bytes (not base64, not JSON-shaped) — see
    `services.export_llm_payload` for the wire-shape step:
      ``{"page_pass", "page_rule", "width", "height", "page_jpeg_bytes",
      "panels": [...], "segments": [...]}``.

    PANEL DEDUPLICATION: a panel row is produced at most once per distinct
    `panel_id` actually assigned to an INCLUDED segment — not once per
    `session.panels` regardless of use. A page can carry panels nothing
    captioned ever falls inside; cropping and encoding those would be pure
    waste for a payload whose whole point is "what this LLM call needs".

    PASS SOURCING, the one place this genuinely diverges from
    `build_sf_payload`'s panel-crop behaviour: Artifact 2 (`panel_crop`) is
    always cut from Artifact 1's page, the session's shared macro-context
    image — never from the assigned mask's own pass, unlike
    `build_sf_payload`'s `panel_crop_png_bytes` call, which deliberately
    uses each mask's own pass for ITS purpose (consistency with that mask's
    own object crop in a training sample). Here the contract is different:
    every segment in one submission shares ONE page, so its panel context
    must come from that same page too, not drift per mask. Artifact 3
    (`segment_crop`) is the one still sourced from the mask's own pass
    image, via the existing `image_for_pass` — pass-aware exactly like
    `build_sf_payload`.
    """
    if include not in ("keep", "held_or_keep"):
        raise ValueError(f"include must be 'keep' or 'held_or_keep', got {include!r}")

    page_pass, page_rule = resolve_artifact1_pass(session.wb_scrub_pass)
    page_image = session.image_for_pass(page_pass).convert("RGB")
    width, height = page_image.size
    page_jpeg_bytes = _jpeg_bytes(_downscaled_to_long_side(page_image, ARTIFACT1_MAX_LONG_SIDE))

    def included(mask: MaskRecord) -> bool:
        if mask.dataset_status is DatasetStatus.KEEP:
            return True
        return include == "held_or_keep" and mask.held

    # Decoded once per pass actually touched by an included segment, not
    # once per mask — same reuse-across-rows shape as build_sf_payload's own
    # pass_images cache, just built lazily since which passes are needed
    # depends on which masks `included()` keeps.
    pass_images: dict[int, Image.Image] = {}

    def image_for(pass_: int) -> Image.Image:
        if pass_ not in pass_images:
            pass_images[pass_] = session.image_for_pass(pass_).convert("RGBA")
        return pass_images[pass_]

    segments: list[dict] = []
    panel_crops: dict[str, dict] = {}  # panel_id -> {"bbox", "crop_jpeg_bytes"}

    for mask in session.masks:
        if not included(mask):
            continue
        bbox = _geometry_bbox(mask.geometry)
        crop_png_bytes = segment_crop_png_bytes(image_for(mask.pass_), mask.geometry)

        panel = assign_panel(bbox, session.panels)
        panel_id = panel.panel_id if panel is not None else None
        if panel is not None and panel_id not in panel_crops:
            panel_crops[panel_id] = {
                "bbox": panel.as_box(),
                "crop_jpeg_bytes": _panel_crop_jpeg_bytes(page_image, panel.as_box()),
            }

        segments.append({
            "mask_id": mask.mask_id,
            "caption": mask.caption or "",
            "pass": mask.pass_,
            "bbox": bbox,
            "panel_id": panel_id,
            "crop_png_bytes": crop_png_bytes,
            "pre_wb": session.wb_scrub_pass is not None and mask.pass_ < session.wb_scrub_pass,
        })

    panels = [
        {"panel_id": pid, "bbox": v["bbox"], "crop_jpeg_bytes": v["crop_jpeg_bytes"]}
        for pid, v in panel_crops.items()
    ]

    return {
        "page_pass": page_pass,
        "page_rule": page_rule,
        "width": width,
        "height": height,
        "page_jpeg_bytes": page_jpeg_bytes,
        "panels": panels,
        "segments": segments,
    }


def build_sf_payload(session: SFSession) -> tuple[list[str], list[str], list]:
    """Serialize an `SFSession`'s kept work into the SF-AA export payload.

    Only `dataset_status == keep` records are exported. Discard is implicit:
    anything held and scrubbed without ever being captioned just isn't here.

    Returns D4M.jl-round-tripped parallel (rows, cols, vals) triples — hand
    to `d4m_juliacall_bridge.save_parquet` to persist, or transport as-is.

    Row keys are prefixed with `session_id` — a bare `"context"` row, or a
    bare `mask_id`, would collide across every other session's payload the
    moment two are ever combined via `⊕`, silently mixing one session's
    image bytes into another's row. `mask_id` already embeds pass
    (`f"{pass}:{uuid4()}"`), so this doesn't re-prefix pass separately.

    Values are strings throughout (the Parquet column convention), so `pass`
    is written as a numeric string. Absent values are omitted rather than
    written empty: `background_image_bytes` at pass 0 (nothing scrubbed yet)
    and `text_tag` on a box/point selected with no text_substitute.

    `crop_png_bytes` is the segmented object itself — `original_image`
    (or the relevant pass's image; see below) cut out via the mask as an
    alpha channel — not mask geometry. Real training/handoff data needs to
    see the object as it actually looks, not a binary black/white silhouette.
    """
    context_row = f"{session.session_id}:context"
    rows: list[str] = [context_row]
    cols: list[str] = ["original_image_bytes"]
    vals: list = [_b64(session.original.bytes)]
    if session.pass_ != ORIGINAL_PASS:
        rows.append(context_row)
        cols.append("background_image_bytes")
        vals.append(_b64(_png_bytes(session.image_for_pass(session.pass_))))

    # Decoded once per pass and reused across every mask row below, not
    # re-decoded per mask — matches aa_persistence.py's own
    # _build_segments_from_masks. Keyed by pass, not just pass 0: a mask
    # kept from an earlier pass has to be cropped from THAT pass's image,
    # since a later scrub can repaint pixels a kept mask's own crop must
    # not pick up (session.image_for_pass already retains every pass's
    # image for exactly this reason — see its own docstring).
    pass_images: dict[int, Image.Image] = {
        ORIGINAL_PASS: Image.open(io.BytesIO(session.original.bytes)).convert("RGBA"),
    }
    for p in range(1, session.pass_ + 1):
        pass_images[p] = session.image_for_pass(p).convert("RGBA")

    for mask in session.masks:
        if mask.dataset_status != DatasetStatus.KEEP:
            continue
        base_image = pass_images.get(mask.pass_, pass_images[ORIGINAL_PASS])
        row_key = f"{session.session_id}:{mask.mask_id}"
        rows += [row_key, row_key, row_key]
        cols += ["pass", "caption", "crop_png_bytes"]
        vals += [str(mask.pass_), mask.caption, _b64(_crop_png_bytes(base_image, mask.geometry))]
        if mask.text_tag:
            rows.append(row_key)
            cols.append("text_tag")
            vals.append(mask.text_tag)

        # The panel this mask sits in (v4 §4): pure layout context for the
        # LoRA, carrying no caption of its own. Cropped from `base_image` —
        # the mask's OWN pass image, the same one its object crop above
        # comes from, never the session's current pass: a keeper captioned
        # in pass 1 of a session now at pass 4 must show the page as it
        # looked when that object was there, and a later scrub may have
        # repainted the very pixels its panel context depends on.
        #
        # `panel_source` is always written, never omitted, because the
        # difference between "this is panel 3" and "no panel covered this
        # mask, so here is the whole page" is not recoverable from the crop
        # bytes themselves — a full-page fallback and a page-sized panel
        # produce identical pixels. A consumer that silently treated the
        # fallback as a real panel would be training on the wrong claim.
        panel = assign_panel(_geometry_bbox(mask.geometry), session.panels)
        if panel is None:
            panel_box = (0, 0, base_image.width, base_image.height)
            panel_source = PANEL_SOURCE_FULL_PAGE
        else:
            panel_box = panel.as_box()
            panel_source = panel.panel_id
        rows += [row_key, row_key]
        cols += ["panel_crop_png_bytes", "panel_source"]
        vals += [_b64(panel_crop_png_bytes(base_image, panel_box)), panel_source]

    return bridge.build_triples(rows, cols, vals)
