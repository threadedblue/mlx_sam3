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
from PIL import Image

import d4m_juliacall_bridge as bridge

ORIGINAL_PASS = 0


class DatasetStatus(str, Enum):
    UNASSIGNED = "unassigned"
    KEEP = "keep"


class UnknownMaskError(KeyError):
    """No MaskRecord with that mask_id exists in this session."""


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
class ScrubRecord:
    """Audit trail for the most recent scrub — enough for LBSCard's "Last
    scrub: Pass N → Pass N+1" line, and for callers that need the batch's
    mask_ids after `held` has already been cleared on them."""
    from_pass: int
    to_pass: int
    mask_ids: tuple[str, ...]


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


class LamaInpainter:
    """LaMa (Large Mask Inpainting via Fast Fourier Convolutions).

    Loads its checkpoint lazily on first construction rather than at module
    import — `simple-lama-inpainting` pulls a multi-hundred-MB torchscript
    checkpoint on first use, which importing this module shouldn't trigger
    on its own.
    """

    def __init__(self) -> None:
        from simple_lama_inpainting import SimpleLama  # noqa: PLC0415

        self._lama = SimpleLama()

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        result = self._lama(image.convert("RGB"), mask_image)
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


def _mask_png_bytes(mask: np.ndarray) -> bytes:
    return _png_bytes(Image.fromarray(mask * 255, mode="L"))


def _norm_cxcywh_to_pixel_xyxy(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, w, h = box
    return ((cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height)


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
    """Index of the returned instance the user actually clicked inside."""
    if not geometries:
        return None
    x, y = point
    px = min(max(int(round(x * width)), 0), width - 1)
    py = min(max(int(round(y * height)), 0), height - 1)
    hits = [i for i, g in enumerate(geometries) if g.shape[0] > py and g.shape[1] > px and g[py, px]]
    if not hits:
        return _argmax_score(scores)
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
        self._pass = ORIGINAL_PASS
        # Scrubbed image per pass >= 1. Pass 0's image is `original`. Every
        # pass is kept, not just the latest: a record's crop has to come from
        # the image it was traced against, and records from any earlier pass
        # (e.g. keepers captioned in pass 1 of a session now at pass 4) are
        # still exported.
        self._pass_images: dict[int, Image.Image] = {}
        self.last_scrub: Optional[ScrubRecord] = None
        # The mask id a box/point call most recently matched-or-created —
        # set by `_select_single` only (text prompts are multi-instance, so
        # there's no single "the" mask to point at). `serialize_sf_masks`'s
        # response is the full current-pass list, not a diff, so this is
        # the caller's only way to know which one THIS call was about —
        # needed to route the Prompt card's caption-mode call at the right
        # mask_id after a box/point selection.
        self.last_touched_mask_id: Optional[str] = None

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
    ) -> None:
        """Reconstruct state from persistence in one call, rather than the
        caller reaching into `_pass`/`_pass_images`/`masks` directly.

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

        # A negative ("Avoid") prompt is a grounding hint, not a selection —
        # it tells SAM3 what to exclude from the concept. Recording a mask
        # for it would mint a phantom selection over a region the user was
        # explicitly ruling out, so the state updates but nothing is logged.
        if not label:
            return []

        raw_masks = self.state.get("masks")
        raw_masks = [] if raw_masks is None else raw_masks
        raw_boxes = self.state.get("boxes")
        raw_boxes = [] if raw_boxes is None else raw_boxes
        raw_scores = self.state.get("scores")
        raw_scores = [] if raw_scores is None else raw_scores

        geometries = [_binary_mask(m) for m in raw_masks]
        if not geometries:
            return []

        width = self.state.get("original_width") or self.original.size[0]
        height = self.state.get("original_height") or self.original.size[1]
        idx = pick(geometries, raw_boxes, raw_scores, width, height)
        if idx is None:
            return []

        geometry = geometries[idx]
        score = float(np.array(raw_scores[idx])) if idx < len(raw_scores) else None

        # Re-selecting an object already recorded in this pass refreshes its
        # geometry rather than minting a duplicate record.
        prior = [m for m in self.masks if m.pass_ == self._pass]
        match = _best_match(geometry, prior, set())
        if match is not None:
            self.masks = [
                replace(m, geometry=geometry) if m.mask_id == match.mask_id else m
                for m in self.masks
            ]
            self.last_touched_mask_id = match.mask_id
            return []

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
    ) -> list[MaskRecord]:
        return self._select_single(
            text_substitute, "box", label,
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
        self.last_scrub = ScrubRecord(from_pass, to_pass, tuple(m.mask_id for m in batch))
        return WorkingCopy(self.image_for_pass(to_pass))


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
    """
    context_row = f"{session.session_id}:context"
    rows: list[str] = [context_row]
    cols: list[str] = ["original_image_bytes"]
    vals: list = [_b64(session.original.bytes)]
    if session.pass_ != ORIGINAL_PASS:
        rows.append(context_row)
        cols.append("background_image_bytes")
        vals.append(_b64(_png_bytes(session.image_for_pass(session.pass_))))

    for mask in session.masks:
        if mask.dataset_status != DatasetStatus.KEEP:
            continue
        row_key = f"{session.session_id}:{mask.mask_id}"
        rows += [row_key, row_key, row_key]
        cols += ["pass", "caption", "mask_png_bytes"]
        vals += [str(mask.pass_), mask.caption, _b64(_mask_png_bytes(mask.geometry))]
        if mask.text_tag:
            rows.append(row_key)
            cols.append("text_tag")
            vals.append(mask.text_tag)

    return bridge.build_triples(rows, cols, vals)
