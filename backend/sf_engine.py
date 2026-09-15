"""SF (Segmentation & Inpainting) engine: multi-pass mask state machine,
LaMa background scrub, and export-AA serialization.

Three stages, matching the pipeline this module implements:

1. **Routing** (`SFSession.add_*_selection`) — an open-vocabulary text prompt
   goes straight to SAM3 for its object boundary mask; a point/box selection
   is bound to a caller-supplied text-substitute string as that region's
   metadata tag. Every selection is classified Green (`MaskType.IN`) or Red
   (`MaskType.OUT`) by the caller, independently of SAM3's own positive/
   negative grounding `label`.
2. **Buffer isolation + LaMa** (`SFSession.run_lama_pass`) — Pass 1's Red
   out-masks are unioned and scrubbed via LaMa on a fresh `WorkingCopy`;
   `ImmutableOriginal` is never touched. This is a one-shot transition into
   Pass 2 (peeled background), where subsequent selections run against the
   scrubbed image instead of the original.
3. **Export serialization** (`build_sf_payload`) — bundles the original
   reference, the scrubbed background, and every Green in-mask (both
   passes) into a D4M.jl-backed AA payload via `d4m_juliacall_bridge`, never
   a hand-rolled triples reimplementation. Red out-masks are consumed by
   stage 2 and don't appear in the export — SF's job stops at serialization;
   prompt polishing/enhancement is a downstream DN node's job.

AA design notes (see the associative-arrays skill / aa-spec.pdf for why):
- Mask row keys are minted as `f"{pass}:{uuid4()}"`, unique across both
  passes — d4m_juliacall_bridge.build_triples combines duplicate (row, col)
  pairs via D4M.jl's default operator, which would silently merge two
  distinct masks if IDs collided.
- Pass 1 vs. Pass 2 is a single typed `pass` column (string value per row),
  not one-hot columns per pass. A one-hot scheme would leave a fully-empty
  column on any session where one pass finds zero masks — a no-empty-column
  invariant violation the typed form can't hit, since every mask row has
  exactly one value in `pass` regardless of how counts split.
"""

from __future__ import annotations

import base64
import io
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional, Protocol, Sequence

import numpy as np
from PIL import Image

import d4m_juliacall_bridge as bridge


class MaskType(str, Enum):
    IN = "in"    # Green — inclusion
    OUT = "out"  # Red — exclusion / obscurator


class Pass(str, Enum):
    FOREGROUND = "foreground"  # Pass 1
    BACKGROUND = "background"  # Pass 2, after LaMa peel


@dataclass(frozen=True)
class MaskRecord:
    mask_id: str
    mask_type: MaskType
    pass_: Pass
    geometry: np.ndarray  # binary mask, shape (H, W)
    text_tag: str         # SAM3 prompt, or caller-supplied text substitute
    source: str           # "text_prompt" | "point" | "box"
    score: Optional[float] = None  # SAM3's confidence for this mask, if reported

    @property
    def render_style(self) -> str:
        color = "green" if self.mask_type is MaskType.IN else "red"
        return f"{color}-diagonal-hatch"


class ImmutableOriginal:
    """Overlay 1 / $I_{orig}$. Never mutated; every op works on a copy."""

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
    """An isolated, mutable canvas buffer derived from `ImmutableOriginal`."""

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


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _mask_png_bytes(mask: np.ndarray) -> bytes:
    return _png_bytes(Image.fromarray(mask * 255, mode="L"))


class SFSession:
    """Multi-pass segmentation + inpainting state machine for one SF session.

    `state` is the SAM3 processor state already produced by the caller's own
    `segmentor.set_image(...)` — image loading isn't this module's concern,
    only what happens to selections made against it.
    """

    def __init__(self, session_id: str, original: ImmutableOriginal, segmentor: Sam3Like, state: dict):
        self.session_id = session_id
        self.original = original
        self.segmentor = segmentor
        self.state = state
        self.masks: list[MaskRecord] = []
        self._pass = Pass.FOREGROUND
        self._working_copy: Optional[WorkingCopy] = None

    @property
    def pass_(self) -> Pass:
        return self._pass

    @property
    def working_copy(self) -> Optional[WorkingCopy]:
        """The Pass 2 scrubbed background, or None if run_lama_pass() hasn't run yet.

        Non-raising, unlike `background` below — for callers (e.g.
        persistence) that need to check without it being a hard precondition.
        """
        return self._working_copy

    def _select(
        self,
        mask_type: MaskType,
        text_tag: str,
        source: str,
        apply: Callable[[dict], dict],
    ) -> list[MaskRecord]:
        """Apply one SAM3 call and reconcile its resulting masks against this
        pass's existing MaskRecords by geometry (IoU), not position or count.

        Neither `set_text_prompt` (replaces the prior text prompt's masks)
        nor `add_geometric_prompt`/`add_point_prompt` (re-grounds the whole
        accumulated prompt set on every call) guarantees append-only growth
        or stable ordering — see aa_persistence.py's module docstring. So a
        post-call mask that still overlaps a pre-call MaskRecord keeps that
        record's identity (mask_id/mask_type/text_tag/source) with its
        geometry refreshed; one with no match is a new selection tagged with
        this call's mask_type/text_tag; a pre-call record with no post-call
        match was dropped by SAM3's re-ground and is removed. Other passes'
        records are untouched throughout.
        """
        prior = [m for m in self.masks if m.pass_ is self._pass]
        self.state = apply(self.state)
        raw_masks = self.state.get("masks") or []
        raw_scores = self.state.get("scores") or []
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
                    mask_id=f"{self._pass.value}:{uuid.uuid4()}",
                    mask_type=mask_type,
                    pass_=self._pass,
                    geometry=geometry,
                    text_tag=text_tag,
                    source=source,
                    score=float(raw_scores[i]) if i < len(raw_scores) else None,
                )
                reconciled.append(record)
                new_records.append(record)

        self.masks = [m for m in self.masks if m.pass_ is not self._pass] + reconciled
        return new_records

    def add_text_selection(self, prompt: str, mask_type: MaskType) -> list[MaskRecord]:
        return self._select(
            mask_type, prompt, "text_prompt",
            lambda s: self.segmentor.set_text_prompt(prompt, s),
        )

    def add_box_selection(
        self, box: list[float], label: bool, text_substitute: str, mask_type: MaskType,
    ) -> list[MaskRecord]:
        return self._select(
            mask_type, text_substitute, "box",
            lambda s: self.segmentor.add_geometric_prompt(box, label, s),
        )

    def add_point_selection(
        self, point: list[float], label: bool, text_substitute: str, mask_type: MaskType,
    ) -> list[MaskRecord]:
        return self._select(
            mask_type, text_substitute, "point",
            lambda s: self.segmentor.add_point_prompt(point, label, s),
        )

    def run_lama_pass(self, inpainter: InpaintingEngine) -> WorkingCopy:
        """Scrub Pass 1's Red out-masks and advance into Pass 2 (background).

        One-shot: this is SF's single foreground -> background transition,
        not a repeatable operation.
        """
        if self._pass is not Pass.FOREGROUND:
            raise RuntimeError("run_lama_pass() has already run for this session.")

        out_masks = [
            m.geometry for m in self.masks
            if m.mask_type is MaskType.OUT and m.pass_ is Pass.FOREGROUND
        ]
        working_copy = self.original.new_working_copy()
        if out_masks:
            working_copy.image = inpainter.inpaint(working_copy.image, _union_mask(out_masks))

        self._working_copy = working_copy
        self.state = self.segmentor.set_image(working_copy.image)
        self._pass = Pass.BACKGROUND
        return working_copy

    @property
    def background(self) -> WorkingCopy:
        if self._working_copy is None:
            raise RuntimeError("run_lama_pass() has not been called yet; no scrubbed background to export.")
        return self._working_copy


def build_sf_payload(session: SFSession) -> tuple[list[str], list[str], list]:
    """Serialize a completed `SFSession` into the SF-AA export payload.

    Returns D4M.jl-round-tripped parallel (rows, cols, vals) triples — hand
    to `d4m_juliacall_bridge.save_parquet` to persist, or transport as-is.

    Row keys are prefixed with `session_id` — a bare `"context"` row, or a
    bare `mask_id`, would collide across every other session's payload the
    moment two are ever combined via `⊕` (e.g. saved into a shared store),
    silently mixing one session's image bytes into another's row. This
    matches aa_persistence.py's own row-key convention
    (`f"{session_id}:{target_id}"`), which is also `mask.mask_id` itself
    once aa_persistence.py's schema carries pass/mask_type — see that
    module's docstring. `mask_id` already embeds pass
    (`f"{pass}:{uuid4()}"`), so this doesn't re-prefix pass separately.
    """
    context_row = f"{session.session_id}:context"
    rows: list[str] = [context_row, context_row]
    cols: list[str] = ["original_image_bytes", "background_image_bytes"]
    vals: list = [_b64(session.original.bytes), _b64(_png_bytes(session.background.image))]

    for mask in session.masks:
        if mask.mask_type is not MaskType.IN:
            continue
        row_key = f"{session.session_id}:{mask.mask_id}"
        rows += [row_key] * 3
        cols += ["pass", "text_tag", "mask_png_bytes"]
        vals += [mask.pass_.value, mask.text_tag, _b64(_mask_png_bytes(mask.geometry))]

    return bridge.build_triples(rows, cols, vals)
