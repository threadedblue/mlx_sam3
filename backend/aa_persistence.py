"""Segment/Linkage/Registry AA persistence for SegForge sessions.

Schema v2 — see DoubleNaught/doc/Clippings/sf-model-v2-design.md §6, which
this implements. `segment.parquet` is the literal source of truth for
compiled training data (see `compile_training_set.py`), not merely a
resume convenience: a round-trip failure on `dataset_status`/`caption`
means captioned work silently never reaches the training set, not just
that a resumed session looks wrong.

Replaces the old JSON+PNG save flow (session.json / original.png / masks/*.png)
with three D4M.jl Assocs, written/read via the shared juliacall bridge —
never a hand-rolled pyarrow reimplementation.

Row/column scheme (SF is one-image-per-session; no `image_id` layer):

  registry.parquet — row `session_id`, one column per metadata field (name,
                      description, created_at, image_url, original_filename,
                      width, height).
  segment.parquet  — row `session_id:_source`, column `image_bytes` (base64
                      of the original image, pass 0) for the source-image
                      sentinel; row `session_id:_pass_image:<pass>` (pass
                      >= 1), column `image_bytes`, one per pass that has ever
                      been scrubbed — needed because passes are unbounded
                      (§3): a mask selected in pass 2 needs pass 2's image to
                      re-crop from on reload, and that's a *different* image
                      than pass 5's, so "just the latest" isn't enough once
                      the session has scrubbed more than once. Row
                      `session_id:<segment_id>`, columns:
                        - `crop_bytes` / `mask_bytes` (base64 PNG each)
                        - `bbox` (JSON `[x0, y0, x1, y1]`)
                        - `score` (str(float), SAM3's confidence) — omitted
                          when absent, not written empty
                        - `pass` (numeric string, unbounded — "0", "1", ...)
                        - `dataset_status` (`"unassigned"` | `"keep"`)
                        - `held` — value `"true"` when the record is queued
                          for the next scrub batch; the column itself is
                          OMITTED (not written `"false"`) once unheld — see
                          design doc §6, "present/true only ... absent or
                          false once that batch has run"
                        - `caption` — present only when dataset_status is
                          `"keep"` (a keep row with no caption is exactly
                          what the compile step must not treat as kept)
                        - `text_tag` — SAM3 prompt or user substitute string,
                          omitted when the selection carried none
                      per selected segment. `score` isn't in the
                      originally-discussed column list — added because
                      DoubleNaught's Seg Forge node (`SegForgeMapping.linkage`,
                      Dart, unrelated to this Parquet persistence) reads
                      `results.scores` back out of `/loadSession/{id}` to
                      build its own `confidence` Linkage rows; dropping it
                      would silently empty those rows on every session DN
                      touches after this change.

                      `segment_id` is `sf_engine.MaskRecord.mask_id` itself
                      (`f"{pass}:{uuid4()}"`, already unique across every
                      pass) rather than a fresh uuid4 per Save — see
                      `_build_segments_from_masks`. Rows are sourced from
                      `SFSession.masks` (every pass), not the flat
                      `state["masks"]`: `state` only reflects the *current*
                      pass (`run_lama_pass` re-`set_image`s it), so once a
                      later pass starts, an earlier pass's masks would
                      silently vanish from every subsequent Save if derived
                      from `state` instead.

                      Rows written before this schema existed (v1, the
                      `MaskType.IN`/`OUT` model) have no `dataset_status`/
                      `held`/`caption` at all, and `pass` is the string
                      `"foreground"`/`"background"` rather than numeric.
                      `read_session_raw` migrates on read: `pass` maps
                      `"foreground"` -> `"0"`, `"background"` -> `"1"`
                      (matching how the two-pass model corresponds to the
                      unbounded one — old Pass 1 was pre-scrub, old Pass 2
                      was the one scrub's result); `dataset_status` defaults
                      to `"unassigned"` (there is no caption to recover, so
                      it cannot default to `"keep"` regardless of the old
                      `mask_type`); `held` defaults to absent/false. A
                      pre-v2 background image (old `_background` sentinel)
                      is read as pass 1's image if no `_pass_image:1` row
                      exists.
  linkage.parquet  — row `session_id:_global` for every prompt. SF's SAM3
                      pipeline has no way to target one already-identified
                      segment — `add_geometric_prompt` re-runs grounding over
                      the *whole* accumulated prompt set on every call, so
                      every prompt refines the whole image's state, never one
                      pre-existing segment. Column key is
                      `f"{index:04d}:{promptId}"`: D4M.jl's Assoc stores
                      columns in sorted-key order, which would otherwise
                      silently discard the original replay order — prompts
                      are not commutative (`set_text_prompt` replaces the
                      prior text prompt; `add_geometric_prompt` accumulates
                      onto whatever came before), so losing order would
                      reconstruct a different grounding state on Load than
                      what was actually being viewed at Save time. Sorting
                      column keys on read restores it. Value is the
                      JSON-encoded prompt entry (bare string for a text
                      prompt, dict for box/point).

`segment_id`/`promptId` are freshly minted (uuid4) on every Save — SAM3 has
no persistent instance identity between calls, so there is no "this segment"
or "this prompt" to keep an id stable for across a session's save history.
The guarantee this provides: ids are internally consistent *within one Save's
snapshot* only. (Investigated and confirmed nothing in DoubleNaught or
SegForge depends on cross-save id stability.)

`segment.parquet`/`linkage.parquet` are omitted entirely when there's nothing
to put in them — D4M.jl's `saveParquet` refuses to write an empty Assoc, and
by convention here a missing file on Load means zero rows, not an error.
"""

from __future__ import annotations

import base64
import io
import json
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

import d4m_juliacall_bridge as bridge
import sf_engine

STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage" / "sf" / "sessions"

_SOURCE_TARGET = "_source"
_BACKGROUND_TARGET = "_background"  # v1 sentinel, read-only migration path
_PASS_IMAGE_PREFIX = "_pass_image"  # v2: one row per pass >= 1, "_pass_image:<pass>"
_GLOBAL_TARGET = "_global"

_REGISTRY_FIELDS = ("name", "description", "created_at", "image_url", "original_filename")


def session_dir(session_id: str) -> Path:
    d = STORAGE_ROOT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


def _mask_binary(mask_mx) -> np.ndarray:
    mask_np = np.array(mask_mx)
    mask_binary = (mask_np > 0.5).astype(np.uint8)
    if mask_binary.ndim == 3:
        mask_binary = mask_binary[0]
    return mask_binary


def _mask_png_bytes(mask_mx) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(_mask_binary(mask_mx) * 255, "L").save(buf, format="PNG")
    return buf.getvalue()


def _crop_png_bytes(original_image: Image.Image, mask_mx) -> bytes:
    mask_image = Image.fromarray(_mask_binary(mask_mx) * 255, "L")
    segment_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    segment_image.paste(original_image, (0, 0), mask_image)
    buf = io.BytesIO()
    segment_image.save(buf, format="PNG")
    return buf.getvalue()


def _build_registry(session_id: str, session: dict) -> tuple[list, list, list]:
    fields = {name: str(session.get(name) or "") for name in _REGISTRY_FIELDS}
    width, height = session.get("image_size") or (None, None)
    if width is not None:
        fields["width"] = str(width)
    if height is not None:
        fields["height"] = str(height)
    cols = list(fields.keys())
    return [session_id] * len(cols), cols, [fields[c] for c in cols]


def _build_segments(
    session_id: str, session: dict, sf_session: "Optional[sf_engine.SFSession]" = None,
) -> tuple[list, list, list]:
    """Dispatch to the SFSession-aware builder when one exists for this
    session, else the legacy flat-state builder. See module docstring."""
    if sf_session is not None:
        return _build_segments_from_masks(session_id, sf_session)
    return _build_segments_legacy(session_id, session)


def _build_segments_legacy(session_id: str, session: dict) -> tuple[list, list, list]:
    rows: list[str] = []
    cols: list[str] = []
    vals: list[str] = []

    image_bytes = session.get("original_image_bytes")
    if image_bytes:
        rows.append(f"{session_id}:{_SOURCE_TARGET}")
        cols.append("image_bytes")
        vals.append(_b64(image_bytes))

    state = session.get("state") or {}
    masks = state.get("masks") or []
    boxes = state.get("boxes") or []
    scores = state.get("scores") or []
    if masks and image_bytes:
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        for i, mask_mx in enumerate(masks):
            # Fresh per Save — see module docstring.
            row_key = f"{session_id}:{uuid.uuid4()}"
            bbox = boxes[i] if i < len(boxes) else None
            rows += [row_key, row_key, row_key]
            cols += ["crop_bytes", "mask_bytes", "bbox"]
            vals += [
                _b64(_crop_png_bytes(original_image, mask_mx)),
                _b64(_mask_png_bytes(mask_mx)),
                json.dumps(bbox),
            ]
            if i < len(scores):
                rows.append(row_key)
                cols.append("score")
                vals.append(str(float(scores[i])))
    return rows, cols, vals


def _bbox_from_geometry(geometry: np.ndarray) -> Optional[list[int]]:
    ys, xs = np.nonzero(geometry)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _build_segments_from_masks(session_id: str, sf_session: "sf_engine.SFSession") -> tuple[list, list, list]:
    """Segment rows sourced from `SFSession.masks` — see module docstring
    for why this must not derive from `state["masks"]` once a later pass
    exists.

    Persists every pass's image (0 is `original`; 1..sf_session.pass_ each
    get their own `_pass_image:<pass>` row) — not just the current one —
    because a mask selected in an earlier pass still needs *that* pass's
    image to crop from, and passes are unbounded.
    """
    rows: list[str] = []
    cols: list[str] = []
    vals: list[str] = []

    image_bytes = sf_session.original.bytes
    rows.append(f"{session_id}:{_SOURCE_TARGET}")
    cols.append("image_bytes")
    vals.append(_b64(image_bytes))

    pass_images: dict[int, Image.Image] = {0: Image.open(io.BytesIO(image_bytes)).convert("RGBA")}
    for p in range(1, sf_session.pass_ + 1):
        image = sf_session.image_for_pass(p).convert("RGBA")
        pass_images[p] = image
        rows.append(f"{session_id}:{_PASS_IMAGE_PREFIX}:{p}")
        cols.append("image_bytes")
        vals.append(_b64(_png_bytes(image)))

    for mask in sf_session.masks:
        base_image = pass_images.get(mask.pass_, pass_images[0])
        # mask.mask_id already embeds pass (f"{pass}:{uuid4()}") — see
        # module docstring; this is the convention sf_engine.py's own
        # mask_id format already covers, not a fresh uuid4 per Save.
        row_key = f"{session_id}:{mask.mask_id}"
        rows += [row_key, row_key, row_key, row_key]
        cols += ["crop_bytes", "mask_bytes", "bbox", "pass"]
        vals += [
            _b64(_crop_png_bytes(base_image, mask.geometry)),
            _b64(_mask_png_bytes(mask.geometry)),
            json.dumps(_bbox_from_geometry(mask.geometry)),
            str(mask.pass_),
        ]
        rows.append(row_key)
        cols.append("dataset_status")
        vals.append(mask.dataset_status.value)
        if mask.held:
            rows.append(row_key)
            cols.append("held")
            vals.append("true")
        if mask.dataset_status is sf_engine.DatasetStatus.KEEP:
            rows.append(row_key)
            cols.append("caption")
            vals.append(mask.caption or "")
        if mask.text_tag:
            rows.append(row_key)
            cols.append("text_tag")
            vals.append(mask.text_tag)
        if mask.score is not None:
            rows.append(row_key)
            cols.append("score")
            vals.append(str(float(mask.score)))

    return rows, cols, vals


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _build_linkage(session_id: str, session: dict) -> tuple[list, list, list]:
    prompts = session.get("prompts") or []
    rows: list[str] = []
    cols: list[str] = []
    vals: list[str] = []
    for i, prompt in enumerate(prompts):
        rows.append(f"{session_id}:{_GLOBAL_TARGET}")
        cols.append(f"{i:04d}:{uuid.uuid4()}")
        vals.append(json.dumps(prompt))
    return rows, cols, vals


def save_session(
    session_id: str, session: dict, sf_session: "Optional[sf_engine.SFSession]" = None,
) -> dict:
    """Persist *session*'s current state to storage/sf/sessions/<id>/.

    Always writes registry.parquet. Writes segment.parquet/linkage.parquet
    only when there's something to put in them; removes any stale file left
    from a previous, non-empty Save of this same session so Load doesn't
    resurrect data that's no longer current.

    *sf_session*, when given, sources segment.parquet from its `.masks`
    (both passes, with mask_type/pass/text_tag) instead of the flat
    `session["state"]` — see module docstring and `_build_segments`.
    """
    d = session_dir(session_id)

    r_rows, r_cols, r_vals = _build_registry(session_id, session)
    bridge.save_parquet(str(d / "registry.parquet"), r_rows, r_cols, r_vals)
    written = ["registry.parquet"]

    seg_path = d / "segment.parquet"
    seg_rows, seg_cols, seg_vals = _build_segments(session_id, session, sf_session)
    # TEMP DEBUG (3/4) — what _build_segments actually returned, before it
    # reaches bridge.save_parquet, plus which branch (write vs. delete-
    # stale) is about to fire. If seg_rows is empty here and the file
    # already had content, the "unlink" branch below is exactly what
    # would silently make segment.parquet vanish with no error at all.
    seg_path_existed = seg_path.exists()
    print(f"[SAVE-DEBUG 3] save_session: session_id={session_id!r} — "
          f"_build_segments returned rows={len(seg_rows)}, cols={len(seg_cols)}, vals={len(seg_vals)} "
          f"(sf_session={'present' if sf_session is not None else 'None'}); "
          f"segment.parquet currently exists on disk: {seg_path_existed}")
    if seg_rows:
        print(f"[SAVE-DEBUG 3] save_session: WRITE branch — calling bridge.save_parquet for "
              f"{seg_path} with {len(seg_rows)} rows")
        bridge.save_parquet(str(seg_path), seg_rows, seg_cols, seg_vals)
        written.append("segment.parquet")
        print(f"[SAVE-DEBUG 3] save_session: bridge.save_parquet for segment.parquet RETURNED")
    else:
        print(f"[SAVE-DEBUG 3] save_session: EMPTY branch — seg_rows is empty, "
              f"{'DELETING existing segment.parquet (had real content!)' if seg_path_existed else 'unlinking (file did not exist / already absent — no-op)'}")
        seg_path.unlink(missing_ok=True)

    link_path = d / "linkage.parquet"
    link_rows, link_cols, link_vals = _build_linkage(session_id, session)
    if link_rows:
        bridge.save_parquet(str(link_path), link_rows, link_cols, link_vals)
        written.append("linkage.parquet")
    else:
        link_path.unlink(missing_ok=True)

    return {"session_dir": str(d), "written": written}


def save_session_settings(session_id: str, settings: dict) -> None:
    """Merge *settings* into this session's persisted UI state.

    UI display settings (e.g. layer-visibility toggles) aren't AA-shaped
    domain data, so rather than inventing a fourth persisted file they live
    as an opaque JSON blob in registry.parquet's `ui_settings` column.

    Raises FileNotFoundError if the session has never been saved — same
    contract as the JSON-backed implementation this replaces.
    """
    registry_path = STORAGE_ROOT / session_id / "registry.parquet"
    if not registry_path.exists():
        raise FileNotFoundError(f"No saved session {session_id}; cannot save settings.")

    _, cols, vals = bridge.load_parquet(str(registry_path))
    fields = dict(zip(cols, vals))
    existing_ui = json.loads(fields.get("ui_settings") or "{}")
    existing_ui.update(settings)
    fields["ui_settings"] = json.dumps(existing_ui)

    new_cols = list(fields.keys())
    bridge.save_parquet(
        str(registry_path),
        [session_id] * len(new_cols),
        new_cols,
        [fields[c] for c in new_cols],
    )


def _migrate_pass_value(raw: Optional[str]) -> str:
    """Numeric-string `pass`, migrating v1's two-value scheme.

    v1 wrote `"foreground"`/`"background"`; v2 writes `"0"`, `"1"`, ... A
    row with neither (pre-mask_type-era) was, by construction, made before
    any scrub existed — pass 0.
    """
    if raw is None:
        return "0"
    return {"foreground": "0", "background": "1"}.get(raw, raw)


def read_session_raw(session_id: str) -> Optional[dict[str, Any]]:
    """Read back everything persisted for *session_id*.

    Returns None when this session has never been saved (no registry.parquet
    at all). A missing segment.parquet or linkage.parquet — normal for a
    zero-segment or zero-prompt save — reads as zero rows, not an error.
    """
    d = STORAGE_ROOT / session_id
    registry_path = d / "registry.parquet"
    if not registry_path.exists():
        return None

    _, r_cols, r_vals = bridge.load_parquet(str(registry_path))
    fields = dict(zip(r_cols, r_vals))

    image_bytes: Optional[bytes] = None
    pass_images: dict[int, bytes] = {}
    segments: list[dict[str, Any]] = []
    seg_path = d / "segment.parquet"
    if seg_path.exists():
        seg_rows, seg_cols, seg_vals = bridge.load_parquet(str(seg_path))
        by_row: dict[str, dict[str, str]] = {}
        for row, col, val in zip(seg_rows, seg_cols, seg_vals):
            by_row.setdefault(row, {})[col] = val
        for row, attrs in by_row.items():
            target_id = row.split(":", 1)[1] if ":" in row else row
            if target_id == _SOURCE_TARGET:
                image_bytes = _unb64(attrs["image_bytes"])
                continue
            if target_id == _BACKGROUND_TARGET:
                # v1 sentinel: one scrubbed image, corresponding to pass 1
                # in the unbounded model (see module docstring). Only used
                # if a v2 `_pass_image:1` row hasn't already supplied it.
                pass_images.setdefault(1, _unb64(attrs["background_image_bytes"]))
                continue
            if target_id.startswith(f"{_PASS_IMAGE_PREFIX}:"):
                pass_num = int(target_id.split(":", 1)[1])
                pass_images[pass_num] = _unb64(attrs["image_bytes"])
                continue
            dataset_status = attrs.get("dataset_status", "unassigned")
            segments.append({
                "segment_id": target_id,
                "crop_bytes": _unb64(attrs["crop_bytes"]) if "crop_bytes" in attrs else None,
                "mask_bytes": _unb64(attrs["mask_bytes"]) if "mask_bytes" in attrs else None,
                "bbox": json.loads(attrs["bbox"]) if "bbox" in attrs else None,
                "score": float(attrs["score"]) if "score" in attrs else None,
                "pass": _migrate_pass_value(attrs.get("pass")),
                "dataset_status": dataset_status,
                "held": attrs.get("held") == "true",
                # Present only when dataset_status is "keep" — see
                # module docstring; None otherwise, not "".
                "caption": attrs.get("caption") if dataset_status == "keep" else None,
                # text_tag defaults to "" (unknown) — never recorded on
                # rows written before this column existed.
                "text_tag": attrs.get("text_tag", ""),
            })

    prompts: list[Any] = []
    link_path = d / "linkage.parquet"
    if link_path.exists():
        _, link_cols, link_vals = bridge.load_parquet(str(link_path))
        # See module docstring: column key's zero-padded index prefix
        # restores the original replay order after D4M.jl's internal sort.
        ordered = sorted(zip(link_cols, link_vals), key=lambda cv: cv[0])
        prompts = [json.loads(val) for _, val in ordered]

    width = int(fields["width"]) if fields.get("width") else None
    height = int(fields["height"]) if fields.get("height") else None

    return {
        "name": fields.get("name") or None,
        "description": fields.get("description") or None,
        "created_at": fields.get("created_at") or None,
        "image_url": fields.get("image_url") or None,
        "original_filename": fields.get("original_filename") or None,
        "width": width,
        "height": height,
        "image_bytes": image_bytes,
        "pass_images": pass_images,
        "segments": segments,
        "prompts": prompts,
        "ui_settings": json.loads(fields["ui_settings"]) if fields.get("ui_settings") else {},
    }


def list_registries() -> list[dict[str, str]]:
    """List every saved session's registry metadata, for a session picker."""
    out: list[dict[str, str]] = []
    if not STORAGE_ROOT.exists():
        return out
    for d in sorted(STORAGE_ROOT.iterdir()):
        registry_path = d / "registry.parquet"
        if not (d.is_dir() and registry_path.exists()):
            continue
        try:
            _, cols, vals = bridge.load_parquet(str(registry_path))
        except Exception:
            continue
        fields = dict(zip(cols, vals))
        out.append({
            "session_id": d.name,
            "name": fields.get("name") or "",
            "description": fields.get("description") or "",
            "created_at": fields.get("created_at") or "",
            "image_url": fields.get("image_url") or "",
        })
    return out
