"""Segment/Linkage/Registry AA persistence for SegForge sessions.

Replaces the old JSON+PNG save flow (session.json / original.png / masks/*.png)
with three D4M.jl Assocs, written/read via the shared juliacall bridge
(~/d4m.Wk/d4m_juliacall_bridge) — never a hand-rolled pyarrow reimplementation.

Row/column scheme (SF is one-image-per-session; no `image_id` layer):

  registry.parquet — row `session_id`, one column per metadata field (name,
                      description, created_at, image_url, original_filename,
                      width, height).
  segment.parquet  — row `session_id:_source`, column `image_bytes` (base64
                      of the original image) for the source-image sentinel;
                      row `session_id:_background`, column
                      `background_image_bytes` (base64 PNG) for the
                      LaMa-scrubbed working copy, present only once
                      sf_engine.SFSession.run_lama_pass() has run; row
                      `session_id:<segment_id>`, columns `crop_bytes` /
                      `mask_bytes` (base64 PNG each) / `bbox` (JSON
                      `[x0, y0, x1, y1]`) / `score` (str(float), parallel to
                      SAM3's own per-segment confidence) / `mask_type`
                      (`"in"`/`"out"`) / `pass` (`"foreground"`/
                      `"background"`) / `text_tag` (SAM3 prompt or user
                      substitute string) per detected segment. `score` isn't
                      in the originally-discussed 3-column list — added
                      because DoubleNaught's Seg Forge node
                      (`SegForgeMapping.linkage`, Dart, unrelated to this
                      Parquet persistence) reads `results.scores` back out
                      of `/loadSession/{id}` to build its own `confidence`
                      Linkage rows; dropping it would silently empty those
                      rows on every session DN touches after this change.

                      When an `sf_engine.SFSession` is available for this
                      session, `segment_id` is `sf_engine.MaskRecord.mask_id`
                      itself (`f"{pass}:{uuid4()}"`, already unique across
                      both passes) rather than a fresh uuid4 per Save — see
                      `_build_segments_from_masks`. Rows are then sourced
                      from `SFSession.masks` (both passes), not the flat
                      `state["masks"]`: `state` only reflects the *current*
                      pass (`run_lama_pass` re-`set_image`s it), so once Pass
                      2 starts, Pass 1's masks would silently vanish from
                      every subsequent Save if derived from `state` instead.
                      `mask_type`/`pass`/`text_tag`/`score` are absent (not
                      empty) on rows written before this schema existed —
                      `read_session_raw` defaults a row with no `mask_type`/
                      `pass` to `"in"`/`"foreground"`: every mask made before
                      LaMa scrubbing existed was, by construction, an
                      undifferentiated Pass-1 inclusion. `text_tag` defaults
                      to `""` (unknown) since the old schema never recorded
                      which prompt produced which specific mask.
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
_BACKGROUND_TARGET = "_background"
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
    for why this must not derive from `state["masks"]` once Pass 2 exists.
    """
    rows: list[str] = []
    cols: list[str] = []
    vals: list[str] = []

    image_bytes = sf_session.original.bytes
    rows.append(f"{session_id}:{_SOURCE_TARGET}")
    cols.append("image_bytes")
    vals.append(_b64(image_bytes))

    working_copy = sf_session.working_copy
    if working_copy is not None:
        rows.append(f"{session_id}:{_BACKGROUND_TARGET}")
        cols.append("background_image_bytes")
        vals.append(_b64(_png_bytes(working_copy.image)))

    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    background_image = working_copy.image.convert("RGBA") if working_copy is not None else None

    for mask in sf_session.masks:
        base_image = (
            background_image if mask.pass_ is sf_engine.Pass.BACKGROUND and background_image is not None
            else original_image
        )
        # mask.mask_id already embeds pass (f"{pass}:{uuid4()}") — see
        # module docstring; this is the "third convention" sf_engine.py's
        # own mask_id format already covers, not a fresh uuid4 per Save.
        row_key = f"{session_id}:{mask.mask_id}"
        rows += [row_key, row_key, row_key, row_key, row_key]
        cols += ["crop_bytes", "mask_bytes", "bbox", "mask_type", "pass"]
        vals += [
            _b64(_crop_png_bytes(base_image, mask.geometry)),
            _b64(_mask_png_bytes(mask.geometry)),
            json.dumps(_bbox_from_geometry(mask.geometry)),
            mask.mask_type.value,
            mask.pass_.value,
        ]
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
    if seg_rows:
        bridge.save_parquet(str(seg_path), seg_rows, seg_cols, seg_vals)
        written.append("segment.parquet")
    else:
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
    background_image_bytes: Optional[bytes] = None
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
                background_image_bytes = _unb64(attrs["background_image_bytes"])
                continue
            segments.append({
                "segment_id": target_id,
                "crop_bytes": _unb64(attrs["crop_bytes"]) if "crop_bytes" in attrs else None,
                "mask_bytes": _unb64(attrs["mask_bytes"]) if "mask_bytes" in attrs else None,
                "bbox": json.loads(attrs["bbox"]) if "bbox" in attrs else None,
                "score": float(attrs["score"]) if "score" in attrs else None,
                # Migration default: rows saved before this schema existed
                # have no mask_type/pass at all — every mask made before
                # LaMa scrubbing existed was an undifferentiated Pass-1
                # inclusion, so that's what they read back as. text_tag
                # defaults to "" (unknown) — never recorded pre-migration.
                "mask_type": attrs.get("mask_type", "in"),
                "pass": attrs.get("pass", "foreground"),
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
        "background_image_bytes": background_image_bytes,
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
