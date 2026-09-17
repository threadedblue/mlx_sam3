"""Compiles a LoRA training set from every saved SF session's kept,
captioned masks — see sf-model-v2-design.md §5.

A read-time operation over `aa_persistence.py`'s existing per-image
`storage/sf/sessions/<session_id>/` directories, which already are the
corpus (`list_registries()` already walks all of them; no new cross-image
store is needed). Idempotent and re-runnable: no session needs to reach a
"finished" state first, and running this again with new sessions saved (or
new captions added to old ones) just re-derives a fresh, complete snapshot —
compiling from one session vs. many is the same operation at different N.

Output matches `lora_trainer.py.validate_inputs`'s contract: a
`metadata.jsonl` with `file_name` (relative to the dataset dir) and `text`
per line, and the referenced image actually present at that path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aa_persistence

IMAGES_SUBDIR = "images"


def _safe_filename(session_id: str, segment_id: str) -> str:
    """`segment_id` is `f"{pass}:{uuid4()}"` — replace the filesystem-unsafe
    `:` rather than strip it, so pass-0's and pass-1's copy of a similarly-
    shaped uuid (astronomically unlikely, but the mask_id format doesn't
    itself rule it out across passes) can't collide on disk."""
    return f"{session_id}_{segment_id.replace(':', '_')}.png"


def compile_training_set(output_dir: str) -> dict[str, Any]:
    """Walk every saved session, filter to `dataset_status == "keep"` masks
    that have a non-empty caption, and materialize `output_dir/images/*.png`
    + `output_dir/metadata.jsonl`.

    Overwrites any prior compile at `output_dir` — the snapshot is always
    derived fresh from current disk state, never accumulated onto a stale
    one (a session whose caption was edited or whose keep status changed
    must not leave a stale duplicate/contradictory entry behind).
    """
    out = Path(output_dir)
    images_dir = out / IMAGES_SUBDIR
    images_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    sessions_scanned = 0
    sessions_with_keepers = 0

    for registry in aa_persistence.list_registries():
        session_id = registry["session_id"]
        sessions_scanned += 1
        raw = aa_persistence.read_session_raw(session_id)
        if raw is None:
            continue

        session_entries: list[dict[str, Any]] = []
        for seg in raw["segments"]:
            if seg["dataset_status"] != "keep":
                continue
            caption = (seg.get("caption") or "").strip()
            if not caption:
                continue  # a keep row with no caption is not export-eligible
            if seg["crop_bytes"] is None:
                continue

            filename = _safe_filename(session_id, seg["segment_id"])
            (images_dir / filename).write_bytes(seg["crop_bytes"])
            session_entries.append({
                "file_name": f"{IMAGES_SUBDIR}/{filename}",
                "text": caption,
                "session_id": session_id,
                "segment_id": seg["segment_id"],
            })

        if session_entries:
            sessions_with_keepers += 1
            entries.extend(session_entries)

    metadata_path = out / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return {
        "dataset_dir": str(out),
        "metadata_path": str(metadata_path),
        "entry_count": len(entries),
        "sessions_scanned": sessions_scanned,
        "sessions_with_keepers": sessions_with_keepers,
    }
