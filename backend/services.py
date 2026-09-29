import io
import json
import os
import re
import shutil
import base64
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import jsonschema

from fastapi import UploadFile
from openai import AsyncOpenAI

import numpy as np
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor

import aa_persistence
import sf_engine


class SessionLoadError(Exception):
    """Raised when an operation refuses to proceed because this session's
    SAM3/prompt reconstruction is known to have failed — see
    `SegmentationService._broken_sessions`."""


def _img_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def mask_to_rle(mask: np.ndarray) -> dict:
    """
    Encode a binary mask to RLE (Run-Length Encoding) format.
    
    Args:
        mask: 2D binary numpy array (H, W) with values 0 or 1
        
    Returns:
        dict with 'counts' (list of run lengths) and 'size' [H, W]
    """
    # Flatten the mask in row-major (C) order
    flat = mask.flatten()
    
    # Find where values change
    diff = np.diff(flat)
    change_indices = np.where(diff != 0)[0] + 1
    
    # Build run lengths
    run_starts = np.concatenate([[0], change_indices])
    run_ends = np.concatenate([change_indices, [len(flat)]])
    run_lengths = (run_ends - run_starts).tolist()
    
    # If mask starts with 1, prepend a 0-length run for background
    if flat[0] == 1:
        run_lengths = [0] + run_lengths
    
    return {
        "counts": run_lengths,
        "size": list(mask.shape)  # [H, W]
    }


def serialize_state(state: dict) -> dict:
    """Convert state arrays to JSON-serializable format."""
    result = {
        "original_width": state.get("original_width"),
        "original_height": state.get("original_height"),
    }
    
    if "masks" in state and state["masks"] is not None:
        masks = state["masks"]
        boxes = state["boxes"]
        scores = state["scores"]
        
        masks_list = []
        boxes_list = []
        scores_list = []
        
        for i in range(len(scores)):
            mask_np = np.array(masks[i])
            box_np = np.array(boxes[i])
            score_np = float(np.array(scores[i]))
            
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            if mask_binary.ndim == 3:
                mask_binary = mask_binary[0]
            
            rle = mask_to_rle(mask_binary)
            masks_list.append(rle)
            boxes_list.append(box_np.tolist())
            scores_list.append(score_np)
        
        result["masks"] = masks_list
        result["boxes"] = boxes_list
        result["scores"] = scores_list
    
    if "prompted_boxes" in state:
        result["prompted_boxes"] = state["prompted_boxes"]
    
    return result


def _bbox_from_geometry(mask: np.ndarray) -> List[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# Process-lifetime cache of each mask's crop PNG, keyed by mask_id — not
# per-session, so it survives independently of any one SFSession object's
# own lifetime. Stores the base64-encoded string, matching every other
# image field on the wire (`_b64(...)`), so a cache HIT never re-encodes,
# only a genuine miss does. Crop generation (compositing the mask as an
# alpha channel over its pass's image) is the expensive part of rebuilding
# the AA preview on every mutation; mask_ids are unique per selection
# (minted as f"{pass}:{uuid4()}", never reused), so once computed a crop is
# valid forever UNLESS that same mask_id's geometry is later refreshed in
# place by an Avoid-refinement (see SFSession.last_refreshed_mask_ids) —
# caption/hold/dataset_status changes never touch geometry and must not
# evict this, or the cache would recompute on every table rebuild, defeating
# the point.
_crop_cache: Dict[str, str] = {}


def _cached_crop_png_b64(sf_session, mask: "sf_engine.MaskRecord") -> str:
    """This mask's crop PNG, base64-encoded — computed once per mask_id for
    the life of the process, reused after that.

    Pass-aware like `build_sf_payload`'s own crop step (`sf_engine.py`):
    crops from the image `mask.pass_` was actually traced against, not
    always pass 0 — `SFSession.image_for_pass` retains every pass's image
    for exactly this reason, and a wrong-pass crop would silently show
    stale content for a mask kept in an earlier pass.
    """
    cached = _crop_cache.get(mask.mask_id)
    if cached is None:
        base_image = sf_session.image_for_pass(mask.pass_).convert("RGBA")
        cached = _b64(sf_engine._crop_png_bytes(base_image, mask.geometry))
        _crop_cache[mask.mask_id] = cached
    return cached


def _invalidate_stale_crops(sf_session) -> None:
    """Evict any cached crop whose mask_id had its geometry refreshed since
    the last call.

    Drains `SFSession.last_refreshed_mask_ids` (set fresh by `_select`/
    `_select_single` on every text/box/point call) rather than reading it
    non-destructively, so a mask_id already invalidated once isn't
    re-invalidated — and therefore re-cropped — on every subsequent
    unrelated mutation.
    """
    for mask_id in sf_session.last_refreshed_mask_ids:
        _crop_cache.pop(mask_id, None)
    sf_session.last_refreshed_mask_ids = set()


def serialize_sf_masks(sf_session, state: dict) -> dict:
    """Wire shape for the /segment/* responses, built from SFSession's
    *recorded selections* rather than raw `state["masks"]`.

    Two reasons this can't just be `serialize_state`:

    1. A box/point prompt leaves every exemplar generalisation in
       `state["masks"]` (9, on the page this was root-caused against) even
       though only one of them is the user's selection — rendering straight
       from state would still draw all 9.
    2. `state` carries no dataset_status/held, so keep/unassigned and
       pending-scrub styling are unrepresentable from it. `mask_ids`/
       `dataset_statuses`/`held_flags`/`captions` here are parallel to
       `masks`/`boxes`/`scores`, which is what the canvas indexes by —
       `mask_ids` in particular is what the hold/caption endpoints need
       back from the frontend to name which record they act on.

    Only the current pass's records are emitted: an earlier pass's masks
    were traced against a different image than the current pass's, so
    mixing them would draw one pass's geometry over another's picture.

    `crop_png_bytes` is the segmented object itself, same convention as
    `build_sf_payload`'s export column — the AA Preview tab renders it
    inline per mask. Pulled from the process-lifetime cache
    (`_cached_crop_png_b64`), not recomputed here on every call; stale
    entries from an in-place geometry refresh are evicted first so a
    refined Avoid selection never serves a pre-refinement crop.
    """
    _invalidate_stale_crops(sf_session)
    records = [m for m in sf_session.masks if m.pass_ == sf_session.pass_]
    result = {
        "original_width": state.get("original_width"),
        "original_height": state.get("original_height"),
        "masks": [mask_to_rle(m.geometry) for m in records],
        "boxes": [_bbox_from_geometry(m.geometry) for m in records],
        "scores": [float(m.score) if m.score is not None else 0.0 for m in records],
        "mask_ids": [m.mask_id for m in records],
        "dataset_statuses": [m.dataset_status.value for m in records],
        "held_flags": [m.held for m in records],
        "captions": [m.caption for m in records],
        "text_tags": [m.text_tag for m in records],
        "crop_png_bytes": [_cached_crop_png_b64(sf_session, m) for m in records],
        # Every record here is already filtered to sf_session.pass_ (see the
        # comprehension above) — an unbounded plain int in the v2 model, not
        # the old two-value enum — so this is a constant repeated once per
        # record, not per-record data. Included anyway, parallel to the
        # other fields, because the AA preview table (SegForge/frontend's
        # sf_aa_preview_adapter.dart) has no other way to know which pass
        # a row belongs to; nothing before this consumed a `pass` field.
        "passes": [sf_session.pass_ for _ in records],
    }
    if "prompted_boxes" in state:
        result["prompted_boxes"] = state["prompted_boxes"]
    return result


def serialize_sf_masks_all_passes(sf_session) -> dict:
    """Every `SFSession` mask record, across every pass — not filtered to
    the current one like `serialize_sf_masks`.

    For the AA Preview tab (SegForge/frontend's sf_aa_preview_adapter.dart),
    which previews session-wide status, not what's paintable on the canvas
    right now. `serialize_sf_masks`'s current-pass-only filter exists
    because its geometry has to match the image the canvas is currently
    displaying — mixing passes there would draw one pass's mask shapes
    over another pass's picture. That constraint doesn't apply here: this
    payload carries no geometry/masks/boxes at all, only per-record status
    fields, so there's nothing to draw "over" anything.

    Confirmed live: a mask held, captioned (dataset_status -> keep), then
    scrubbed vanished from every subsequent serialize_sf_masks response —
    correct for that function's own current-pass contract, but it left the
    AA Preview tab with no way to ever show the very data it exists to
    preview, since the moment something is scrubbed is exactly when it
    stops being "the current pass". This function is the fix: called
    fresh after every mutation, independent of which pass is current.

    No dataset_status filter of its own, unlike `build_sf_payload`: a mask
    that was held and scrubbed without ever being captioned still appears
    here (dataset_status "unassigned", held False) rather than
    disappearing — implicit discard is an EXPORT-time filter
    (`build_sf_payload`'s keep-only pass), never a deletion from
    `SFSession.masks`, which keeps every record as a permanent audit
    trail. This ledger is meant to show that trail, not just what will
    eventually be exported.

    `crop_png_bytes` reuses `_cached_crop_png_b64` — same cache
    `serialize_sf_masks` fills, keyed by mask_id — rather than a second
    cache. Crucially, each record's crop still comes from ITS OWN pass's
    image (`_cached_crop_png_b64` sources `sf_session.image_for_pass(
    mask.pass_)` per mask, not the session's *current* pass): this
    function spans every pass, so a pass-0 mask and a pass-1 mask in the
    same response must not collapse onto one image. This is the same
    multi-pass crop-correctness class of bug already found and fixed once
    for `build_sf_payload` — a keeper from a later pass was being cropped
    from pass 0's image before that fix.

    Stale crops are invalidated here too, not just in `serialize_sf_masks`:
    three callers (`/mask/hold`, `/mask/caption`, the scrub response) read
    `all_masks` without calling `serialize_sf_masks` first in the same
    request, so this can't assume that call already drained
    `last_refreshed_mask_ids`. Idempotent when it does run right after —
    the set is already empty by then.
    """
    _invalidate_stale_crops(sf_session)
    records = sf_session.masks
    return {
        "mask_ids": [m.mask_id for m in records],
        "dataset_statuses": [m.dataset_status.value for m in records],
        "held_flags": [m.held for m in records],
        "captions": [m.caption for m in records],
        "text_tags": [m.text_tag for m in records],
        "scores": [float(m.score) if m.score is not None else 0.0 for m in records],
        "boxes": [_bbox_from_geometry(m.geometry) for m in records],
        "crop_png_bytes": [_cached_crop_png_b64(sf_session, m) for m in records],
        "passes": [m.pass_ for m in records],
    }


class SegmentationService:
    ORIGINAL_IMAGE_FILENAME = "original.png"

    def __init__(self, storage_dir: Path, processor: Sam3Processor):
        self.storage_dir = storage_dir
        self.processor = processor
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.sf_sessions: Dict[str, sf_engine.SFSession] = {}
        # session_ids whose SAM3/prompt reconstruction threw on the most
        # recent attempt — see `_load_session_into_memory`'s except clause
        # and `save_session_settings` below. Deliberately NOT populated for
        # a session that simply has no image yet (staged via /initSession):
        # that is an expected, benign state (`load_session_from_disk`'s own
        # docstring: "a session the caller can still use"), not a failure.
        self._broken_sessions: set[str] = set()
        self.segment_prompt_dir = self.storage_dir.parent / "segment_prompt"
        schema_path = Path(__file__).parent.parent / "schemas" / "metadata-schema.json"
        with schema_path.open() as _f:
            self._metadata_schema = json.load(_f)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session from memory.
        If not in memory, it attempts to load it from disk.
        """
        session = self.sessions.get(session_id)
        if session:
            return session

        # Session not in memory, try to load from disk
        print(f"Session {session_id} not in memory, attempting to load from disk.")
        return self._load_session_into_memory(session_id)

    def get_or_create_sf_session(self, session_id: str) -> Optional[sf_engine.SFSession]:
        """The SFSession paired with this session, if one can exist yet.

        None only for a session with no image/state at all (e.g. staged via
        /initSession but never uploaded to) — there's nothing to segment.
        A session reloaded from disk gets its SFSession reconstructed by
        `_load_session_into_memory` (from persisted dataset_status/held/
        pass/text_tag, not by replay — see aa_persistence's module
        docstring), not here.
        """
        if session_id in self.sf_sessions:
            return self.sf_sessions[session_id]
        session = self.sessions.get(session_id)
        if not session or session.get("state") is None or not session.get("original_image_bytes"):
            return None
        sf_session = sf_engine.SFSession(
            session_id=session_id,
            original=sf_engine.ImmutableOriginal(session["original_image_bytes"]),
            segmentor=self.processor,
            state=session["state"],
        )
        self.sf_sessions[session_id] = sf_session
        return sf_session

    def _reconstruct_sf_session(
        self, session_id: str, raw: Dict[str, Any], replayed_state: dict,
    ) -> sf_engine.SFSession:
        """Rebuild the SFSession's mask index from persisted AA columns.

        The mask index itself is not from prompt replay: replay (in
        `_load_session_into_memory`) only reconstructs SAM3's raw mask list,
        not SF's own dataset_status/held/pass classification, which exists
        only in segment.parquet's columns.

        `state`, however, deliberately IS `replayed_state` at pass 0 — same
        object as `session_data["state"]`, no second source of truth (an
        earlier decision, unaffected by v2). For a session at a later pass,
        this can't hold: `replayed_state` was built by replaying *every*
        flat prompt (linkage.parquet has no per-prompt pass marker) against
        the *original* image, which is pass-0-shaped and wrong for any later
        pass. Best effort there: re-`set_image` on that pass's actual image
        so future selections at least segment the right picture, but SAM3's
        accumulated-prompt history for that pass starts empty rather than
        resuming where the session left off — a real, flagged gap (fixing
        it needs linkage.parquet to track pass per prompt, out of scope
        here), not a silent one.
        """
        pass_images_bytes: Dict[int, bytes] = raw.get("pass_images") or {}
        segments = raw["segments"]
        current_pass = max([0] + [int(seg["pass"]) for seg in segments] + list(pass_images_bytes.keys()))

        pass_images: Dict[int, Image.Image] = {
            p: Image.open(io.BytesIO(b)).convert("RGBA") for p, b in pass_images_bytes.items()
        }

        if current_pass == 0:
            state = replayed_state
        else:
            if current_pass not in pass_images:
                raise ValueError(
                    f"session {session_id!r} is at pass {current_pass} but has no persisted image for it"
                )
            state = self.processor.set_image(pass_images[current_pass].convert("RGB"))

        sf_session = sf_engine.SFSession(
            session_id=session_id,
            original=sf_engine.ImmutableOriginal(raw["image_bytes"]),
            segmentor=self.processor,
            state=state,
        )

        masks: list[sf_engine.MaskRecord] = []
        for seg in segments:
            if seg.get("mask_bytes") is None:
                continue
            mask_img = Image.open(io.BytesIO(seg["mask_bytes"])).convert("L")
            geometry = (np.array(mask_img) > 127).astype(np.uint8)
            pass_value = int(seg["pass"])
            # Old (v1) rows' segment_id is a bare uuid4 with no pass prefix;
            # v2 rows' segment_id already IS mask.mask_id verbatim
            # (f"{pass}:{uuid4()}") — see aa_persistence's module docstring.
            mask_id = seg["segment_id"] if ":" in seg["segment_id"] else f"{pass_value}:{seg['segment_id']}"
            masks.append(sf_engine.MaskRecord(
                mask_id=mask_id,
                pass_=pass_value,
                geometry=geometry,
                text_tag=seg.get("text_tag") or None,
                # Not persisted (audit-only, no downstream algorithmic use —
                # see aa_persistence's module docstring): the original
                # text_prompt/box/point distinction can't be recovered.
                source="loaded",
                score=seg.get("score"),
                dataset_status=sf_engine.DatasetStatus(seg.get("dataset_status", "unassigned")),
                held=bool(seg.get("held", False)),
                caption=seg.get("caption"),
            ))

        sf_session.restore(current_pass, pass_images, masks)
        return sf_session

    def _load_session_into_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Reads a session from its persisted Segment/Linkage/Registry AAs,
        reconstructs its MLX state by replaying prompts in original order,
        and loads it into the in-memory session cache.
        """
        raw = aa_persistence.read_session_raw(session_id)
        if raw is None or raw["image_bytes"] is None:
            return None

        try:
            # 1. Load image bytes
            image = Image.open(io.BytesIO(raw["image_bytes"])).convert("RGB")

            # 2. Re-initialize model state with the image
            state = self.processor.set_image(image)

            # 3. Re-apply prompts, in original order, to reconstruct the full state
            prompts = raw["prompts"]
            for idx, p in enumerate(prompts):
                try:
                    if isinstance(p, str):  # Text prompt
                        state = self.processor.set_text_prompt(p, state)
                    elif isinstance(p, dict) and p.get("type") in ["box", "point"]:
                        is_positive = p.get("label") == "positive"
                        # Dispatch by TYPE, not by "whichever geometry key is
                        # present". Replaying a point through
                        # add_geometric_prompt sends a 2-element [x, y] into
                        # the box path, which reshapes to (1, 1, 4) and throws
                        # "Cannot reshape array of size 2 into shape (1,1,4)" —
                        # failing the whole load, which the caller then reports
                        # as a bare 404 "Session not found". Confirmed live on
                        # 3 of 7 stored sessions. Points also have their own
                        # trained pathway (see add_point_prompt's docstring);
                        # the box path was never a valid input shape for them
                        # even when the arity happened to line up.
                        if p.get("type") == "point":
                            point = p.get("point")
                            if point:
                                state = self.processor.add_point_prompt(point, is_positive, state)
                        else:
                            box = p.get("box")
                            if box:
                                state = self.processor.add_geometric_prompt(box, is_positive, state)
                except Exception as e:
                    # One unreplayable prompt must not cost the whole session.
                    # Resuming with a partially reconstructed SAM3 state is
                    # strictly better than a 404 that reads as "your work is
                    # gone" — the masks themselves are restored from persisted
                    # columns, not from replay (see this class's docstring).
                    print(
                        f"Warning: skipping unreplayable prompt {idx} "
                        f"({p!r}) in session {session_id}: {e}"
                    )

            # 4. Construct the session object and store it in memory
            session_data = {
                "state": state,
                "original_image_bytes": raw["image_bytes"],
                "original_filename": raw["original_filename"],
                "image_size": (raw["width"], raw["height"]),
                "created_at": raw["created_at"],
                "prompts": prompts,
                "name": raw["name"],
                "description": raw["description"],
                "image_url": raw["image_url"],
            }
            self.sessions[session_id] = session_data
            self.sf_sessions[session_id] = self._reconstruct_sf_session(session_id, raw, state)

            # Restore the single-object invariant most of this codebase
            # already assumes (session["state"] and sf_session.state are
            # the SAME object — see e.g. _reconstruct_sf_session's own
            # docstring). For a session resumed at pass 0 this is already
            # true (`state` above IS `replayed_state`, unchanged). For
            # pass > 0, `_reconstruct_sf_session` re-`set_image`s onto that
            # pass's actual image and returns a DIFFERENT state object —
            # `session_data["state"]` above still points at the stale,
            # pass-0-shaped replay. Confirmed live this was never a
            # correctness risk (every live selection/scrub/save handler
            # rebinds `state = sf_session.state` before doing anything
            # real, so they self-healed it on first use) but it did leak
            # through /reset (which reads `session["state"]` directly and
            # never re-aliases, so Clear Prompts silently operated on the
            # wrong object) and dropped the very first prompted_boxes
            # display marker after a pass>0 resume (appended to the stale
            # object, then serialized from the correct one). Re-aliasing
            # here, once, closes both — and removes the fragility in
            # `get_or_create_sf_session`, which would otherwise rebuild a
            # brand-new, MASKLESS SFSession from this stale state if
            # `self.sf_sessions` and `self.sessions` were ever evicted
            # asymmetrically.
            session_data["state"] = self.sf_sessions[session_id].state

            print(f"Successfully loaded session {session_id} from disk into memory.")
            # A retry (e.g. after a code fix, or a transient error) that now
            # succeeds must clear any earlier broken mark — this is a
            # liveness flag for "as of the last attempt", not a permanent
            # blacklist.
            self._broken_sessions.discard(session_id)
            return session_data
        except Exception as e:
            # Using print for visibility in logs, but proper logging is better
            print(f"Error loading session {session_id} into memory: {e}")
            # Distinct from the early `image_bytes is None` return above:
            # this session HAS a registry row and an image, but reconstructing
            # its SAM3/prompt state threw partway through. Marked broken so
            # save_session_settings (and any future write path that doesn't
            # itself depend on a healthy in-memory session) can refuse to
            # persist against it, rather than silently "succeeding" while the
            # backend has no valid state for this session — see that
            # function's docstring for the incident this closes.
            self._broken_sessions.add(session_id)
            return None

    def create_session(self) -> str:
        """Create a new session ID and initialize storage directories."""
        session_id = str(uuid.uuid4())
        
        # Create directory structure
        session_dir = self.storage_dir / session_id
        (session_dir / "masks").mkdir(parents=True, exist_ok=True)
        (session_dir / "segments_raw").mkdir(parents=True, exist_ok=True)
        (session_dir / "segments_work").mkdir(parents=True, exist_ok=True)
        (session_dir / "segments_final").mkdir(parents=True, exist_ok=True)
        
        return session_id

    def register_session_data(self, session_id: str, data: Dict[str, Any]):
        """Register session data in memory and save its initial state to disk.

        Eagerly pairs an SFSession with this session_id once both `state`
        and `original_image_bytes` are present (i.e. right after /upload) —
        so every mask made from here on is dataset_status/held/pass-tracked
        from its very first selection, not just retroactively. Sessions that already
        had masks before this feature existed only get one on disk reload,
        via `_reconstruct_sf_session`'s migration path.

        Defense in depth: this merges into whatever is already cached for
        `session_id` rather than replacing it (`.update()` below, and the
        `session_id not in self.sf_sessions` guard further down) — but that
        alone only protects state already IN MEMORY. A caller that reaches
        this (via /upload) for a session that's genuinely saved on disk but
        not yet cached — e.g. one that skips /loadSession, whose own call
        into `get_session` is what normally warms this first — would
        otherwise still see `session_id not in self.sessions` as vacuously
        true and overwrite real name/description/prompts/masks with a
        blank session and a fresh, empty SFSession. Confirmed live: this is
        exactly what destroyed a session's data on every resume before
        /loadSession was fixed to warm the cache. Restoring from disk FIRST
        when nothing is cached yet closes that gap independently of
        whatever the caller did or didn't call beforehand; for a genuinely
        new session_id (nothing on disk), `get_session` returns None and
        this is a harmless no-op.
        """
        if session_id not in self.sessions:
            self.get_session(session_id)
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        self.sessions[session_id].update(data)
        session = self.sessions[session_id]
        if (
            session_id not in self.sf_sessions
            and session.get("state") is not None
            and session.get("original_image_bytes")
        ):
            self.sf_sessions[session_id] = sf_engine.SFSession(
                session_id=session_id,
                original=sf_engine.ImmutableOriginal(session["original_image_bytes"]),
                segmentor=self.processor,
                state=session["state"],
            )
        try:
            self.save_session_to_disk(session_id)
        except Exception as e:
            # Log this error, but don't fail the request
            print(f"Warning: Failed to save initial state for session {session_id}: {e}")

    def save_session_to_disk(self, session_id: str):
        """Persists the full session state as Segment/Linkage/Registry AAs.

        Replaces the old JSON+PNG flow — session.json/original.png are no
        longer written here. See aa_persistence for the schema.
        """
        session = self.get_session(session_id)
        if not session:
            print(f"Warning: Cannot save state for non-existent in-memory session {session_id}")
            return

        sf_session = self.sf_sessions.get(session_id)
        aa_persistence.save_session(session_id, session, sf_session)

    def save_session_settings(self, session_id: str, settings: Dict[str, Any]):
        """Merges UI-specific settings into the session's persisted registry.

        Refuses when this session's SAM3/prompt reconstruction is known to
        have failed (`_broken_sessions`) — `aa_persistence.save_session_settings`
        itself only checks that registry.parquet exists, with no idea whether
        the backend actually has a valid in-memory session for this id.
        Confirmed live: a session whose replay threw (e.g. the point-prompt
        reshape bug) could still have its view-layer toggles "saved"
        successfully, silently implying the session was fine when it wasn't
        — harmless in that specific case (this write never touches anything
        but the ui_settings field, and preserves every other field's existing
        value), but the wrong invariant to leave standing in general.

        `self.get_session` first ensures a load has actually been attempted
        in this process — so this works whether or not /loadSession already
        ran — before consulting the flag it sets. A session that legitimately
        has no image yet (staged via /initSession) is NOT broken and is
        unaffected: `get_session` returns None for it too, but nothing marks
        it broken (see `_load_session_into_memory`'s early return).
        """
        self.get_session(session_id)
        if session_id in self._broken_sessions:
            raise SessionLoadError(
                f"Session {session_id} failed to load; refusing to persist "
                "settings until it loads successfully."
            )
        aa_persistence.save_session_settings(session_id, settings)

    def load_session_from_disk(self, session_id: str) -> Dict[str, Any]:
        """Load session state and image from the persisted AAs.

        Reconstructs the same wire shape the old JSON+PNG flow produced:
        image_b64, width/height, results (masks as RLE / boxes / scores),
        prompts, name/description/image_url/created_at.

        Also warms the in-memory caches (`self.sessions`/`self.sf_sessions`)
        as a side effect, via `get_session` — the same restoration
        `_load_session_into_memory` already gives any other caller (real
        SAM3 state, replayed prompts, and a reconstructed SFSession with
        every persisted mask). Confirmed live: without this, /loadSession
        left both caches empty, so the frontend's follow-up /upload call
        (which only exists to (re-)initialize SAM3's in-memory state — see
        main.dart's `_loadImageFromUrl`) found nothing here to merge into
        and rebuilt the session from scratch, then auto-saved that empty
        result over the correct on-disk data — silently destroying a
        session's name, description, prompts and masks on every resume,
        before the user touched anything. `get_session` returning None
        here (a session staged via /initSession but never uploaded to, or
        one with no image at all) is fine — the raw-based response below
        already handles that case on its own.
        """
        self.get_session(session_id)

        raw = aa_persistence.read_session_raw(session_id)
        if raw is None:
            raise FileNotFoundError(f"State file not found for session {session_id}")

        # `self.get_session` above (via `_load_session_into_memory` ->
        # `_reconstruct_sf_session`) already warmed `self.sf_sessions` with
        # every persisted mask when an image exists — reused here for
        # crop_png_bytes rather than a third crop cache. Looked up by
        # mask_id rather than zipped by list position: a live in-memory
        # session (already cached before this call, e.g. via /updateState
        # on a session with unsaved edits) can have masks this fresh
        # `raw["segments"]` read doesn't know about yet, or vice versa: a
        # missing entry becomes None, not a misaligned crop for some other
        # mask.
        sf_session = self.sf_sessions.get(session_id)
        crop_by_mask_id = (
            {m.mask_id: m for m in sf_session.masks} if sf_session is not None else {}
        )

        def _resolved_mask_id(seg: Dict[str, Any]) -> str:
            # Mirrors _reconstruct_sf_session's own v1/v2 normalization: a v2
            # segment_id already carries its pass prefix (f"{pass}:{uuid4()}"),
            # unchanged; a v1 row's segment_id is a bare uuid and needs one
            # prepended to match the mask_id sf_session.masks actually uses.
            seg_id = seg["segment_id"]
            return seg_id if ":" in seg_id else f"{int(seg['pass'])}:{seg_id}"

        def _crop_for(seg: Dict[str, Any]) -> Optional[str]:
            # Each mask's crop still comes from ITS OWN pass's image
            # (_cached_crop_png_b64 sources sf_session.image_for_pass(
            # mask.pass_) per mask) — a resumed session can span multiple
            # passes same as the live all_masks endpoints, so this must not
            # collapse onto pass 0 or any one "current" pass.
            mask = crop_by_mask_id.get(_resolved_mask_id(seg))
            return _cached_crop_png_b64(sf_session, mask) if mask is not None else None

        # A session registered by /initSession but never uploaded to has
        # metadata and no image. That is a session the caller can still use --
        # the frontend shows its name while the image is on its way -- so the
        # image is optional here rather than a 404 for the whole session.
        image_b64 = (
            base64.b64encode(raw["image_bytes"]).decode("utf-8")
            if raw["image_bytes"] else None
        )

        masks_rle: list = []
        boxes: list = []
        scores: list = []
        for seg in raw["segments"]:
            if seg["mask_bytes"] is not None:
                mask_img = Image.open(io.BytesIO(seg["mask_bytes"])).convert("L")
                mask_np = (np.array(mask_img) > 127).astype(np.uint8)
                masks_rle.append(mask_to_rle(mask_np))
            if seg["bbox"] is not None:
                boxes.append(seg["bbox"])
            if seg["score"] is not None:
                scores.append(seg["score"])

        response = {
            "session_id": session_id,
            "image_b64": image_b64,
            "width": raw["width"],
            "height": raw["height"],
            "results": {
                "masks": masks_rle,
                "boxes": boxes,
                "scores": scores,
            },
            # Same shape/purpose as serialize_sf_masks_all_passes' field of
            # the same name on the live endpoints — a reloaded session's AA
            # Preview tab needs this too, not just a live in-progress one.
            # Two fields need normalizing before they match that live
            # contract, not just `segment["segment_id"]` as read_session_raw
            # hands it back:
            #   - `mask_ids` must be _resolved_mask_id(seg), not the raw
            #     segment_id — a v1 row's segment_id is a bare uuid with no
            #     pass prefix, while serialize_sf_masks_all_passes always
            #     emits sf_session.masks' already-normalized (pass-prefixed)
            #     mask_id for the same record. Left un-normalized, the same
            #     v1 mask would be addressable under two different ids
            #     depending on which endpoint answered, and /mask/hold and
            #     /mask/caption route by whatever id they're given.
            #   - `passes` must be int(seg["pass"]), not the numeric-string
            #     form aa_persistence._migrate_pass_value produces — every
            #     live /segment/*, /mask/*, and /lama/scrub response's
            #     `all_masks.passes` is already int (sf_engine.MaskRecord.
            #     pass_ is a plain int), and the frontend adapter casts this
            #     field `as int?` accordingly. Confirmed live: sending the
            #     string form here made sfResultToAaPayload throw a
            #     TypeError on every resumed session, which made the AA
            #     Preview tab render nothing at all — not just missing
            #     images, the whole tab silently empty.
            "all_masks": {
                "mask_ids": [_resolved_mask_id(seg) for seg in raw["segments"]],
                "dataset_statuses": [seg["dataset_status"] for seg in raw["segments"]],
                "held_flags": [seg["held"] for seg in raw["segments"]],
                "captions": [seg["caption"] for seg in raw["segments"]],
                "text_tags": [seg["text_tag"] or None for seg in raw["segments"]],
                "scores": [seg["score"] or 0.0 for seg in raw["segments"]],
                "boxes": [seg["bbox"] for seg in raw["segments"]],
                "crop_png_bytes": [_crop_for(seg) for seg in raw["segments"]],
                "passes": [int(seg["pass"]) for seg in raw["segments"]],
            },
            "prompts": raw["prompts"],
            "created_at": raw["created_at"],
            "name": raw["name"],
            "description": raw["description"],
            "image_url": raw["image_url"],
        }

        view_layers = raw["ui_settings"].get("view_layers")
        if view_layers is not None:
            response["view_layers"] = view_layers

        return response

    def delete_session_memory(self, session_id: str) -> bool:
        """Remove session from memory — both caches, not just `self.sessions`.

        `self.sf_sessions` used to survive this: `/mask/hold` and
        `/mask/caption` call `get_or_create_sf_session` directly, which
        checks `self.sf_sessions` FIRST, before ever consulting
        `self.sessions` — so leaving the SFSession behind let those two
        endpoints keep succeeding against a "deleted" session's zombie
        object indefinitely (only a process restart actually cleared it).
        Returns True if either cache held something to remove.
        """
        had_session = self.sessions.pop(session_id, None) is not None
        had_sf_session = self.sf_sessions.pop(session_id, None) is not None
        return had_session or had_sf_session

    def list_disk_sessions(self) -> List[str]:
        """List all sessions saved to disk."""
        if not self.storage_dir.exists():
            return []
        
        sessions_list = [
            d.name for d in self.storage_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ]
        sessions_list.sort(reverse=True)
        return sessions_list

    def delete_disk_session(self, session_id: str) -> bool:
        """Delete a session's storage directory."""
        session_dir = self.storage_dir / session_id
        if session_dir.exists() and session_dir.is_dir():
            shutil.rmtree(session_dir)
            return True
        return False

    def save_masks_to_disk(self, session_id: str) -> Dict[str, Any]:
        """Save current session state (image, masks, metadata) to disk."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        start_time = time.perf_counter()
        state = session["state"]

        session_dir = self.storage_dir / session_id
        masks_dir = session_dir / "masks"

        # Ensure directories exist (idempotent)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save original image
        image_path = session_dir / self.ORIGINAL_IMAGE_FILENAME
        if "original_image_bytes" in session:
            image_path.write_bytes(session["original_image_bytes"])

        # 2. Save masks
        if "masks" in state:
            masks = state["masks"]
            for i, mask_mx in enumerate(masks):
                mask_np = np.array(mask_mx)
                mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                if mask_binary.ndim == 3:
                    mask_binary = mask_binary[0]
                
                mask_image = Image.fromarray(mask_binary, mode='L')
                mask_image.save(masks_dir / f"mask_{i:03d}.png")

        return {
            "path": str(session_dir),
            "processing_time_ms": (time.perf_counter() - start_time) * 1000
        }

    def create_segments(self, session_id: str) -> Dict[str, Any]:
        """Generate segment images from masks and original image."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        if "original_image_bytes" not in session or "state" not in session or session["state"].get("masks") is None:
            raise ValueError("Image or masks not available")

        start_time = time.perf_counter()

        original_image = Image.open(io.BytesIO(session["original_image_bytes"])).convert("RGBA")
        masks = session["state"]["masks"]

        # Each original image gets its own subdir named after the file (no extension)
        original_filename = session.get("original_filename") or "image"
        image_stem = Path(original_filename).stem

        segments_dir = self.storage_dir / session_id / "segments_raw" / image_stem
        segments_dir.mkdir(parents=True, exist_ok=True)
        # Clear only this image's previous segments, preserving other images' subdirs
        for f in segments_dir.glob('*.png'):
            f.unlink()

        for i, mask_mx in enumerate(masks):
            mask_np = np.array(mask_mx)
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            if mask_binary.ndim == 3:
                mask_binary = mask_binary[0]
            mask_image = Image.fromarray(mask_binary * 255, 'L')
            segment_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
            segment_image.paste(original_image, (0, 0), mask_image)
            segment_image.save(segments_dir / f"segment_{i:03d}.png")

        return {
            "count": len(masks),
            "path": str(segments_dir),
            "image_stem": image_stem,
            "processing_time_ms": (time.perf_counter() - start_time) * 1000
        }

    async def append_lora_entry(self, session_id: str, segment_index: int, original_prompt: str) -> Dict[str, Any]:
        """Call OpenAI to caption one segment, validate against schema, append to metadata.jsonl."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        original_filename = session.get("original_filename") or "image"
        image_stem = Path(original_filename).stem

        segment_path = self.storage_dir / session_id / "segments_raw" / image_stem / f"segment_{segment_index:03d}.png"
        if not segment_path.exists():
            raise ValueError(f"Segment {segment_index} not found under '{image_stem}'. Run 'Create Segments' first.")

        original_image_path = self.storage_dir / session_id / self.ORIGINAL_IMAGE_FILENAME
        if not original_image_path.exists():
            raise ValueError("Original image not found for this session.")

        original_img = Image.open(original_image_path).convert("RGB")
        segment_img = Image.open(segment_path).convert("RGBA")
        seg_w, seg_h = segment_img.size

        file_name = f"segments_raw/{image_stem}/segment_{segment_index:03d}.png"

        client = AsyncOpenAI(api_key=os.getenv("CHATGPT_API_KEY"))
        instruction = (
            f"The concept being trained is: '{original_prompt}'. "
            "Using the original image for context and the segment cutout as the subject, "
            "generate a caption, tags, trigger word, and flip_augmentation value for this training sample.\n\n"
            "Output exactly ONE line — a single valid JSON object, no markdown fences, no extra text.\n"
            "Fields to include:\n"
            "  text (string): detailed comma-separated caption that describes the subject.\n"
            "  trigger_word (string): short ALL_CAPS unique token for this concept.\n"
            "  tags (array of strings): 3–8 semantic tags.\n"
            "  flip_augmentation (boolean): false only if the subject is clearly asymmetric.\n"
        )
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": _img_to_data_url(original_img)}},
                {"type": "image_url", "image_url": {"url": _img_to_data_url(segment_img)}},
            ]}],
            max_tokens=512,
        )

        raw = response.choices[0].message.content.strip()
        raw_clean = re.sub(r'^```[a-zA-Z]*\n?', '', raw)
        raw_clean = re.sub(r'\n?```$', '', raw_clean.strip())

        try:
            parsed = json.loads(raw_clean)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini returned invalid JSON: {exc}\nRaw response:\n{raw}")

        refined_text = parsed.get("text", "").strip() or original_prompt

        entry: Dict[str, Any] = {
            "file_name": file_name,
            "text": refined_text,
            "subset": "train",
            "weight": 1.0,
            "drop_caption_probability": 0.05,
            "resolution": {"width": seg_w, "height": seg_h},
        }
        for opt in ("trigger_word", "tags", "flip_augmentation"):
            if parsed.get(opt) is not None:
                entry[opt] = parsed[opt]

        try:
            jsonschema.validate(instance=entry, schema=self._metadata_schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(
                f"Schema validation failed: {exc.message}\nRaw Gemini response:\n{raw}"
            )

        jsonl_path = self.storage_dir / session_id / "metadata.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        entry_count = sum(1 for line in jsonl_path.open("r", encoding="utf-8") if line.strip())

        return {
            "entry": entry,
            "metadata_path": str(jsonl_path),
            "entry_count": entry_count,
        }

    async def caption_all_segments(self, session_id: str, prompt: str) -> Dict[str, Any]:
        """Caption every segment in a session and write all entries to metadata.jsonl."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        original_filename = session.get("original_filename") or "image"
        image_stem = Path(original_filename).stem
        segments_dir = self.storage_dir / session_id / "segments_raw" / image_stem
        segment_files = sorted(segments_dir.glob("segment_*.png"))
        if not segment_files:
            raise ValueError("No segments found. Run 'Create Segments' first.")

        results = []
        for seg_file in segment_files:
            idx = int(seg_file.stem.split("_")[1])
            r = await self.append_lora_entry(session_id, idx, prompt)
            results.append(r)

        return {
            "entries_added": len(results),
            "metadata_path": str(self.storage_dir / session_id / "metadata.jsonl"),
            "dataset_dir": str(self.storage_dir / session_id),
            "refined_prompts": [r["entry"]["text"] for r in results],
        }

    async def generate_caption(self, file: UploadFile) -> str:
        """
        Generates a caption for an image using Gemini, saves the image and caption.
        """
        # Ensure the target directory exists
        self.segment_prompt_dir.mkdir(exist_ok=True)

        # Read file content
        contents = await file.read()
        filename = file.filename if file.filename else "untitled.jpg"

        # Save the original image
        image_path = self.segment_prompt_dir / filename
        image_path.write_bytes(contents)

        img = Image.open(io.BytesIO(contents))
        client = AsyncOpenAI(api_key=os.getenv("CHATGPT_API_KEY"))
        prompt_text = (
            "Generate a detailed, descriptive caption for this image, suitable for training a Stable Diffusion LoRA. "
            "The caption should be a series of comma-separated keywords and phrases. "
            "Start with the main subject, then describe their appearance, clothing, pose, and the background. "
            "Mention the style of the image (e.g., photo, illustration, 3d render) and any notable lighting or color schemes. "
            "Be concise but comprehensive."
        )
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": _img_to_data_url(img, fmt="JPEG")}},
            ]}],
            max_tokens=256,
        )
        caption = response.choices[0].message.content.strip()

        # Save the generated prompt as a text file
        prompt_path = self.segment_prompt_dir / Path(filename).with_suffix('.txt')
        prompt_path.write_text(caption)

        return caption