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


class SegmentationService:
    ORIGINAL_IMAGE_FILENAME = "original.png"

    def __init__(self, storage_dir: Path, processor: Sam3Processor):
        self.storage_dir = storage_dir
        self.processor = processor
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.sf_sessions: Dict[str, sf_engine.SFSession] = {}
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
        `_load_session_into_memory` (from persisted mask_type/pass/text_tag,
        not by replay — see aa_persistence's module docstring), not here.
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
        not SF's own mask_type/pass classification, which exists only in
        segment.parquet's mask_type/pass/text_tag columns.

        `state`, however, deliberately IS `replayed_state` for a
        Pass-1-only session — same object as `session_data["state"]`, no
        second source of truth (PART 1's decision). For a session that
        reached Pass 2, this can't hold: `replayed_state` was built by
        replaying *every* flat prompt (linkage.parquet has no per-prompt
        pass marker) against the *original* image, which is Pass-1-shaped
        and wrong for Pass 2. Best effort there: re-`set_image` on the
        correct background so future selections at least segment the right
        picture, but SAM3's accumulated-prompt history for Pass 2 starts
        empty rather than resuming where the session left off — a real,
        flagged gap (fixing it needs linkage.parquet to track pass per
        prompt, out of scope here), not a silent one.
        """
        background_bytes = raw.get("background_image_bytes")
        has_background_segment = any(seg.get("pass") == "background" for seg in raw["segments"])
        current_pass = (
            sf_engine.Pass.BACKGROUND if (background_bytes or has_background_segment)
            else sf_engine.Pass.FOREGROUND
        )

        if current_pass is sf_engine.Pass.FOREGROUND:
            state = replayed_state
        else:
            state = self.processor.set_image(Image.open(io.BytesIO(background_bytes)).convert("RGB"))

        sf_session = sf_engine.SFSession(
            session_id=session_id,
            original=sf_engine.ImmutableOriginal(raw["image_bytes"]),
            segmentor=self.processor,
            state=state,
        )
        sf_session._pass = current_pass
        if background_bytes:
            sf_session._working_copy = sf_engine.WorkingCopy(
                Image.open(io.BytesIO(background_bytes)).convert("RGBA")
            )

        masks: list[sf_engine.MaskRecord] = []
        for seg in raw["segments"]:
            if seg.get("mask_bytes") is None:
                continue
            mask_img = Image.open(io.BytesIO(seg["mask_bytes"])).convert("L")
            geometry = (np.array(mask_img) > 127).astype(np.uint8)
            pass_value = seg.get("pass", "foreground")
            # Old rows' segment_id is a bare uuid4 with no pass prefix;
            # new rows' segment_id already IS mask.mask_id verbatim
            # (f"{pass}:{uuid4()}") — see aa_persistence's module docstring.
            mask_id = seg["segment_id"] if ":" in seg["segment_id"] else f"{pass_value}:{seg['segment_id']}"
            masks.append(sf_engine.MaskRecord(
                mask_id=mask_id,
                mask_type=sf_engine.MaskType(seg.get("mask_type", "in")),
                pass_=sf_engine.Pass(pass_value),
                geometry=geometry,
                text_tag=seg.get("text_tag") or "",
                # Not persisted (audit-only, no downstream algorithmic use —
                # see aa_persistence's module docstring): the original
                # text_prompt/box/point distinction can't be recovered.
                source="loaded",
                score=seg.get("score"),
            ))
        sf_session.masks = masks
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
            for p in prompts:
                if isinstance(p, str):  # Text prompt
                    state = self.processor.set_text_prompt(p, state)
                elif isinstance(p, dict) and p.get("type") in ["box", "point"]:
                    prompt_geom = p.get("box") or p.get("point")
                    is_positive = p.get("label") == "positive"
                    if prompt_geom:
                        state = self.processor.add_geometric_prompt(prompt_geom, is_positive, state)

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
            print(f"Successfully loaded session {session_id} from disk into memory.")
            return session_data
        except Exception as e:
            # Using print for visibility in logs, but proper logging is better
            print(f"Error loading session {session_id} into memory: {e}")
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
        so every mask made from here on is mask_type/pass-tracked from its
        very first selection, not just retroactively. Sessions that already
        had masks before this feature existed only get one on disk reload,
        via `_reconstruct_sf_session`'s migration path.
        """
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

        aa_persistence.save_session(session_id, session, self.sf_sessions.get(session_id))

    def save_session_settings(self, session_id: str, settings: Dict[str, Any]):
        """Merges UI-specific settings into the session's persisted registry."""
        aa_persistence.save_session_settings(session_id, settings)

    def load_session_from_disk(self, session_id: str) -> Dict[str, Any]:
        """Load session state and image from the persisted AAs.

        Reconstructs the same wire shape the old JSON+PNG flow produced:
        image_b64, width/height, results (masks as RLE / boxes / scores),
        prompts, name/description/image_url/created_at.
        """
        raw = aa_persistence.read_session_raw(session_id)
        if raw is None:
            raise FileNotFoundError(f"State file not found for session {session_id}")

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
        """Remove session from memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

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