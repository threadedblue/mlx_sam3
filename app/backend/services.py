import io
import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image


class SegmentationService:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        # In-memory session storage
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session from memory."""
        return self.sessions.get(session_id)

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
        """Register or update session data in memory."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        self.sessions[session_id].update(data)

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
        if "original_image_bytes" in session:
            (session_dir / "original.png").write_bytes(session["original_image_bytes"])

        # 2. Save masks
        mask_count = 0
        if "masks" in state:
            masks = state["masks"]
            mask_count = len(masks)
            for i, mask_mx in enumerate(masks):
                mask_np = np.array(mask_mx)
                mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                if mask_binary.ndim == 3:
                    mask_binary = mask_binary[0]
                
                mask_image = Image.fromarray(mask_binary, mode='L')
                mask_image.save(masks_dir / f"mask_{i:03d}.png")

        # 3. Save metadata
        session_data = {
            "session_id": session_id,
            "created_at": session.get("created_at"),
            "original_filename": session.get("original_filename"),
            "prompts": session.get("prompts", []),
            "mask_count": mask_count
        }
        (session_dir / "session.json").write_text(json.dumps(session_data, indent=2))

        return {
            "path": str(session_dir),
            "processing_time_ms": (time.perf_counter() - start_time) * 1000
        }

    def create_segments(self, session_id: str) -> Dict[str, Any]:
        """Generate segment images from masks and original image."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")
            
        if "original_image_bytes" not in session or "state" not in session or "masks" not in session["state"]:
            raise ValueError("Image or masks not available")

        start_time = time.perf_counter()
        
        original_image = Image.open(io.BytesIO(session["original_image_bytes"])).convert("RGBA")
        masks = session["state"]["masks"]
        
        segments_dir = self.storage_dir / session_id / "segments_raw"
        # Clear old segments
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
            "processing_time_ms": (time.perf_counter() - start_time) * 1000
        }