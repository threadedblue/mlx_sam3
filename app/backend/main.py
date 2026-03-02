"""
FastAPI backend for SAM3 segmentation model.
Provides endpoints for image upload, text prompts, box prompts, and segmentation results.
"""

import io
import os
import sys
import json
from datetime import datetime
import time
import uuid
from contextlib import asynccontextmanager

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

# Add parent directory to path to import sam3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from services import SegmentationService

# Global model and processor
model = None
processor = None
service = None

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR.parent.parent / "storage" / "sessions"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, processor
    
    sam3_root = os.path.dirname(sam3.__file__)
    
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("SAM3 model loaded successfully!")
    
    # Initialize Service Layer
    global service
    service = SegmentationService(STORAGE_DIR)
    
    yield
    
    # Cleanup if needed

app = FastAPI(
    title="SAM3 Segmentation API for MLX",
    description="API for interactive image segmentation using SAM3 model",
    version="1.0.0",
    lifespan=lifespan
)

FLUTTER_WEB_DIR = BASE_DIR.parent / "frontend" / "build" / "web"
STORAGE_PARENT_DIR = STORAGE_DIR.parent
print(f"===> {FLUTTER_WEB_DIR}")


# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextPromptRequest(BaseModel):
    session_id: str
    prompt: str


class BoxPromptRequest(BaseModel):
    session_id: str
    box: list[float]  # [center_x, center_y, width, height] normalized
    label: bool  # True for positive, False for negative


class ConfidenceRequest(BaseModel):
    session_id: str
    threshold: float


class SessionRequest(BaseModel):
    session_id: str


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
    
    if "masks" in state:
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
            
            # Convert mask to binary and get the 2D mask (handle [1, H, W] shape)
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            if mask_binary.ndim == 3:
                mask_binary = mask_binary[0]  # Take first channel
            
            # Encode as RLE
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


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/")
async def root():
    return {"message": "SAM3 Segmentation API", "status": "running"}
    
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and initialize a session."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Create session
        session_id = str(uuid.uuid4())
        
        # Process image through model (timed)
        start_time = time.perf_counter()
        state = processor.set_image(image)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Register session in service
        service.register_session_data(session_id, {
            "state": state,
            "original_image_bytes": contents,
            "original_filename": file.filename,
            "image_size": image.size,
            "created_at": datetime.utcnow().isoformat(),
        })
        
        return {
            "session_id": session_id,
            "width": image.size[0],
            "height": image.size[1],
            "message": "Image uploaded and processed successfully",
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/segment/text")
async def segment_with_text(request: TextPromptRequest):
    """Segment image using text prompt."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        start_time = time.perf_counter()
        state = processor.set_text_prompt(request.prompt, session["state"])
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        session.setdefault("prompts", []).append(request.prompt)
        session["state"] = state
        start = time.perf_counter()
        results = serialize_state(state)
        end = time.perf_counter()
        print(f"Serialization took {end - start:.4f} seconds")
        
        return {
            "session_id": request.session_id,
            "prompt": request.prompt,
            "results": results,
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")


@app.post("/segment/box")
async def add_box_prompt(request: BoxPromptRequest):
    """Add a box prompt (positive or negative) and re-segment."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        state = session["state"]
        
        # Store prompted box for display
        if "prompted_boxes" not in state:
            state["prompted_boxes"] = []
        
        # Convert from normalized cxcywh to pixel xyxy for display
        img_w = state["original_width"]
        img_h = state["original_height"]
        cx, cy, w, h = request.box
        x_min = (cx - w / 2) * img_w
        y_min = (cy - h / 2) * img_h
        x_max = (cx + w / 2) * img_w
        y_max = (cy + h / 2) * img_h
        
        state["prompted_boxes"].append({
            "box": [x_min, y_min, x_max, y_max],
            "label": request.label
        })
        
        session.setdefault("prompts", []).append({
            "type": "box",
            "box": request.box,
            "label": "positive" if request.label else "negative"
        })
        
        start_time = time.perf_counter()
        state = processor.add_geometric_prompt(request.box, request.label, state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        session["state"] = state
        
        return {
            "session_id": request.session_id,
            "box_type": "positive" if request.label else "negative",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding box prompt: {str(e)}")


@app.post("/reset")
async def reset_prompts(request: SessionRequest):
    """Reset all prompts for a session."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        state = session["state"]
        
        start_time = time.perf_counter()
        processor.reset_all_prompts(state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        if "prompted_boxes" in state:
            del state["prompted_boxes"]
        if "prompts" in session:
            session["prompts"] = []
        
        return {
            "session_id": request.session_id,
            "message": "All prompts reset",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting prompts: {str(e)}")


@app.post("/saveMasks")
async def save_masks(request: SessionRequest):
    """Saves the session data, including original image and masks, to the filesystem."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        result = service.save_masks_to_disk(request.session_id)
        return {
            "session_id": request.session_id,
            "message": f"Session saved to {result['path']}",
            "processing_time_ms": round(result['processing_time_ms'], 2),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving session: {str(e)}")

@app.post("/confidence")
async def set_confidence(request: ConfidenceRequest):
    """Update confidence threshold (note: requires re-running inference)."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update processor threshold
    processor.confidence_threshold = request.threshold
    
    return {
        "session_id": request.session_id,
        "threshold": request.threshold,
        "message": "Confidence threshold updated. Re-run segmentation to apply."
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free memory."""
    if service.delete_session_memory(session_id):
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/listSessions")
async def list_sessions():
    """List all saved sessions in storage."""
    if not STORAGE_DIR.exists():
        return []
    
    # List directories only
    sessions_list = [
        d.name for d in STORAGE_DIR.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ]
    # Sort by modification time (newest first) if possible, or just name
    sessions_list.sort(reverse=True)
    return sessions_list


@app.get("/newSession")
async def new_session():
    """Creates a new session ID and its storage directories."""
    session_id = service.create_session()
    return {"session_id": session_id}


@app.delete("/deleteSession/{session_id}")
async def delete_saved_session(session_id: str):
    """Deletes the storage directory for a specific session."""
    session_dir = STORAGE_DIR / session_id
    if session_dir.exists() and session_dir.is_dir():
        shutil.rmtree(session_dir)
        return {"message": f"Session {session_id} deleted", "session_id": session_id}
    
    raise HTTPException(status_code=404, detail="Saved session not found")


@app.post("/createSegments")
async def create_segments(request: SessionRequest):
    """Creates segmented image files from the current masks."""
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if "original_image_bytes" not in session or "state" not in session or "masks" not in session["state"]:
        raise HTTPException(status_code=400, detail="Image or masks not available in session. Please generate masks first.")

    try:
        start_time = time.perf_counter()
        
        original_image = Image.open(io.BytesIO(session["original_image_bytes"])).convert("RGBA")
        masks = session["state"]["masks"]
        
        session_dir = STORAGE_DIR / request.session_id
        segments_dir = session_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

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

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        return {
            "message": f"{len(masks)} segments created in {segments_dir}",
            "segment_count": len(masks),
            "processing_time_ms": round(processing_time_ms, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating segments: {str(e)}")


@app.get("/showSegments/{session_id}")
async def show_segments(session_id: str):
    """Returns a list of URLs for the generated segments."""
    segments_dir = service.storage_dir / session_id / "segments_raw"
    if not segments_dir.exists():
        return []

    segment_files = sorted(segments_dir.glob("*.png"))
    base_path = f"/storage/sessions/{session_id}/segments_raw" # This path is for the static mount
    urls = [f"{base_path}/{f.name}" for f in segment_files]
    return urls


app.mount(
    "/web",
    StaticFiles(directory=FLUTTER_WEB_DIR, html=True),
    name="frontend",
)

app.mount(
    "/storage",
    StaticFiles(directory=STORAGE_PARENT_DIR),
    name="storage"
)