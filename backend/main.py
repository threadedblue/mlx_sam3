"""
FastAPI backend for SAM3 segmentation model.
Provides endpoints for image upload, text prompts, box prompts, and segmentation results.
"""

import os
import io
import re
import sys
import json
import asyncio
import shutil
import traceback
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import uuid
from contextlib import asynccontextmanager

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path

# Add parent directory to path to import sam3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from dotenv import load_dotenv

from services import SegmentationService, serialize_state
from lora_inferencer import (
    validate_inference_inputs,
    run_inference as _run_inference_fn,
    release_pipeline as _release_inference_pipeline,
    get_cached_model_path,
    free_vram_mb,
    sha256_of_file as _infer_sha256,
    load_provider_from_config,
    list_providers,
    get_active_provider_name,
    set_active_provider,
)

# Global model and processor
model = None
processor = None
service = None

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR.parent.parent / "storage" / "sessions"
INFERENCE_DIR = BASE_DIR.parent.parent / "storage" / "inference"
INFERENCE_CONFIG = BASE_DIR.parent.parent / "storage" / "inference_config.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_dotenv()

    global model, processor

    sam3_root = os.path.dirname(sam3.__file__)

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("SAM3 model loaded successfully!")

    # Initialize Service Layer
    global service
    service = SegmentationService(STORAGE_DIR, processor)

    if not os.getenv("CHATGPT_API_KEY"):
        print("Warning: CHATGPT_API_KEY not set. AI captioning will be disabled.")
    else:
        print("ChatGPT API key loaded.")

    # Restore last-used inference provider and Cloud Run URL from config.
    load_provider_from_config()

    # Pre-load last-used inference model in background (non-blocking)
    _maybe_preload_inference_model()

    yield

    # Release inference pipeline VRAM on shutdown
    try:
        _release_inference_pipeline()
        print("Inference pipeline released.")
    except Exception as exc:
        print(f"Warning: error releasing inference pipeline: {exc}")

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


class PointPromptRequest(BaseModel):
    session_id: str
    point: list[float]  # [x, y] normalized
    label: bool  # True for positive, False for negative


class ConfidenceRequest(BaseModel):
    session_id: str
    threshold: float


class SessionRequest(BaseModel):
    session_id: str


class SessionSettingsRequest(BaseModel):
    session_id: str
    settings: Dict[str, Any]


class InitSessionRequest(BaseModel):
    session_id: str
    name: str
    description: str
    image_url: str


class LoraAppendRequest(BaseModel):
    session_id: str
    segment_index: int
    prompt: str


@app.get("/health")
async def health():
    info: Dict[str, Any] = {
        "status": "healthy",
        "model_loaded": model is not None,
        "inference_model_loaded": get_cached_model_path() is not None,
        "inference_cached_model": get_cached_model_path(),
    }
    vram = free_vram_mb()
    if vram is not None:
        info["vram_free_mb"] = vram
    return info


@app.get("/")
async def root():
    return {"message": "SAM3 Segmentation API", "status": "running"}
    
@app.post("/upload")
async def upload_image(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    """Upload an image into an existing session or create a new one."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Reuse the supplied session or allocate a new one
        if session_id:
            # Ensure subdirectories exist in case this session was created
            # before the backend last restarted
            for sub in ("masks", "segments_raw", "segments_work", "segments_final"):
                (STORAGE_DIR / session_id / sub).mkdir(parents=True, exist_ok=True)
        else:
            session_id = service.create_session()

        start_time = time.perf_counter()
        state = processor.set_image(image)
        processing_time_ms = (time.perf_counter() - start_time) * 1000

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


@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """Generate a descriptive caption for an image and save both."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    if not os.getenv("CHATGPT_API_KEY"):
        raise HTTPException(status_code=503, detail="AI captioning is not configured on the server (CHATGPT_API_KEY is missing).")

    try:
        # The service method will handle reading the file and calling the AI model
        caption = await service.generate_caption(file)
        return {"caption": caption}
    except Exception as e:
        # Log the full error for debugging
        print(f"Error during image processing: {e}")
        traceback.print_exc()
        # Raise a generic but informative error to the client
        raise HTTPException(status_code=500, detail=f"Error processing image with AI: {str(e)}")


@app.post("/segment/text")
async def segment_with_text(request: TextPromptRequest):
    """Segment image using text prompt."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = service.get_session(request.session_id)
    print(f"session:id=={request.session_id}")
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        start_time = time.perf_counter()
        state = processor.set_text_prompt(request.prompt, session["state"])
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        session.setdefault("prompts", []).append(request.prompt)
        session["state"] = state
        service.save_session_to_disk(request.session_id)
        start = time.perf_counter()
        results = serialize_state(state)
        end = time.perf_counter()
        mask_count = len(results.get("masks") or [])
        print(f"Serialization took {end - start:.4f} seconds | masks={mask_count} | inference={processing_time_ms:.1f}ms")
        
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
        service.save_session_to_disk(request.session_id)
        
        return {
            "session_id": request.session_id,
            "box_type": "positive" if request.label else "negative",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding box prompt: {str(e)}")


@app.post("/segment/point")
async def add_point_prompt(request: PointPromptRequest):
    """Add a point prompt by encoding it as a zero-size box."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    session = service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        state = session["state"]
        x, y = request.point

        # Store for display alongside box prompts
        if "prompted_boxes" not in state:
            state["prompted_boxes"] = []
        img_w = state["original_width"]
        img_h = state["original_height"]
        state["prompted_boxes"].append({
            "box": [x * img_w, y * img_h, x * img_w, y * img_h],
            "label": request.label,
            "is_point": True,
        })

        session.setdefault("prompts", []).append({
            "type": "point",
            "point": request.point,
            "label": "positive" if request.label else "negative",
        })

        # Encode as degenerate box [cx, cy, 0, 0]
        start_time = time.perf_counter()
        state = processor.add_geometric_prompt([x, y, 0.0, 0.0], request.label, state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        session["state"] = state
        service.save_session_to_disk(request.session_id)

        return {
            "session_id": request.session_id,
            "point_type": "positive" if request.label else "negative",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding point prompt: {str(e)}")


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
        
        service.save_session_to_disk(request.session_id)
        
        return {
            "session_id": request.session_id,
            "message": "All prompts reset",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting prompts: {str(e)}")


@app.post("/session/settings")
async def save_session_settings(request: SessionSettingsRequest):
    """Saves arbitrary UI settings (like layer visibility) to the session file."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    try:
        service.save_session_settings(request.session_id, request.settings)
        return {"message": "Settings saved", "session_id": request.session_id}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/saveSession")
async def save_session(request: SessionRequest):
    """Manually save the current session state to disk."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    
    try:
        service.save_session_to_disk(request.session_id)
        return {"message": "Session saved", "session_id": request.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/newSession")
async def new_session():
    """Creates a new session ID and its storage directories."""
    session_id = service.create_session()
    return {"session_id": session_id}


@app.post("/initSession")
async def init_session(request: InitSessionRequest):
    """
    Initialize a session with metadata before the SegForge app launches.

    Called by DoubleNaught after allocating a session but before launching
    the SF frontend. Stores session name, description, and image URL so they
    are available to the frontend via /getSession/{id}.
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        service.register_session_data(
            request.session_id,
            {
                "name": request.name,
                "description": request.description,
                "image_url": request.image_url,
            }
        )
        return {
            "session_id": request.session_id,
            "name": request.name,
            "description": request.description,
            "image_url": request.image_url,
            "ready": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize session: {str(e)}")


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
    if service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    try:
        result = service.create_segments(request.session_id)
        return {
            "message": f"{result['count']} segments created in {result['path']}",
            "segment_count": result['count'],
            "processing_time_ms": round(result['processing_time_ms'], 2),
        }
    except ValueError as e:
        # This will catch "Session not found" or "Image or masks not available"
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating segments: {str(e)}")


@app.get("/showSegments/{session_id}")
async def show_segments(session_id: str):
    """Returns a list of URLs for the generated segments, grouped by source image subdir."""
    segments_dir = service.storage_dir / session_id / "segments_raw"
    if not segments_dir.exists():
        return []

    # Collect segment files from all image-named subdirectories
    segment_files = []
    for subdir in sorted(segments_dir.iterdir()):
        if subdir.is_dir():
            segment_files.extend(sorted(subdir.glob("segment_*.png")))

    base_path = f"/storage/sessions/{session_id}/segments_raw"
    urls = [f"{base_path}/{f.parent.name}/{f.name}" for f in segment_files]
    return urls

@app.get("/loadSession/{session_id}", response_model=Dict[str, Any])
async def load_session(session_id: str):
    """
    Loads the complete state for a given session_id from disk.

    This endpoint reads the session's state.json file, finds the original image,
    encodes it to base64, and returns it along with other session metadata like
    image dimensions and prompt results. This allows the frontend to fully
    reconstruct and display a previously saved session.
    """
    try:
        return service.load_session_from_disk(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # A general catch-all for other potential file or system errors.
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/updateState", response_model=Dict[str, Any])
async def update_state(request: SessionRequest):
    """
    Loads the complete state for a given session_id.

    This endpoint reads the session's state.json file, finds the original image,
    encodes it to base64, and returns it along with other session metadata like
    image dimensions and prompt results. This allows the frontend to fully
    reconstruct and display a previously saved session.
    """
    try:
        return service.load_session_from_disk(request.session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # A general catch-all for other potential file or system errors.
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# ── LoRA training run state ──────────────────────────────────────────────────
import threading
from lora_trainer import validate_inputs, train_lora, sha256_of_file

_lora_runs: Dict[str, Dict[str, Any]] = {}


class LoraTrainRequest(BaseModel):
    session_id: str
    dataset_dir: str = ""   # auto-derived from session_id if empty
    output_dir: str = ""    # auto-derived from session_id if empty
    model_path: str = "runwayml/stable-diffusion-v1-5"
    script_path: str = ""
    rank: int = 16
    alpha: int = 16
    learning_rate: float = 1e-4
    num_train_epochs: int = 1
    batch_size: int = 1
    resolution: int = 512
    mixed_precision: str = "no"

    def resolve_paths(self, storage_dir: Path) -> "LoraTrainRequest":
        updates: dict = {}
        if not self.dataset_dir:
            updates["dataset_dir"] = str(storage_dir / self.session_id)
        if not self.output_dir:
            updates["output_dir"] = str(storage_dir / self.session_id / "lora_output")
        resolved = self.model_copy(update=updates) if updates else self
        Path(resolved.output_dir).mkdir(parents=True, exist_ok=True)
        return resolved

    @property
    def output_path(self) -> str:
        return str(Path(self.output_dir) / "lora_output.safetensors")


async def _run_training(run_id: str, req: LoraTrainRequest) -> None:
    run = _lora_runs[run_id]

    def log_cb(line: str) -> None:
        run["logs"].append(line)
        m = re.search(r"Steps:\s*(\d+)/(\d+)", line)
        if m:
            run["current_step"] = int(m.group(1))
            run["total_steps"] = int(m.group(2))

    def progress_cb(step: int, total: int, loss: float) -> None:
        run["current_step"] = step
        run["total_steps"] = total

    cancel_event: threading.Event = run["cancel_event"]

    try:
        output_path = await asyncio.to_thread(
            train_lora,
            dataset_dir=req.dataset_dir,
            output_path=req.output_path,
            model_path=req.model_path,
            rank=req.rank,
            alpha=req.alpha,
            learning_rate=req.learning_rate,
            num_epochs=req.num_train_epochs,
            batch_size=req.batch_size,
            resolution=req.resolution,
            mixed_precision=req.mixed_precision,
            log_cb=log_cb,
            progress_cb=progress_cb,
            cancelled=cancel_event.is_set,
        )
        run["output_path"] = output_path
        run["output_checksum"] = sha256_of_file(output_path)
        run["status"] = "done"

    except RuntimeError as exc:
        run["status"] = "failed"
        run["last_error"] = str(exc)
        _cleanup_partial_output(req.output_path)

    except Exception as exc:
        run["status"] = "failed"
        run["last_error"] = str(exc)
        run["logs"].append(f"EXCEPTION: {traceback.format_exc()}")
        _cleanup_partial_output(req.output_path)

    finally:
        run["completed_at"] = time.time()


def _cleanup_partial_output(output_path: str) -> None:
    try:
        p = Path(output_path)
        if p.exists():
            p.unlink()
    except OSError:
        pass


@app.get("/pipeline/lora_train/ready")
async def lora_train_ready(session_id: str):
    """Check whether metadata.jsonl exists for the session."""
    jsonl_path = STORAGE_DIR / session_id / "metadata.jsonl"
    if not jsonl_path.exists():
        return {"ready": False, "image_count": 0, "file_size_bytes": 0}
    lines = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {
        "ready": True,
        "image_count": len(lines),
        "file_size_bytes": jsonl_path.stat().st_size,
    }


@app.post("/pipeline/lora_train/run")
async def lora_train_run(request: LoraTrainRequest):
    """Validate inputs, then launch in-process LoRA training and return a run_id."""
    request = request.resolve_paths(STORAGE_DIR)
    error = validate_inputs(
        dataset_dir=request.dataset_dir,
        output_path=request.output_path,
        model_path=request.model_path,
    )
    if error:
        raise HTTPException(status_code=422, detail=error)

    run_id = str(uuid.uuid4())
    _lora_runs[run_id] = {
        "status": "running",
        "current_step": 0,
        "total_steps": 0,
        "started_at": time.time(),
        "completed_at": None,
        "logs": [],
        "output_path": None,
        "output_checksum": None,
        "last_error": None,
        "cancel_event": threading.Event(),
    }
    asyncio.create_task(_run_training(run_id, request))
    return {"run_id": run_id}


@app.post("/pipeline/lora_train/cancel/{run_id}")
async def lora_train_cancel(run_id: str):
    """Signal the training thread to stop at the next cancellation check."""
    run = _lora_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] != "running":
        return {"message": "Run is not active"}
    run["cancel_event"].set()
    return {"message": "Cancellation requested"}


@app.get("/pipeline/lora_train/status/{run_id}")
async def lora_train_status(run_id: str):
    """Return current training status, progress, and checksum on completion."""
    run = _lora_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    elapsed = int((run["completed_at"] or time.time()) - run["started_at"])
    return {
        "status": run["status"],
        "current_step": run["current_step"],
        "total_steps": run["total_steps"],
        "elapsed_seconds": elapsed,
        "output_path": run["output_path"],
        "output_checksum": run.get("output_checksum"),
        "last_error": run["last_error"],
    }


@app.get("/pipeline/lora_train/logs/{run_id}")
async def lora_train_logs(run_id: str):
    """Return all accumulated log lines for a run."""
    run = _lora_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run["logs"]


@app.post("/lora/append")
async def append_lora_entry(request: LoraAppendRequest):
    """Refine a segment caption via Gemini and append it to the LoRA JSONL dataset."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    if not os.getenv("CHATGPT_API_KEY"):
        raise HTTPException(status_code=503, detail="CHATGPT_API_KEY not set")
    try:
        result = await service.append_lora_entry(request.session_id, request.segment_index, request.prompt)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error appending LoRA entry: {str(e)}")


@app.get("/lora/count")
async def get_lora_count():
    """Return the number of entries in the global LoRA JSONL dataset."""
    jsonl_path = STORAGE_DIR.parent / "lora_dataset.jsonl"
    if not jsonl_path.exists():
        return {"count": 0}
    count = sum(1 for line in jsonl_path.open("r", encoding="utf-8") if line.strip())
    return {"count": count}


class LoraCaptionAllRequest(BaseModel):
    session_id: str
    prompt: str


@app.post("/lora/caption_all")
async def caption_all_segments(request: LoraCaptionAllRequest):
    """Caption all segments for a session via Gemini and write to metadata.jsonl."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    if not os.getenv("CHATGPT_API_KEY"):
        raise HTTPException(status_code=503, detail="CHATGPT_API_KEY not set")
    try:
        result = await service.caption_all_segments(request.session_id, request.prompt)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error captioning segments: {str(e)}")


# ── Inference provider management ────────────────────────────────────────────

class ProviderSwitchRequest(BaseModel):
    provider: str              # "mlx" | "cloud_run"
    url: str = ""              # required when provider == "cloud_run"


@app.get("/inference/provider")
async def get_provider():
    """Return the active provider and status of all registered providers."""
    return {
        "active": get_active_provider_name(),
        "providers": list_providers(),
    }


@app.post("/inference/provider")
async def switch_provider(request: ProviderSwitchRequest):
    """Switch the active inference provider (persisted to inference_config.json)."""
    try:
        set_active_provider(request.provider, url=request.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "active": get_active_provider_name(),
        "providers": list_providers(),
    }


# ── LoRA inference run state ─────────────────────────────────────────────────

_infer_runs: Dict[str, Dict[str, Any]] = {}


class InferenceRequest(BaseModel):
    lora_path: str
    model_path: str = "black-forest-labs/FLUX.1-dev"
    prompt: str
    lora_strength: float = 0.8
    steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    output_dir: str = ""
    continuity_image_b64: Optional[str] = None
    denoise_strength: float = 0.75


def _save_inference_config(model_path: str) -> None:
    try:
        INFERENCE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        INFERENCE_CONFIG.write_text(
            json.dumps({"last_model_path": model_path}), encoding="utf-8"
        )
    except OSError:
        pass


def _maybe_preload_inference_model() -> None:
    """Background pre-load of the last-used inference model, if recorded."""
    if not INFERENCE_CONFIG.exists():
        return
    try:
        cfg = json.loads(INFERENCE_CONFIG.read_text(encoding="utf-8"))
        last_model = cfg.get("last_model_path", "")
    except Exception:
        return
    if not last_model:
        return

    async def _preload():
        try:
            print(f"[startup] pre-loading inference model {last_model!r} …")
            from lora_inferencer import _ensure_pipeline  # noqa: PLC0415
            # We need at least one LoRA path — skip if there's no recorded LoRA.
            last_lora = cfg.get("last_lora_path", "")
            if not last_lora or not Path(last_lora).exists():
                print("[startup] no valid last LoRA recorded; skipping pre-load.")
                return
            await asyncio.to_thread(
                _ensure_pipeline, last_model, last_lora, False, print
            )
            print(f"[startup] inference model ready: {last_model!r}")
        except Exception as exc:
            print(f"[startup] inference pre-load failed (non-fatal): {exc}")

    asyncio.ensure_future(_preload())


async def _run_inference_task(run_id: str, req: InferenceRequest, output_path: str) -> None:
    run = _infer_runs[run_id]

    def log_cb(line: str) -> None:
        run["logs"].append(line)

    def progress_cb(step: int, total: int) -> None:
        run["current_step"] = step
        run["total_steps"] = total

    cancel_event: threading.Event = run["cancel_event"]
    continuity_bytes: Optional[bytes] = (
        base64.b64decode(req.continuity_image_b64)
        if req.continuity_image_b64
        else None
    )

    try:
        result_path = await asyncio.to_thread(
            _run_inference_fn,
            lora_path=req.lora_path,
            model_path=req.model_path,
            prompt=req.prompt,
            lora_strength=req.lora_strength,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            output_path=output_path,
            continuity_image_bytes=continuity_bytes,
            denoise_strength=req.denoise_strength,
            log_cb=log_cb,
            progress_cb=progress_cb,
            cancelled=cancel_event.is_set,
        )
        run["output_path"] = result_path
        run["output_checksum"] = _infer_sha256(result_path)
        run["status"] = "done"
        _save_inference_config(req.model_path)
    except RuntimeError as exc:
        msg = str(exc).lower()
        run["status"] = "cancelled" if "cancel" in msg else "failed"
        run["last_error"] = str(exc)
    except Exception as exc:
        run["status"] = "failed"
        run["last_error"] = str(exc)
        run["logs"].append(f"EXCEPTION: {traceback.format_exc()}")
    finally:
        run["completed_at"] = time.time()


@app.post("/inference/generate")
async def inference_generate(request: InferenceRequest):
    """Validate inputs, start inference in the background, return a run_id."""
    error = validate_inference_inputs(
        lora_path=request.lora_path,
        model_path=request.model_path,
        prompt=request.prompt,
        output_dir=request.output_dir,
    )
    if error:
        raise HTTPException(status_code=422, detail=error)

    out_dir = request.output_dir.strip() or str(INFERENCE_DIR)
    output_path = str(Path(out_dir) / f"inference_{int(time.time()*1000)}.png")

    run_id = str(uuid.uuid4())
    _infer_runs[run_id] = {
        "status": "running",
        "current_step": 0,
        "total_steps": request.steps,
        "started_at": time.time(),
        "completed_at": None,
        "logs": [],
        "output_path": None,
        "output_checksum": None,
        "last_error": None,
        "cancel_event": threading.Event(),
    }
    asyncio.create_task(_run_inference_task(run_id, request, output_path))
    return {"run_id": run_id}


@app.get("/inference/status/{run_id}")
async def inference_status(run_id: str):
    """Return progress, status, and result path for a generation run."""
    run = _infer_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    total = run["total_steps"] or 1
    elapsed = int((run["completed_at"] or time.time()) - run["started_at"])
    return {
        "status": run["status"],
        "progress": min(run["current_step"] / total, 1.0),
        "current_step": run["current_step"],
        "total_steps": run["total_steps"],
        "elapsed_seconds": elapsed,
        "output_path": run["output_path"],
        "output_checksum": run.get("output_checksum"),
        "last_error": run["last_error"],
    }


@app.get("/inference/image/{run_id}")
async def inference_image(run_id: str):
    """Return the generated PNG as raw bytes once the run is done."""
    run = _infer_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] != "done" or not run["output_path"]:
        raise HTTPException(status_code=404, detail="Image not yet available")
    path = Path(run["output_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    return FileResponse(str(path), media_type="image/png")


@app.post("/inference/cancel/{run_id}")
async def inference_cancel(run_id: str):
    """Signal the inference thread to stop at its next cancellation check."""
    run = _infer_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] != "running":
        return {"message": "Run is not active"}
    run["cancel_event"].set()
    return {"message": "Cancellation requested"}


@app.get("/inference/logs/{run_id}")
async def inference_logs(run_id: str):
    """Return all accumulated log lines for an inference run."""
    run = _infer_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run["logs"]


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