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
from fastapi import FastAPI, File, UploadFile, HTTPException
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
import google.generativeai as genai
from dotenv import load_dotenv

from services import SegmentationService, serialize_state

# Global model and processor
model = None
processor = None
service = None

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR.parent.parent / "storage" / "sessions"

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

    # Configure Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not set. AI captioning will be disabled.")
    else:
        genai.configure(api_key=api_key)
        print("Gemini SDK configured.")

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


class SessionSettingsRequest(BaseModel):
    session_id: str
    settings: Dict[str, Any]


class LoraAppendRequest(BaseModel):
    session_id: str
    segment_index: int
    prompt: str


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


@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """Generate a descriptive caption for an image and save both."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=503, detail="AI captioning is not configured on the server (GEMINI_API_KEY is missing).")

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
    """Returns a list of URLs for the generated segments."""
    segments_dir = service.storage_dir / session_id / "segments_raw"
    if not segments_dir.exists():
        return []

    segment_files = sorted(segments_dir.glob("*.png"))
    base_path = f"/storage/sessions/{session_id}/segments_raw" # This path is for the static mount
    urls = [f"{base_path}/{f.name}" for f in segment_files]
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
_lora_runs: Dict[str, Dict[str, Any]] = {}


class LoraTrainRequest(BaseModel):
    dataset_dir: str
    output_dir: str
    script_path: str = "train_dreambooth_lora_flux.py"
    rank: int = 16
    learning_rate: float = 1e-4
    num_train_epochs: int = 1
    resolution: int = 1024
    mixed_precision: str = "bf16"


async def _run_training(run_id: str, req: LoraTrainRequest) -> None:
    run = _lora_runs[run_id]
    cmd = [
        sys.executable, req.script_path,
        "--pretrained_model_name_or_path", "black-forest-labs/FLUX.1-dev",
        "--instance_data_dir", req.dataset_dir,
        "--output_dir", req.output_dir,
        "--resolution", str(req.resolution),
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--num_train_epochs", str(req.num_train_epochs),
        "--learning_rate", str(req.learning_rate),
        "--lr_scheduler", "constant",
        "--mixed_precision", req.mixed_precision,
        "--rank", str(req.rank),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            run["logs"].append(line)
            m = re.search(r"Steps:\s*(\d+)/(\d+)", line)
            if m:
                run["current_step"] = int(m.group(1))
                run["total_steps"] = int(m.group(2))
        await proc.wait()
        if proc.returncode == 0:
            safetensors = sorted(
                Path(req.output_dir).glob("**/*.safetensors"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            run["output_path"] = str(safetensors[0]) if safetensors else None
            run["status"] = "done"
        else:
            run["status"] = "failed"
            run["last_error"] = next(
                (l for l in reversed(run["logs"]) if l.strip()), "Unknown error"
            )
    except Exception as exc:
        run["status"] = "failed"
        run["last_error"] = str(exc)
        run["logs"].append(f"EXCEPTION: {exc}")
    finally:
        run["completed_at"] = time.time()


@app.get("/pipeline/lora_train/ready")
async def lora_train_ready(dataset_dir: str):
    """Check whether metadata.jsonl exists in dataset_dir."""
    jsonl_path = Path(dataset_dir) / "metadata.jsonl"
    if not jsonl_path.exists():
        return {"ready": False, "image_count": 0, "file_size_bytes": 0}
    lines = [l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {
        "ready": True,
        "image_count": len(lines),
        "file_size_bytes": jsonl_path.stat().st_size,
    }


@app.post("/pipeline/lora_train/run")
async def lora_train_run(request: LoraTrainRequest):
    """Launch training as a background subprocess and return a run_id."""
    run_id = str(uuid.uuid4())
    _lora_runs[run_id] = {
        "status": "running",
        "current_step": 0,
        "total_steps": 0,
        "started_at": time.time(),
        "completed_at": None,
        "logs": [],
        "output_path": None,
        "last_error": None,
    }
    asyncio.create_task(_run_training(run_id, request))
    return {"run_id": run_id}


@app.get("/pipeline/lora_train/status/{run_id}")
async def lora_train_status(run_id: str):
    """Return current training status and progress."""
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
        "last_error": run["last_error"],
    }


@app.get("/pipeline/lora_train/logs/{run_id}")
async def lora_train_logs(run_id: str):
    """Return all accumulated stdout/stderr lines for a run."""
    run = _lora_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run["logs"]


@app.post("/lora/append")
async def append_lora_entry(request: LoraAppendRequest):
    """Refine a segment caption via Gemini and append it to the LoRA JSONL dataset."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not set")
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