"""
Cloud Run inference service — FLUX.1-dev on NVIDIA L4 via diffusers.

Endpoints mirror the local backend so CloudRunProvider can proxy transparently:
  POST /generate          → { "run_id" }
  GET  /status/{run_id}   → { status, progress, current_step, total_steps,
                               elapsed_seconds, image_b64, output_checksum,
                               last_error }
  POST /cancel/{run_id}   → { "message" }
  GET  /health            → { status, model_loaded, vram_free_mb }

Key differences from the local backend:
  - No local filesystem access: LoRA and init images arrive as base64 in the body.
  - Generated image is returned as base64 in the status response (no persistent disk).
  - Model is loaded once at startup and kept warm between requests.
  - HUGGING_FACE_HUB_TOKEN env var must be set (FLUX.1-dev is a gated model).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import os
import tempfile
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Diffusers pipeline — loaded once at startup
# ---------------------------------------------------------------------------

try:
    import torch
    from diffusers import FluxPipeline, FluxImg2ImgPipeline
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

_pipeline: Any = None           # FluxPipeline (txt2img base)
_img2img_pipeline: Any = None   # FluxImg2ImgPipeline (shares weights)
_cached_lora: Optional[str] = None   # temp path of current LoRA file


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_base_pipeline(log=print) -> None:
    global _pipeline, _img2img_pipeline
    model_id = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev")
    log(f"[startup] loading {model_id!r} …")
    _pipeline = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
    ).to(_device())
    _img2img_pipeline = FluxImg2ImgPipeline(**_pipeline.components)
    log(f"[startup] model ready on {_device()}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _DEPS_OK:
        try:
            await asyncio.to_thread(_load_base_pipeline)
        except Exception as exc:
            print(f"[startup] model load failed (inference will error): {exc}")
    yield
    if _pipeline is not None:
        try:
            _pipeline.to("cpu")
        except Exception:
            pass


app = FastAPI(
    title="FLUX Inference — Cloud Run",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------

_runs: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    lora_b64: Optional[str] = None      # safetensors file as base64
    lora_strength: float = 0.8
    steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    continuity_image_b64: Optional[str] = None
    denoise_strength: float = 0.75

# ---------------------------------------------------------------------------
# Core generation (blocking — runs in asyncio.to_thread)
# ---------------------------------------------------------------------------

_pipeline_lock = threading.Lock()   # one generation at a time on this instance


def _generate(run_id: str, req: GenerateRequest) -> None:
    run = _runs[run_id]

    def log(msg: str) -> None:
        run["logs"].append(msg)
        print(msg)

    def progress(step: int, total: int) -> None:
        run["current_step"] = step
        run["total_steps"] = total

    cancel_event: threading.Event = run["cancel_event"]

    try:
        if _pipeline is None:
            raise RuntimeError("Pipeline not loaded — check startup logs.")

        # ── LoRA ──────────────────────────────────────────────────────────────
        global _cached_lora
        lora_tmp: Optional[str] = None
        with _pipeline_lock:
            if req.lora_b64:
                lora_bytes = base64.b64decode(req.lora_b64)
                with tempfile.NamedTemporaryFile(
                    suffix=".safetensors", delete=False
                ) as tf:
                    tf.write(lora_bytes)
                    lora_tmp = tf.name
                log(f"[generate] LoRA written to {lora_tmp} ({len(lora_bytes)//1024} KB)")

                if _cached_lora != lora_tmp:
                    if _cached_lora:
                        _pipeline.unload_lora_weights()
                    _pipeline.load_lora_weights(lora_tmp)
                    _cached_lora = lora_tmp
            elif _cached_lora:
                _pipeline.unload_lora_weights()
                _cached_lora = None

            generator = torch.Generator(_device()).manual_seed(req.seed)
            steps_done = [0]
            interrupted = [False]

            def step_cb(pipeline, step_index: int, timestep, cb_kwargs: dict):
                steps_done[0] = step_index + 1
                progress(steps_done[0], req.steps)
                log(f"[generate] step {steps_done[0]}/{req.steps}")
                if cancel_event.is_set():
                    interrupted[0] = True
                    pipeline._interrupt = True
                return cb_kwargs

            if req.continuity_image_b64:
                # ── img2img ───────────────────────────────────────────────────
                init_bytes = base64.b64decode(req.continuity_image_b64)
                init_image = Image.open(io.BytesIO(init_bytes)).convert("RGB")
                output = _img2img_pipeline(
                    prompt=req.prompt,
                    image=init_image,
                    strength=req.denoise_strength,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    generator=generator,
                    joint_attention_kwargs={"scale": req.lora_strength},
                    callback_on_step_end=step_cb,
                    callback_on_step_end_tensor_inputs=[],
                )
            else:
                # ── txt2img ───────────────────────────────────────────────────
                output = _pipeline(
                    prompt=req.prompt,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    generator=generator,
                    joint_attention_kwargs={"scale": req.lora_strength},
                    callback_on_step_end=step_cb,
                    callback_on_step_end_tensor_inputs=[],
                )

        if interrupted[0]:
            raise RuntimeError("Generation cancelled.")

        # ── encode result as base64 (no persistent disk on Cloud Run) ─────────
        image = output.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        run["image_b64"] = base64.b64encode(image_bytes).decode()
        run["output_checksum"] = hashlib.sha256(image_bytes).hexdigest()
        run["status"] = "done"
        log("[generate] done")

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
        # Clean up temp LoRA file.
        if lora_tmp and Path(lora_tmp).exists():
            try:
                Path(lora_tmp).unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    vram: Optional[float] = None
    if _DEPS_OK and torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            vram = round(free / (1024 ** 2), 1)
        except Exception:
            pass
    return {
        "status": "healthy",
        "model_loaded": _pipeline is not None,
        "device": _device() if _DEPS_OK else "unavailable",
        "vram_free_mb": vram,
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    if not _DEPS_OK:
        raise HTTPException(status_code=503, detail="diffusers not installed")

    run_id = str(uuid.uuid4())
    _runs[run_id] = {
        "status": "running",
        "current_step": 0,
        "total_steps": request.steps,
        "started_at": time.time(),
        "completed_at": None,
        "logs": [],
        "image_b64": None,
        "output_checksum": None,
        "last_error": None,
        "cancel_event": threading.Event(),
    }
    asyncio.create_task(asyncio.to_thread(_generate, run_id, request))
    return {"run_id": run_id}


@app.get("/status/{run_id}")
async def status(run_id: str):
    run = _runs.get(run_id)
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
        "image_b64": run["image_b64"],           # non-null only when done
        "output_checksum": run["output_checksum"],
        "last_error": run["last_error"],
    }


@app.post("/cancel/{run_id}")
async def cancel(run_id: str):
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] != "running":
        return {"message": "Run is not active"}
    run["cancel_event"].set()
    return {"message": "Cancellation requested"}


@app.get("/logs/{run_id}")
async def logs(run_id: str):
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run["logs"]
