"""
In-process LoRA inference for FLUX.1-dev and compatible diffusion models.

Supports txt2img and img2img with LoRA adapters in safetensors format.
Caches the base pipeline in memory; swaps LoRA adapters without full reload.

Public surface:
    validate_inference_inputs(...) -> Optional[str]
    run_inference(**kwargs) -> str          # returns output_path; raises RuntimeError
    release_pipeline()                     # free GPU/MPS memory on shutdown
    get_cached_model_path() -> Optional[str]
"""

from __future__ import annotations

import hashlib
import io
import threading
from pathlib import Path
from typing import Callable, Optional

try:
    import torch
    from diffusers import FluxPipeline
    try:
        from diffusers import FluxImg2ImgPipeline
        _IMG2IMG_AVAILABLE = True
    except ImportError:
        _IMG2IMG_AVAILABLE = False
    _INFERENCE_DEPS_AVAILABLE = True
except ImportError:
    _INFERENCE_DEPS_AVAILABLE = False
    _IMG2IMG_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pipeline cache (module-level singleton; guarded by a threading lock)
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cached_model_path: Optional[str] = None
_cached_txt2img = None       # FluxPipeline instance
_cached_img2img = None       # FluxImg2ImgPipeline instance (shares components)
_cached_lora_path: Optional[str] = None


def get_cached_model_path() -> Optional[str]:
    return _cached_model_path


def release_pipeline() -> None:
    """Move models to CPU and drop references so GC can reclaim memory."""
    global _cached_txt2img, _cached_img2img, _cached_model_path, _cached_lora_path
    with _cache_lock:
        for pipe in (_cached_txt2img, _cached_img2img):
            if pipe is not None:
                try:
                    pipe.to("cpu")
                except Exception:
                    pass
        _cached_txt2img = None
        _cached_img2img = None
        _cached_model_path = None
        _cached_lora_path = None
        _flush_device_cache()


def _flush_device_cache() -> None:
    if not _INFERENCE_DEPS_AVAILABLE:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------

def _device() -> str:
    if not _INFERENCE_DEPS_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_flux(model_path: str) -> bool:
    return "flux" in model_path.lower()


def free_vram_mb() -> Optional[float]:
    """Return approximate free VRAM in MB, or None if unavailable."""
    if not _INFERENCE_DEPS_AVAILABLE:
        return None
    try:
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return round(free / (1024 ** 2), 1)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Pipeline loading / LoRA management
# ---------------------------------------------------------------------------

def _ensure_pipeline(
    model_path: str,
    lora_path: str,
    need_img2img: bool,
    log_cb: Callable[[str], None],
):
    """Return (txt2img_pipe, img2img_pipe_or_None), loading or swapping as needed.

    The lock is held for the duration of any load/swap so concurrent requests
    queue rather than double-loading.
    """
    global _cached_model_path, _cached_txt2img, _cached_img2img, _cached_lora_path

    with _cache_lock:
        # ── reload base model when path changes ──────────────────────────────
        if _cached_model_path != model_path:
            log_cb(f"[inferencer] loading base model {model_path!r}")
            dtype = torch.bfloat16 if _is_flux(model_path) else torch.float32
            pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)
            device = _device()
            log_cb(f"[inferencer] moving to {device}")
            pipe = pipe.to(device)
            _cached_txt2img = pipe
            _cached_img2img = None
            _cached_model_path = model_path
            _cached_lora_path = None  # force LoRA reload after base change

        # ── swap LoRA when path changes ───────────────────────────────────────
        if _cached_lora_path != lora_path:
            if _cached_lora_path is not None:
                log_cb("[inferencer] unloading previous LoRA")
                _cached_txt2img.unload_lora_weights()
                _cached_img2img = None  # img2img must be rebuilt after LoRA swap
            log_cb(f"[inferencer] loading LoRA {lora_path!r}")
            _cached_txt2img.load_lora_weights(lora_path)
            _cached_lora_path = lora_path
            _cached_img2img = None

        # ── build img2img from shared components (zero extra VRAM) ───────────
        if need_img2img and _cached_img2img is None:
            if not _IMG2IMG_AVAILABLE:
                raise RuntimeError(
                    "FluxImg2ImgPipeline not available — upgrade diffusers >= 0.28.0"
                )
            log_cb("[inferencer] building img2img pipeline from txt2img components")
            _cached_img2img = FluxImg2ImgPipeline(**_cached_txt2img.components)

        return _cached_txt2img, _cached_img2img


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inference_inputs(
    lora_path: str,
    model_path: str,
    prompt: str,
    output_dir: str = "",
) -> Optional[str]:
    """Return an error string on failure, None on success. Does not load models."""
    if not prompt.strip():
        return "Prompt must not be empty"

    lora = Path(lora_path)
    if not lora.exists():
        return f"LoRA file not found: {lora_path!r}"
    if lora.suffix.lower() not in {".safetensors", ".bin"}:
        return f"LoRA must be a .safetensors file, got: {lora_path!r}"

    if output_dir:
        out = Path(output_dir)
        try:
            out.mkdir(parents=True, exist_ok=True)
            probe = out / ".write_probe"
            probe.touch()
            probe.unlink()
        except OSError as exc:
            return f"Output directory not writable: {exc}"

    local = Path(model_path)
    if local.is_dir() and not (local / "model_index.json").exists():
        return f"Local model directory {model_path!r} is missing model_index.json"

    return None


# ---------------------------------------------------------------------------
# Inference — blocking, intended to run via asyncio.to_thread
# ---------------------------------------------------------------------------

def run_inference(
    *,
    lora_path: str,
    model_path: str,
    prompt: str,
    lora_strength: float = 0.8,
    steps: int = 28,
    guidance_scale: float = 3.5,
    seed: int = 42,
    output_path: str,
    continuity_image_bytes: Optional[bytes] = None,
    denoise_strength: float = 0.75,
    log_cb: Callable[[str], None] = print,
    progress_cb: Callable[[int, int], None] = lambda *_: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> str:
    """
    Run txt2img (no continuity_image_bytes) or img2img (with continuity_image_bytes).
    Saves the result to output_path and returns it.
    Raises RuntimeError on failure, cancellation, or OOM.
    """
    if not _INFERENCE_DEPS_AVAILABLE:
        raise RuntimeError(
            "Inference dependencies not installed. "
            "Run: pip install diffusers transformers torch accelerate safetensors"
        )

    need_img2img = continuity_image_bytes is not None
    mode = "img2img" if need_img2img else "txt2img"
    log_cb(f"[inferencer] mode={mode} model={model_path!r} lora={lora_path!r}")
    log_cb(f"[inferencer] steps={steps} guidance={guidance_scale} "
           f"lora_strength={lora_strength} seed={seed}")

    try:
        txt2img_pipe, img2img_pipe = _ensure_pipeline(
            model_path, lora_path, need_img2img, log_cb
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc

    generator = torch.Generator(_device()).manual_seed(seed)
    _steps_done = [0]
    _interrupted = [False]

    def _step_cb(pipeline, step_index: int, timestep, callback_kwargs: dict):
        _steps_done[0] = step_index + 1
        progress_cb(_steps_done[0], steps)
        log_cb(f"[inferencer] step {_steps_done[0]}/{steps}")
        if cancelled():
            _interrupted[0] = True
            pipeline._interrupt = True
        return callback_kwargs

    try:
        if not need_img2img:
            # ── txt2img ──────────────────────────────────────────────────────
            output = txt2img_pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                joint_attention_kwargs={"scale": lora_strength},
                callback_on_step_end=_step_cb,
                callback_on_step_end_tensor_inputs=[],
            )
        else:
            # ── img2img ──────────────────────────────────────────────────────
            if img2img_pipe is None:
                raise RuntimeError("img2img pipeline is unavailable")
            init_image = _bytes_to_pil(continuity_image_bytes)
            output = img2img_pipe(
                prompt=prompt,
                image=init_image,
                strength=denoise_strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                joint_attention_kwargs={"scale": lora_strength},
                callback_on_step_end=_step_cb,
                callback_on_step_end_tensor_inputs=[],
            )
    except Exception as exc:
        if _interrupted[0]:
            raise RuntimeError("Generation cancelled.") from exc
        msg = str(exc).lower()
        if "out of memory" in msg or "mps backend" in msg or "oom" in msg:
            raise RuntimeError(
                f"Out of memory — try reducing steps or image size. Details: {exc}"
            ) from exc
        raise RuntimeError(f"Generation failed: {exc}") from exc

    if _interrupted[0]:
        raise RuntimeError("Generation cancelled.")

    image = output.images[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    log_cb(f"[inferencer] saved → {output_path}")
    return output_path


def _bytes_to_pil(data: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# Checksum helper (mirrors lora_trainer.sha256_of_file)
# ---------------------------------------------------------------------------

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
