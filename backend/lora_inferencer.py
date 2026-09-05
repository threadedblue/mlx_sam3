"""
Inference routing layer.

Reads the active provider from storage/inference_config.json and delegates
every call to it.  Providers live in inference_providers.py.

Public surface (unchanged from the original — main.py imports only these):
    validate_inference_inputs(...) -> Optional[str]
    run_inference(**kwargs) -> str
    release_pipeline()
    get_cached_model_path() -> Optional[str]
    free_vram_mb() -> Optional[float]
    sha256_of_file(path) -> str

Provider management (new — used by the /inference/provider endpoints):
    list_providers() -> List[Dict]
    get_active_provider_name() -> str
    set_active_provider(name, url="") -> None
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from inference_providers import (
    InferenceProvider,
    MlxLocalProvider,
    CloudRunProvider,
    PytorchSd15Provider,
)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_mlx_provider    = MlxLocalProvider()
_cloud_provider  = CloudRunProvider()
_sd15_provider   = PytorchSd15Provider()

_REGISTRY: Dict[str, InferenceProvider] = {
    "mlx":           _mlx_provider,
    "cloud_run":     _cloud_provider,
    "pytorch_sd15":  _sd15_provider,
}

_active_provider_name: str = "pytorch_sd15"

# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

# Resolved at call-time to avoid import-order issues with main.py's BASE_DIR.
def _config_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "storage" / "inference_config.json"


def _load_config() -> Dict[str, Any]:
    p = _config_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_config(data: Dict[str, Any]) -> None:
    p = _config_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = _load_config()
        existing.update(data)
        p.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except OSError:
        pass


def load_provider_from_config() -> None:
    """Called at startup to restore last-used provider and Cloud Run URL."""
    global _active_provider_name
    cfg = _load_config()
    name = cfg.get("provider", "pytorch_sd15")
    url  = cfg.get("cloud_run_url", "")
    if url:
        _cloud_provider.url = url
    if name in _REGISTRY:
        _active_provider_name = name


# ---------------------------------------------------------------------------
# Provider management API
# ---------------------------------------------------------------------------

def list_providers() -> List[Dict[str, Any]]:
    return [
        {
            "name":   name,
            "active": name == _active_provider_name,
            **p.status_dict(),
        }
        for name, p in _REGISTRY.items()
    ]


def get_active_provider_name() -> str:
    return _active_provider_name


def set_active_provider(name: str, url: str = "") -> None:
    global _active_provider_name
    if name not in _REGISTRY:
        raise ValueError(f"Unknown provider {name!r}. Choose from: {list(_REGISTRY)}")
    if name == "cloud_run" and url:
        _cloud_provider.url = url
    _active_provider_name = name
    data: Dict[str, Any] = {"provider": name}
    if name == "cloud_run" and url:
        data["cloud_run_url"] = url
    _save_config(data)


def _active() -> InferenceProvider:
    return _REGISTRY[_active_provider_name]


# ---------------------------------------------------------------------------
# Public surface — unchanged API used by main.py
# ---------------------------------------------------------------------------

def validate_inference_inputs(
    lora_path: str,
    model_path: str,
    prompt: str,
    output_dir: str = "",
) -> Optional[str]:
    """Provider-agnostic pre-flight checks."""
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
    """Delegate to the active provider. Blocking — run via asyncio.to_thread."""
    provider = _active()
    log_cb(f"[inferencer] provider={provider.name!r}")
    return provider.run_inference(
        lora_path=lora_path,
        model_path=model_path,
        prompt=prompt,
        lora_strength=lora_strength,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        output_path=output_path,
        continuity_image_bytes=continuity_image_bytes,
        denoise_strength=denoise_strength,
        log_cb=log_cb,
        progress_cb=progress_cb,
        cancelled=cancelled,
    )


def release_pipeline() -> None:
    """Release resources for all providers."""
    for p in _REGISTRY.values():
        try:
            p.release()
        except Exception:
            pass


def get_cached_model_path() -> Optional[str]:
    return _active().cached_model()


def free_vram_mb() -> Optional[float]:
    """Best-effort free VRAM estimate (CUDA only)."""
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return round(free / (1024 ** 2), 1)
    except Exception:
        pass
    return None


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
