"""
Inference provider implementations.

MlxLocalProvider   — in-process FLUX on Apple Silicon via mflux
CloudRunProvider   — HTTP proxy to a remote Cloud Run / GCE service

Both implement the same blocking run_inference() signature so main.py can
wrap either with asyncio.to_thread unchanged.

Provider selection and config live in lora_inferencer.py; this file is pure
implementation with no FastAPI or config-file dependencies.
"""

from __future__ import annotations

import base64
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# mflux — MLX-native FLUX inference (Apple Silicon)
# ---------------------------------------------------------------------------

try:
    from mflux import Flux1, Config, ModelConfig
    try:
        from mflux import StopImageGenerationException as _StopGen
    except Exception:
        _StopGen = None  # older mflux: raise a plain RuntimeError instead
    _MFLUX_AVAILABLE = True
except Exception:
    # Catch broadly: mflux and its deps can raise RuntimeError, not just ImportError.
    _MFLUX_AVAILABLE = False

# ---------------------------------------------------------------------------
# httpx — used by CloudRunProvider for blocking HTTP calls inside a thread
# ---------------------------------------------------------------------------

try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class InferenceProvider(Protocol):
    """Minimal interface every provider must satisfy."""
    name: str

    def run_inference(
        self,
        *,
        lora_path: str,
        model_path: str,
        prompt: str,
        lora_strength: float,
        steps: int,
        guidance_scale: float,
        seed: int,
        output_path: str,
        continuity_image_bytes: Optional[bytes],
        denoise_strength: float,
        log_cb: Callable[[str], None],
        progress_cb: Callable[[int, int], None],
        cancelled: Callable[[], bool],
    ) -> str: ...

    def is_available(self) -> bool: ...
    def release(self) -> None: ...
    def cached_model(self) -> Optional[str]: ...
    def status_dict(self) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_argmaxinc_snapshot() -> Optional[str]:
    """Return the local snapshot path of argmaxinc/mlx-FLUX.1-dev, or None."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for variant in ("models--argmaxinc--mlx-FLUX.1-dev",):
        snaps = hub / variant / "snapshots"
        if not snaps.exists():
            continue
        for snap in snaps.iterdir():
            if (snap / "flux1-dev.safetensors").exists():
                return str(snap)
    return None


# ---------------------------------------------------------------------------
# MlxLocalProvider
# ---------------------------------------------------------------------------

class MlxLocalProvider:
    """
    Runs FLUX.1-dev inference in-process on Apple Silicon using mflux.

    Model weights are loaded once and cached.  LoRA is applied per-request
    (mflux re-instantiates the transformer with LoRA fused, but the VAE and
    text encoders are shared via the module-level cache when the same base
    model is used).

    Caching strategy:
      - Base model stays loaded across requests.
      - LoRA changes trigger a new Flux1 instance (mflux does not hot-swap LoRA).
    """

    name = "mlx"

    def __init__(self) -> None:
        self._flux: Any = None          # Flux1 instance
        self._cached_lora: Optional[str] = None
        self._cached_model_alias: Optional[str] = None
        self._local_path: Optional[str] = _find_argmaxinc_snapshot()

    # ── public ────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return _MFLUX_AVAILABLE

    def cached_model(self) -> Optional[str]:
        return self._cached_model_alias

    def release(self) -> None:
        self._flux = None
        self._cached_lora = None
        self._cached_model_alias = None

    def status_dict(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "model_loaded": self._flux is not None,
            "cached_model": self._cached_model_alias,
            "local_weights": self._local_path,
        }

    def run_inference(
        self,
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
        if not _MFLUX_AVAILABLE:
            raise RuntimeError(
                "mflux is not installed. Run: pip install mflux>=0.9.0"
            )

        alias = self._model_alias(model_path)
        lora_paths = [lora_path] if lora_path else []
        lora_scales = [lora_strength] if lora_paths else []

        flux = self._get_flux(alias, lora_paths, lora_scales, log_cb)

        mode = "img2img" if continuity_image_bytes else "txt2img"
        log_cb(f"[mlx] {mode} alias={alias!r} lora={lora_path!r} steps={steps} seed={seed}")

        _done = [False]
        _interrupted = [False]

        def _step_cb(step: int, total: int, *_):
            progress_cb(step, total)
            log_cb(f"[mlx] step {step}/{total}")
            if cancelled():
                _interrupted[0] = True
                if _StopGen is not None:
                    raise _StopGen("cancelled")
                raise RuntimeError("Generation cancelled.")

        cfg_kwargs: Dict[str, Any] = dict(
            num_inference_steps=steps,
            guidance=guidance_scale,
        )

        if continuity_image_bytes:
            # Write init image to a temp file so mflux can read it by path.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tf.write(continuity_image_bytes)
                init_path = tf.name
            cfg_kwargs["image_path"] = init_path
            cfg_kwargs["image_strength"] = denoise_strength

        config = Config(**cfg_kwargs)

        try:
            result = flux.generate_image(
                seed=seed,
                prompt=prompt,
                config=config,
                step_callback=_step_cb,
            )
        except Exception as exc:
            if _interrupted[0] or "cancel" in str(exc).lower():
                raise RuntimeError("Generation cancelled.") from exc
            msg = str(exc).lower()
            if "out of memory" in msg or "oom" in msg:
                raise RuntimeError(
                    f"Out of memory — reduce steps or image size. Details: {exc}"
                ) from exc
            raise RuntimeError(f"MLX generation failed: {exc}") from exc

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(path=output_path)
        log_cb(f"[mlx] saved → {output_path}")
        return output_path

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _model_alias(model_path: str) -> str:
        p = model_path.lower()
        if "schnell" in p:
            return "flux-1-schnell"
        return "flux-1-dev"

    def _get_flux(
        self,
        alias: str,
        lora_paths: list,
        lora_scales: list,
        log_cb: Callable[[str], None],
    ):
        lora_key = (tuple(lora_paths), tuple(lora_scales))
        if (
            self._flux is not None
            and self._cached_model_alias == alias
            and self._cached_lora == lora_key
        ):
            return self._flux

        log_cb(f"[mlx] loading model alias={alias!r} lora={lora_paths}")
        kwargs: Dict[str, Any] = dict(quantize=8)
        if self._local_path:
            log_cb(f"[mlx] using local weights at {self._local_path!r}")
            kwargs["local_path"] = self._local_path
        if lora_paths:
            kwargs["lora_paths"] = lora_paths
            kwargs["lora_scales"] = lora_scales

        self._flux = Flux1.from_alias(alias, **kwargs)
        self._cached_model_alias = alias
        self._cached_lora = lora_key
        return self._flux


# ---------------------------------------------------------------------------
# CloudRunProvider
# ---------------------------------------------------------------------------

class CloudRunProvider:
    """
    Proxies inference to a remote Cloud Run (or GCE) service.

    The remote service speaks a simplified API:
      POST  /generate          → { "run_id" }
      GET   /status/{run_id}   → { status, progress, current_step, total_steps,
                                   elapsed_seconds, image_b64, output_checksum,
                                   last_error }
      POST  /cancel/{run_id}   → { "message" }

    The LoRA file is sent as base64 in the request body because the remote
    container cannot access the local filesystem.  The image is returned as
    base64 in the status response and written to a local output_path.
    """

    name = "cloud_run"
    _POLL_INTERVAL = 2.0   # seconds between status polls
    _TIMEOUT = 3600        # max seconds to wait for a result

    def __init__(self, url: str = "") -> None:
        self.url = url.rstrip("/")

    # ── public ────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return bool(self.url) and _HTTPX_AVAILABLE

    def cached_model(self) -> Optional[str]:
        return "FLUX.1-dev (remote)" if self.url else None

    def release(self) -> None:
        pass  # stateless proxy

    def status_dict(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "url": self.url or "(not configured)",
        }

    def run_inference(
        self,
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
        if not _HTTPX_AVAILABLE:
            raise RuntimeError("httpx is not installed. Run: pip install httpx")
        if not self.url:
            raise RuntimeError(
                "Cloud Run URL is not configured. "
                "Set it via POST /inference/provider."
            )

        log_cb(f"[cloud_run] sending to {self.url!r}")

        body: Dict[str, Any] = {
            "prompt": prompt,
            "lora_strength": lora_strength,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "denoise_strength": denoise_strength,
        }

        # Encode LoRA as base64 so the remote container can load it.
        if lora_path:
            lora_bytes = Path(lora_path).read_bytes()
            body["lora_b64"] = base64.b64encode(lora_bytes).decode()
            log_cb(f"[cloud_run] lora payload {len(lora_bytes) / 1024:.0f} KB")

        if continuity_image_bytes:
            body["continuity_image_b64"] = base64.b64encode(
                continuity_image_bytes
            ).decode()

        with _httpx.Client(timeout=30) as client:
            resp = client.post(f"{self.url}/generate", json=body)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Remote /generate failed ({resp.status_code}): {resp.text[:200]}"
            )
        run_id: str = resp.json()["run_id"]
        log_cb(f"[cloud_run] run_id={run_id!r}")

        # Poll for completion.
        deadline = time.time() + self._TIMEOUT
        with _httpx.Client(timeout=30) as client:
            while time.time() < deadline:
                if cancelled():
                    try:
                        client.post(f"{self.url}/cancel/{run_id}")
                    except Exception:
                        pass
                    raise RuntimeError("Generation cancelled.")

                time.sleep(self._POLL_INTERVAL)
                try:
                    resp = client.get(f"{self.url}/status/{run_id}")
                except Exception as exc:
                    log_cb(f"[cloud_run] poll error: {exc}")
                    continue

                if resp.status_code != 200:
                    continue

                st = resp.json()
                progress_cb(st.get("current_step", 0), st.get("total_steps", steps))
                log_cb(
                    f"[cloud_run] {st['status']} "
                    f"{st.get('current_step', 0)}/{st.get('total_steps', steps)}"
                )

                if st["status"] == "done":
                    image_b64 = st.get("image_b64")
                    if not image_b64:
                        raise RuntimeError(
                            "Remote returned done but no image_b64 in response"
                        )
                    image_bytes = base64.b64decode(image_b64)
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(output_path).write_bytes(image_bytes)
                    log_cb(f"[cloud_run] saved → {output_path}")
                    return output_path

                if st["status"] in ("failed", "cancelled"):
                    raise RuntimeError(
                        st.get("last_error") or f"Remote run {st['status']}"
                    )

        raise RuntimeError(f"Timed out waiting for Cloud Run after {self._TIMEOUT}s")
