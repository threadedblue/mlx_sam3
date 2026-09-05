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
# PyTorch + diffusers — SD 1.5 LoRA inference on CPU
# ---------------------------------------------------------------------------

try:
    import torch as _torch
    from diffusers import StableDiffusionPipeline as _SDPipeline
    from peft import LoraConfig as _LoraConfig, get_peft_model as _get_peft_model
    from safetensors.torch import load_file as _sf_load
    _PYTORCH_SD15_AVAILABLE = True
except Exception:
    _PYTORCH_SD15_AVAILABLE = False

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


# ---------------------------------------------------------------------------
# PytorchSd15Provider
# ---------------------------------------------------------------------------

def _sd15_device() -> str:
    if not _PYTORCH_SD15_AVAILABLE:
        return "cpu"
    if _torch.cuda.is_available():
        return "cuda"
    if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PytorchSd15Provider:
    """
    Runs SD 1.5 UNet-LoRA inference on CPU via PyTorch + diffusers + PEFT.

    The LoRA adapter was saved by lora_trainer.py as a flat safetensors file
    with PEFT-format keys (base_model.model.{...}.lora_A/B.default.weight).
    We re-apply the same LoRA config, load the weights, and merge before
    running inference so each forward pass is a standard UNet call.

    Caching: the merged pipeline is cached by (model_path, lora_path,
    lora_strength) — any change triggers a reload.
    """

    name = "pytorch_sd15"

    def __init__(self) -> None:
        self._pipe: Any = None
        self._cached_key: Optional[tuple] = None

    def is_available(self) -> bool:
        return _PYTORCH_SD15_AVAILABLE

    def cached_model(self) -> Optional[str]:
        return self._cached_key[0] if self._cached_key else None

    def release(self) -> None:
        self._pipe = None
        self._cached_key = None

    def status_dict(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "model_loaded": self._pipe is not None,
            "cached_model": self.cached_model(),
        }

    def run_inference(
        self,
        *,
        lora_path: str,
        model_path: str,
        prompt: str,
        lora_strength: float = 1.0,
        steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = 42,
        output_path: str,
        continuity_image_bytes: Optional[bytes] = None,
        denoise_strength: float = 0.75,
        log_cb: Callable[[str], None] = print,
        progress_cb: Callable[[int, int], None] = lambda *_: None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> str:
        if not _PYTORCH_SD15_AVAILABLE:
            raise RuntimeError(
                "SD 1.5 inference requires: pip install diffusers peft safetensors torch"
            )

        cache_key = (model_path, lora_path, lora_strength)
        if self._pipe is None or self._cached_key != cache_key:
            self._pipe = None  # release previous before loading new
            device = _sd15_device()
            dtype = _torch.float16 if device == "mps" else _torch.float32
            log_cb(f"[sd15] loading pipeline from {model_path!r} device={device} dtype={dtype}")
            pipe = _SDPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)

            log_cb(f"[sd15] applying LoRA from {lora_path!r} (strength={lora_strength})")
            lora_sd = _sf_load(lora_path)

            # Infer rank from first lora_A tensor
            rank = 16
            for k, v in lora_sd.items():
                if "lora_A" in k:
                    rank = int(v.shape[0])
                    break

            # Scale lora_B weights by lora_strength so merged UNet reflects
            # the desired contribution magnitude. Cast to pipeline dtype.
            lora_sd = {
                k: (v * lora_strength if "lora_B" in k else v).to(dtype)
                for k, v in lora_sd.items()
            }

            lora_cfg = _LoraConfig(
                r=rank,
                lora_alpha=rank,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"],
                lora_dropout=0.0,
                bias="none",
            )
            pipe.unet = _get_peft_model(pipe.unet, lora_cfg)
            missing, unexpected = pipe.unet.load_state_dict(lora_sd, strict=False)
            log_cb(f"[sd15] LoRA state: {len(missing)} missing, {len(unexpected)} unexpected")

            # Merge LoRA into base UNet weights and unwrap PEFT for fast inference.
            pipe.unet.merge_adapter()
            pipe.unet = pipe.unet.base_model.model
            log_cb("[sd15] LoRA merged into UNet")

            self._pipe = pipe
            self._cached_key = cache_key

        pipe = self._pipe
        # MPS doesn't support on-device generators; CPU generator works for seeding.
        generator = _torch.Generator("cpu").manual_seed(seed)

        def _on_step_end(pipeline, step: int, timestep, callback_kwargs):
            progress_cb(step + 1, steps)
            log_cb(f"[sd15] step {step + 1}/{steps}")
            if cancelled():
                raise RuntimeError("Generation cancelled.")
            return callback_kwargs

        log_cb(f"[sd15] generating: steps={steps} guidance={guidance_scale} seed={seed}")

        if continuity_image_bytes:
            from PIL import Image as _PIL_Image
            import io as _io
            from diffusers import StableDiffusionImg2ImgPipeline as _Img2Img
            img2img = _Img2Img(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
            init_image = _PIL_Image.open(_io.BytesIO(continuity_image_bytes)).convert("RGB")
            result = img2img(
                prompt=prompt,
                image=init_image,
                strength=denoise_strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=_on_step_end,
            )
        else:
            result = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=_on_step_end,
            )

        image = result.images[0]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        log_cb(f"[sd15] saved → {output_path}")
        return output_path
