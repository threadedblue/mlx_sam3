"""
In-process LoRA trainer for diffusion models.

Supports UNet-based models (SD 1.5, SDXL).
FLUX.1 requires the external script approach — detect and raise clearly.

Public surface:
    validate_inputs(dataset_dir, output_path, model_path) -> Optional[str]
    train_lora(**kwargs) -> str          # returns output_path
    sha256_of_file(path) -> str
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Callable, Optional

# Training deps — imported at module level so tests can patch them.
# FastAPI startup still works if they're absent; the run endpoint validates first.
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer
    from safetensors.torch import save_file as sf_save
    _TRAINING_DEPS_AVAILABLE = True
except Exception:
    # diffusers and its dependencies can raise RuntimeError (not just ImportError)
    # when a transitive dependency is missing or version-mismatched.
    _TRAINING_DEPS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs(
    dataset_dir: str,
    output_path: str,
    model_path: str,
) -> Optional[str]:
    """
    Pre-flight validation. Returns an error string, or None if all checks pass.
    Fast — no model loading happens here.
    """
    jsonl = Path(dataset_dir) / "metadata.jsonl"

    if not jsonl.exists():
        return f"metadata.jsonl not found in {dataset_dir!r}"

    # Validate every JSONL line and referenced image
    with jsonl.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                return f"metadata.jsonl line {lineno}: invalid JSON — {exc}"
            for key in ("file_name", "text"):
                if key not in rec:
                    return f"metadata.jsonl line {lineno}: missing {key!r}"
            img = Path(dataset_dir) / rec["file_name"]
            if not img.exists():
                return f"Image not found: {img}"

    # Output directory must be writable (create it if needed)
    out_dir = Path(output_path).parent
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".write_probe"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        return f"Output directory not writable: {exc}"

    # Refuse FLUX early with a clear message
    if _is_flux(model_path):
        return (
            "FLUX.1 training requires the external script "
            "(train_dreambooth_lora_flux.py). "
            "Use a UNet-based model such as "
            "'runwayml/stable-diffusion-v1-5' or 'stabilityai/stable-diffusion-xl-base-1.0', "
            "or set script_path to the diffusers FLUX training script."
        )

    # Model must exist locally or on the Hub
    err = _check_model(model_path)
    if err:
        return err

    return None


def _is_flux(model_path: str) -> bool:
    p = model_path.lower()
    return "flux" in p


def _check_model(model_path: str) -> Optional[str]:
    local = Path(model_path)
    if local.is_dir():
        if not (local / "model_index.json").exists():
            return f"Local model directory {model_path!r} has no model_index.json"
        return None
    try:
        from huggingface_hub import model_info
        model_info(model_path)
        return None
    except Exception as exc:
        return f"Cannot resolve model {model_path!r}: {exc}"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MetadataDataset:
    """PyTorch-compatible dataset loaded from metadata.jsonl."""

    def __init__(self, dataset_dir: str, resolution: int) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.records: list[dict] = []

        jsonl = self.dataset_dir / "metadata.jsonl"
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image  # Pillow is always available
        rec = self.records[idx]
        img = Image.open(self.dataset_dir / rec["file_name"]).convert("RGB")
        return {"pixel_values": self.transform(img), "caption": rec["text"]}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lora(
    *,
    dataset_dir: str,
    output_path: str,
    model_path: str,
    rank: int = 16,
    alpha: int = 16,
    learning_rate: float = 1e-4,
    num_epochs: int = 1,
    batch_size: int = 1,
    resolution: int = 512,
    mixed_precision: str = "no",
    log_cb: Callable[[str], None] = print,
    progress_cb: Callable[[int, int, float], None] = lambda *_: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> str:
    """
    Train a LoRA adapter and save it to output_path using safetensors.
    Blocking — run via asyncio.to_thread.

    Returns output_path on success. Raises RuntimeError on failure or cancellation.
    """
    if not _TRAINING_DEPS_AVAILABLE:
        raise RuntimeError(
            "Training dependencies not installed. "
            "Run: pip install diffusers peft transformers safetensors torch torchvision"
        )

    device = _device()
    dtype = _dtype(mixed_precision, device)
    log_cb(f"[trainer] device={device} dtype={dtype} rank={rank} alpha={alpha}")

    # Release any MLX/MPS cached memory before loading PyTorch models so both
    # frameworks don't compete for the same unified memory budget.
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
        log_cb("[trainer] MLX metal cache cleared")
    except Exception:
        pass

    # ── load components ────────────────────────────────────────────────────
    log_cb(f"[trainer] loading tokenizer from {model_path!r}")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")

    # Keep text_encoder and VAE on CPU — they are frozen and only need a
    # forward pass; moving results to `device` per-batch saves ~1.3 GiB MPS.
    log_cb("[trainer] loading text_encoder (cpu)")
    text_enc = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(dtype)
    text_enc.requires_grad_(False)

    log_cb("[trainer] loading vae (cpu)")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(dtype)
    vae.requires_grad_(False)

    log_cb("[trainer] loading unet")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

    noise_sched = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    # ── apply LoRA ─────────────────────────────────────────────────────────
    # Wrap with PEFT *before* moving to device so LoRA layers land on the
    # same device as the base weights (moving after ensures all params move).
    log_cb(f"[trainer] applying LoRA r={rank} alpha={alpha}")
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet = unet.to(device, dtype)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    actual_device = next(unet.parameters()).device
    log_cb(f"[trainer] trainable parameters: {trainable:,}")
    log_cb(f"[trainer] UNet confirmed on: {actual_device}  (target: {device})")

    # ── dataset & optimizer ────────────────────────────────────────────────
    dataset = MetadataDataset(dataset_dir, resolution)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    total_steps = len(loader) * num_epochs

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    log_cb(f"[trainer] {len(dataset)} images · {total_steps} steps")

    # ── training loop ──────────────────────────────────────────────────────
    unet.train()
    global_step = 0

    for epoch in range(num_epochs):
        log_cb(f"[trainer] epoch {epoch + 1}/{num_epochs}")

        for batch in loader:
            if cancelled():
                raise RuntimeError("Training cancelled.")

            pixels   = batch["pixel_values"].to(dtype)  # CPU for VAE encode
            captions = batch["caption"]

            with torch.no_grad():
                # Encode on CPU then move result to MPS (saves ~1.3 GiB)
                latents = (vae.encode(pixels).latent_dist.sample()
                           * vae.config.scaling_factor).to(device)
                tok_out = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                cond = text_enc(tok_out.input_ids)[0].to(device)

            noise     = torch.randn_like(latents)
            t         = torch.randint(0, noise_sched.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_lat = noise_sched.add_noise(latents, noise, t)

            # autocast device_type must be "cuda" or "cpu" — MPS uses "cpu" path.
            ac_device = "cuda" if device == "cuda" else "cpu"
            with torch.autocast(device_type=ac_device,
                                 enabled=(mixed_precision != "no" and device == "cuda"),
                                 dtype=dtype):
                pred = unet(noisy_lat, t, cond).sample
                loss = F.mse_loss(pred.float(), noise.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in unet.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            loss_val = loss.item()
            log_cb(f"Steps: {global_step}/{total_steps}  loss: {loss_val:.4f}")
            progress_cb(global_step, total_steps, loss_val)

    # ── save ──────────────────────────────────────────────────────────────
    log_cb("[trainer] saving LoRA weights")
    lora_sd: dict[str, torch.Tensor] = {
        name: param.data.cpu().float()
        for name, param in unet.named_parameters()
        if "lora_" in name and param.requires_grad
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf_save(lora_sd, output_path)
    log_cb(f"[trainer] saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _dtype(mixed_precision: str, device: str):
    import torch
    if device == "cuda":
        if mixed_precision == "bf16":
            return torch.bfloat16
        if mixed_precision == "fp16":
            return torch.float16
    if device == "mps":
        # float16 causes NaN loss on MPS during backward pass; use float32.
        return torch.float32
    return torch.float32
