"""
Unit tests for the LoRA training backend.

Run with:  pytest app/backend/tests/test_lora_train.py -v
"""

import json
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    """A minimal valid dataset directory: one image + metadata.jsonl."""
    from PIL import Image

    img_path = tmp_path / "img_001.png"
    Image.new("RGB", (64, 64), color=(128, 64, 32)).save(img_path)

    meta = tmp_path / "metadata.jsonl"
    meta.write_text(
        json.dumps({"file_name": "img_001.png", "text": "a small red cube"}) + "\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# validate_inputs tests
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_passes_for_valid_sd_dataset(self, dataset_dir, output_dir):
        from lora_trainer import validate_inputs

        with patch("lora_trainer._check_model", return_value=None):
            result = validate_inputs(
                dataset_dir=str(dataset_dir),
                output_path=str(output_dir / "lora.safetensors"),
                model_path="runwayml/stable-diffusion-v1-5",
            )
        assert result is None

    def test_fails_when_jsonl_missing(self, tmp_path, output_dir):
        from lora_trainer import validate_inputs

        err = validate_inputs(
            dataset_dir=str(tmp_path),
            output_path=str(output_dir / "lora.safetensors"),
            model_path="runwayml/stable-diffusion-v1-5",
        )
        assert err is not None
        assert "metadata.jsonl" in err

    def test_fails_for_invalid_jsonl(self, tmp_path, output_dir):
        from lora_trainer import validate_inputs

        (tmp_path / "metadata.jsonl").write_text("not valid json\n")
        err = validate_inputs(
            dataset_dir=str(tmp_path),
            output_path=str(output_dir / "lora.safetensors"),
            model_path="runwayml/stable-diffusion-v1-5",
        )
        assert err is not None
        assert "JSON" in err

    def test_fails_when_image_missing(self, tmp_path, output_dir):
        from lora_trainer import validate_inputs

        (tmp_path / "metadata.jsonl").write_text(
            json.dumps({"file_name": "ghost.png", "text": "nothing"}) + "\n"
        )
        err = validate_inputs(
            dataset_dir=str(tmp_path),
            output_path=str(output_dir / "lora.safetensors"),
            model_path="runwayml/stable-diffusion-v1-5",
        )
        assert err is not None
        assert "ghost.png" in err

    def test_rejects_flux_model_with_clear_message(self, dataset_dir, output_dir):
        from lora_trainer import validate_inputs

        err = validate_inputs(
            dataset_dir=str(dataset_dir),
            output_path=str(output_dir / "lora.safetensors"),
            model_path="black-forest-labs/FLUX.1-dev",
        )
        assert err is not None
        assert "FLUX" in err


# ---------------------------------------------------------------------------
# sha256_of_file test
# ---------------------------------------------------------------------------

def test_sha256_is_deterministic(tmp_path):
    from lora_trainer import sha256_of_file

    f = tmp_path / "data.bin"
    f.write_bytes(b"\x00\x01\x02\x03" * 1024)

    h1 = sha256_of_file(str(f))
    h2 = sha256_of_file(str(f))
    assert h1 == h2
    assert len(h1) == 64   # SHA-256 hex digest length


# ---------------------------------------------------------------------------
# train_lora: mocked end-to-end tests
# ---------------------------------------------------------------------------

import struct as _struct


def _write_fake_safetensors(path: str) -> None:
    """Write a minimal valid-looking safetensors file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode()
    p.write_bytes(_struct.pack("<Q", len(header)) + header + b"\x00" * 16)


def _make_train_mocks():
    """
    Return a dict of properly-chained mocks for train_lora.

    Key design decisions:
    - .to() on model mocks returns self, so attribute chains survive the call.
    - Latents and tokenizer ids are real tensors so torch.randn_like / tensor ops work.
    - The optimizer receives a real nn.Parameter so AdamW does not raise.
    - loss mock has .item() = 0.05 so the log f-string formats without error.
    """
    import torch

    latents   = torch.zeros(1, 4, 8, 8)   # real tensor for randn_like
    cond      = torch.zeros(1, 77, 768)
    real_ids  = torch.zeros(1, 77, dtype=torch.long)
    real_param = torch.nn.Parameter(torch.zeros(4, 4))  # real tensor for AdamW

    fake_loss = MagicMock()
    fake_loss.item.return_value = 0.05
    fake_loss.backward = MagicMock()

    fake_F = MagicMock()
    fake_F.mse_loss.return_value = fake_loss

    tokenizer = MagicMock()
    tokenizer.model_max_length = 77
    tokenizer.return_value.input_ids = real_ids  # .to(device) on real tensor works

    text_enc = MagicMock()
    text_enc.to.return_value = text_enc      # survives .to() call
    text_enc.return_value = (cond,)           # text_enc(ids) → (cond,)

    vae = MagicMock()
    vae.to.return_value = vae
    vae.config.scaling_factor = 0.18215
    # vae.encode(x).latent_dist.sample() → latents (real tensor)
    vae.encode.return_value.latent_dist.sample.return_value = latents

    sched = MagicMock()
    sched.config.num_train_timesteps = 1000
    sched.add_noise.return_value = latents   # noisy_latents (real tensor)

    raw_unet = MagicMock()
    raw_unet.to.return_value = raw_unet

    peft_unet = MagicMock()
    peft_unet.return_value.sample = latents   # pred (real tensor)
    peft_unet.parameters.return_value = [real_param]
    peft_unet.named_parameters.return_value = [
        ("unet.lora_A.weight", real_param),   # "lora_" substring → included in lora_sd
    ]

    batch = {"pixel_values": torch.zeros(1, 3, 64, 64), "caption": ["a cat"]}

    return dict(
        tokenizer=tokenizer, text_enc=text_enc, vae=vae, sched=sched,
        raw_unet=raw_unet, peft_unet=peft_unet, fake_F=fake_F, batch=batch,
    )


def _train_lora_patches(mocks: dict):
    """Return a list of patch context managers using the provided mocks."""
    _c = dict(create=True)
    peft_unet = mocks["peft_unet"]
    return [
        patch("lora_trainer._TRAINING_DEPS_AVAILABLE", True),
        patch("lora_trainer.CLIPTokenizer",         **_c),
        patch("lora_trainer.CLIPTextModel",          **_c),
        patch("lora_trainer.AutoencoderKL",          **_c),
        patch("lora_trainer.UNet2DConditionModel",   **_c),
        patch("lora_trainer.DDPMScheduler",          **_c),
        patch("lora_trainer.get_peft_model", return_value=peft_unet, **_c),
        patch("lora_trainer.LoraConfig",             **_c),
        patch("lora_trainer.F",                      **_c),
        patch("lora_trainer.DataLoader",
              return_value=[mocks["batch"]], **_c),
    ]


def _configure_cls_mocks(patches_entered: list, mocks: dict) -> None:
    """Wire from_pretrained() return values after patches are active."""
    # patches_entered order matches _train_lora_patches (indices 1-5 are the class mocks)
    tok_cls, enc_cls, vae_cls, unet_cls, sched_cls = patches_entered[1:6]
    tok_cls.from_pretrained.return_value  = mocks["tokenizer"]
    enc_cls.from_pretrained.return_value  = mocks["text_enc"]
    vae_cls.from_pretrained.return_value  = mocks["vae"]
    unet_cls.from_pretrained.return_value = mocks["raw_unet"]
    sched_cls.from_pretrained.return_value = mocks["sched"]
    # Patch F
    patches_entered[8].__dict__.update(mocks["fake_F"].__dict__)


def test_train_lora_writes_file_and_returns_path(dataset_dir, output_dir):
    """
    With all heavy ML deps mocked, train_lora() must:
      - complete without error
      - write the safetensors file to the requested path
      - return that path
      - emit at least one Steps: log line
    """
    from contextlib import ExitStack
    from lora_trainer import train_lora, sha256_of_file

    out  = str(output_dir / "lora.safetensors")
    logs: list[str] = []
    mocks = _make_train_mocks()

    with ExitStack() as stack:
        entered = [stack.enter_context(p) for p in _train_lora_patches(mocks)]
        _configure_cls_mocks(entered, mocks)

        # sf_save side-effect writes a real file
        sf_mock = stack.enter_context(
            patch("lora_trainer.sf_save", side_effect=lambda sd, path: _write_fake_safetensors(path),
                  create=True)
        )

        result = train_lora(
            dataset_dir=str(dataset_dir), output_path=out,
            model_path="runwayml/stable-diffusion-v1-5",
            rank=4, alpha=4, learning_rate=1e-4,
            num_epochs=1, batch_size=1, resolution=64,
            mixed_precision="no",
            log_cb=logs.append, progress_cb=lambda *_: None,
            cancelled=lambda: False,
        )

    assert result == out, f"returned {result!r}, expected {out!r}"
    assert Path(out).exists(), "safetensors file was not written to disk"

    checksum = sha256_of_file(out)
    assert len(checksum) == 64
    assert all(c in "0123456789abcdef" for c in checksum)

    step_logs = [ln for ln in logs if "Steps:" in ln]
    assert step_logs, f"No 'Steps:' log found. Full logs: {logs}"


def test_train_lora_cancels_cleanly(dataset_dir, output_dir):
    """
    When cancelled() returns True at the first batch, train_lora must raise
    RuntimeError containing 'cancel' (case-insensitive) before doing any work.
    """
    from contextlib import ExitStack
    from lora_trainer import train_lora

    out   = str(output_dir / "lora_cancelled.safetensors")
    mocks = _make_train_mocks()

    with ExitStack() as stack:
        entered = [stack.enter_context(p) for p in _train_lora_patches(mocks)]
        _configure_cls_mocks(entered, mocks)
        stack.enter_context(patch("lora_trainer.sf_save", create=True))

        with pytest.raises(RuntimeError, match="(?i)cancel"):
            train_lora(
                dataset_dir=str(dataset_dir), output_path=out,
                model_path="runwayml/stable-diffusion-v1-5",
                rank=4, alpha=4, learning_rate=1e-4,
                num_epochs=1, batch_size=1, resolution=64,
                mixed_precision="no",
                log_cb=lambda _: None, progress_cb=lambda *_: None,
                cancelled=lambda: True,   # cancel at first batch check
            )
