"""
Unit tests for the LoRA inference backend.

Run with:  pytest app/backend/tests/test_lora_infer.py -v
"""

import io
import json
import struct
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_pipeline_cache():
    """Clear module-level pipeline cache before and after every test."""
    import lora_inferencer as _m
    _m._cached_model_path = None
    _m._cached_txt2img = None
    _m._cached_img2img = None
    _m._cached_lora_path = None
    yield
    _m._cached_model_path = None
    _m._cached_txt2img = None
    _m._cached_img2img = None
    _m._cached_lora_path = None


@pytest.fixture()
def lora_file(tmp_path: Path) -> Path:
    """Minimal fake .safetensors file on disk."""
    p = tmp_path / "adapter.safetensors"
    header = json.dumps({"__metadata__": {}}).encode()
    p.write_bytes(struct.pack("<Q", len(header)) + header)
    return p


@pytest.fixture()
def output_path(tmp_path: Path) -> str:
    return str(tmp_path / "out.png")


@pytest.fixture()
def continuity_bytes() -> bytes:
    """A tiny in-memory PNG that acts as the continuity / init image."""
    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), color=(0, 128, 255)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_output_image() -> PILImage.Image:
    return PILImage.new("RGB", (512, 512), color=(200, 100, 50))


def _make_txt2img_mock():
    """A FluxPipeline mock that produces a real PIL image when called."""
    fake_out = MagicMock()
    fake_out.images = [_fake_output_image()]

    pipe = MagicMock()
    pipe.return_value = fake_out
    pipe.to.return_value = pipe
    pipe.load_lora_weights = MagicMock()
    pipe.unload_lora_weights = MagicMock()
    pipe.components = {"transformer": MagicMock(), "vae": MagicMock()}
    return pipe


def _make_img2img_mock():
    """A FluxImg2ImgPipeline mock that produces a real PIL image when called."""
    fake_out = MagicMock()
    fake_out.images = [_fake_output_image()]

    pipe = MagicMock()
    pipe.return_value = fake_out
    pipe.to.return_value = pipe
    return pipe


# ---------------------------------------------------------------------------
# validate_inference_inputs
# ---------------------------------------------------------------------------

class TestValidateInferenceInputs:
    def test_passes_for_valid_inputs(self, lora_file, tmp_path):
        from lora_inferencer import validate_inference_inputs
        err = validate_inference_inputs(
            lora_path=str(lora_file),
            model_path="black-forest-labs/FLUX.1-dev",
            prompt="a red sphere",
            output_dir=str(tmp_path / "out"),
        )
        assert err is None

    def test_fails_when_prompt_empty(self, lora_file):
        from lora_inferencer import validate_inference_inputs
        err = validate_inference_inputs(
            lora_path=str(lora_file),
            model_path="black-forest-labs/FLUX.1-dev",
            prompt="   ",
        )
        assert err is not None
        assert "prompt" in err.lower()

    def test_fails_when_lora_missing(self, tmp_path):
        from lora_inferencer import validate_inference_inputs
        err = validate_inference_inputs(
            lora_path=str(tmp_path / "ghost.safetensors"),
            model_path="black-forest-labs/FLUX.1-dev",
            prompt="a test",
        )
        assert err is not None
        assert "not found" in err.lower()

    def test_fails_for_wrong_lora_extension(self, tmp_path):
        from lora_inferencer import validate_inference_inputs
        bad = tmp_path / "adapter.ckpt"
        bad.write_bytes(b"\x00" * 8)
        err = validate_inference_inputs(
            lora_path=str(bad),
            model_path="black-forest-labs/FLUX.1-dev",
            prompt="a test",
        )
        assert err is not None
        assert "safetensors" in err.lower()

    def test_fails_when_local_model_dir_has_no_index(self, tmp_path, lora_file):
        from lora_inferencer import validate_inference_inputs
        model_dir = tmp_path / "mymodel"
        model_dir.mkdir()
        err = validate_inference_inputs(
            lora_path=str(lora_file),
            model_path=str(model_dir),
            prompt="a test",
        )
        assert err is not None
        assert "model_index.json" in err

    def test_output_dir_created_on_validation(self, lora_file, tmp_path):
        from lora_inferencer import validate_inference_inputs
        new_dir = tmp_path / "created_now" / "sub"
        err = validate_inference_inputs(
            lora_path=str(lora_file),
            model_path="black-forest-labs/FLUX.1-dev",
            prompt="a test",
            output_dir=str(new_dir),
        )
        assert err is None
        assert new_dir.is_dir()


# ---------------------------------------------------------------------------
# sha256_of_file
# ---------------------------------------------------------------------------

def test_sha256_matches_known_value(tmp_path):
    from lora_inferencer import sha256_of_file
    import hashlib
    data = b"\xDE\xAD\xBE\xEF" * 4096
    f = tmp_path / "blob.bin"
    f.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_of_file(str(f)) == expected


# ---------------------------------------------------------------------------
# run_inference — txt2img
# ---------------------------------------------------------------------------

class TestRunInferenceTxt2Img:
    def test_writes_png_and_returns_path(self, lora_file, output_path):
        from lora_inferencer import run_inference

        txt2img = _make_txt2img_mock()

        with ExitStack() as stack:
            stack.enter_context(patch("lora_inferencer._INFERENCE_DEPS_AVAILABLE", True))
            flux_cls = stack.enter_context(patch("lora_inferencer.FluxPipeline", create=True))
            flux_cls.from_pretrained.return_value = txt2img
            stack.enter_context(patch("lora_inferencer._IMG2IMG_AVAILABLE", True))

            result = run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="a red sphere",
                lora_strength=0.8,
                steps=4,
                guidance_scale=3.5,
                seed=0,
                output_path=output_path,
            )

        assert result == output_path
        assert Path(output_path).exists(), "PNG was not written to disk"
        img = PILImage.open(output_path)
        assert img.size == (512, 512)

    def test_txt2img_calls_txt2img_pipeline_not_img2img(self, lora_file, output_path):
        from lora_inferencer import run_inference

        txt2img = _make_txt2img_mock()
        img2img_cls = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(patch("lora_inferencer._INFERENCE_DEPS_AVAILABLE", True))
            flux_cls = stack.enter_context(patch("lora_inferencer.FluxPipeline", create=True))
            flux_cls.from_pretrained.return_value = txt2img
            stack.enter_context(
                patch("lora_inferencer.FluxImg2ImgPipeline", img2img_cls, create=True)
            )
            stack.enter_context(patch("lora_inferencer._IMG2IMG_AVAILABLE", True))

            run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="a test",
                steps=2, seed=1,
                output_path=output_path,
            )

        # txt2img pipe was called; img2img constructor was NOT
        txt2img.assert_called_once()
        img2img_cls.assert_not_called()

    def test_progress_cb_called_for_each_step(self, lora_file, output_path):
        from lora_inferencer import run_inference

        txt2img = _make_txt2img_mock()
        progress_calls: list[tuple[int, int]] = []

        # Make the mock invoke the callback to simulate step progress.
        def side_effect(**kwargs):
            cb = kwargs.get("callback_on_step_end")
            if cb:
                for step in range(kwargs.get("num_inference_steps", 0)):
                    cb(txt2img, step, None, {})
            fake_out = MagicMock()
            fake_out.images = [_fake_output_image()]
            return fake_out

        txt2img.side_effect = side_effect

        with ExitStack() as stack:
            stack.enter_context(patch("lora_inferencer._INFERENCE_DEPS_AVAILABLE", True))
            flux_cls = stack.enter_context(patch("lora_inferencer.FluxPipeline", create=True))
            flux_cls.from_pretrained.return_value = txt2img
            stack.enter_context(patch("lora_inferencer._IMG2IMG_AVAILABLE", True))

            run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="a sphere",
                steps=3, seed=0,
                output_path=output_path,
                progress_cb=lambda s, t: progress_calls.append((s, t)),
            )

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    def test_cancellation_raises_runtime_error(self, lora_file, output_path):
        from lora_inferencer import run_inference

        txt2img = _make_txt2img_mock()

        def side_effect(**kwargs):
            cb = kwargs.get("callback_on_step_end")
            if cb:
                cb(txt2img, 0, None, {})  # first callback sets _interrupt
            fake_out = MagicMock()
            fake_out.images = [_fake_output_image()]
            return fake_out

        txt2img.side_effect = side_effect

        with ExitStack() as stack:
            stack.enter_context(patch("lora_inferencer._INFERENCE_DEPS_AVAILABLE", True))
            flux_cls = stack.enter_context(patch("lora_inferencer.FluxPipeline", create=True))
            flux_cls.from_pretrained.return_value = txt2img
            stack.enter_context(patch("lora_inferencer._IMG2IMG_AVAILABLE", True))

            # cancelled() always returns True → should raise after first step
            with pytest.raises(RuntimeError, match="(?i)cancel"):
                run_inference(
                    lora_path=str(lora_file),
                    model_path="black-forest-labs/FLUX.1-dev",
                    prompt="a sphere",
                    steps=4, seed=0,
                    output_path=output_path,
                    cancelled=lambda: True,
                )

    def test_lora_loaded_and_unloaded_on_swap(self, lora_file, tmp_path, output_path):
        """Swapping LoRA path calls unload_lora_weights then load_lora_weights."""
        from lora_inferencer import run_inference

        lora2 = tmp_path / "adapter2.safetensors"
        header = json.dumps({}).encode()
        lora2.write_bytes(struct.pack("<Q", len(header)) + header)

        txt2img = _make_txt2img_mock()

        with ExitStack() as stack:
            stack.enter_context(patch("lora_inferencer._INFERENCE_DEPS_AVAILABLE", True))
            flux_cls = stack.enter_context(patch("lora_inferencer.FluxPipeline", create=True))
            flux_cls.from_pretrained.return_value = txt2img
            stack.enter_context(patch("lora_inferencer._IMG2IMG_AVAILABLE", True))

            run_inference(
                lora_path=str(lora_file), model_path="black-forest-labs/FLUX.1-dev",
                prompt="first", steps=2, seed=0, output_path=output_path,
            )
            # Now swap LoRA — cache already holds lora_file
            out2 = str(tmp_path / "out2.png")
            run_inference(
                lora_path=str(lora2), model_path="black-forest-labs/FLUX.1-dev",
                prompt="second", steps=2, seed=0, output_path=out2,
            )

        txt2img.unload_lora_weights.assert_called_once()
        assert txt2img.load_lora_weights.call_count == 2


# ---------------------------------------------------------------------------
# run_inference — img2img
# ---------------------------------------------------------------------------

class TestRunInferenceImg2Img:
    def test_writes_png_and_returns_path(self, lora_file, output_path, continuity_bytes):
        from lora_inferencer import run_inference

        txt2img  = _make_txt2img_mock()
        img2img  = _make_img2img_mock()
        img2img_cls = MagicMock(return_value=img2img)

        with ExitStack() as stack:
            stack.enter_context(patch("lora_inferencer._INFERENCE_DEPS_AVAILABLE", True))
            flux_cls = stack.enter_context(patch("lora_inferencer.FluxPipeline", create=True))
            flux_cls.from_pretrained.return_value = txt2img
            stack.enter_context(
                patch("lora_inferencer.FluxImg2ImgPipeline", img2img_cls, create=True)
            )
            stack.enter_context(patch("lora_inferencer._IMG2IMG_AVAILABLE", True))

            result = run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="an oil painting",
                steps=4, seed=7,
                output_path=output_path,
                continuity_image_bytes=continuity_bytes,
                denoise_strength=0.6,
            )

        assert result == output_path
        assert Path(output_path).exists()
        img = PILImage.open(output_path)
        assert img.size == (512, 512)

    def test_img2img_pipeline_used_not_txt2img(self, lora_file, output_path, continuity_bytes):
        from lora_inferencer import run_inference

        txt2img  = _make_txt2img_mock()
        img2img  = _make_img2img_mock()
        img2img_cls = MagicMock(return_value=img2img)

        with ExitStack() as stack:
            stack.enter_context(patch("lora_inferencer._INFERENCE_DEPS_AVAILABLE", True))
            flux_cls = stack.enter_context(patch("lora_inferencer.FluxPipeline", create=True))
            flux_cls.from_pretrained.return_value = txt2img
            stack.enter_context(
                patch("lora_inferencer.FluxImg2ImgPipeline", img2img_cls, create=True)
            )
            stack.enter_context(patch("lora_inferencer._IMG2IMG_AVAILABLE", True))

            run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="an oil painting",
                steps=4, seed=0,
                output_path=output_path,
                continuity_image_bytes=continuity_bytes,
            )

        # img2img pipe was called; txt2img pipe was NOT called for inference
        img2img.assert_called_once()
        txt2img.assert_not_called()

    def test_denoise_strength_passed_to_img2img(self, lora_file, output_path, continuity_bytes):
        from lora_inferencer import run_inference

        txt2img  = _make_txt2img_mock()
        img2img  = _make_img2img_mock()
        img2img_cls = MagicMock(return_value=img2img)

        with ExitStack() as stack:
            stack.enter_context(patch("lora_inferencer._INFERENCE_DEPS_AVAILABLE", True))
            flux_cls = stack.enter_context(patch("lora_inferencer.FluxPipeline", create=True))
            flux_cls.from_pretrained.return_value = txt2img
            stack.enter_context(
                patch("lora_inferencer.FluxImg2ImgPipeline", img2img_cls, create=True)
            )
            stack.enter_context(patch("lora_inferencer._IMG2IMG_AVAILABLE", True))

            run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="style transfer",
                steps=4, seed=0,
                output_path=output_path,
                continuity_image_bytes=continuity_bytes,
                denoise_strength=0.42,
            )

        _, call_kwargs = img2img.call_args
        assert call_kwargs.get("strength") == pytest.approx(0.42)

    def test_img2img_unavailable_raises_clearly(self, lora_file, output_path, continuity_bytes):
        from lora_inferencer import run_inference

        txt2img = _make_txt2img_mock()

        with ExitStack() as stack:
            stack.enter_context(patch("lora_inferencer._INFERENCE_DEPS_AVAILABLE", True))
            flux_cls = stack.enter_context(patch("lora_inferencer.FluxPipeline", create=True))
            flux_cls.from_pretrained.return_value = txt2img
            # FluxImg2ImgPipeline must be patchable even though it won't be called.
            stack.enter_context(
                patch("lora_inferencer.FluxImg2ImgPipeline", MagicMock(), create=True)
            )
            stack.enter_context(patch("lora_inferencer._IMG2IMG_AVAILABLE", False))

            with pytest.raises(RuntimeError, match="(?i)img2img"):
                run_inference(
                    lora_path=str(lora_file),
                    model_path="black-forest-labs/FLUX.1-dev",
                    prompt="test",
                    steps=2, seed=0,
                    output_path=output_path,
                    continuity_image_bytes=continuity_bytes,
                )

    def test_output_checksum_is_stable(self, lora_file, output_path, continuity_bytes):
        """sha256_of_file returns a 64-char hex string for the written PNG."""
        from lora_inferencer import run_inference, sha256_of_file

        txt2img  = _make_txt2img_mock()
        img2img  = _make_img2img_mock()
        img2img_cls = MagicMock(return_value=img2img)

        with ExitStack() as stack:
            stack.enter_context(patch("lora_inferencer._INFERENCE_DEPS_AVAILABLE", True))
            flux_cls = stack.enter_context(patch("lora_inferencer.FluxPipeline", create=True))
            flux_cls.from_pretrained.return_value = txt2img
            stack.enter_context(
                patch("lora_inferencer.FluxImg2ImgPipeline", img2img_cls, create=True)
            )
            stack.enter_context(patch("lora_inferencer._IMG2IMG_AVAILABLE", True))

            result = run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="checksum test",
                steps=2, seed=0,
                output_path=output_path,
                continuity_image_bytes=continuity_bytes,
            )

        checksum = sha256_of_file(result)
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)
        # Deterministic: same file, same checksum
        assert sha256_of_file(result) == checksum
