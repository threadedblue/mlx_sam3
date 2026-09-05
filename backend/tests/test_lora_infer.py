"""
Unit tests for the LoRA inference backend (post-provider-refactor).

Structure:
  TestValidateInferenceInputs  — provider-agnostic pre-flight checks
  TestSha256                   — checksum helper
  TestMlxProvider              — MlxLocalProvider with mocked mflux
  TestCloudRunProvider         — CloudRunProvider with mocked httpx responses

Run with:  pytest app/backend/tests/test_lora_infer.py -v
"""

from __future__ import annotations

import io
import json
import struct
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mlx_provider_cache():
    """Clear the MlxLocalProvider instance cache before and after each test."""
    from inference_providers import MlxLocalProvider
    import lora_inferencer as _m
    # Reset module-level provider instances so LoRA/model caches don't bleed.
    _m._mlx_provider = MlxLocalProvider()
    yield
    _m._mlx_provider = MlxLocalProvider()


@pytest.fixture()
def lora_file(tmp_path: Path) -> Path:
    p = tmp_path / "adapter.safetensors"
    header = json.dumps({"__metadata__": {}}).encode()
    p.write_bytes(struct.pack("<Q", len(header)) + header)
    return p


@pytest.fixture()
def output_path(tmp_path: Path) -> str:
    return str(tmp_path / "out.png")


@pytest.fixture()
def continuity_bytes() -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), color=(0, 128, 255)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Mock helpers — mflux API
# ---------------------------------------------------------------------------

class _FakeConfig:
    """Minimal stand-in for mflux.Config so tests don't need mflux installed."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_mflux_mocks(steps: int = 4):
    """Return (flux_cls, flux_instance, result) mocks with a working save()."""
    def _save(path):
        PILImage.new("RGB", (64, 64), (200, 100, 50)).save(path, format="PNG")

    result = MagicMock()
    result.save = MagicMock(side_effect=lambda path: _save(path))

    flux_instance = MagicMock()
    flux_instance.generate_image.return_value = result

    flux_cls = MagicMock()
    flux_cls.from_alias.return_value = flux_instance

    return flux_cls, flux_instance, result


def _flux_patches(flux_cls):
    """Return context managers that make inference_providers think mflux is available."""
    return [
        patch("inference_providers._MFLUX_AVAILABLE", True),
        patch("inference_providers.Flux1", flux_cls, create=True),
        patch("inference_providers.Config", _FakeConfig, create=True),
        patch("inference_providers._StopGen", RuntimeError, create=True),
    ]


# ---------------------------------------------------------------------------
# TestValidateInferenceInputs
# ---------------------------------------------------------------------------

class TestValidateInferenceInputs:
    def test_passes_for_valid_inputs(self, lora_file, tmp_path):
        from lora_inferencer import validate_inference_inputs
        assert validate_inference_inputs(
            lora_path=str(lora_file),
            model_path="black-forest-labs/FLUX.1-dev",
            prompt="a red sphere",
            output_dir=str(tmp_path / "out"),
        ) is None

    def test_fails_when_prompt_empty(self, lora_file):
        from lora_inferencer import validate_inference_inputs
        err = validate_inference_inputs(
            lora_path=str(lora_file),
            model_path="black-forest-labs/FLUX.1-dev",
            prompt="   ",
        )
        assert err and "prompt" in err.lower()

    def test_fails_when_lora_missing(self, tmp_path):
        from lora_inferencer import validate_inference_inputs
        err = validate_inference_inputs(
            lora_path=str(tmp_path / "ghost.safetensors"),
            model_path="black-forest-labs/FLUX.1-dev",
            prompt="a test",
        )
        assert err and "not found" in err.lower()

    def test_fails_for_wrong_lora_extension(self, tmp_path):
        from lora_inferencer import validate_inference_inputs
        bad = tmp_path / "adapter.ckpt"
        bad.write_bytes(b"\x00" * 8)
        err = validate_inference_inputs(
            lora_path=str(bad),
            model_path="black-forest-labs/FLUX.1-dev",
            prompt="a test",
        )
        assert err and "safetensors" in err.lower()

    def test_output_dir_created_on_validation(self, lora_file, tmp_path):
        from lora_inferencer import validate_inference_inputs
        new_dir = tmp_path / "created" / "sub"
        assert validate_inference_inputs(
            lora_path=str(lora_file),
            model_path="black-forest-labs/FLUX.1-dev",
            prompt="a test",
            output_dir=str(new_dir),
        ) is None
        assert new_dir.is_dir()


# ---------------------------------------------------------------------------
# TestSha256
# ---------------------------------------------------------------------------

def test_sha256_matches_known_value(tmp_path):
    import hashlib
    from lora_inferencer import sha256_of_file
    data = b"\xDE\xAD\xBE\xEF" * 4096
    f = tmp_path / "blob.bin"
    f.write_bytes(data)
    assert sha256_of_file(str(f)) == hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# TestMlxProvider — tests MlxLocalProvider directly
# ---------------------------------------------------------------------------

class TestMlxProvider:
    def test_txt2img_writes_png_and_returns_path(self, lora_file, output_path):
        from inference_providers import MlxLocalProvider
        provider = MlxLocalProvider()
        flux_cls, flux_inst, _ = _make_mflux_mocks()

        with ExitStack() as stack:
            for p in _flux_patches(flux_cls):
                stack.enter_context(p)

            result = provider.run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="a red sphere",
                steps=4, seed=0,
                output_path=output_path,
            )

        assert result == output_path
        assert Path(output_path).exists()
        assert PILImage.open(output_path).size == (64, 64)

    def test_flux_loaded_from_alias(self, lora_file, output_path):
        from inference_providers import MlxLocalProvider
        provider = MlxLocalProvider()
        flux_cls, _, _ = _make_mflux_mocks()

        with ExitStack() as stack:
            for p in _flux_patches(flux_cls):
                stack.enter_context(p)
            provider.run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="test", steps=2, seed=0, output_path=output_path,
            )

        flux_cls.from_alias.assert_called_once()
        alias_arg = flux_cls.from_alias.call_args[0][0]
        assert alias_arg == "flux-1-dev"

    def test_progress_cb_called_per_step(self, lora_file, output_path):
        from inference_providers import MlxLocalProvider
        provider = MlxLocalProvider()
        flux_cls, flux_inst, result = _make_mflux_mocks(steps=3)
        calls: list = []

        def side_effect(**kwargs):
            cb = kwargs.get("step_callback")
            if cb:
                for i in range(3):
                    cb(i + 1, 3)
            return result

        flux_inst.generate_image.side_effect = side_effect

        with ExitStack() as stack:
            for p in _flux_patches(flux_cls):
                stack.enter_context(p)
            provider.run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="test", steps=3, seed=0,
                output_path=output_path,
                progress_cb=lambda s, t: calls.append((s, t)),
            )

        assert len(calls) == 3
        assert calls[-1] == (3, 3)

    def test_cancellation_raises_runtime_error(self, lora_file, output_path):
        from inference_providers import MlxLocalProvider
        provider = MlxLocalProvider()
        flux_cls, flux_inst, result = _make_mflux_mocks()

        def side_effect(**kwargs):
            cb = kwargs.get("step_callback")
            if cb:
                cb(1, 4)   # first step triggers the cancelled() check
            return result

        flux_inst.generate_image.side_effect = side_effect

        with ExitStack() as stack:
            for p in _flux_patches(flux_cls):
                stack.enter_context(p)
            with pytest.raises(RuntimeError, match="(?i)cancel"):
                provider.run_inference(
                    lora_path=str(lora_file),
                    model_path="black-forest-labs/FLUX.1-dev",
                    prompt="test", steps=4, seed=0,
                    output_path=output_path,
                    cancelled=lambda: True,
                )

    def test_img2img_passes_image_path_to_config(self, lora_file, output_path, continuity_bytes):
        from inference_providers import MlxLocalProvider
        provider = MlxLocalProvider()
        flux_cls, flux_inst, _ = _make_mflux_mocks()
        configs_seen: list = []

        def side_effect(**kwargs):
            configs_seen.append(kwargs.get("config"))
            return MagicMock(save=MagicMock(
                side_effect=lambda path: PILImage.new("RGB", (64, 64)).save(path, format="PNG")
            ))

        flux_inst.generate_image.side_effect = side_effect

        with ExitStack() as stack:
            for p in _flux_patches(flux_cls):
                stack.enter_context(p)
            provider.run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="style transfer", steps=4, seed=0,
                output_path=output_path,
                continuity_image_bytes=continuity_bytes,
                denoise_strength=0.42,
            )

        assert configs_seen, "generate_image was not called"
        cfg = configs_seen[0]
        assert hasattr(cfg, "image_path"), "Config missing image_path"
        assert cfg.image_strength == pytest.approx(0.42)

    def test_lora_reloaded_on_path_swap(self, lora_file, tmp_path, output_path):
        from inference_providers import MlxLocalProvider
        provider = MlxLocalProvider()
        flux_cls, _, _ = _make_mflux_mocks()

        lora2 = tmp_path / "adapter2.safetensors"
        lora2.write_bytes(struct.pack("<Q", 2) + b"{}")

        with ExitStack() as stack:
            for p in _flux_patches(flux_cls):
                stack.enter_context(p)

            provider.run_inference(
                lora_path=str(lora_file), model_path="black-forest-labs/FLUX.1-dev",
                prompt="first", steps=2, seed=0, output_path=output_path,
            )
            out2 = str(tmp_path / "out2.png")
            provider.run_inference(
                lora_path=str(lora2), model_path="black-forest-labs/FLUX.1-dev",
                prompt="second", steps=2, seed=0, output_path=out2,
            )

        # Flux1.from_alias called twice: once per distinct LoRA
        assert flux_cls.from_alias.call_count == 2


# ---------------------------------------------------------------------------
# TestCloudRunProvider — tests CloudRunProvider directly via mocked httpx
# ---------------------------------------------------------------------------

class TestCloudRunProvider:
    _URL = "http://cloud-run.example.com"

    def _make_httpx_mock(self, image_bytes: bytes):
        """Build a mock httpx.Client whose POST returns run_id and GET returns done."""
        import base64, hashlib

        image_b64 = base64.b64encode(image_bytes).decode()
        checksum   = hashlib.sha256(image_bytes).hexdigest()

        post_resp  = MagicMock(status_code=200)
        post_resp.json.return_value = {"run_id": "test-run-123"}

        done_resp  = MagicMock(status_code=200)
        done_resp.json.return_value = {
            "status":          "done",
            "progress":        1.0,
            "current_step":    4,
            "total_steps":     4,
            "elapsed_seconds": 10,
            "image_b64":       image_b64,
            "output_checksum": checksum,
            "last_error":      None,
        }

        client_instance = MagicMock()
        client_instance.__enter__ = MagicMock(return_value=client_instance)
        client_instance.__exit__  = MagicMock(return_value=False)
        # First call to post → /generate; subsequent GET → status done.
        client_instance.post.return_value = post_resp
        client_instance.get.return_value  = done_resp

        return client_instance

    @pytest.fixture()
    def raw_image_bytes(self) -> bytes:
        buf = io.BytesIO()
        PILImage.new("RGB", (64, 64), (50, 100, 200)).save(buf, format="PNG")
        return buf.getvalue()

    def test_txt2img_writes_png_and_returns_path(self, lora_file, output_path, raw_image_bytes):
        from inference_providers import CloudRunProvider
        provider = CloudRunProvider(url=self._URL)
        client   = self._make_httpx_mock(raw_image_bytes)

        with patch("inference_providers._httpx") as mock_httpx, \
             patch("inference_providers._HTTPX_AVAILABLE", True):
            mock_httpx.Client.return_value = client

            result = provider.run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="a sphere",
                steps=4, seed=0,
                output_path=output_path,
            )

        assert result == output_path
        assert Path(output_path).exists()
        assert Path(output_path).read_bytes() == raw_image_bytes

    def test_lora_sent_as_base64(self, lora_file, output_path, raw_image_bytes):
        import base64
        from inference_providers import CloudRunProvider
        provider = CloudRunProvider(url=self._URL)
        client   = self._make_httpx_mock(raw_image_bytes)

        with patch("inference_providers._httpx") as mock_httpx, \
             patch("inference_providers._HTTPX_AVAILABLE", True):
            mock_httpx.Client.return_value = client
            provider.run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="test", steps=4, seed=0, output_path=output_path,
            )

        post_kwargs = client.post.call_args
        body = post_kwargs[1]["json"] if post_kwargs[1] else post_kwargs[0][1]
        assert "lora_b64" in body
        decoded = base64.b64decode(body["lora_b64"])
        assert decoded == Path(lora_file).read_bytes()

    def test_remote_failure_raises_runtime_error(self, lora_file, output_path):
        from inference_providers import CloudRunProvider
        provider = CloudRunProvider(url=self._URL)

        failed_resp = MagicMock(status_code=200)
        failed_resp.json.return_value = {
            "status": "failed", "progress": 0.5,
            "current_step": 2, "total_steps": 4,
            "elapsed_seconds": 5,
            "image_b64": None, "output_checksum": None,
            "last_error": "OOM on remote",
        }

        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__  = MagicMock(return_value=False)
        client.post.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value={"run_id": "x"})
        )
        client.get.return_value = failed_resp

        with patch("inference_providers._httpx") as mock_httpx, \
             patch("inference_providers._HTTPX_AVAILABLE", True):
            mock_httpx.Client.return_value = client
            with pytest.raises(RuntimeError, match="OOM on remote"):
                provider.run_inference(
                    lora_path=str(lora_file),
                    model_path="black-forest-labs/FLUX.1-dev",
                    prompt="test", steps=4, seed=0, output_path=output_path,
                )

    def test_missing_url_raises_clearly(self, lora_file, output_path):
        from inference_providers import CloudRunProvider
        provider = CloudRunProvider(url="")
        with patch("inference_providers._HTTPX_AVAILABLE", True):
            with pytest.raises(RuntimeError, match="(?i)url"):
                provider.run_inference(
                    lora_path=str(lora_file),
                    model_path="black-forest-labs/FLUX.1-dev",
                    prompt="test", steps=4, seed=0, output_path=output_path,
                )

    def test_output_checksum_matches_written_file(self, lora_file, output_path, raw_image_bytes):
        from inference_providers import CloudRunProvider
        from lora_inferencer import sha256_of_file
        provider = CloudRunProvider(url=self._URL)
        client   = self._make_httpx_mock(raw_image_bytes)

        with patch("inference_providers._httpx") as mock_httpx, \
             patch("inference_providers._HTTPX_AVAILABLE", True):
            mock_httpx.Client.return_value = client
            result = provider.run_inference(
                lora_path=str(lora_file),
                model_path="black-forest-labs/FLUX.1-dev",
                prompt="test", steps=4, seed=0, output_path=output_path,
            )

        import hashlib
        assert sha256_of_file(result) == hashlib.sha256(raw_image_bytes).hexdigest()
