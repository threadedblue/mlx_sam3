"""`register_session_data` must never merge onto a blank session when a real
one exists on disk.

Found while round-trip-testing mode, but NOT a mode bug: `/upload` has called
this since long before modes existed, so name, description, image_url,
created_at and original_filename were all exposed to the same wipe. Mode just
joined the list of casualties.

Three ways the cache can still be cold after the restore attempt, which the
old code treated identically:

  (a) genuinely new session_id — blank IS correct;
  (b) reconstruction threw — must fail loudly, never blank-merge;
  (c) staged via /initSession, metadata on disk but no image yet —
      `_load_session_into_memory` returns None for it, which is right for
      "nothing to segment" and wrong as "nothing exists". Its metadata must
      survive.

(c) is the one that bites in ordinary use: /initSession then /upload with a
restart in between is the documented DoubleNaught launch flow, and it does
not throw or log anything — it just quietly emptied every registry field.
"""

from __future__ import annotations

import io
import pathlib

import pytest
from PIL import Image

import aa_persistence
from services import SegmentationService, SessionLoadError

SID = "guard-session"


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


class FakeBridge:
    def __init__(self):
        self.files: dict[str, tuple] = {}

    def save_parquet(self, path, rows, cols, vals):
        self.files[str(path)] = (list(rows), list(cols), list(vals))
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).touch()

    def load_parquet(self, path):
        if str(path) not in self.files:
            raise FileNotFoundError(path)
        return self.files[str(path)]

    def build_triples(self, rows, cols, vals):
        return (rows, cols, vals)


class WorkingProcessor:
    def set_image(self, image):
        return {"backbone_out": True,
                "original_width": image.width, "original_height": image.height}

    def set_text_prompt(self, prompt, state):
        return state

    def add_geometric_prompt(self, box, label, state):
        return state

    def add_point_prompt(self, point, label, state):
        return state


class ExplodingProcessor(WorkingProcessor):
    """Reconstruction fails for a real, production-plausible reason — a torch/
    MPS fault or an OOM inside set_image, not a contrived None processor."""

    def set_image(self, image):
        raise RuntimeError("MPS backend fault during set_image")


@pytest.fixture()
def bridge(tmp_path, monkeypatch):
    fake = FakeBridge()
    monkeypatch.setattr(aa_persistence, "bridge", fake)
    monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf")
    return fake


@pytest.fixture()
def service(bridge, tmp_path):
    return SegmentationService(tmp_path / "sessions", WorkingProcessor())


def _registry(bridge_) -> dict:
    for path, (_rows, cols, vals) in bridge_.files.items():
        if path.endswith("registry.parquet"):
            return dict(zip(cols, vals))
    raise AssertionError("no registry.parquet written")


def _stage_via_init_session(service):
    """What /initSession writes: metadata, deliberately no image."""
    service.register_session_data(SID, {
        "name": "My Page", "description": "a nemo page",
        "image_url": "http://example/p.png", "mode": "comix",
    })


def _upload_payload():
    """What /upload merges in: image state, and no metadata at all."""
    return {"state": {}, "original_image_bytes": _png(), "image_size": (8, 8)}


# ── (c) the staged-session wipe: silent, and the documented launch flow ──────


class TestStagedSessionMetadataSurvives:
    def test_upload_after_restart_keeps_the_staged_metadata(self, service, bridge):
        _stage_via_init_session(service)
        service.sessions.clear()
        service.sf_sessions.clear()

        service.register_session_data(SID, _upload_payload())

        reg = _registry(bridge)
        assert reg["name"] == "My Page"
        assert reg["description"] == "a nemo page"
        assert reg["image_url"] == "http://example/p.png"
        assert reg["mode"] == "comix"

    def test_the_upload_still_lands(self, service, bridge):
        """Preserving metadata must not come at the cost of the actual upload."""
        _stage_via_init_session(service)
        service.sessions.clear()

        service.register_session_data(SID, _upload_payload())

        reg = _registry(bridge)
        assert reg["width"] == "8" and reg["height"] == "8"
        assert service.sessions[SID]["original_image_bytes"] is not None

    def test_it_is_not_mode_specific(self, service, bridge):
        """Pins item 3: this predates modes. A session carrying NO mode at all
        must still keep its other fields."""
        service.register_session_data(SID, {
            "name": "Legacy Page", "description": "pre-modes",
            "image_url": "http://example/legacy.png",
        })
        service.sessions.clear()

        service.register_session_data(SID, _upload_payload())

        reg = _registry(bridge)
        assert reg["name"] == "Legacy Page"
        assert reg["description"] == "pre-modes"
        assert reg["image_url"] == "http://example/legacy.png"


# ── (b) genuine reconstruction failure: loud, never a wipe ───────────────────


class TestFailedRestoreRefusesToWipe:
    def _saved_full_session(self, service):
        service.register_session_data(SID, {
            "name": "My Page", "description": "a nemo page",
            "image_url": "http://example/p.png", "mode": "comix",
            **_upload_payload(),
        })

    def test_a_failed_restore_raises_instead_of_blanking(self, service, bridge):
        self._saved_full_session(service)
        service.sessions.clear()
        service.sf_sessions.clear()
        service.processor = ExplodingProcessor()

        with pytest.raises(SessionLoadError, match="could not be restored"):
            service.register_session_data(SID, _upload_payload())

    def test_the_on_disk_session_is_untouched_after_the_refusal(
        self, service, bridge
    ):
        """The point of raising: the data is still there afterwards."""
        self._saved_full_session(service)
        before = dict(_registry(bridge))
        service.sessions.clear()
        service.sf_sessions.clear()
        service.processor = ExplodingProcessor()

        with pytest.raises(SessionLoadError):
            service.register_session_data(SID, _upload_payload())

        assert _registry(bridge) == before


# ── (a) the genuinely-new case must be completely unaffected ─────────────────


class TestSuccessPathUnchanged:
    def test_a_brand_new_session_still_registers_from_blank(self, service, bridge):
        service.register_session_data(SID, {
            "name": "Fresh", "description": "", **_upload_payload(),
        })

        reg = _registry(bridge)
        assert reg["name"] == "Fresh"
        assert service.sessions[SID]["original_image_bytes"] is not None

    def test_a_warm_cached_session_never_reaches_the_guard(self, service, bridge):
        """When the cache is already warm the guard is not consulted at all —
        the ordinary merge path, untouched."""
        self_sessions_before = service.register_session_data(SID, {
            "name": "Warm", "description": "", **_upload_payload(),
        })
        assert self_sessions_before is None  # no return value, as before

        service.processor = ExplodingProcessor()  # would raise IF consulted
        service.register_session_data(SID, {"description": "edited"})

        assert service.sessions[SID]["name"] == "Warm"
        assert service.sessions[SID]["description"] == "edited"

    def test_a_successful_restore_merges_normally(self, service, bridge):
        """The already-working case: session on disk, restore succeeds, merge
        proceeds. Must behave exactly as before the fix."""
        self._full = service.register_session_data(SID, {
            "name": "Roundtrip", "description": "d",
            "image_url": "u", "mode": "comix", **_upload_payload(),
        })
        service.sessions.clear()
        service.sf_sessions.clear()

        service.register_session_data(SID, {"description": "edited"})

        reg = _registry(bridge)
        assert reg["name"] == "Roundtrip"
        assert reg["description"] == "edited"
        assert reg["mode"] == "comix"


# ── (class 1) a throwing read was already loud — confirm it stays that way ───


def test_a_throwing_registry_read_still_propagates(service, bridge):
    """`read_session_raw` sits OUTSIDE `_load_session_into_memory`'s try, so a
    corrupt parquet or a Julia bridge fault already propagated rather than
    wiping. This pins that it still does, and is not quietly swallowed by the
    new guard."""
    service.register_session_data(SID, {"name": "n", **_upload_payload()})
    service.sessions.clear()

    def boom(path):
        raise RuntimeError("Julia precompile lock")

    bridge.load_parquet = boom

    with pytest.raises(RuntimeError, match="Julia precompile lock"):
        service.register_session_data(SID, _upload_payload())
