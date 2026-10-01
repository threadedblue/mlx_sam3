"""v4 §3: a session's mode must survive save -> restart -> resume without
DoubleNaught re-supplying it.

This does NOT lean on image_url's existing coverage. image_url follows the
same registry.parquet path and still had a real, separately-discovered bug in
exactly this round trip (the resume-destruction investigation: a later call
merging an empty value over the stored one, wiping it before the user touched
anything). Mode goes through the same merge, carries the same empty-on-resume
convention from DN, and therefore has the same failure mode available to it.
So it gets its own reproduction of that scenario class.

The bridge is stubbed per-test rather than hitting real Julia parquet I/O —
test_sf_wiring.py, which does hit it, hangs in this environment. What is being
verified here is the merge/read/clobber logic, which is where the bug class
actually lives; the bridge is a dumb key-value round trip either way.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

import aa_persistence
import main
from services import SegmentationService


SID = "mode-round-trip-session"


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class FakeBridge:
    """A real round trip through a dict, standing in for the Julia parquet
    bridge: writes land, reads come back, and a value written once stays
    written. That is the only property these tests need from it."""

    def __init__(self):
        self.files: dict[str, tuple[list, list, list]] = {}

    def save_parquet(self, path, rows, cols, vals):
        self.files[str(path)] = (list(rows), list(cols), list(vals))
        # read_session_raw gates on the file actually existing before it
        # reads, so the stand-in has to leave a real file behind too.
        from pathlib import Path as _P
        _P(path).parent.mkdir(parents=True, exist_ok=True)
        _P(path).touch()

    def load_parquet(self, path):
        if str(path) not in self.files:
            raise FileNotFoundError(path)
        return self.files[str(path)]

    def build_triples(self, rows, cols, vals):
        return (rows, cols, vals)


class StubProcessor:
    """Enough of Sam3Processor for a session with no prompts to restore.

    Not optional: `register_session_data` protects a stored session by
    restoring it from disk BEFORE merging (see its docstring). That guard runs
    through the processor, so a null processor makes the restore fail and the
    merge falls through to the blank-session path — which would make these
    tests pass or fail for the wrong reason entirely.
    """

    def set_image(self, image):
        return {"backbone_out": True,
                "original_width": image.width, "original_height": image.height}

    def set_text_prompt(self, prompt, state):
        return state

    def add_geometric_prompt(self, box, label, state):
        return state

    def add_point_prompt(self, point, label, state):
        return state


@pytest.fixture()
def bridge(tmp_path, monkeypatch):
    fake = FakeBridge()
    monkeypatch.setattr(aa_persistence, "bridge", fake)
    monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf")
    return fake


def _registry_of(bridge_, session_id=SID) -> dict:
    for path, (rows, cols, vals) in bridge_.files.items():
        if path.endswith("registry.parquet") and session_id in path:
            return dict(zip(cols, vals))
    raise AssertionError("no registry.parquet written")


# ── the registry column itself ───────────────────────────────────────────────


def test_mode_is_written_to_the_registry(bridge):
    aa_persistence.save_session(SID, {"name": "n", "mode": "comix"})

    assert _registry_of(bridge)["mode"] == "comix"


def test_mode_reads_back_off_the_registry(bridge):
    aa_persistence.save_session(SID, {"name": "n", "mode": "comix"})

    assert aa_persistence.read_session_raw(SID)["mode"] == "comix"


def test_an_absent_mode_reads_back_as_none_not_empty_string(bridge):
    """None is what the frontend keys 'legacy/unfiltered' off. An empty
    string would parse to no mode anyway, but None is the honest shape and
    is what `SfMode.parse` is specified against."""
    aa_persistence.save_session(SID, {"name": "n"})

    assert aa_persistence.read_session_raw(SID)["mode"] is None


def test_a_session_saved_before_modes_existed_still_loads(bridge):
    """Old registries have no `mode` column at all — that must read as None,
    not raise."""
    d = aa_persistence.session_dir(SID)
    bridge.save_parquet(
        str(d / "registry.parquet"), [SID], ["name"], ["legacy session"],
    )

    raw = aa_persistence.read_session_raw(SID)
    assert raw["mode"] is None
    assert raw["name"] == "legacy session"


# ── the full round trip, including the clobber bug class ─────────────────────


class TestFullRoundTrip:
    """Create with mode -> save -> restart -> resume, with DN out of the loop
    on resume exactly as it is in production."""

    @pytest.fixture()
    def service(self, bridge, tmp_path, monkeypatch):
        svc = SegmentationService(tmp_path / "sessions", StubProcessor())
        monkeypatch.setattr(main, "service", svc)
        return svc

    def _create_with_mode(self, service, mode="comix"):
        """What /initSession does for a FRESH launch: DN supplies the mode."""
        service.register_session_data(SID, {
            "name": "page 26",
            "description": "",
            "image_url": "http://example/page.png",
            "mode": mode,
            "original_image_bytes": _png(),
            "image_size": (16, 16),
        })

    def test_mode_survives_save_restart_and_resume(self, service):
        self._create_with_mode(service)
        service.save_session_to_disk(SID)

        # Restart: every in-memory cache is gone, only the parquet remains.
        service.sessions.clear()
        service.sf_sessions.clear()

        raw = aa_persistence.read_session_raw(SID)
        assert raw["mode"] == "comix", (
            "mode did not survive the save/restart boundary"
        )

    def test_resume_recovers_mode_with_dn_sending_nothing(self, service):
        """The production resume shape: DN re-launches SF against an existing
        session and sends mode="" because it is not creating anything. SF must
        read its own stored value."""
        self._create_with_mode(service)
        service.save_session_to_disk(SID)
        service.sessions.clear()

        # DN's /initSession on resume — empty mode, empty image_url.
        init_data = {"name": "page 26", "description": "",
                     "image_url": "http://example/page.png"}
        mode_from_dn = ""
        if mode_from_dn:                      # the guard in main.init_session
            init_data["mode"] = mode_from_dn
        service.register_session_data(SID, init_data)

        assert aa_persistence.read_session_raw(SID)["mode"] == "comix", (
            "DN sending an empty mode on resume wiped the stored one — this is "
            "the exact clobber that bit image_url in the resume-destruction bug"
        )

    def test_an_unguarded_empty_mode_would_have_destroyed_it(self, service):
        """Pins WHY the guard exists, by showing the failure it prevents. If
        someone later drops `if request.mode:` from init_session, the guarded
        test above starts failing and this one explains what broke."""
        self._create_with_mode(service)
        service.save_session_to_disk(SID)
        service.sessions.clear()

        # Deliberately WITHOUT the guard — the bug being prevented.
        service.register_session_data(SID, {
            "name": "page 26", "description": "",
            "image_url": "http://example/page.png",
            "mode": "",
        })

        assert aa_persistence.read_session_raw(SID)["mode"] is None, (
            "sanity: an unguarded empty mode really does clobber, which is "
            "what makes the guard load-bearing rather than decorative"
        )

    def test_a_real_mode_from_dn_does_update_an_existing_session(self, service):
        """The guard must not freeze the value: DN genuinely changing the mode
        still has to land."""
        self._create_with_mode(service, mode="comix")
        service.save_session_to_disk(SID)
        service.sessions.clear()

        init_data = {"name": "page 26", "description": "", "image_url": "x"}
        mode_from_dn = "poster"
        if mode_from_dn:
            init_data["mode"] = mode_from_dn
        service.register_session_data(SID, init_data)

        assert aa_persistence.read_session_raw(SID)["mode"] == "poster"

    def test_resume_response_carries_the_mode_to_the_frontend(self, service):
        """The last link: the value has to reach the app, not just the
        registry. This is what the card list and Display Layers read."""
        self._create_with_mode(service)
        service.save_session_to_disk(SID)
        service.sessions.clear()
        service.sf_sessions.clear()

        response = service.load_session_from_disk(SID)

        assert response["mode"] == "comix"

    def test_resume_response_mode_is_none_for_a_legacy_session(self, service):
        service.register_session_data(SID, {
            "name": "old", "description": "", "image_url": "",
            "original_image_bytes": _png(), "image_size": (16, 16),
        })
        service.save_session_to_disk(SID)
        service.sessions.clear()
        service.sf_sessions.clear()

        response = service.load_session_from_disk(SID)

        assert response["mode"] is None, (
            "a pre-mode session must resume unfiltered, not as some default"
        )
