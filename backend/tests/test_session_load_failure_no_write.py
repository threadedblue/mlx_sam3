"""A session load that fails partway must not let ANY write proceed as if
it had succeeded.

Follow-up to the point-prompt resume fix (test_session_resume_prompts.py).
Traced where registry.parquet actually gets written relative to the reshape
crash: nowhere in the load path itself. `_load_session_into_memory`,
`load_session_from_disk` and `_reconstruct_sf_session` contain no write call
at all, before or after that fix — so there was never a "write metadata,
then try to replay prompts, then fail" sequence to reorder.

The write that was actually observed on the three broken sessions came from
a completely separate endpoint, `/session/settings`
(`aa_persistence.save_session_settings`), most likely triggered by an
ordinary layer-visibility toggle shortly after the session opened. That
function only checks that registry.parquet EXISTS on disk — it has no idea
whether the backend's in-memory SAM3/prompt state for that session is
actually valid, which is exactly why it "succeeded" against a session whose
resume was silently broken (and, incidentally, why it caused no data loss:
it only ever touches the `ui_settings` field, preserving every other field
from the file it reads, and never touches segment.parquet/linkage.parquet).

The fix: `SegmentationService` tracks which session_ids most recently failed
reconstruction (`_broken_sessions`, set only on a genuine exception — never
on the benign "no image yet" /initSession-staged case) and
`save_session_settings` refuses to persist anything for one of them.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

import aa_persistence
from services import SegmentationService, SessionLoadError

SID = "load-failure-test-session"


def _png_bytes(size=64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class BrokenProcessor:
    """Fails at the very first step, `set_image` — outside the per-prompt
    try/except that `_load_session_into_memory` uses to skip individual
    unreplayable prompts (test_session_resume_prompts.py covers that path;
    a single bad prompt is no longer fatal to the whole load). This
    simulates a genuine WHOLE-load failure, the kind the outer except
    clause — and therefore `_broken_sessions` — is actually for."""

    def set_image(self, image):
        raise ValueError("simulated reconstruction failure")

    def set_text_prompt(self, prompt, state):
        return state

    def add_geometric_prompt(self, box, label, state):
        return state

    def add_point_prompt(self, point, label, state):
        return state


class WorkingProcessor:
    def set_image(self, image):
        return {"backbone_out": True, "original_width": image.width,
                "original_height": image.height}

    def set_text_prompt(self, prompt, state):
        return state

    def add_geometric_prompt(self, box, label, state):
        return state

    def add_point_prompt(self, point, label, state):
        return state


def _raw(prompts, image_bytes=b"present"):
    return {
        "name": "load failure probe",
        "description": "",
        "created_at": "2026-09-29T13:56:35",
        "image_url": None,
        "original_filename": "image.png",
        "width": 64,
        "height": 64,
        "image_bytes": _png_bytes() if image_bytes else None,
        "pass_images": {},
        "segments": [],
        "prompts": prompts,
        "ui_settings": {},
    }


@pytest.fixture()
def service(tmp_path, monkeypatch):
    svc = SegmentationService(tmp_path / "sessions", BrokenProcessor())
    monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf_storage")
    return svc


def _write_spy(monkeypatch):
    """Replaces the real (bridge-backed) persistence call with a spy, so
    these tests prove the causal chain — the guard prevents the CALL from
    ever happening — without needing real parquet I/O. (The real function's
    on-disk behavior is covered separately, live, against the actual
    formerly-broken sessions.)"""
    calls = []
    monkeypatch.setattr(
        aa_persistence,
        "save_session_settings",
        lambda sid, settings: calls.append((sid, settings)),
    )
    return calls


def test_a_load_failure_marks_the_session_broken(service, monkeypatch):
    monkeypatch.setattr(aa_persistence, "read_session_raw", lambda sid: _raw(["x"]))

    loaded = service._load_session_into_memory(SID)

    assert loaded is None
    assert SID in service._broken_sessions


def test_a_staged_session_with_no_image_yet_is_not_marked_broken(
    service, monkeypatch
):
    """/initSession's legitimate no-image-yet state must not be conflated
    with a genuine reconstruction failure."""
    monkeypatch.setattr(
        aa_persistence, "read_session_raw", lambda sid: _raw([], image_bytes=None)
    )

    loaded = service.get_session(SID)

    assert loaded is None  # still nothing to warm up, as before
    assert SID not in service._broken_sessions


def test_save_session_settings_refuses_after_a_load_failure_and_writes_nothing(
    service, monkeypatch
):
    monkeypatch.setattr(aa_persistence, "read_session_raw", lambda sid: _raw(["x"]))
    calls = _write_spy(monkeypatch)

    with pytest.raises(SessionLoadError):
        service.save_session_settings(SID, {"view_layers": {"raw": False}})

    assert calls == [], (
        "the underlying persistence call must never fire for a session "
        "whose reconstruction is known to have failed"
    )


def test_save_session_settings_refuses_even_when_already_cached_as_broken(
    service, monkeypatch
):
    """The guard must not depend on this exact request being the one that
    discovers the failure — an earlier /loadSession in the same process
    already marked it, and this call must still refuse."""
    monkeypatch.setattr(aa_persistence, "read_session_raw", lambda sid: _raw(["x"]))
    service._load_session_into_memory(SID)  # the earlier, failing attempt
    assert SID in service._broken_sessions

    calls = _write_spy(monkeypatch)
    with pytest.raises(SessionLoadError):
        service.save_session_settings(SID, {"view_layers": {"raw": False}})
    assert calls == []


def test_save_session_settings_still_works_for_a_healthy_session(
    service, monkeypatch
):
    """The fix must not regress the ordinary case."""
    service.processor = WorkingProcessor()
    monkeypatch.setattr(aa_persistence, "read_session_raw", lambda sid: _raw(["x"]))
    calls = _write_spy(monkeypatch)

    service.save_session_settings(SID, {"view_layers": {"raw": False}})

    assert calls == [(SID, {"view_layers": {"raw": False}})]


def test_save_session_settings_still_works_for_a_staged_no_image_session(
    service, monkeypatch
):
    """The other case the fix must not regress: settings saved for a session
    staged via /initSession, before any image/upload exists."""
    monkeypatch.setattr(
        aa_persistence, "read_session_raw", lambda sid: _raw([], image_bytes=None)
    )
    calls = _write_spy(monkeypatch)

    service.save_session_settings(SID, {"view_layers": {"raw": False}})

    assert calls == [(SID, {"view_layers": {"raw": False}})]


def test_a_later_successful_load_clears_the_broken_mark(service, monkeypatch):
    """A session broken on one attempt (e.g. before a code fix ships, or a
    transient error) must not stay permanently blacklisted once it loads."""
    monkeypatch.setattr(aa_persistence, "read_session_raw", lambda sid: _raw(["x"]))
    service._load_session_into_memory(SID)
    assert SID in service._broken_sessions

    service.processor = WorkingProcessor()
    service.sessions.pop(SID, None)
    service._load_session_into_memory(SID)

    assert SID not in service._broken_sessions

    calls = _write_spy(monkeypatch)
    service.save_session_settings(SID, {"view_layers": {"raw": False}})
    assert calls == [(SID, {"view_layers": {"raw": False}})]
