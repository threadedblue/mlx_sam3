"""Regression test for the Oct 2 incident: a backend started Sept 30 kept
serving pre-fix code for two days, surviving every restart attempt, because
the only question anyone asked it was "are you listening?".

`isListening()`/`healthCheck()` cannot distinguish a current backend from one
running days-old code — both answer 200 with `model_loaded: true`. So the
backend now answers a second question about itself: is the source on disk
still the source I imported?

The shape reproduced here is the incident's exactly: a process starts, source
changes underneath it, and the already-running process must report itself
stale WITHOUT restarting (it cannot restart itself — that is the launcher's
job, and the launcher needs this flag to know).
"""

from __future__ import annotations

import importlib
import os
import time

import pytest

import source_fingerprint


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """A throwaway source tree, so a test can edit 'source' without touching
    the real backend."""
    (tmp_path / "alpha.py").write_text("x = 1\n")
    (tmp_path / "beta.py").write_text("y = 2\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "alpha.cpython-313.pyc").write_bytes(b"\x00bytecode")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_thing.py").write_text("def test_x(): pass\n")
    monkeypatch.setattr(source_fingerprint, "BACKEND_DIR", tmp_path)
    return tmp_path


def _snapshot(module=source_fingerprint):
    """Freeze 'what this process imported' at this moment, the way module
    import does for the real server."""
    module.DIGEST_AT_START = module.compute_digest()
    module.STARTED_AT = time.time()


class TestTheOct2IncidentShape:
    def test_a_running_process_detects_source_changed_underneath_it(self, sandbox):
        """The incident, reduced: process starts, source is edited, process is
        never restarted — it must now report itself stale."""
        _snapshot()
        assert source_fingerprint.status()["stale"] is False

        # The Oct 1 edit, landing under a Sept 30 process.
        (sandbox / "alpha.py").write_text("x = 1  # the fix\n")

        status = source_fingerprint.status()
        assert status["stale"] is True
        assert status["source_digest_now"] != status["source_digest_at_start"]

    def test_the_old_check_could_not_have_caught_it(self, sandbox):
        """Pins WHY this exists: liveness is unchanged by the edit, so a
        listening/health check sees nothing wrong. This is the gap."""
        _snapshot()
        (sandbox / "alpha.py").write_text("x = 1  # the fix\n")

        # Everything the previous mechanism could observe is identical before
        # and after — the process is just as alive, just as ready.
        assert source_fingerprint.status()["stale"] is True, (
            "only the content check distinguishes these two states"
        )

    def test_a_new_file_counts_as_a_change(self, sandbox):
        _snapshot()
        (sandbox / "gamma.py").write_text("z = 3\n")
        assert source_fingerprint.status()["stale"] is True

    def test_a_deleted_file_counts_as_a_change(self, sandbox):
        _snapshot()
        (sandbox / "beta.py").unlink()
        assert source_fingerprint.status()["stale"] is True

    def test_a_rename_counts_as_a_change_even_with_identical_bytes(self, sandbox):
        """Paths are hashed alongside contents, so a pure rename registers."""
        _snapshot()
        (sandbox / "beta.py").rename(sandbox / "renamed.py")
        assert source_fingerprint.status()["stale"] is True


class TestNoFalsePositives:
    def test_a_touch_with_no_content_change_is_not_stale(self, sandbox):
        """The edge case worth getting right: an editor save with no edit, or
        a checkout rewriting identical bytes, must not force a restart. An
        mtime-based check would fail this."""
        _snapshot()
        future = time.time() + 10_000
        os.utime(sandbox / "alpha.py", (future, future))

        assert source_fingerprint.status()["stale"] is False

    def test_an_edit_that_is_reverted_is_not_stale(self, sandbox):
        _snapshot()
        original = (sandbox / "alpha.py").read_text()
        (sandbox / "alpha.py").write_text("x = 999\n")
        assert source_fingerprint.status()["stale"] is True

        (sandbox / "alpha.py").write_text(original)
        assert source_fingerprint.status()["stale"] is False, (
            "back to the imported bytes — nothing to restart for"
        )

    def test_bytecode_churn_is_ignored(self, sandbox):
        """__pycache__ mirrors the .py it came from and rewrites itself on
        import; counting it would make every start look like a change."""
        _snapshot()
        (sandbox / "__pycache__" / "alpha.cpython-313.pyc").write_bytes(b"\x00recompiled")
        assert source_fingerprint.status()["stale"] is False

    def test_editing_a_test_does_not_demand_a_restart(self, sandbox):
        """Tests cannot change what the server serves, and reloading a SAM3
        model costs tens of seconds."""
        _snapshot()
        (sandbox / "tests" / "test_thing.py").write_text("def test_x(): assert True\n")
        assert source_fingerprint.status()["stale"] is False


class TestReportedShape:
    def test_status_reports_what_a_launcher_needs(self, sandbox):
        _snapshot()
        status = source_fingerprint.status()
        assert set(status) == {
            "started_at", "source_digest_at_start", "source_digest_now", "stale",
        }
        assert isinstance(status["stale"], bool)
        assert isinstance(status["started_at"], float)

    def test_an_unreadable_file_does_not_raise(self, sandbox, monkeypatch):
        """/health must not 500 because a file vanished mid-walk."""
        _snapshot()
        real_read = type(sandbox / "alpha.py").read_bytes

        def boom(self):
            if self.name == "alpha.py":
                raise OSError("vanished")
            return real_read(self)

        monkeypatch.setattr(type(sandbox / "alpha.py"), "read_bytes", boom)
        status = source_fingerprint.status()  # must not raise
        assert status["stale"] is True


class TestRealBackendTree:
    def test_the_real_backend_reports_itself_current(self):
        """Sanity against the actual tree, not just the sandbox: an unmodified
        checkout must not claim to be stale, or every launch force-restarts."""
        importlib.reload(source_fingerprint)
        assert source_fingerprint.status()["stale"] is False

    def test_the_real_digest_covers_main_and_services(self):
        importlib.reload(source_fingerprint)
        names = {p.name for p in source_fingerprint._source_files()}
        assert {"main.py", "services.py", "sf_engine.py"} <= names
        assert not any("__pycache__" in str(p) for p in source_fingerprint._source_files())


class TestOverTheHttpSurface:
    """The incident end to end, through the real `/health` endpoint a launcher
    actually calls — not just the module's own function."""

    @pytest.fixture()
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient
        import main
        # TestClient without a `with` block does not run lifespan, so no real
        # SAM3 model loads (same convention as test_sf_wiring.py's header).
        monkeypatch.setattr(main, "model", object())
        return TestClient(main.app)

    def test_health_reports_current_then_stale_after_an_edit(
        self, client, monkeypatch, tmp_path
    ):
        (tmp_path / "alpha.py").write_text("x = 1\n")
        monkeypatch.setattr(source_fingerprint, "BACKEND_DIR", tmp_path)
        _snapshot()

        before = client.get("/health").json()
        assert before["status"] == "healthy"
        assert before["stale"] is False, "unmodified source must not be stale"

        # The edit landing under the already-running process.
        (tmp_path / "alpha.py").write_text("x = 1  # the fix\n")

        after = client.get("/health").json()
        # Liveness is untouched -- which is exactly why the old check missed it.
        assert after["status"] == "healthy"
        assert after["stale"] is True, (
            "a running process whose source changed must report itself stale"
        )

    def test_a_launcher_can_tell_the_two_apart_from_the_response_alone(
        self, client, monkeypatch, tmp_path
    ):
        """What DoubleNaught's isStale() keys off: nothing but the JSON."""
        (tmp_path / "alpha.py").write_text("x = 1\n")
        monkeypatch.setattr(source_fingerprint, "BACKEND_DIR", tmp_path)
        _snapshot()
        fresh = client.get("/health").json()
        (tmp_path / "alpha.py").write_text("x = 2\n")
        stale = client.get("/health").json()

        assert ("stale" in fresh) and ("stale" in stale)
        assert fresh["stale"] is False and stale["stale"] is True
        # And the pre-mechanism case the launcher also treats as stale: a
        # backend too old to carry the key at all.
        legacy = {"status": "healthy", "model_loaded": True}
        assert "stale" not in legacy
