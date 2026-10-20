"""Reproduces the real user-reported bug: WB-scrubbed sessions saved, then
reopened showing every balloon back as if never scrubbed, with Scrub
producing no visible response.

Confirmed by direct inspection of a real affected session (9bd355ea) before
any fix: `_source` (original upload) and `_pass_image:1` (the scrubbed pass)
were BOTH correctly present on disk with correct pixel content — this was
never data loss. `load_session_from_disk` was sending `_source` regardless of
how many passes had run (bug 1), and `_loadLaunchSession` on the frontend
never populated `_result`/`_segments` at all (bug 2), so even a correctly-sent
image would have left the Scrub button permanently disabled.

This file reproduces 9bd355ea's exact real shape: a session selected and held
at pass 0, scrubbed to pass 1 (masks recorded at pass 0, session's current
pass is 1) — and additionally covers the case these two fixes do NOT touch:
a session whose masks genuinely sit at its own current pass (bug 3, the
pass-reconciliation gap, is explicitly out of scope here).

The bridge is stubbed rather than hitting real Julia parquet I/O —
test_sf_wiring.py, which does, hangs in this environment.
"""

from __future__ import annotations

import base64
import io
import pathlib

import numpy as np
import pytest
from PIL import Image

import aa_persistence
import sf_engine as sfe
from services import SegmentationService
from tests.test_sf_engine import DRAWN, PaintingInpainter, _norm_box

SID = "resume-scrub-session"
IMG_SIZE = 100


def _colour_png(size=IMG_SIZE) -> bytes:
    """Non-uniform so pixel comparisons can't pass by coincidence (a flat
    fixture would make _source and a scrubbed pass accidentally identical
    wherever the scrub repainted with a colour that happened to match)."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            arr[y, x] = (10 + x * 2 % 200, 60 + y * 3 % 180, 120 + (x + y) % 100)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
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


class FakeSam3:
    """Mirrors tests.test_sf_engine.FakeSam3 — reimplemented locally rather
    than imported, since that one is tightly coupled to SHAPE=(10,10) and
    this file needs a larger, non-uniform, pixel-comparable canvas."""

    def __init__(self, size=IMG_SIZE):
        self.size = size
        self.geo_rects: list[tuple] = []

    def _state(self, rects):
        return {
            "masks": [self._rect_mask(r) for r in rects],
            "boxes": [[float(x0), float(y0), float(x1), float(y1)] for (y0, y1, x0, x1) in rects],
            "scores": [0.9] * len(rects),
            "original_width": self.size,
            "original_height": self.size,
        }

    def _rect_mask(self, rect):
        y0, y1, x0, x1 = rect
        m = np.zeros((self.size, self.size), dtype=np.float32)
        m[y0:y1, x0:x1] = 1.0
        return m

    def set_image(self, image):
        return self._state([])

    def set_text_prompt(self, prompt, state):
        return self._state([])

    def add_geometric_prompt(self, box, label, state):
        return self._state(self.geo_rects)

    def add_point_prompt(self, point, label, state):
        return self._state(self.geo_rects)


def _norm_box_100(rect) -> list[float]:
    y0, y1, x0, x1 = rect
    return [((x0 + x1) / 2) / IMG_SIZE, ((y0 + y1) / 2) / IMG_SIZE,
            (x1 - x0) / IMG_SIZE, (y1 - y0) / IMG_SIZE]


RECT_A = (10, 30, 10, 30)   # a box in the pass-0 selection
RECT_B = (50, 70, 50, 70)   # a second box, selected AFTER the scrub (pass 1)


@pytest.fixture()
def bridge(tmp_path, monkeypatch):
    fake = FakeBridge()
    monkeypatch.setattr(aa_persistence, "bridge", fake)
    monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf")
    return fake


@pytest.fixture()
def service(bridge, tmp_path):
    seg = FakeSam3()
    svc = SegmentationService(tmp_path / "sessions", seg)
    return svc


def _upload(service, contents=None):
    contents = contents or _colour_png()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    state = service.processor.set_image(image)
    service.register_session_data(SID, {
        "name": "McCay page", "description": "", "image_url": "",
        "state": state, "original_image_bytes": contents,
        "image_size": image.size,
    })
    return service.sf_sessions[SID]


def _restart(service):
    """Simulate a process restart: every in-memory cache gone, only the
    parquet files remain."""
    service.sessions.clear()
    service.sf_sessions.clear()


# ── Bug 1: the image sent on resume ──────────────────────────────────────────


class TestResumeSendsTheCurrentPassImage:
    def test_a_wb_scrubbed_session_resumes_showing_the_scrubbed_image(
        self, service
    ):
        """9bd355ea's exact real shape: selected+held+scrubbed at pass 0,
        producing pass 1. On disk: _source (original) and _pass_image:1
        (scrubbed) both present, masks recorded at pass 0."""
        sf_session = _upload(service)
        service.processor.geo_rects = [RECT_A]
        rec = sf_session.add_box_selection(_norm_box_100(RECT_A), True)[0]
        sf_session.set_held(rec.mask_id, True)
        sf_session.run_lama_pass(PaintingInpainter(color=(255, 0, 0, 255)))
        service.sessions[SID]["state"] = sf_session.state
        assert sf_session.pass_ == 1
        assert rec.pass_ == 0  # the real-world shape: mask at an earlier pass

        service.save_session_to_disk(SID)
        _restart(service)

        response = service.load_session_from_disk(SID)

        returned = base64.b64decode(response["image_b64"])
        reloaded = service.sf_sessions[SID]
        scrubbed = sfe._png_bytes(reloaded.image_for_pass(1))
        original = reloaded.original.bytes

        assert returned == scrubbed, "resume must send the scrubbed pass image"
        assert returned != original, (
            "resume must NOT send the original — this is the literal bug: "
            "balloons reappearing because _source was sent unconditionally"
        )

    def test_a_fresh_never_scrubbed_session_still_sends_source_unchanged(
        self, service
    ):
        """Regression check for the common case: no scrub history at all."""
        contents = _colour_png()
        sf_session = _upload(service, contents)
        assert sf_session.pass_ == 0

        service.save_session_to_disk(SID)
        _restart(service)

        response = service.load_session_from_disk(SID)

        returned = base64.b64decode(response["image_b64"])
        assert returned == contents, (
            "a never-scrubbed session must send the exact original bytes, "
            "not a PIL re-encoding of pass 0's image"
        )


# ── Bug 2: _result / segments on resume ──────────────────────────────────────


class TestResumeResultsPopulateSegments:
    def test_results_reflect_masks_at_the_sessions_current_pass(self, service):
        """The case bug 2's fix is actually for: a mask that genuinely sits
        at the session's current pass. This is NOT the bug-3 shape."""
        sf_session = _upload(service)
        service.processor.geo_rects = [RECT_A]
        rec_a = sf_session.add_box_selection(_norm_box_100(RECT_A), True)[0]
        sf_session.set_held(rec_a.mask_id, True)
        sf_session.run_lama_pass(PaintingInpainter(color=(255, 0, 0, 255)))
        service.sessions[SID]["state"] = sf_session.state
        assert sf_session.pass_ == 1

        # Selected AFTER the scrub -> this one's pass_ == 1 == current pass.
        service.processor.geo_rects = [RECT_B]
        rec_b = sf_session.add_box_selection(_norm_box_100(RECT_B), True)[0]
        service.sessions[SID]["state"] = sf_session.state
        assert rec_b.pass_ == 1

        service.save_session_to_disk(SID)
        _restart(service)

        response = service.load_session_from_disk(SID)
        results = response["results"]

        assert results["mask_ids"] == [rec_b.mask_id], (
            "only the current-pass mask belongs in `results` -- an earlier "
            "pass's mask (rec_a) was traced against a different image"
        )
        assert len(results["masks"]) == 1  # a real RLE entry, not empty
        assert len(results["boxes"]) == 1

    def test_all_masks_still_carries_every_pass_regardless(self, service):
        """`all_masks` (the AA Preview tab's field) is unfiltered by design --
        confirms the fix did not accidentally narrow it too."""
        sf_session = _upload(service)
        service.processor.geo_rects = [RECT_A]
        rec_a = sf_session.add_box_selection(_norm_box_100(RECT_A), True)[0]
        sf_session.set_held(rec_a.mask_id, True)
        sf_session.run_lama_pass(PaintingInpainter(color=(255, 0, 0, 255)))
        service.sessions[SID]["state"] = sf_session.state
        service.processor.geo_rects = [RECT_B]
        rec_b = sf_session.add_box_selection(_norm_box_100(RECT_B), True)[0]
        service.sessions[SID]["state"] = sf_session.state

        service.save_session_to_disk(SID)
        _restart(service)

        response = service.load_session_from_disk(SID)

        assert set(response["all_masks"]["mask_ids"]) == {rec_a.mask_id, rec_b.mask_id}


# ── Bug 3: must remain visible, not hidden or crashed ────────────────────────


class TestBug3RemainsVisibleNotPaperedOver:
    """Explicit confirmation per the task: after fixing 1 and 2, a session
    whose masks sit at an earlier pass than its current one must show a
    clean, empty, non-crashing result -- not a silent success, not an
    exception, and not accidentally working by coincidence."""

    def test_resumed_results_are_empty_when_every_mask_predates_the_current_pass(
        self, service
    ):
        sf_session = _upload(service)
        service.processor.geo_rects = [RECT_A]
        rec = sf_session.add_box_selection(_norm_box_100(RECT_A), True)[0]
        sf_session.set_held(rec.mask_id, True)
        sf_session.run_lama_pass(PaintingInpainter(color=(255, 0, 0, 255)))
        service.sessions[SID]["state"] = sf_session.state
        assert sf_session.pass_ == 1 and rec.pass_ == 0

        service.save_session_to_disk(SID)
        _restart(service)

        response = service.load_session_from_disk(SID)

        # The image is STILL correctly scrubbed (bug 1's fix) ...
        returned = base64.b64decode(response["image_b64"])
        assert returned == sfe._png_bytes(service.sf_sessions[SID].image_for_pass(1))
        # ... but `results` -- what the canvas selects from and what gates
        # Scrub's enabled state -- is cleanly empty, not an error and not a
        # misleading non-empty value borrowed from the wrong pass.
        assert response["results"]["mask_ids"] == []
        assert response["results"]["masks"] == []
        assert response["results"]["boxes"] == []
        # The mask is NOT gone -- all_masks (session-wide, pass-aware) still
        # has it. Only the current-pass-filtered `results` is empty.
        assert rec.mask_id in response["all_masks"]["mask_ids"]

    def test_loading_such_a_session_raises_nothing(self, service):
        """No exception anywhere in the resume path for this shape -- a
        clean empty result, never a crash."""
        sf_session = _upload(service)
        service.processor.geo_rects = [RECT_A]
        rec = sf_session.add_box_selection(_norm_box_100(RECT_A), True)[0]
        sf_session.set_held(rec.mask_id, True)
        sf_session.run_lama_pass(PaintingInpainter(color=(255, 0, 0, 255)))
        service.sessions[SID]["state"] = sf_session.state
        service.save_session_to_disk(SID)
        _restart(service)

        response = service.load_session_from_disk(SID)  # must not raise
        assert response is not None


# ── Bug 3 audit fix #1: the `pass` field /loadSession never sent ────────────


class TestResumeReportsCurrentPass:
    """/loadSession carried no current-pass number at all, so the frontend
    had no way to distinguish a resumed post-scrub session from a fresh one
    and show the same scrub label a live scrub already shows. One added
    field — nothing else about the response changes."""

    def test_a_scrubbed_session_reports_its_post_scrub_pass(self, service):
        sf_session = _upload(service)
        service.processor.geo_rects = [RECT_A]
        rec = sf_session.add_box_selection(_norm_box_100(RECT_A), True)[0]
        sf_session.set_held(rec.mask_id, True)
        sf_session.run_lama_pass(PaintingInpainter(color=(255, 0, 0, 255)))
        service.sessions[SID]["state"] = sf_session.state
        assert sf_session.pass_ == 1

        service.save_session_to_disk(SID)
        _restart(service)

        response = service.load_session_from_disk(SID)
        assert response["pass"] == 1

    def test_a_never_scrubbed_session_reports_pass_zero(self, service):
        _upload(service)
        service.save_session_to_disk(SID)
        _restart(service)

        response = service.load_session_from_disk(SID)
        assert response["pass"] == 0

    def test_a_session_scrubbed_twice_reports_the_latest_pass(self, service):
        sf_session = _upload(service)
        service.processor.geo_rects = [RECT_A]
        rec_a = sf_session.add_box_selection(_norm_box_100(RECT_A), True)[0]
        sf_session.set_held(rec_a.mask_id, True)
        sf_session.run_lama_pass(PaintingInpainter(color=(255, 0, 0, 255)))
        service.processor.geo_rects = [RECT_B]
        rec_b = sf_session.add_box_selection(_norm_box_100(RECT_B), True)[0]
        sf_session.set_held(rec_b.mask_id, True)
        sf_session.run_lama_pass(PaintingInpainter(color=(0, 255, 0, 255)))
        service.sessions[SID]["state"] = sf_session.state
        assert sf_session.pass_ == 2

        service.save_session_to_disk(SID)
        _restart(service)

        response = service.load_session_from_disk(SID)
        assert response["pass"] == 2

    def test_a_session_staged_but_never_uploaded_to_reports_pass_zero(
        self, service, monkeypatch
    ):
        """No SFSession is ever reconstructed for this shape (no image —
        see load_session_from_disk's own docstring) — `pass` must still be
        present and 0, not missing or raising. Same raw shape
        test_session_load_failure_no_write.py's own staged-session case
        uses."""
        monkeypatch.setattr(
            aa_persistence,
            "read_session_raw",
            lambda sid: {
                "name": "staged only", "description": "",
                "created_at": "2026-01-01T00:00:00", "image_url": None,
                "original_filename": "image.png", "width": 0, "height": 0,
                "image_bytes": None, "pass_images": {}, "segments": [],
                "prompts": [], "ui_settings": {},
            },
        )

        response = service.load_session_from_disk(SID)
        assert response["pass"] == 0


# ── Resume does not replay prompts it would throw away ──────────────────────


class TestResumeSkipsDiscardedReplay:
    """A session past pass 0 had its prompts replayed against the ORIGINAL image
    and the result thrown away (`_reconstruct_sf_session` re-`set_image`s on the
    current pass). That replay is a SAM3 forward per saved prompt; on a scrubbed
    session with many prompts it was minutes of unused work."""

    def _count_calls(self, service):
        calls = {"text": 0, "set_image": 0}
        seg = service.processor
        orig_text, orig_set = seg.set_text_prompt, seg.set_image

        def text(prompt, state):
            calls["text"] += 1
            return orig_text(prompt, state)

        def set_image(image):
            calls["set_image"] += 1
            return orig_set(image)

        seg.set_text_prompt, seg.set_image = text, set_image
        return calls

    def test_a_scrubbed_session_does_not_replay_its_prompts(self, service):
        sf_session = _upload(service)
        service.processor.geo_rects = [RECT_A]
        rec = sf_session.add_box_selection(_norm_box_100(RECT_A), True)[0]
        sf_session.set_held(rec.mask_id, True)
        sf_session.run_lama_pass(PaintingInpainter(color=(255, 0, 0, 255)))
        service.sessions[SID]["state"] = sf_session.state
        service.sessions[SID]["prompts"] = ["balloon", "title", "cloud"]
        service.save_session_to_disk(SID)
        _restart(service)

        calls = self._count_calls(service)
        response = service.load_session_from_disk(SID)

        assert calls["text"] == 0, "prompts were replayed into a state that is then discarded"
        assert calls["set_image"] == 1, "only the current pass's image needs embedding"
        assert rec.mask_id in response["all_masks"]["mask_ids"]
        # The session is still fully usable: it has a real state of its own.
        assert service.sf_sessions[SID].state is service.sessions[SID]["state"]
        assert service.sf_sessions[SID].state is not None

    def test_a_pass_zero_session_still_replays_its_prompts(self, service):
        _upload(service)
        service.sessions[SID]["prompts"] = ["balloon", "title"]
        service.save_session_to_disk(SID)
        _restart(service)

        calls = self._count_calls(service)
        service.load_session_from_disk(SID)

        assert calls["text"] == 2
