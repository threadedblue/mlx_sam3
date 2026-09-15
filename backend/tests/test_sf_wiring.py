"""End-to-end tests for SFSession wired into the running backend:
session-state mask_type tracking (PART 1), the /lama/scrub endpoint
(PART 2), and aa_persistence.py's schema extension (PART 3).

Drives real HTTP endpoints via FastAPI's TestClient against `main.app`,
with `main.model`/`main.processor`/`main.service` swapped for fakes so no
real SAM3/LaMa model ever loads — `TestClient(app)` used without a `with`
block does not run `lifespan`, confirmed separately.

Run with:  pytest backend/tests/test_sf_wiring.py -v
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import aa_persistence
import d4m_juliacall_bridge as bridge
import main
import sf_engine
from services import SegmentationService

IMG_SIZE = 100


def _mask_for_px_box(px_box: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = px_box
    m = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    m[int(y0):int(y1), int(x0):int(x1)] = 1.0
    return m


class FakeProcessor:
    """Minimal Sam3Processor double with the same non-append-only semantics
    documented in aa_persistence.py's module docstring: `set_text_prompt`
    replaces, `add_geometric_prompt`/`add_point_prompt` re-ground the whole
    accumulated box/point history every call."""

    def __init__(self):
        self.text_boxes: dict[str, list[tuple[float, float, float, float]]] = {}
        self._geo_boxes: list[tuple] = []

    def set_image(self, image):
        w, h = image.size
        self._geo_boxes = []
        return {"masks": [], "boxes": [], "scores": [], "original_width": w, "original_height": h}

    def set_text_prompt(self, prompt, state):
        px_boxes = self.text_boxes.get(prompt, [])
        masks = [_mask_for_px_box(b) for b in px_boxes]
        return {**state, "masks": masks, "boxes": [list(b) for b in px_boxes], "scores": [0.9] * len(masks)}

    def add_geometric_prompt(self, box, label, state):
        self._geo_boxes.append(tuple(box))
        px_boxes = [self._to_px(b) for b in self._geo_boxes]
        masks = [_mask_for_px_box(b) for b in px_boxes]
        return {**state, "masks": masks, "boxes": [list(b) for b in px_boxes], "scores": [0.9] * len(masks)}

    def add_point_prompt(self, point, label, state):
        x, y = point
        return self.add_geometric_prompt([x, y, 0.05, 0.05], label, state)

    @staticmethod
    def _to_px(cxcywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        cx, cy, w, h = cxcywh
        return (
            (cx - w / 2) * IMG_SIZE, (cy - h / 2) * IMG_SIZE,
            (cx + w / 2) * IMG_SIZE, (cy + h / 2) * IMG_SIZE,
        )


class FakeInpainter:
    def inpaint(self, image, mask):
        return image  # no-op scrub — orchestration is what's under test, not pixels


def _upload_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (IMG_SIZE, IMG_SIZE), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    fake_processor = FakeProcessor()
    fake_service = SegmentationService(tmp_path / "sessions", fake_processor)

    monkeypatch.setattr(main, "model", object())
    monkeypatch.setattr(main, "processor", fake_processor)
    monkeypatch.setattr(main, "service", fake_service)
    monkeypatch.setattr(main, "_lama_inpainter", FakeInpainter())
    monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf_storage")

    return TestClient(main.app)


def _upload(client) -> str:
    files = {"file": ("test.png", _upload_png_bytes(), "image/png")}
    r = client.post("/upload", files=files)
    assert r.status_code == 200
    return r.json()["session_id"]


def _drive_full_mask_flow(client) -> str:
    """Runs mark-in/mark-out/scrub/mark-in-again via the real HTTP
    endpoints, asserting the whole way; returns the session_id for callers
    (TestSaveReloadRoundTrip) that need to inspect what got persisted."""
    session_id = _upload(client)

    main.processor.text_boxes = {"widget": [(10, 10, 30, 30)]}
    r = client.post("/segment/text", json={
        "session_id": session_id, "prompt": "widget", "mask_type": "in",
    })
    assert r.status_code == 200
    assert len(r.json()["results"]["masks"]) == 1

    r = client.post("/segment/box", json={
        "session_id": session_id,
        "box": [0.7, 0.7, 0.2, 0.2],
        "label": True,
        "mask_type": "out",
        "text_substitute": "word balloon",
    })
    assert r.status_code == 200

    sf_session = main.service.sf_sessions[session_id]
    assert len(sf_session.masks) == 2
    assert {m.mask_type.value for m in sf_session.masks} == {"in", "out"}
    out_record = next(m for m in sf_session.masks if m.mask_type is sf_engine.MaskType.OUT)
    assert out_record.text_tag == "word balloon"

    # Scrub
    r = client.post("/lama/scrub", json={"session_id": session_id})
    assert r.status_code == 200
    body = r.json()
    assert body["image_b64"]
    assert body["consumed_out_mask_ids"] == [out_record.mask_id]
    assert sf_session.pass_ is sf_engine.Pass.BACKGROUND

    # Two-pass lock: a second scrub is a 409, not a 500
    r = client.post("/lama/scrub", json={"session_id": session_id})
    assert r.status_code == 409
    assert "already run" in r.json()["detail"]

    # Pass 2 "out" is rejected outright, not silently accepted as a no-op
    r = client.post("/segment/text", json={
        "session_id": session_id, "prompt": "anything", "mask_type": "out",
    })
    assert r.status_code == 422
    assert len(sf_session.masks) == 2  # rejected before touching state

    # Pass 2: mark newly-uncovered background as IN
    main.processor.text_boxes = {"lamp": [(50, 50, 70, 70)]}
    r = client.post("/segment/text", json={
        "session_id": session_id, "prompt": "lamp", "mask_type": "in",
    })
    assert r.status_code == 200

    assert len(sf_session.masks) == 3
    pass2_masks = [m for m in sf_session.masks if m.pass_ is sf_engine.Pass.BACKGROUND]
    assert len(pass2_masks) == 1
    assert pass2_masks[0].text_tag == "lamp"
    assert pass2_masks[0].mask_type is sf_engine.MaskType.IN

    # Pass 1's records are untouched by the Pass 2 call.
    pass1_masks = [m for m in sf_session.masks if m.pass_ is sf_engine.Pass.FOREGROUND]
    assert len(pass1_masks) == 2

    # session["state"] and sf_session.state never diverged (PART 1).
    flat_session = main.service.sessions[session_id]
    assert flat_session["state"] is sf_session.state

    return session_id


class TestHttpMaskFlow:
    """Deliverable 1: mark-in/mark-out/scrub/mark-in-again through the real
    HTTP endpoints (SFSession itself is already covered by test_sf_engine.py
    — this exercises main.py's routing, not the engine's own logic)."""

    def test_full_flow(self, client):
        _drive_full_mask_flow(client)


class TestSaveReloadRoundTrip:
    """Deliverable 2: mask_type and pass survive a save/reload round trip."""

    def test_mask_type_and_pass_survive_direct_read(self, client):
        session_id = _drive_full_mask_flow(client)

        raw = aa_persistence.read_session_raw(session_id)
        assert raw is not None
        assert raw["background_image_bytes"] is not None
        by_tag = {seg["text_tag"]: seg for seg in raw["segments"] if seg["text_tag"]}

        assert by_tag["widget"]["mask_type"] == "in"
        assert by_tag["widget"]["pass"] == "foreground"
        assert by_tag["word balloon"]["mask_type"] == "out"
        assert by_tag["word balloon"]["pass"] == "foreground"
        assert by_tag["lamp"]["mask_type"] == "in"
        assert by_tag["lamp"]["pass"] == "background"

    def test_reload_reconstructs_sf_session_across_both_passes(self, client):
        session_id = _drive_full_mask_flow(client)
        original_sf_session = main.service.sf_sessions[session_id]
        original_mask_ids = {m.mask_id for m in original_sf_session.masks}

        # Simulate a process restart: evict the in-memory cache entirely.
        del main.service.sessions[session_id]
        del main.service.sf_sessions[session_id]

        reloaded_flat = main.service.get_session(session_id)
        assert reloaded_flat is not None
        reloaded_sf = main.service.sf_sessions[session_id]

        assert reloaded_sf.pass_ is sf_engine.Pass.BACKGROUND
        assert reloaded_sf.working_copy is not None
        assert {m.mask_id for m in reloaded_sf.masks} == original_mask_ids

        by_tag = {m.text_tag: m for m in reloaded_sf.masks if m.text_tag}
        assert by_tag["widget"].mask_type is sf_engine.MaskType.IN
        assert by_tag["widget"].pass_ is sf_engine.Pass.FOREGROUND
        assert by_tag["word balloon"].mask_type is sf_engine.MaskType.OUT
        assert by_tag["lamp"].pass_ is sf_engine.Pass.BACKGROUND

        # Foreground-reload single-source-of-truth guarantee doesn't apply
        # here (this session reached Pass 2 — see services.py's
        # _reconstruct_sf_session docstring on the flagged Pass-2 gap), but
        # the reconstructed state must still target the right image.
        assert reloaded_sf.state["original_width"] == IMG_SIZE


class TestOldSessionMigration:
    """Deliverable 3: sessions saved before this schema existed read back
    with defined defaults, not undefined/missing behavior."""

    def test_pre_migration_segment_defaults_to_in_and_foreground(self, tmp_path, monkeypatch):
        monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf_storage")
        session_id = "legacy-session"
        d = aa_persistence.session_dir(session_id)

        # Hand-write registry.parquet + an OLD-format segment.parquet: bare
        # uuid4 row key, no mask_type/pass/text_tag columns at all.
        bridge.save_parquet(str(d / "registry.parquet"), [session_id], ["name"], ["Legacy"])
        old_mask_id = "old-mask-uuid"
        row = f"{session_id}:{old_mask_id}"
        tiny_png = _upload_png_bytes()
        bridge.save_parquet(
            str(d / "segment.parquet"),
            [f"{session_id}:_source", row, row],
            ["image_bytes", "crop_bytes", "mask_bytes"],
            [
                aa_persistence._b64(tiny_png),
                aa_persistence._b64(tiny_png),
                aa_persistence._b64(tiny_png),
            ],
        )

        raw = aa_persistence.read_session_raw(session_id)
        assert raw is not None
        assert len(raw["segments"]) == 1
        seg = raw["segments"][0]
        assert seg["segment_id"] == old_mask_id
        assert seg["mask_type"] == "in"
        assert seg["pass"] == "foreground"
        assert seg["text_tag"] == ""
        assert raw["background_image_bytes"] is None

    def test_reconstructed_mask_id_gets_a_synthesized_pass_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf_storage")
        fake_processor = FakeProcessor()
        fake_service = SegmentationService(tmp_path / "sessions", fake_processor)

        session_id = "legacy-session-2"
        d = aa_persistence.session_dir(session_id)
        bridge.save_parquet(str(d / "registry.parquet"), [session_id], ["name"], ["Legacy"])
        old_mask_id = "bare-uuid-no-pass-prefix"
        row = f"{session_id}:{old_mask_id}"
        tiny_png = _upload_png_bytes()
        bridge.save_parquet(
            str(d / "segment.parquet"),
            [f"{session_id}:_source", row, row],
            ["image_bytes", "crop_bytes", "mask_bytes"],
            [aa_persistence._b64(tiny_png)] * 3,
        )

        loaded = fake_service.get_session(session_id)
        assert loaded is not None
        sf_session = fake_service.sf_sessions[session_id]
        assert len(sf_session.masks) == 1
        migrated = sf_session.masks[0]
        assert migrated.mask_id == f"foreground:{old_mask_id}"
        assert migrated.mask_type is sf_engine.MaskType.IN
        assert migrated.pass_ is sf_engine.Pass.FOREGROUND
        assert sf_session.pass_ is sf_engine.Pass.FOREGROUND
