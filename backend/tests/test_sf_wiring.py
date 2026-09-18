"""End-to-end tests for the v2 model (sf-model-v2-design.md) wired into the
running backend: select/hold/caption/scrub over the real HTTP endpoints,
aa_persistence.py's schema (pass/dataset_status/held/caption), and the
training-set compile step.

Drives real HTTP endpoints via FastAPI's TestClient against `main.app`,
with `main.model`/`main.processor`/`main.service` swapped for fakes so no
real SAM3/LaMa model ever loads — `TestClient(app)` used without a `with`
block does not run `lifespan`, confirmed separately.

Persistence is explicit-Save-only (design doc §5): nothing here relies on
an endpoint silently autosaving — every test that checks disk state calls
/saveSession (or `service.save_session_to_disk`) itself.

Run with:  pytest backend/tests/test_sf_wiring.py -v
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import aa_persistence
import compile_training_set
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

    def reset_all_prompts(self, state):
        """Mirrors the real Sam3Processor.reset_all_prompts: drops the
        accumulated geometric-prompt history and the last grounding result.
        Was missing entirely until /reset's own test coverage needed it —
        every prior test exercising /reset went through it via `_reset`
        only implicitly, none actually asserted on its response."""
        self._geo_boxes = []
        for key in ("geometric_prompt", "boxes", "masks", "masks_logits", "scores"):
            state.pop(key, None)

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


def _box_select(client, session_id, box, text_substitute=None) -> str:
    """Selects via /segment/box and returns the mask id THIS call touched.

    Not `results["mask_ids"][0]`/length: `results` is the full current-pass
    list (serialize_sf_masks is not a diff — see its docstring), so it
    already has every prior selection in it too once more than one exists.
    `selected_mask_id` is the field that exists specifically to answer
    "which one was this call about."
    """
    r = client.post("/segment/box", json={
        "session_id": session_id, "box": box, "label": True,
        **({"text_substitute": text_substitute} if text_substitute else {}),
    })
    assert r.status_code == 200
    mask_id = r.json()["selected_mask_id"]
    assert mask_id is not None, f"expected a selection, got {r.json()}"
    return mask_id


def _hold(client, session_id, mask_id, held=True):
    r = client.post("/mask/hold", json={"session_id": session_id, "mask_id": mask_id, "held": held})
    assert r.status_code == 200, r.text
    return r.json()


def _caption(client, session_id, mask_id, caption):
    r = client.post("/mask/caption", json={"session_id": session_id, "mask_id": mask_id, "caption": caption})
    assert r.status_code == 200, r.text
    return r.json()


class TestHttpSelectHoldCaptionScrub:
    """The actual acceptance test from the prompt: select (no caption) ->
    hold -> caption via the second mask's Prompt-card-equivalent call ->
    hold a second, uncaptioned mask -> scrub -> confirm both scrubbed."""

    def test_full_flow(self, client):
        session_id = _upload(client)

        keeper_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        balloon_id = _box_select(client, session_id, [0.7, 0.7, 0.2, 0.2], text_substitute="word balloon")
        assert keeper_id != balloon_id

        # Selected, but neither held nor captioned yet.
        sf_session = main.service.sf_sessions[session_id]
        assert all(not m.held and m.dataset_status == sf_engine.DatasetStatus.UNASSIGNED for m in sf_session.masks)

        _hold(client, session_id, keeper_id, True)
        caption_resp = _caption(client, session_id, keeper_id, "a red egg")
        assert caption_resp["dataset_status"] == "keep"
        assert sf_session.get_mask(keeper_id).held is True  # captioning didn't touch held

        _hold(client, session_id, balloon_id, True)
        assert sf_session.get_mask(balloon_id).dataset_status == sf_engine.DatasetStatus.UNASSIGNED  # never captioned

        r = client.post("/lama/scrub", json={"session_id": session_id})
        assert r.status_code == 200
        body = r.json()
        assert body["from_pass"] == 0 and body["to_pass"] == 1
        assert set(body["scrubbed_mask_ids"]) == {keeper_id, balloon_id}

        assert sf_session.pass_ == 1
        assert sf_session.get_mask(keeper_id).held is False
        assert sf_session.get_mask(balloon_id).held is False
        # Scrubbing didn't touch dataset_status either direction.
        assert sf_session.get_mask(keeper_id).dataset_status == sf_engine.DatasetStatus.KEEP
        assert sf_session.get_mask(balloon_id).dataset_status == sf_engine.DatasetStatus.UNASSIGNED

    def test_results_include_the_current_pass_number(self, client):
        """Fix: serialize_sf_masks never returned a pass number at all —
        every record in a response is implicitly "whatever pass the
        session is currently at", but nothing surfaced that number itself.
        Needed by SegForge/frontend's AA preview adapter, which has no
        other way to label which pass a row belongs to."""
        session_id = _upload(client)

        r = client.post("/segment/box", json={"session_id": session_id, "box": [0.2, 0.2, 0.2, 0.2], "label": True})
        assert r.json()["results"]["passes"] == [0]

        _box_select(client, session_id, [0.7, 0.7, 0.2, 0.2])
        r = client.post("/segment/box", json={"session_id": session_id, "box": [0.2, 0.2, 0.2, 0.2], "label": True})
        # Both current-pass records report the same pass — a constant
        # repeated once per row, not per-record data — matching
        # serialize_sf_masks' own current-pass-only filter.
        assert r.json()["results"]["passes"] == [0, 0]

        client.post("/lama/scrub", json={"session_id": session_id})
        r = client.post("/segment/box", json={"session_id": session_id, "box": [0.5, 0.5, 0.1, 0.1], "label": True})
        assert r.json()["results"]["passes"] == [1]

    def test_scrub_is_repeatable_no_two_pass_cap(self, client):
        session_id = _upload(client)
        for _ in range(3):
            r = client.post("/lama/scrub", json={"session_id": session_id})
            assert r.status_code == 200
        assert main.service.sf_sessions[session_id].pass_ == 3

    def test_hold_unknown_mask_is_404_not_500(self, client):
        session_id = _upload(client)
        r = client.post("/mask/hold", json={"session_id": session_id, "mask_id": "0:nope", "held": True})
        assert r.status_code == 404

    def test_caption_unknown_mask_is_404_and_empty_caption_is_422(self, client):
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])

        r = client.post("/mask/caption", json={"session_id": session_id, "mask_id": "0:nope", "caption": "x"})
        assert r.status_code == 404

        r = client.post("/mask/caption", json={"session_id": session_id, "mask_id": mask_id, "caption": "   "})
        assert r.status_code == 422

    def test_holding_a_record_from_an_earlier_pass_is_422(self, client):
        session_id = _upload(client)
        old_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        client.post("/lama/scrub", json={"session_id": session_id})  # advances to pass 1

        r = client.post("/mask/hold", json={"session_id": session_id, "mask_id": old_id, "held": True})
        assert r.status_code == 422

    def test_delete_session_clears_the_sf_session_zombie(self, client):
        """/mask/hold and /mask/caption call get_or_create_sf_session
        directly — they never call get_session() first — and that method
        checks the sf_sessions cache before anything else. If DELETE
        /session/{id} only cleared `service.sessions`, both endpoints would
        keep succeeding against the deleted session's stale SFSession
        object indefinitely (only a process restart would actually clear
        it). This is the exact zombie this test pins.
        """
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        assert session_id in main.service.sf_sessions

        r = client.delete(f"/session/{session_id}")
        assert r.status_code == 200

        assert session_id not in main.service.sessions
        assert session_id not in main.service.sf_sessions

        r = client.post("/mask/hold", json={"session_id": session_id, "mask_id": mask_id, "held": True})
        assert r.status_code == 404

        r = client.post("/mask/caption", json={"session_id": session_id, "mask_id": mask_id, "caption": "x"})
        assert r.status_code == 404

    def test_delete_unknown_session_is_404(self, client):
        r = client.delete("/session/never-existed")
        assert r.status_code == 404

    def test_reset_clears_selected_mask_id_and_the_deleted_mask_404s(self, client):
        """Confirmed live: after /reset, selected_mask_id kept echoing the
        id of a mask reset had just discarded, in every later /segment/*
        response, until the backend restarted -- because /reset discarded
        the record but never cleared sf_session.last_touched_mask_id, the
        field those responses echo."""
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])

        r = client.post("/reset", json={"session_id": session_id})
        assert r.status_code == 200
        assert r.json().get("selected_mask_id") is None

        # The record itself is gone -- confirms this isn't just the
        # top-level field being blanked while the object underneath survives.
        r = client.post("/mask/hold", json={"session_id": session_id, "mask_id": mask_id, "held": True})
        assert r.status_code == 404

        r = client.post("/mask/caption", json={"session_id": session_id, "mask_id": mask_id, "caption": "x"})
        assert r.status_code == 404

        # A later no-op call (Avoid with nothing to match) must not revive
        # the stale id either -- this is the exact case F/G replay from the
        # investigation that originally surfaced it.
        r = client.post("/segment/point", json={"session_id": session_id, "point": [0.9, 0.05], "label": False})
        assert r.status_code == 200
        assert r.json().get("selected_mask_id") is None


class TestSaveReloadRoundTrip:
    """dataset_status, held, caption, and pass must survive a full
    save_session -> read_session_raw cycle unchanged — a correctness bug,
    not a nice-to-have, per the design doc (§6): a failure here means
    captioned work silently never reaches the training set."""

    def _build_session(self, client) -> tuple[str, str, str]:
        session_id = _upload(client)
        keeper_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        balloon_id = _box_select(client, session_id, [0.7, 0.7, 0.2, 0.2], text_substitute="word balloon")
        _hold(client, session_id, keeper_id, True)
        _caption(client, session_id, keeper_id, "a red egg")
        _hold(client, session_id, balloon_id, True)
        client.post("/lama/scrub", json={"session_id": session_id})  # pass 0 -> 1

        lamp_id = _box_select(client, session_id, [0.4, 0.4, 0.1, 0.1], text_substitute="lamp")
        _caption(client, session_id, lamp_id, "a brass lamp")
        return session_id, keeper_id, lamp_id

    def test_fields_survive_direct_save_and_read(self, client):
        session_id, keeper_id, lamp_id = self._build_session(client)
        r = client.post("/saveSession", json={"session_id": session_id})
        assert r.status_code == 200

        raw = aa_persistence.read_session_raw(session_id)
        assert raw is not None
        assert set(raw["pass_images"].keys()) == {1}  # pass 0 has no scrub image; only pass 1 does

        by_id = {seg["segment_id"]: seg for seg in raw["segments"]}

        keeper = by_id[keeper_id]
        assert keeper["pass"] == "0"
        assert keeper["dataset_status"] == "keep"
        assert keeper["caption"] == "a red egg"
        assert keeper["held"] is False  # cleared by the scrub

        lamp = by_id[lamp_id]
        assert lamp["pass"] == "1"
        assert lamp["dataset_status"] == "keep"
        assert lamp["caption"] == "a brass lamp"

        # The uncaptioned balloon has no caption column at all (None, not "").
        balloon = next(s for s in raw["segments"] if s["text_tag"] == "word balloon")
        assert balloon["dataset_status"] == "unassigned"
        assert balloon["caption"] is None
        assert balloon["held"] is False

    def test_full_reload_reconstructs_sf_session_across_passes(self, client):
        session_id, keeper_id, lamp_id = self._build_session(client)
        client.post("/saveSession", json={"session_id": session_id})

        # Simulate a process restart: evict the in-memory cache entirely.
        del main.service.sessions[session_id]
        del main.service.sf_sessions[session_id]

        assert main.service.get_session(session_id) is not None
        reloaded = main.service.sf_sessions[session_id]

        assert reloaded.pass_ == 1
        assert reloaded.working_copy is not None
        assert reloaded.image_for_pass(0) is not None
        assert reloaded.image_for_pass(1) is not None

        keeper = reloaded.get_mask(keeper_id)
        assert (keeper.pass_, keeper.dataset_status, keeper.caption, keeper.held) == (0, sf_engine.DatasetStatus.KEEP, "a red egg", False)
        lamp = reloaded.get_mask(lamp_id)
        assert (lamp.pass_, lamp.dataset_status, lamp.caption) == (1, sf_engine.DatasetStatus.KEEP, "a brass lamp")

    def test_held_record_survives_a_save_reload_cycle(self, client):
        """A record still IN the pending batch (not yet scrubbed) must
        reload as held — this is the state a user leaves mid-session."""
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        _hold(client, session_id, mask_id, True)
        client.post("/saveSession", json={"session_id": session_id})

        del main.service.sessions[session_id]
        del main.service.sf_sessions[session_id]
        main.service.get_session(session_id)

        assert main.service.sf_sessions[session_id].get_mask(mask_id).held is True


class TestOldSessionMigration:
    """v1 (`MaskType.IN`/`OUT`, two-pass) rows read back with defined v2
    defaults, not undefined/missing behavior."""

    def test_pre_v2_segment_migrates_to_unassigned_unheld_pass_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf_storage")
        session_id = "legacy-session"
        d = aa_persistence.session_dir(session_id)

        # Hand-write registry.parquet + a v1-format segment.parquet: bare
        # uuid4 row key, mask_type column, "foreground" pass — no
        # dataset_status/held/caption columns at all.
        bridge.save_parquet(str(d / "registry.parquet"), [session_id], ["name"], ["Legacy"])
        old_mask_id = "old-mask-uuid"
        row = f"{session_id}:{old_mask_id}"
        tiny_png = _upload_png_bytes()
        bridge.save_parquet(
            str(d / "segment.parquet"),
            [f"{session_id}:_source", row, row, row, row],
            ["image_bytes", "crop_bytes", "mask_bytes", "mask_type", "pass"],
            [aa_persistence._b64(tiny_png), aa_persistence._b64(tiny_png), aa_persistence._b64(tiny_png),
             "in", "foreground"],
        )

        raw = aa_persistence.read_session_raw(session_id)
        assert raw is not None
        assert len(raw["segments"]) == 1
        seg = raw["segments"][0]
        assert seg["segment_id"] == old_mask_id
        assert seg["pass"] == "0"
        assert seg["dataset_status"] == "unassigned"
        assert seg["held"] is False
        assert seg["caption"] is None
        assert raw["pass_images"] == {}

    def test_pre_v2_background_sentinel_migrates_to_pass_one_image(self, tmp_path, monkeypatch):
        monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf_storage")
        session_id = "legacy-session-bg"
        d = aa_persistence.session_dir(session_id)
        bridge.save_parquet(str(d / "registry.parquet"), [session_id], ["name"], ["Legacy"])
        tiny_png = _upload_png_bytes()
        bridge.save_parquet(
            str(d / "segment.parquet"),
            [f"{session_id}:_source", f"{session_id}:_background"],
            ["image_bytes", "background_image_bytes"],
            [aa_persistence._b64(tiny_png), aa_persistence._b64(tiny_png)],
        )

        raw = aa_persistence.read_session_raw(session_id)
        assert set(raw["pass_images"].keys()) == {1}

    def test_reconstructed_session_resumes_at_pass_zero(self, tmp_path, monkeypatch):
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
        assert migrated.mask_id == f"0:{old_mask_id}"
        assert migrated.pass_ == 0
        assert migrated.dataset_status == sf_engine.DatasetStatus.UNASSIGNED
        assert sf_session.pass_ == 0


class TestCompileTrainingSet:
    """Only dataset_status == keep AND captioned masks reach the compiled
    snapshot — the implicit-discard behavior, proven end to end through the
    real HTTP + compile pipeline, not just at the sf_engine unit level."""

    def test_only_captioned_keeper_is_compiled(self, client, tmp_path):
        session_id = _upload(client)
        keeper_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        balloon_id = _box_select(client, session_id, [0.7, 0.7, 0.2, 0.2], text_substitute="word balloon")
        _hold(client, session_id, keeper_id, True)
        _caption(client, session_id, keeper_id, "a red egg")
        _hold(client, session_id, balloon_id, True)  # held, but never captioned

        client.post("/lama/scrub", json={"session_id": session_id})
        client.post("/saveSession", json={"session_id": session_id})

        out_dir = tmp_path / "compiled"
        r = client.post("/dataset/compile", json={"output_dir": str(out_dir)})
        assert r.status_code == 200
        result = r.json()
        assert result["entry_count"] == 1

        lines = (out_dir / "metadata.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        import json as _json
        entry = _json.loads(lines[0])
        assert entry["text"] == "a red egg"
        assert entry["segment_id"] == keeper_id
        assert (out_dir / entry["file_name"]).exists()

    def test_compile_is_idempotent_across_two_sessions(self, client, tmp_path):
        s1 = _upload(client)
        m1 = _box_select(client, s1, [0.2, 0.2, 0.2, 0.2])
        _caption(client, s1, m1, "first image's keeper")
        client.post("/saveSession", json={"session_id": s1})

        out_dir = tmp_path / "compiled"
        r1 = compile_training_set.compile_training_set(str(out_dir))
        assert r1["entry_count"] == 1

        s2 = _upload(client)
        m2 = _box_select(client, s2, [0.3, 0.3, 0.2, 0.2])
        _caption(client, s2, m2, "second image's keeper")
        client.post("/saveSession", json={"session_id": s2})

        r2 = compile_training_set.compile_training_set(str(out_dir))
        assert r2["entry_count"] == 2  # re-run picked up both, not appended onto the first run's file

        lines = (out_dir / "metadata.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
