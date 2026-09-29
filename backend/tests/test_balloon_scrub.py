"""Tests for POST /balloon-scrub — the automatic balloon-detect-and-scrub flow.

The detector is mocked throughout: these assert the pipeline's wiring and its
safety properties, not YOLO's accuracy (that was established separately by
zero-shot validation against real Little Nemo pages).

The safety properties matter more than usual here because this flow has no
per-detection human review: nothing is captioned, nothing is saved, and no
session other than the one it creates may be read or written.
"""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import aa_persistence
import balloon_detector
import main
from services import SegmentationService
from tests.test_sf_wiring import FakeInpainter, FakeProcessor, IMG_SIZE


def _png_bytes(size=IMG_SIZE) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (10, 20, 30)).save(buf, format="PNG")
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


def _detects(monkeypatch, boxes):
    """Point the endpoint's detector at a fixed list of pixel-xyxy boxes."""
    dets = [
        {"box": list(b), "confidence": c, "cls": "text_bubble"} for b, c in boxes
    ]
    monkeypatch.setattr(balloon_detector, "detect", lambda image, confidence: dets)
    return dets


def _post(client, size=IMG_SIZE):
    return client.post(
        "/balloon-scrub",
        files={"file": ("page.png", _png_bytes(size), "image/png")},
    )


def test_zero_detections_is_a_clean_no_op(client, monkeypatch):
    _detects(monkeypatch, [])
    original = _png_bytes()

    r = client.post(
        "/balloon-scrub", files={"file": ("page.png", original, "image/png")}
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scrubbed"] is False
    assert body["detections"] == []
    assert body["scrubbed_mask_ids"] == []
    # The image comes back unchanged, not an error and not an inpainted copy.
    returned = Image.open(io.BytesIO(base64.b64decode(body["image_b64"])))
    assert returned.size == Image.open(io.BytesIO(original)).size
    assert list(returned.convert("RGB").getdata()) == list(
        Image.open(io.BytesIO(original)).convert("RGB").getdata()
    )


def test_zero_detections_never_invokes_lama(client, monkeypatch):
    _detects(monkeypatch, [])

    calls = []
    monkeypatch.setattr(
        main, "_get_lama_inpainter", lambda: calls.append(1) or FakeInpainter()
    )

    assert _post(client).status_code == 200
    assert calls == [], "LaMa must not be constructed or run with nothing held"


def test_detections_are_held_and_scrubbed(client, monkeypatch):
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91), ((50, 50, 70, 70), 0.62)])

    r = _post(client)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["scrubbed"] is True
    assert len(body["scrubbed_mask_ids"]) == 2
    assert len(body["detections"]) == 2
    for entry in body["detections"]:
        assert entry["mask_id"] in body["scrubbed_mask_ids"]


def test_nothing_is_ever_captioned(client, monkeypatch):
    """Held-and-scrubbed-without-a-caption is the implicit-discard state a word
    balloon must end in; a caption would write it into the AA as kept."""
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91)])

    captioned = []
    monkeypatch.setattr(
        main.sf_engine.SFSession,
        "attach_caption",
        lambda self, mask_id, caption: captioned.append((mask_id, caption)),
    )

    r = _post(client)
    assert r.status_code == 200
    assert captioned == [], "no caption call may fire in this flow"

    session_id = r.json()["session_id"]
    sf_session = main.service.sf_sessions[session_id]
    for mask in sf_session.masks:
        assert mask.caption in (None, ""), f"{mask.mask_id} was captioned"
        assert mask.dataset_status != "keep", f"{mask.mask_id} was marked keep"


def test_the_session_is_never_saved(client, monkeypatch):
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91)])

    saved = []
    monkeypatch.setattr(
        SegmentationService,
        "save_session_to_disk",
        lambda self, session_id: saved.append(session_id),
    )
    monkeypatch.setattr(
        aa_persistence,
        "save_session",
        lambda *a, **k: saved.append("aa_persistence.save_session"),
    )

    r = _post(client)
    assert r.status_code == 200
    assert saved == [], f"nothing may be persisted, but got {saved}"


def test_no_other_session_is_read_or_touched(client, monkeypatch):
    """The run must be hermetic: pre-existing sessions stay byte-identical and
    are never even read."""
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91)])

    # Two pre-existing sessions, one of them with real in-memory content.
    other_a = main.service.create_session()
    other_b = main.service.create_session()
    main.service.sessions[other_a] = {"sentinel": "do-not-touch"}
    before = {
        other_a: dict(main.service.sessions[other_a]),
        other_b: main.service.sessions.get(other_b),
    }

    read: list[str] = []
    real_get = SegmentationService.get_session

    def spy_get(self, session_id):
        read.append(session_id)
        return real_get(self, session_id)

    monkeypatch.setattr(SegmentationService, "get_session", spy_get)

    r = _post(client)
    assert r.status_code == 200
    new_id = r.json()["session_id"]

    assert [s for s in read if s != new_id] == [], (
        f"only the freshly created session may be read, but saw {read}"
    )
    assert main.service.sessions[other_a] == before[other_a]
    assert main.service.sessions.get(other_b) == before[other_b]
    assert other_a not in main.service.sf_sessions
    assert other_b not in main.service.sf_sessions


def test_each_run_creates_its_own_fresh_session(client, monkeypatch):
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91)])

    first = _post(client).json()["session_id"]
    second = _post(client).json()["session_id"]

    assert first != second
    assert main.service.sf_sessions[first] is not main.service.sf_sessions[second]


def test_audit_list_records_box_and_confidence(client, monkeypatch):
    """With no human reviewing individual detections, this list is the only
    record of what got scrubbed and why."""
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91), ((50, 50, 70, 70), 0.62)])

    body = _post(client).json()
    assert body["confidence_threshold"] == main.BALLOON_SCRUB_CONFIDENCE

    confs = [d["confidence"] for d in body["detections"]]
    boxes = [d["box"] for d in body["detections"]]
    assert confs == [0.91, 0.62]
    assert boxes == [[10, 10, 30, 30], [50, 50, 70, 70]]


def test_a_detection_that_yields_no_mask_is_recorded_not_dropped(
    client, monkeypatch
):
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91)])
    monkeypatch.setattr(
        main.sf_engine.SFSession, "add_box_selection", lambda self, box, label: []
    )

    body = _post(client).json()

    assert body["scrubbed"] is False
    assert len(body["detections"]) == 1
    assert body["detections"][0]["mask_id"] is None
    assert "no SAM3 mask" in body["detections"][0]["note"]


def test_detector_boxes_are_converted_to_normalized_cxcywh(client, monkeypatch):
    """YOLO reports pixel xyxy; add_box_selection takes normalized cxcywh."""
    _detects(monkeypatch, [((10, 20, 30, 40), 0.91)])

    seen = []
    real = main.sf_engine.SFSession.add_box_selection

    def spy(self, box, label, text_substitute=None):
        seen.append(box)
        return real(self, box, label, text_substitute)

    monkeypatch.setattr(main.sf_engine.SFSession, "add_box_selection", spy)

    assert _post(client).status_code == 200
    assert seen == [[20 / IMG_SIZE, 30 / IMG_SIZE, 20 / IMG_SIZE, 20 / IMG_SIZE]]


def test_dedup_collapses_the_two_classes_onto_one_region():
    """Both classes fire on the same balloon; without this the same balloon
    would be selected, held and scrubbed twice."""
    dets = [
        {"box": [10, 10, 30, 30], "confidence": 0.77, "cls": "text_free"},
        {"box": [11, 10, 30, 31], "confidence": 0.49, "cls": "text_bubble"},
        {"box": [50, 50, 70, 70], "confidence": 0.62, "cls": "text_bubble"},
    ]
    kept = balloon_detector.dedup(dets)

    assert len(kept) == 2
    assert kept[0]["confidence"] == 0.77, "the higher-confidence box survives"
    assert kept[1]["box"] == [50, 50, 70, 70]


# ── POST /segment/balloons ───────────────────────────────────────────────────
#
# The card behind this endpoint only detects and holds. Scrubbing stays with
# the existing LaMa Background Scrub card, so these guard against a second
# scrub path appearing and against the disposable-session behaviour of
# /balloon-scrub leaking in.


def _make_session(client, monkeypatch) -> str:
    """An uploaded session, with persistence stubbed out for setup.

    /upload's last act is register_session_data -> save_session_to_disk, which
    writes real parquet through the Julia bridge: minutes per call here, and
    irrelevant to every assertion below. The one test that cares about saving
    installs its own recording spy AFTER this, so this setup write can never
    pollute it.
    """
    monkeypatch.setattr(
        SegmentationService, "save_session_to_disk", lambda self, sid: None
    )
    r = client.post(
        "/upload", files={"file": ("page.png", _png_bytes(), "image/png")}
    )
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def test_balloons_operate_on_the_existing_session(client, monkeypatch):
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91), ((50, 50, 70, 70), 0.62)])
    session_id = _make_session(client, monkeypatch)
    before = set(main.service.sessions)

    r = client.post("/segment/balloons", json={"session_id": session_id})

    assert r.status_code == 200, r.text
    assert r.json()["session_id"] == session_id
    assert set(main.service.sessions) == before, "no new session may be created"


def test_balloons_hold_every_detection_uncaptioned(client, monkeypatch):
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91), ((50, 50, 70, 70), 0.62)])
    session_id = _make_session(client, monkeypatch)

    body = client.post(
        "/segment/balloons", json={"session_id": session_id}
    ).json()
    assert body["detection_count"] == 2
    assert len(body["held_mask_ids"]) == 2

    sf_session = main.service.sf_sessions[session_id]
    current = [m for m in sf_session.masks if m.pass_ == sf_session.pass_]
    assert len(current) == 2
    for mask in current:
        assert mask.held is True
        assert mask.dataset_status != "keep"
        assert mask.caption in (None, "")


def test_balloons_never_scrub(client, monkeypatch):
    """The card detects and holds; LBSCard scrubs. One scrub path only."""
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91)])
    session_id = _make_session(client, monkeypatch)
    sf_session = main.service.sf_sessions[session_id]
    before_pass = sf_session.pass_

    scrubs = []
    monkeypatch.setattr(
        main.sf_engine.SFSession,
        "run_lama_pass",
        lambda self, inpainter: scrubs.append(1),
    )

    assert client.post(
        "/segment/balloons", json={"session_id": session_id}
    ).status_code == 200

    assert scrubs == [], "run_lama_pass must never fire from this endpoint"
    assert sf_session.pass_ == before_pass, "the pass must not advance"


def test_balloons_zero_detections_is_a_clean_no_op(client, monkeypatch):
    _detects(monkeypatch, [])
    session_id = _make_session(client, monkeypatch)
    sf_session = main.service.sf_sessions[session_id]

    body = client.post(
        "/segment/balloons", json={"session_id": session_id}
    ).json()

    assert body["detection_count"] == 0
    assert body["held_mask_ids"] == []
    assert sf_session.masks == []
    assert body["results"] is not None, "a no-op still returns current masks"


def test_balloons_run_against_the_current_pass_image(client, monkeypatch):
    """A second run after a scrub must see what is on screen now, not pass 0."""
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91)])
    session_id = _make_session(client, monkeypatch)
    client.post("/segment/balloons", json={"session_id": session_id})
    client.post("/lama/scrub", json={"session_id": session_id})

    sf_session = main.service.sf_sessions[session_id]
    seen = []
    real = main.sf_engine.SFSession.image_for_pass

    def spy(self, pass_):
        seen.append(pass_)
        return real(self, pass_)

    monkeypatch.setattr(main.sf_engine.SFSession, "image_for_pass", spy)

    client.post("/segment/balloons", json={"session_id": session_id})

    assert sf_session.pass_ in seen, (
        f"detection must read pass {sf_session.pass_}, but read {seen}"
    )


def test_balloons_touch_no_other_session(client, monkeypatch):
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91)])
    session_id = _make_session(client, monkeypatch)
    other = _make_session(client, monkeypatch)
    other_masks_before = list(main.service.sf_sessions[other].masks)

    read: list[str] = []
    real_get = SegmentationService.get_session

    def spy_get(self, sid):
        read.append(sid)
        return real_get(self, sid)

    monkeypatch.setattr(SegmentationService, "get_session", spy_get)

    assert client.post(
        "/segment/balloons", json={"session_id": session_id}
    ).status_code == 200

    assert [s for s in read if s != session_id] == [], (
        f"only the target session may be read, but saw {read}"
    )
    assert main.service.sf_sessions[other].masks == other_masks_before


def test_balloons_do_not_persist_the_session(client, monkeypatch):
    """Persistence is explicit-Save-only, same as every other selection."""
    _detects(monkeypatch, [((10, 10, 30, 30), 0.91)])
    session_id = _make_session(client, monkeypatch)

    saved = []
    monkeypatch.setattr(
        SegmentationService,
        "save_session_to_disk",
        lambda self, sid: saved.append(sid),
    )

    client.post("/segment/balloons", json={"session_id": session_id})

    assert saved == []
