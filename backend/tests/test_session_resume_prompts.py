"""Resume-path coverage for prompt replay in `_load_session_into_memory`.

Regression: replay dispatched on "whichever geometry key is present" rather
than on the prompt's `type`, so a stored point prompt ([x, y], two numbers)
was replayed through `add_geometric_prompt`, whose box path reshapes to
(1, 1, 4). MLX raised "Cannot reshape array of size 2 into shape (1,1,4)",
the whole load failed, and the caller surfaced that as a bare 404
"Session not found" — indistinguishable from a session that genuinely does
not exist.

Confirmed live against stored data: 3 of 7 real sessions contain at least
one point prompt and were unresumable for this reason. Nothing on disk was
malformed — every stored box carried its 4 numbers and every stored point
its 2; only the replay dispatch was wrong.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

import aa_persistence
from services import SegmentationService

SID = "resume-prompt-test-session"


def _png_bytes(size=64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class StrictProcessor:
    """Mimics the real MLX processor's arity behaviour.

    `add_geometric_prompt` raises on anything that is not a 4-element box,
    exactly as the box path's reshape to (1, 1, 4) does — without this the
    regression cannot be reproduced at all, because a permissive double
    accepts the malformed call and the bug stays invisible.
    """

    def __init__(self):
        self.box_calls: list[list[float]] = []
        self.point_calls: list[list[float]] = []
        self.text_calls: list[str] = []

    def set_image(self, image):
        return {"backbone_out": True, "original_width": image.width,
                "original_height": image.height}

    def set_text_prompt(self, prompt, state):
        self.text_calls.append(prompt)
        return state

    def add_geometric_prompt(self, box, label, state):
        if len(box) != 4:
            raise ValueError(
                f"[reshape] Cannot reshape array of size {len(box)} into shape (1,1,4)"
            )
        self.box_calls.append(list(box))
        return state

    def add_point_prompt(self, point, label, state):
        if len(point) != 2:
            raise ValueError(f"point must be [x, y], got {len(point)} values")
        self.point_calls.append(list(point))
        return state


def _raw(prompts):
    return {
        "name": "resume probe",
        "description": "",
        "created_at": "2026-09-29T13:56:35",
        "image_url": None,
        "original_filename": "image.png",
        "width": 64,
        "height": 64,
        "image_bytes": _png_bytes(),
        "pass_images": {},
        "segments": [],
        "prompts": prompts,
        "ui_settings": {},
    }


@pytest.fixture()
def service(tmp_path, monkeypatch):
    processor = StrictProcessor()
    svc = SegmentationService(tmp_path / "sessions", processor)
    monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf_storage")
    return svc


def _load(service, monkeypatch, prompts):
    monkeypatch.setattr(aa_persistence, "read_session_raw", lambda sid: _raw(prompts))
    return service._load_session_into_memory(SID)


# The exact prompt list stored for session dd95c98a, which is what was
# reported failing: a text prompt, a point, a box, and another point.
_REAL_PROMPTS = [
    "herald",
    {"type": "point", "point": [0.12230697384806966, 0.9116311269263699],
     "label": "positive"},
    {"type": "box", "box": [0.1169675454107642, 0.923908390410959,
                            0.1825668696003269, 0.15264474529109578],
     "label": "positive"},
    {"type": "point", "point": [0.636940285574603, 0.11611729452054789],
     "label": "positive"},
]


def test_a_session_with_point_prompts_resumes(service, monkeypatch):
    loaded = _load(service, monkeypatch, _REAL_PROMPTS)

    assert loaded is not None, (
        "a session with point prompts must resume; returning None here is "
        "what the caller turns into a bogus 404"
    )
    assert SID in service.sessions


def test_points_replay_through_the_point_path_not_the_box_path(
    service, monkeypatch
):
    _load(service, monkeypatch, _REAL_PROMPTS)
    processor = service.processor

    assert processor.point_calls == [
        [0.12230697384806966, 0.9116311269263699],
        [0.636940285574603, 0.11611729452054789],
    ]
    assert processor.box_calls == [
        [0.1169675454107642, 0.923908390410959,
         0.1825668696003269, 0.15264474529109578]
    ]
    assert processor.text_calls == ["herald"]


def test_a_box_only_session_is_unaffected(service, monkeypatch):
    prompts = [
        {"type": "box", "box": [0.1, 0.2, 0.3, 0.4], "label": "positive"},
        {"type": "box", "box": [0.5, 0.6, 0.7, 0.8], "label": "negative"},
    ]
    loaded = _load(service, monkeypatch, prompts)

    assert loaded is not None
    assert service.processor.box_calls == [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ]
    assert service.processor.point_calls == []


def test_negative_points_keep_their_polarity(service, monkeypatch):
    seen = []
    prompts = [
        {"type": "point", "point": [0.1, 0.2], "label": "positive"},
        {"type": "point", "point": [0.3, 0.4], "label": "negative"},
    ]
    processor = StrictProcessor()
    real = processor.add_point_prompt

    def spy(point, label, state):
        seen.append(label)
        return real(point, label, state)

    processor.add_point_prompt = spy
    service.processor = processor

    _load(service, monkeypatch, prompts)

    assert seen == [True, False], "a negative point must not become positive"


def test_one_unreplayable_prompt_does_not_cost_the_whole_session(
    service, monkeypatch
):
    """A malformed prompt is skipped, not fatal.

    Masks are restored from persisted columns rather than from replay, so a
    partially reconstructed SAM3 state still beats a 404 that reads to the
    user as "your work is gone".
    """
    prompts = [
        {"type": "box", "box": [0.1, 0.2, 0.3, 0.4], "label": "positive"},
        {"type": "box", "box": [0.9, 0.9], "label": "positive"},  # malformed
        {"type": "point", "point": [0.5, 0.6], "label": "positive"},
    ]

    loaded = _load(service, monkeypatch, prompts)

    assert loaded is not None, "one bad prompt must not fail the load"
    # The good prompts on both sides of the bad one still replayed.
    assert service.processor.box_calls == [[0.1, 0.2, 0.3, 0.4]]
    assert service.processor.point_calls == [[0.5, 0.6]]
