"""Unit tests for Sam3Processor's box/point prompt injection
(sf-display-and-workflow-v3-spec.md §3 Fix 1).

Isolated from real grounding math on purpose: this only checks WHICH text
concept a box/point call ends up grounded against, not what SAM3 predicts
from it — `_call_grounding` is stubbed out, so no real model/weights are
needed and this stays fast, matching the rest of this project's tests
(main.model/processor are always fakes — see test_sf_wiring.py's module
docstring).

Run with:  pytest backend/tests/test_sam3_image_processor.py -v
"""

from __future__ import annotations

from sam3.model.sam3_image_processor import Sam3Processor


class FakeGeometricPrompt:
    """Stands in for whatever `model._get_dummy_prompt()` returns — this
    fix doesn't touch geometric-prompt handling, so these are no-ops."""

    def __init__(self):
        self.boxes: list = []
        self.points: list = []

    def append_boxes(self, boxes, labels):
        self.boxes.append((boxes, labels))

    def append_points(self, points, labels):
        self.points.append((points, labels))


class FakeBackbone:
    """Records every prompt list `call_text` was invoked with, and returns
    a marker value keyed by the prompt — lets a test see WHICH concept
    ended up in `backbone_out` after a call, without needing real language
    features."""

    def __init__(self):
        self.text_calls: list[list[str]] = []

    def call_text(self, prompts):
        self.text_calls.append(list(prompts))
        return {"language_features": f"features-for:{prompts[0]}"}


class FakeModel:
    def __init__(self):
        self.backbone = FakeBackbone()
        self.inst_interactive_predictor = None

    def _get_dummy_prompt(self, num_prompts=1):
        return FakeGeometricPrompt()


def _processor_with_noop_grounding() -> Sam3Processor:
    """A real Sam3Processor whose `_call_grounding` is stubbed to a no-op
    — isolates the prompt-injection logic this fix changes from the real
    grounding math, which this fix doesn't touch and needs no fake-model
    infrastructure to verify here."""
    processor = Sam3Processor(FakeModel())
    processor._call_grounding = lambda state: state
    return processor


def _state_with_text(processor: Sam3Processor, prompt: str) -> dict:
    state = {"backbone_out": {}}
    processor.set_text_prompt(prompt, state)
    return state


class TestBoxPointAlwaysGroundOnDummyVisual:
    """v3 spec §3 Part 2 finding (b): add_geometric_prompt/add_point_prompt
    used to skip injecting the dummy "visual" text whenever ANY prior text
    prompt — even an unrelated, already-abandoned one — left
    language_features sitting in backbone_out. Confirmed live: this let a
    stale prompt silently condition a later point click's grounding,
    causing point selection to miss the clicked object and (with the old
    _pick_for_point fallback) hijack focus onto an unrelated existing
    record up to 1,500px away. Fix: always (re-)inject "visual" on every
    box/point call, unconditionally overwriting whatever's there — not a
    narrower "only when there's no explicit text_substitute" guard,
    because reconciliation against existing records is purely geometric
    (IoU-based — see sf_engine.py's _best_match), so there's no mechanism
    in this codebase where a box/point deliberately benefits from
    inheriting the active text prompt's conditioning.
    """

    def test_box_call_grounds_on_visual_even_with_no_prior_text(self):
        processor = _processor_with_noop_grounding()
        state = {"backbone_out": {}}

        processor.add_geometric_prompt([0.5, 0.5, 0.1, 0.1], True, state)

        assert processor.model.backbone.text_calls[-1] == ["visual"]
        assert state["backbone_out"]["language_features"] == "features-for:visual"

    def test_point_call_grounds_on_visual_even_with_no_prior_text(self):
        processor = _processor_with_noop_grounding()
        state = {"backbone_out": {}}

        processor.add_point_prompt([0.5, 0.5], True, state)

        assert processor.model.backbone.text_calls[-1] == ["visual"]
        assert state["backbone_out"]["language_features"] == "features-for:visual"

    def test_box_call_overrides_a_stale_unrelated_text_prompt(self):
        """The actual bug: a prior 'rabbit' text selection must not leak
        into a later, unrelated box call."""
        processor = _processor_with_noop_grounding()
        state = _state_with_text(processor, "rabbit")
        assert state["backbone_out"]["language_features"] == "features-for:rabbit"

        processor.add_geometric_prompt([0.2, 0.2, 0.1, 0.1], True, state)

        assert processor.model.backbone.text_calls[-1] == ["visual"], (
            "box/point must ground on the dummy concept, not the leftover "
            "text prompt from an earlier, unrelated selection"
        )
        assert state["backbone_out"]["language_features"] == "features-for:visual"

    def test_point_call_overrides_a_stale_unrelated_text_prompt(self):
        processor = _processor_with_noop_grounding()
        state = _state_with_text(processor, "rabbit")

        processor.add_point_prompt([0.8, 0.8], True, state)

        assert processor.model.backbone.text_calls[-1] == ["visual"]
        assert state["backbone_out"]["language_features"] == "features-for:visual"

    def test_consecutive_geometric_calls_each_reground_on_visual(self):
        """Confirms the injection is unconditional, not merely widened to
        trigger more often — a SECOND geometric call still explicitly
        re-grounds on "visual" rather than reusing the first call's own
        dummy injection because it's already present."""
        processor = _processor_with_noop_grounding()
        state = {"backbone_out": {}}

        processor.add_geometric_prompt([0.2, 0.2, 0.1, 0.1], True, state)
        processor.add_geometric_prompt([0.6, 0.6, 0.1, 0.1], True, state)

        visual_calls = [c for c in processor.model.backbone.text_calls if c == ["visual"]]
        assert len(visual_calls) == 2
