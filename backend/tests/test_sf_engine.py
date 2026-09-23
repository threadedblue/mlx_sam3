"""Unit tests for sf_engine (model v2 — see sf-model-v2-design.md).

Selection paths:
* **Text prompts** -> `_select`: multi-instance, reconciled against the
  pass's existing records by IoU; unmatched records are evicted unless they
  carry user state (keep or held).
* **Box/point prompts** -> `_select_single`: one record for the instance the
  user drew; never evicts.

Held-batch scrubbing: the current pass's held records, whatever their
dataset_status, are unioned and inpainted onto the current pass's image to
produce the next pass — repeatable without limit.

`FakeSam3` is deliberately dumb and fully test-controlled: `text_rects` and
`geo_rects` are the pixel rects it reports as instances.

Run with:  pytest backend/tests/test_sf_engine.py -v
"""

from __future__ import annotations

import base64
import io
import time

import numpy as np
import pytest
from PIL import Image, ImageFilter

import sf_engine as sfe

SHAPE = (10, 10)  # (H, W)
KEEP = sfe.DatasetStatus.KEEP
UNASSIGNED = sfe.DatasetStatus.UNASSIGNED


def _rect_mask(rect: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = rect
    m = np.zeros(SHAPE, dtype=np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


def _geom(rect) -> np.ndarray:
    return _rect_mask(rect).astype(np.uint8)


def _norm_box(rect: tuple[int, int, int, int]) -> list[float]:
    """Pixel rect -> the normalized [cx, cy, w, h] that maps back onto it."""
    y0, y1, x0, x1 = rect
    H, W = SHAPE
    return [((x0 + x1) / 2) / W, ((y0 + y1) / 2) / H, (x1 - x0) / W, (y1 - y0) / H]


def _norm_point(y: int, x: int) -> list[float]:
    H, W = SHAPE
    return [(x + 0.5) / W, (y + 0.5) / H]


class FakeSam3:
    def __init__(self):
        self.text_rects: dict[str, list[tuple]] = {}
        self.geo_rects: list[tuple] = []
        self.geo_scores: list[float] | None = None
        self.images_set: list[Image.Image] = []

    def _state(self, rects, scores=None) -> dict:
        H, W = SHAPE
        return {
            "masks": [_rect_mask(r) for r in rects],
            "boxes": [[float(x0), float(y0), float(x1), float(y1)] for (y0, y1, x0, x1) in rects],
            "scores": list(scores) if scores is not None else [0.9] * len(rects),
            "original_width": W,
            "original_height": H,
        }

    def set_image(self, image) -> dict:
        self.images_set.append(image)
        return self._state([])

    def set_text_prompt(self, prompt: str, state: dict) -> dict:
        return self._state(self.text_rects.get(prompt, []))  # replace semantics

    def add_geometric_prompt(self, box, label: bool, state: dict) -> dict:
        return self._state(self.geo_rects, self.geo_scores)

    def add_point_prompt(self, point, label: bool, state: dict) -> dict:
        return self._state(self.geo_rects, self.geo_scores)


def _tiny_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (SHAPE[1], SHAPE[0]), (0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _session(segmentor: FakeSam3, session_id: str = "sess-1") -> sfe.SFSession:
    return sfe.SFSession(session_id, sfe.ImmutableOriginal(_tiny_png()), segmentor, segmentor.set_image(None))


class NoOpInpainter:
    def inpaint(self, image, mask):
        return image


class PaintingInpainter:
    """Records the union mask it was handed and paints those pixels, so tests
    can see both what fed the union and which image it was applied to."""

    def __init__(self, color=(255, 0, 0, 255)):
        self.color = color
        self.masks: list[np.ndarray] = []

    def inpaint(self, image, mask):
        self.masks.append(mask.copy())
        out = image.copy()
        px = out.load()
        for y, x in zip(*np.nonzero(mask)):
            px[int(x), int(y)] = self.color
        return out


def _select_one(session, seg, name, rect):
    """Text-select exactly one object in the current pass; returns its record.

    Beware calling this twice in a row within one pass: set_text_prompt
    REPLACES, so the second call evicts the first's bare selection (see
    test_reground_that_drops_an_untouched_region_removes_its_record). Tests
    that need several live selections at once use `_select_many`.
    """
    seg.text_rects[name] = [rect]
    recs = session.add_text_selection(name)
    assert len(recs) == 1
    return recs[0]


def _select_many(session, seg, name, rects):
    """Select several objects with ONE multi-instance text prompt, so none of
    them is evicted by the others; returns records in `rects` order."""
    seg.text_rects[name] = list(rects)
    recs = session.add_text_selection(name)
    assert len(recs) == len(rects)
    return recs


def _pixel(image, y, x):
    return image.convert("RGBA").getpixel((x, y))


@pytest.fixture
def passthrough_bridge(monkeypatch):
    """Stand-in for d4m_juliacall_bridge.build_triples that keeps its
    preconditions (so export tests still guard real AA invariants) without
    booting Julia."""

    def build_triples(rows, cols, vals):
        assert len(rows) == len(cols) == len(vals)
        pairs = list(zip(rows, cols))
        assert len(pairs) == len(set(pairs)), "duplicate (row, col) would be merged by D4M.jl"
        assert all(v is not None and v != "" for v in vals), "empty/None value in an AA triple"
        return list(rows), list(cols), list(vals)

    monkeypatch.setattr(sfe.bridge, "build_triples", build_triples)


def _payload_rows(rows, cols, vals) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for r, c, v in zip(rows, cols, vals):
        out.setdefault(r, {})[c] = v
    return out


# ---------------------------------------------------------------------------
# Text path: _select's reconciliation
# ---------------------------------------------------------------------------

class TestTextPromptReconciliation:
    def test_shrinking_result_count_does_not_silently_drop_the_new_selection(self):
        """The original slicing bug: `masks[before:]` on a now-shorter list
        silently yielded [], losing the selection with no error."""
        seg = FakeSam3()
        seg.text_rects = {"cat": [(0, 3, 0, 3), (5, 8, 5, 8)], "dog": [(2, 6, 2, 6)]}
        session = _session(seg)

        assert len(session.add_text_selection("cat")) == 2
        dog = session.add_text_selection("dog")

        assert len(dog) == 1 and dog[0].text_tag == "dog"
        assert [m.text_tag for m in session.masks] == ["dog"]

    def test_growing_result_count_only_returns_the_genuinely_new_ones(self):
        seg = FakeSam3()
        seg.text_rects = {"cat": [(0, 3, 0, 3)], "cats": [(0, 3, 0, 3), (5, 8, 5, 8)]}
        session = _session(seg)

        original_id = session.add_text_selection("cat")[0].mask_id
        second = session.add_text_selection("cats")

        assert len(second) == 1 and second[0].text_tag == "cats"
        assert len(session.masks) == 2
        assert session.get_mask(original_id).text_tag == "cat"

    def test_reground_refines_geometry_without_duplicating_or_corrupting(self):
        seg = FakeSam3()
        seg.text_rects = {"a": [(0, 3, 0, 3)], "b": [(0, 4, 0, 4), (6, 9, 6, 9)]}
        session = _session(seg)

        kept_id = session.add_text_selection("a")[0].mask_id
        second = session.add_text_selection("b")

        assert len(second) == 1 and second[0].text_tag == "b"
        kept = session.get_mask(kept_id)
        assert kept.text_tag == "a"
        np.testing.assert_array_equal(kept.geometry, _geom((0, 4, 0, 4)))

    def test_reground_that_drops_an_untouched_region_removes_its_record(self):
        """Pins CURRENT behaviour on an OPEN design decision.

        A bare selection (unassigned, unheld) from one text prompt is evicted
        by the next text prompt in the same pass, because set_text_prompt
        replaces. So "hold/caption in any order" (design doc §4) is only true
        for records already held or kept — a bare selection must be acted on
        before the next text prompt. That contradicts §4 while matching §7
        ("reconciliation unaffected"). If the decision goes the other way
        (never evict on text replace), this test should flip, and SFSession
        will need an explicit remove/deselect operation, which doesn't exist.
        """
        seg = FakeSam3()
        seg.text_rects = {"a": [(0, 3, 0, 3)], "b": [(6, 9, 6, 9)]}
        session = _session(seg)

        gone_id = session.add_text_selection("a")[0].mask_id
        session.add_text_selection("b")

        with pytest.raises(sfe.UnknownMaskError):
            session.get_mask(gone_id)
        assert [m.text_tag for m in session.masks] == ["b"]

    def test_reground_preserves_caption_and_held_on_a_matched_record(self):
        seg = FakeSam3()
        seg.text_rects = {"a": [(0, 3, 0, 3)], "b": [(0, 4, 0, 4)]}
        session = _session(seg)

        rid = session.add_text_selection("a")[0].mask_id
        session.attach_caption(rid, "a red egg")
        session.set_held(rid, True)
        session.add_text_selection("b")

        rec = session.get_mask(rid)
        assert (rec.dataset_status, rec.caption, rec.held) == (KEEP, "a red egg", True)
        np.testing.assert_array_equal(rec.geometry, _geom((0, 4, 0, 4)))

    def test_keep_record_survives_a_text_reground_that_no_longer_reports_it(self):
        """set_text_prompt REPLACES: a second text prompt reports none of the
        first prompt's objects. Evicting on that would silently delete a
        captioned keeper — the failure the design doc (§6) exists to prevent."""
        seg = FakeSam3()
        seg.text_rects = {"egg": [(0, 3, 0, 3)], "rabbit": [(6, 9, 6, 9)]}
        session = _session(seg)

        egg = session.add_text_selection("egg")[0]
        session.attach_caption(egg.mask_id, "a red egg")
        session.add_text_selection("rabbit")

        kept = session.get_mask(egg.mask_id)
        assert kept.dataset_status == KEEP and kept.caption == "a red egg"
        np.testing.assert_array_equal(kept.geometry, _geom((0, 3, 0, 3)))
        assert len(session.masks) == 2

    def test_held_record_survives_a_text_reground_that_no_longer_reports_it(self):
        """Same hazard for the scrub queue: evicting a held record would
        silently drop it from the batch the user already queued."""
        seg = FakeSam3()
        seg.text_rects = {"balloon": [(0, 3, 0, 3)], "rabbit": [(6, 9, 6, 9)]}
        session = _session(seg)

        balloon = session.add_text_selection("balloon")[0]
        session.set_held(balloon.mask_id, True)
        session.add_text_selection("rabbit")

        assert session.get_mask(balloon.mask_id).held is True


# ---------------------------------------------------------------------------
# Pass isolation across an unbounded chain
# ---------------------------------------------------------------------------

class TestPassIsolation:
    def test_reconciliation_never_touches_a_different_pass(self):
        seg = FakeSam3()
        session = _session(seg)
        pass0 = _select_one(session, seg, "a", (0, 3, 0, 3))

        session.run_lama_pass(NoOpInpainter())
        assert session.pass_ == 1

        # Same geometry in pass 1 must not match against, or evict, pass 0's.
        pass1 = _select_one(session, seg, "lamp", (0, 3, 0, 3))
        assert pass1.pass_ == 1 and pass1.mask_id != pass0.mask_id
        assert {m.mask_id for m in session.masks} == {pass0.mask_id, pass1.mask_id}

    def test_isolation_holds_across_four_passes_with_identical_geometry(self):
        """Adversarial, extended past the old two-pass cap: every pass traces
        a geometrically identical object (IoU 1.0 across passes, the strongest
        false-match trigger). Then a replacing text prompt in the last pass —
        which evicts that pass's unmatched, untouched records — must leave
        every earlier pass's record exactly as it was."""
        seg = FakeSam3()
        session = _session(seg)
        rect = (0, 3, 0, 3)

        earlier = []
        for p in range(3):
            assert session.pass_ == p
            earlier.append(_select_one(session, seg, f"obj{p}", rect))
            session.run_lama_pass(NoOpInpainter())

        assert session.pass_ == 3
        doomed = _select_one(session, seg, "obj3", rect)
        seg.text_rects["elsewhere"] = [(6, 9, 6, 9)]
        session.add_text_selection("elsewhere")

        with pytest.raises(sfe.UnknownMaskError):
            session.get_mask(doomed.mask_id)
        for original in earlier:
            current = session.get_mask(original.mask_id)
            assert current.pass_ == original.pass_
            assert current.text_tag == original.text_tag
            np.testing.assert_array_equal(current.geometry, original.geometry)
        assert sorted(m.pass_ for m in session.masks) == [0, 1, 2, 3]

    def test_isolation_compares_pass_by_value_not_identity(self):
        """Pass scoping was `m.pass_ is self._pass` — correct only against the
        old enum's singletons. On ints it holds for CPython's cached -5..256
        and fails above that, and for any pass rebuilt from storage. Mimics a
        reload at depth: pass values parsed from strings, as Parquet stores
        them, so current pass and record pass are equal but not identical."""
        seg = FakeSam3()
        session = _session(seg)
        rect = (0, 3, 0, 3)

        session._pass = int("300")
        reloaded = sfe.MaskRecord(mask_id="300:reloaded", pass_=int("300"),
                                  geometry=_geom(rect), source="loaded", text_tag="egg")
        other = sfe.MaskRecord(mask_id="299:other", pass_=int("299"),
                               geometry=_geom(rect), source="loaded", text_tag="older")
        session.masks = [other, reloaded]
        assert session.pass_ == reloaded.pass_ and session.pass_ is not reloaded.pass_  # not vacuous

        seg.text_rects["egg"] = [rect]
        new = session.add_text_selection("egg")

        # By value, the reloaded pass-300 record IS this pass's: re-selecting
        # it reconciles rather than minting a duplicate beside it.
        assert new == []
        assert sorted(m.mask_id for m in session.masks) == ["299:other", "300:reloaded"]


# ---------------------------------------------------------------------------
# Box/point path: _select_single's geometric top-1 selection
# ---------------------------------------------------------------------------

DRAWN = (2, 6, 2, 6)
FAR_AWAY = [(0, 2, 8, 10), (8, 10, 0, 2), (0, 1, 0, 1)]


class TestSingleInstanceGeometricSelection:
    def test_box_records_one_selection_despite_exemplar_generalisations(self):
        seg = FakeSam3()
        seg.geo_rects = [DRAWN] + FAR_AWAY
        session = _session(seg)

        recs = session.add_box_selection(_norm_box(DRAWN), True, "widget")

        assert len(recs) == 1
        assert recs[0].text_tag == "widget"
        np.testing.assert_array_equal(recs[0].geometry, _geom(DRAWN))
        assert len(session.masks) == 1

    def test_box_picks_the_overlapping_instance_not_the_highest_scoring_one(self):
        seg = FakeSam3()
        seg.geo_rects = [DRAWN, FAR_AWAY[0]]
        seg.geo_scores = [0.41, 0.99]
        session = _session(seg)

        recs = session.add_box_selection(_norm_box(DRAWN), True)

        assert len(recs) == 1
        np.testing.assert_array_equal(recs[0].geometry, _geom(DRAWN))
        assert recs[0].score == 0.41

    def test_point_picks_the_instance_containing_the_click(self):
        seg = FakeSam3()
        seg.geo_rects = [FAR_AWAY[0], DRAWN]
        seg.geo_scores = [0.99, 0.30]
        session = _session(seg)

        recs = session.add_point_selection(_norm_point(4, 4), True)

        assert len(recs) == 1
        np.testing.assert_array_equal(recs[0].geometry, _geom(DRAWN))
        assert recs[0].source == "point"

    def test_box_selection_never_evicts_earlier_records(self):
        seg = FakeSam3()
        session = _session(seg)
        earlier = _select_one(session, seg, "earlier", (8, 10, 8, 10))

        seg.geo_rects = [DRAWN] + FAR_AWAY
        session.add_box_selection(_norm_box(DRAWN), True)

        assert earlier.mask_id in {m.mask_id for m in session.masks}
        assert len(session.masks) == 2

    def test_reselecting_the_same_object_refreshes_instead_of_duplicating(self):
        seg = FakeSam3()
        seg.geo_rects = [DRAWN]
        session = _session(seg)
        first = session.add_box_selection(_norm_box(DRAWN), True)

        refined = (2, 7, 2, 7)
        seg.geo_rects = [refined]
        again = session.add_box_selection(_norm_box(refined), True)

        assert again == []
        assert [m.mask_id for m in session.masks] == [first[0].mask_id]
        np.testing.assert_array_equal(session.masks[0].geometry, _geom(refined))

    def test_negative_label_records_nothing(self):
        seg = FakeSam3()
        seg.geo_rects = [DRAWN] + FAR_AWAY
        session = _session(seg)

        assert session.add_box_selection(_norm_box(DRAWN), False, "avoid me") == []
        assert session.masks == []

    def test_box_over_empty_background_falls_back_to_top_score(self):
        seg = FakeSam3()
        seg.geo_rects = [FAR_AWAY[0], FAR_AWAY[1]]
        seg.geo_scores = [0.20, 0.80]
        session = _session(seg)

        recs = session.add_box_selection(_norm_box(DRAWN), True)

        assert len(recs) == 1
        np.testing.assert_array_equal(recs[0].geometry, _geom(FAR_AWAY[1]))


# ---------------------------------------------------------------------------
# Point selection: no top-score fallback, no focus hijack
# (sf-display-and-workflow-v3-spec.md §3 Fix 2)
# ---------------------------------------------------------------------------

class TestPointSelectionNoFallbackHijack:
    """Reproduces the investigation's own P1/P2/P3/P5 control study with
    the fake segmentor: _pick_for_point used to fall back to the
    highest-confidence returned instance ANYWHERE when none contained the
    click, and if that instance's geometry happened to IoU-match an
    existing record (as a leaked/mismatched concept's results often do —
    confirmed live, up to 1,500px from the actual click), `_select_single`
    silently reassigned focus to it. `_pick_for_box` is untouched — it
    keeps its own top-score fallback (test_box_over_empty_background_
    falls_back_to_top_score above, unmodified) because a drawn BOX region
    is a much weaker "nothing else to go on" case than a point.
    """

    def test_point_on_the_object_it_was_drawn_for_still_works(self):
        """P1 equivalent: baseline, must stay passing. (Already covered by
        test_point_picks_the_instance_containing_the_click above — this
        just pins the same shape under this class's own name for the
        P1/P2/P3/P5 grouping.)"""
        seg = FakeSam3()
        seg.geo_rects = [FAR_AWAY[0], DRAWN]
        seg.geo_scores = [0.99, 0.30]
        session = _session(seg)

        recs = session.add_point_selection(_norm_point(4, 4), True)

        assert len(recs) == 1
        np.testing.assert_array_equal(recs[0].geometry, _geom(DRAWN))
        assert session.last_touched_mask_id == recs[0].mask_id

    def test_point_with_no_containing_instance_does_not_hijack_an_existing_selection(self):
        """P2/P5 equivalent: a click that hits nothing must not silently
        focus an unrelated EXISTING record just because the (former)
        top-score fallback would have picked an instance that happens to
        geometrically reconcile onto it — exactly the mechanism confirmed
        live with a leaked text prompt, reproduced here without needing
        any text/concept simulation at all."""
        seg = FakeSam3()
        session = _session(seg)
        existing = _select_one(session, seg, "prior-search", FAR_AWAY[0])
        assert session.last_touched_mask_id is None  # text selection never sets focus

        # Neither returned instance contains the click (DRAWN's region) —
        # but the highest-scoring one (index 0) is geometrically the SAME
        # object as `existing`, exactly like a leaked-concept point call
        # returning largely the same candidate set as an earlier search.
        seg.geo_rects = [FAR_AWAY[0], FAR_AWAY[1]]
        seg.geo_scores = [0.95, 0.10]

        recs = session.add_point_selection(_norm_point(4, 4), True)

        assert recs == []
        assert len(session.masks) == 1, "the existing record must not be duplicated or altered"
        assert session.last_touched_mask_id is None, (
            "must not silently focus the pre-existing record just because "
            "the old fallback would have picked an instance that "
            "happens to reconcile onto it"
        )

    def test_point_with_no_containing_instance_records_nothing_on_a_blank_session(self):
        """The simplest case of the same fix: nothing selected yet, click
        hits nothing -- no record, no crash, focus stays None."""
        seg = FakeSam3()
        seg.geo_rects = FAR_AWAY
        seg.geo_scores = [0.99, 0.5, 0.3]
        session = _session(seg)

        recs = session.add_point_selection(_norm_point(4, 4), True)

        assert recs == []
        assert session.masks == []
        assert session.last_touched_mask_id is None

    def test_a_completely_empty_grounding_result_does_not_echo_a_stale_focus_either(self):
        """The second, narrower hijack path the investigation flagged
        explicitly: `_select_single` used to `return []` the instant the
        raw grounding result had ZERO instances at all, before `pick`
        (and therefore before the `idx is None` handling above) ever ran
        — leaving `last_touched_mask_id` exactly as an EARLIER, unrelated
        call left it. A click that finds literally nothing must clear
        focus the same way a click that finds candidates-but-no-match
        does, not echo whatever was focused before."""
        seg = FakeSam3()
        session = _session(seg)
        _select_one(session, seg, "earlier", DRAWN)
        seg.geo_rects = [FAR_AWAY[0]]
        another = session.add_box_selection(_norm_box(FAR_AWAY[0]), True)[0]
        assert session.last_touched_mask_id == another.mask_id  # sanity: focus IS set going in

        seg.geo_rects = []  # SAM3/fake returns NOTHING at all for this click

        recs = session.add_point_selection(_norm_point(1, 1), True)

        assert recs == []
        assert len(session.masks) == 2, "neither existing record touched"
        assert session.last_touched_mask_id is None, (
            "must not keep echoing an EARLIER call's focus as if this "
            "click (which found nothing) had touched it"
        )

    def test_point_on_an_already_selected_object_still_reconciles_via_genuine_containment(self):
        """P3 equivalent: a point clicked ON an object that's already
        selected must still correctly re-focus the existing record via
        genuine containment -- Fix 2 only removes the FALLBACK, not the
        real containment path reconciliation already depends on. The
        unrelated candidate scores HIGHER here, deliberately, to prove
        this is containment-driven, not a score-driven coincidence."""
        seg = FakeSam3()
        session = _session(seg)
        existing = _select_one(session, seg, "already-selected", DRAWN)

        seg.geo_rects = [DRAWN, FAR_AWAY[0]]
        seg.geo_scores = [0.5, 0.99]

        recs = session.add_point_selection(_norm_point(4, 4), True)  # inside DRAWN

        assert recs == [], "no NEW record -- it's the same object"
        assert len(session.masks) == 1
        assert session.last_touched_mask_id == existing.mask_id

    def test_an_avoid_click_that_finds_nothing_still_leaves_focus_untouched(self):
        """Removing the standalone `if not geometries: return []` guard
        must not disturb Avoid's own, separately-documented "focus is
        left wherever it was" behavior (see `_select_single`'s own
        comment) — a negative label with a completely empty grounding
        result still returns via the `if not label` branch, before ever
        reaching the idx/pick logic this fix touches."""
        seg = FakeSam3()
        seg.geo_rects = [DRAWN]
        session = _session(seg)
        existing = session.add_box_selection(_norm_box(DRAWN), True)[0]
        assert session.last_touched_mask_id == existing.mask_id

        seg.geo_rects = []  # Avoid click that finds nothing at all

        recs = session.add_point_selection(_norm_point(1, 1), False)

        assert recs == []
        assert session.last_touched_mask_id == existing.mask_id, (
            "an Avoid call must leave focus exactly where it was, "
            "empty grounding result or not"
        )


class TestAvoidRefinesRecordedGeometry:
    """Fix regression: the live investigation's case D. `_select_single`
    used to return before reconciling on a negative (Avoid) label, so a
    re-ground that refined an already-recorded object's boundary (or
    suppressed unrelated generalisations, as observed live: 17 raw
    instances -> 10, and the targeted figure's own mask 2317px -> 2297px,
    IoU 0.987) left the RECORDED MaskRecord frozen at its pre-refinement
    shape forever. This class pins that the reconciliation the text-prompt
    path already does (match by geometry, refresh in place) also runs for
    box/point calls regardless of label."""

    REFINED = (2, 6, 2, 5)  # DRAWN = (2, 6, 2, 6), shrunk by one column — high IoU, not identical

    def test_avoid_after_target_refreshes_the_recorded_geometry(self):
        seg = FakeSam3()
        seg.geo_rects = [DRAWN]
        session = _session(seg)

        target = session.add_box_selection(_norm_box(DRAWN), True, "figure")
        assert len(target) == 1
        mask_id = target[0].mask_id
        original_geometry = target[0].geometry.copy()

        # Simulate the live scenario: an Avoid prompt re-grounds and comes
        # back with a refined (not identical) boundary for the same figure.
        seg.geo_rects = [self.REFINED]
        avoid_result = session.add_point_selection(_norm_point(4, 3), False, "avoid this part")

        assert avoid_result == []  # still no new selection
        assert len(session.masks) == 1  # not duplicated
        refreshed = session.get_mask(mask_id)
        assert refreshed.mask_id == mask_id  # identity preserved, not a new record
        np.testing.assert_array_equal(refreshed.geometry, _geom(self.REFINED))
        assert not np.array_equal(refreshed.geometry, original_geometry)  # proves it actually changed

    def test_avoid_refresh_preserves_caption_and_held(self):
        seg = FakeSam3()
        seg.geo_rects = [DRAWN]
        session = _session(seg)

        target = session.add_box_selection(_norm_box(DRAWN), True, "figure")[0]
        session.attach_caption(target.mask_id, "a red egg")
        session.set_held(target.mask_id, True)

        seg.geo_rects = [self.REFINED]
        session.add_point_selection(_norm_point(4, 3), False, "avoid this part")

        refreshed = session.get_mask(target.mask_id)
        assert refreshed.dataset_status == sfe.DatasetStatus.KEEP
        assert refreshed.caption == "a red egg"
        assert refreshed.held is True
        np.testing.assert_array_equal(refreshed.geometry, _geom(self.REFINED))

    def test_avoid_with_no_matching_prior_record_still_records_nothing(self):
        """Unrelated Avoid prompts must not be affected by this fix — no
        prior record to refine, so behaviour stays exactly as before."""
        seg = FakeSam3()
        seg.geo_rects = [DRAWN] + FAR_AWAY
        session = _session(seg)

        recs = session.add_box_selection(_norm_box(DRAWN), False, "avoid me")

        assert recs == []
        assert session.masks == []


# ---------------------------------------------------------------------------
# Selection, holding and captioning are independent
# ---------------------------------------------------------------------------

class TestSelectHoldCaptionIndependence:
    def test_new_selection_is_unassigned_unheld_and_uncaptioned(self):
        seg = FakeSam3()
        session = _session(seg)
        rec = _select_one(session, seg, "egg", DRAWN)
        assert (rec.dataset_status, rec.held, rec.caption) == (UNASSIGNED, False, None)

    def test_box_selection_needs_no_text_substitute(self):
        seg = FakeSam3()
        seg.geo_rects = [DRAWN]
        session = _session(seg)

        rec = session.add_box_selection(_norm_box(DRAWN), True)[0]

        assert rec.text_tag is None and rec.caption is None

    def test_holding_does_not_caption_and_captioning_does_not_hold(self):
        seg = FakeSam3()
        session = _session(seg)
        a, b = _select_many(session, seg, "eggs", [(0, 3, 0, 3), (6, 9, 6, 9)])

        held = session.set_held(a.mask_id, True)
        captioned = session.attach_caption(b.mask_id, "a blue egg")

        assert (held.held, held.dataset_status, held.caption) == (True, UNASSIGNED, None)
        assert (captioned.held, captioned.dataset_status, captioned.caption) == (False, KEEP, "a blue egg")

    def test_attach_caption_to_unknown_mask_raises(self):
        session = _session(FakeSam3())
        with pytest.raises(sfe.UnknownMaskError, match="no-such-mask"):
            session.attach_caption("no-such-mask", "anything")

    def test_set_held_on_unknown_mask_raises(self):
        session = _session(FakeSam3())
        with pytest.raises(sfe.UnknownMaskError, match="no-such-mask"):
            session.set_held("no-such-mask", True)

    @pytest.mark.parametrize("caption", ["", "   "])
    def test_empty_caption_is_refused(self, caption):
        seg = FakeSam3()
        session = _session(seg)
        rec = _select_one(session, seg, "egg", DRAWN)

        with pytest.raises(ValueError, match="non-empty"):
            session.attach_caption(rec.mask_id, caption)
        assert session.get_mask(rec.mask_id).dataset_status == UNASSIGNED

    def test_cannot_hold_a_record_from_an_earlier_pass_but_can_unhold_it(self):
        seg = FakeSam3()
        session = _session(seg)
        old = _select_one(session, seg, "old", DRAWN)
        session.run_lama_pass(NoOpInpainter())

        with pytest.raises(ValueError, match="only pass 1"):
            session.set_held(old.mask_id, True)
        assert session.set_held(old.mask_id, False).held is False


# ---------------------------------------------------------------------------
# Reset: discarding SAM3's ungrounded selections without losing committed work
# ---------------------------------------------------------------------------

class TestDiscardUngroundedSelections:
    """/reset (main.py) clears SAM3's own grounding/prompt state and calls
    this to keep sf_session.masks in sync -- an ungrounded selection has
    nothing else backing it and must go, but a captioned/held selection is
    committed user work that resetting the live grounding must not erase."""

    def test_an_uncaptioned_unheld_selection_is_discarded(self):
        seg = FakeSam3()
        session = _session(seg)
        rec = _select_one(session, seg, "egg", DRAWN)

        session.discard_ungrounded_selections()

        assert session.masks == []
        with pytest.raises(sfe.UnknownMaskError):
            session.get_mask(rec.mask_id)

    def test_a_captioned_selection_survives_geometry_and_status_intact(self):
        seg = FakeSam3()
        session = _session(seg)
        rec = _select_one(session, seg, "egg", DRAWN)
        session.attach_caption(rec.mask_id, "a red egg")

        session.discard_ungrounded_selections()

        survivor = session.get_mask(rec.mask_id)
        assert survivor.dataset_status == KEEP
        assert survivor.caption == "a red egg"
        np.testing.assert_array_equal(survivor.geometry, rec.geometry)

    def test_a_held_but_uncaptioned_selection_also_survives(self):
        seg = FakeSam3()
        session = _session(seg)
        rec = _select_one(session, seg, "egg", DRAWN)
        session.set_held(rec.mask_id, True)

        session.discard_ungrounded_selections()

        assert session.get_mask(rec.mask_id).held is True

    def test_only_the_current_pass_is_affected(self):
        seg = FakeSam3()
        session = _session(seg)
        earlier = _select_one(session, seg, "old", DRAWN)  # uncaptioned, unheld
        session.run_lama_pass(NoOpInpainter())
        current = _select_one(session, seg, "new", DRAWN)  # also uncaptioned, unheld

        session.discard_ungrounded_selections()

        # The earlier pass's record survives even though it carries no user
        # state -- discard_ungrounded_selections only scopes to self._pass,
        # matching _select's own eviction scope. The current pass's
        # equally-ungrounded record is still discarded as normal.
        assert session.get_mask(earlier.mask_id).pass_ == 0
        with pytest.raises(sfe.UnknownMaskError):
            session.get_mask(current.mask_id)


# ---------------------------------------------------------------------------
# Held-batch scrubbing over an unbounded pass chain
# ---------------------------------------------------------------------------

class TestHeldBatchScrubbing:
    def test_scrub_is_repeatable_without_limit(self):
        seg = FakeSam3()
        session = _session(seg)
        for expected in range(1, 6):
            session.run_lama_pass(NoOpInpainter())
            assert session.pass_ == expected
        assert session.last_scrub == sfe.ScrubRecord(4, 5, ())

    def test_union_is_the_current_pass_held_set_whatever_its_dataset_status(self):
        seg = FakeSam3()
        session = _session(seg)
        keeper, balloon, unheld = _select_many(
            session, seg, "things", [(0, 3, 0, 3), (6, 9, 6, 9), (0, 2, 7, 9)])
        session.attach_caption(keeper.mask_id, "a red egg")
        session.set_held(keeper.mask_id, True)
        session.set_held(balloon.mask_id, True)

        inpainter = PaintingInpainter()
        session.run_lama_pass(inpainter)

        assert len(inpainter.masks) == 1
        expected = np.logical_or(_geom((0, 3, 0, 3)), _geom((6, 9, 6, 9))).astype(np.uint8)
        np.testing.assert_array_equal(inpainter.masks[0], expected)
        assert not inpainter.masks[0][_geom((0, 2, 7, 9)).astype(bool)].any()
        assert unheld.mask_id not in session.last_scrub.mask_ids

    def test_scrub_clears_held_only_on_the_batch_and_records_the_transition(self):
        seg = FakeSam3()
        session = _session(seg)
        keeper, other = _select_many(session, seg, "things", [(0, 3, 0, 3), (6, 9, 6, 9)])
        session.attach_caption(keeper.mask_id, "a red egg")
        session.set_held(keeper.mask_id, True)

        session.run_lama_pass(NoOpInpainter())

        after = session.get_mask(keeper.mask_id)
        assert (after.held, after.dataset_status, after.caption, after.pass_) == (False, KEEP, "a red egg", 0)
        assert session.get_mask(other.mask_id).held is False
        assert session.last_scrub == sfe.ScrubRecord(0, 1, (keeper.mask_id,))

    def test_each_scrub_builds_on_the_previous_pass_not_the_original(self):
        """Peels must accumulate. Starting every scrub from the original image
        (what the two-pass code did) would silently undo each earlier peel."""
        seg = FakeSam3()
        session = _session(seg)
        a = _select_one(session, seg, "a", (0, 3, 0, 3))
        session.set_held(a.mask_id, True)
        session.run_lama_pass(PaintingInpainter(color=(255, 0, 0, 255)))

        b = _select_one(session, seg, "b", (6, 9, 6, 9))
        session.set_held(b.mask_id, True)
        session.run_lama_pass(PaintingInpainter(color=(0, 0, 255, 255)))

        latest = session.image_for_pass(2)
        assert _pixel(latest, 1, 1) == (255, 0, 0, 255)   # pass 0 -> 1 peel survived
        assert _pixel(latest, 7, 7) == (0, 0, 255, 255)   # pass 1 -> 2 peel applied
        assert seg.images_set[-1] is not None             # SAM3 re-pointed at the new pass

    def test_later_scrubs_do_not_mutate_earlier_pass_images(self):
        seg = FakeSam3()
        session = _session(seg)
        a = _select_one(session, seg, "a", (0, 3, 0, 3))
        session.set_held(a.mask_id, True)
        session.run_lama_pass(PaintingInpainter(color=(255, 0, 0, 255)))
        b = _select_one(session, seg, "b", (6, 9, 6, 9))
        session.set_held(b.mask_id, True)
        session.run_lama_pass(PaintingInpainter(color=(0, 0, 255, 255)))

        assert _pixel(session.image_for_pass(0), 1, 1) == (0, 0, 0, 255)
        assert _pixel(session.image_for_pass(1), 1, 1) == (255, 0, 0, 255)
        assert _pixel(session.image_for_pass(1), 7, 7) == (0, 0, 0, 255)

    def test_failed_inpaint_leaves_the_session_unchanged(self):
        class Exploding:
            def inpaint(self, image, mask):
                raise RuntimeError("LaMa fell over")

        seg = FakeSam3()
        session = _session(seg)
        a = _select_one(session, seg, "a", (0, 3, 0, 3))
        session.set_held(a.mask_id, True)

        with pytest.raises(RuntimeError, match="LaMa fell over"):
            session.run_lama_pass(Exploding())

        assert session.pass_ == 0
        assert session.get_mask(a.mask_id).held is True
        assert session.last_scrub is None
        with pytest.raises(ValueError):
            session.image_for_pass(1)


# ---------------------------------------------------------------------------
# LaMa's device: MPS, not SimpleLama's own cuda-or-cpu default
# ---------------------------------------------------------------------------

class TestLamaInpainterDeviceSelection:
    """Fix regression: SimpleLama's own default only checks
    torch.cuda.is_available() -- no MPS branch at all -- so on this
    Apple Silicon machine it silently ran on CPU despite MPS being
    available. Measured on the same page/mask TestLamaMaskDilation and
    TestLamaPaddingCroppedBeforeNextPass use: 28.92s per scrub on CPU vs
    1.76s on MPS (16.5x), output numerically equivalent (max channel diff
    1/255 across 4.1M pixels). LamaInpainter.__init__ must pass an
    explicit device rather than rely on that default.

    Mocks SimpleLama's constructor rather than loading the real
    multi-hundred-MB torchscript checkpoint -- this only needs to prove
    LamaInpainter asks for the right device; the real inpaint() code path
    (dilation, crop-back) is already covered by the other Lama test
    classes via LamaInpainter.__new__, and end-to-end correctness on MPS
    was verified live in the investigation, not re-proven here.
    """

    def _capture_simplelama_device(self, monkeypatch, device_str: str) -> dict:
        import lora_trainer
        import simple_lama_inpainting

        monkeypatch.setattr(lora_trainer, "_device", lambda: device_str)
        seen: dict = {}

        class FakeSimpleLama:
            def __init__(self, device):
                seen["device"] = device

        monkeypatch.setattr(simple_lama_inpainting, "SimpleLama", FakeSimpleLama)
        return seen

    def test_constructs_simplelama_with_mps_when_lora_trainer_selects_it(self, monkeypatch):
        import torch

        seen = self._capture_simplelama_device(monkeypatch, "mps")

        sfe.LamaInpainter()

        assert seen["device"] == torch.device("mps")

    def test_follows_lora_trainers_cuda_and_cpu_selection_too(self, monkeypatch):
        """Not just MPS -- LamaInpainter defers to _device()'s full
        cuda->mps->cpu precedence, whatever it resolves to, rather than
        hardcoding "mps" itself."""
        import torch

        for device_str in ("cuda", "cpu"):
            seen = self._capture_simplelama_device(monkeypatch, device_str)
            sfe.LamaInpainter()
            assert seen["device"] == torch.device(device_str)


# ---------------------------------------------------------------------------
# LaMa's own padding: cropped back before it becomes a pass image
# ---------------------------------------------------------------------------

class TestLamaPaddingCroppedBeforeNextPass:
    """Fix regression: simple_lama_inpainting pads its input to a multiple
    of 8 before running the model (prepare_img_and_mask/pad_img_to_modulo,
    padding added only at the bottom/right) and returns that padded output
    as-is -- confirmed live on a real scrub: a 1791x2298 source came back
    1792x2304. Left uncropped, that becomes the next pass's image via
    set_image, silently shifting every later pass's mask coordinates
    relative to the true image -- exactly what a recursive peel
    (select-in-revealed-background -> hold -> scrub again) depends on being
    correct.

    `LamaInpainter.__init__` unconditionally constructs a real `SimpleLama`
    (downloads/loads a torchscript checkpoint), so these tests bypass it
    with `__new__` and swap in a fake `_lama` that reproduces the padding
    behaviour without the real model -- exercising the actual `inpaint()`
    crop this fix added, not a reimplementation of it.
    """

    ORIGINAL_SIZE = (1791, 2298)  # (width, height) -- same shape as the live case

    def _padding_lama_inpainter(self) -> sfe.LamaInpainter:
        inpainter = sfe.LamaInpainter.__new__(sfe.LamaInpainter)

        def fake_lama(image, mask_image):
            def ceil_modulo(x, mod):
                return x if x % mod == 0 else (x // mod + 1) * mod
            w, h = image.size
            padded = Image.new("RGB", (ceil_modulo(w, 8), ceil_modulo(h, 8)), (0, 0, 0))
            padded.paste(image, (0, 0))
            return padded

        inpainter._lama = fake_lama
        return inpainter

    def test_inpaint_crops_the_padded_result_back_to_the_input_size(self):
        inpainter = self._padding_lama_inpainter()
        image = Image.new("RGB", self.ORIGINAL_SIZE, (10, 20, 30))
        mask = np.zeros((self.ORIGINAL_SIZE[1], self.ORIGINAL_SIZE[0]), dtype=bool)
        mask[100:200, 100:200] = True

        result = inpainter.inpaint(image, mask)

        assert result.size == self.ORIGINAL_SIZE  # not (1792, 2304)
        assert result.mode == "RGBA"

    def test_scrub_produces_a_correctly_sized_next_pass_image(self):
        """Through the real SFSession.run_lama_pass integration path, not
        just the isolated crop -- this is the regression case the padding
        bug would otherwise produce: a subsequent selection's pixel math
        (in a real Sam3Processor) is computed against whatever size
        `set_image` actually received, so that size has to be right too,
        not just what `image_for_pass` later reports."""
        seg = FakeSam3()
        buf = io.BytesIO()
        Image.new("RGB", self.ORIGINAL_SIZE, (5, 5, 5)).save(buf, format="PNG")
        session = sfe.SFSession("sess-pad", sfe.ImmutableOriginal(buf.getvalue()), seg, seg.set_image(None))
        assert session.image_for_pass(0).size == self.ORIGINAL_SIZE

        a = _select_one(session, seg, "a", (0, 3, 0, 3))
        session.set_held(a.mask_id, True)

        session.run_lama_pass(self._padding_lama_inpainter())

        assert session.image_for_pass(session.pass_).size == self.ORIGINAL_SIZE  # not padded
        # What the (fake, but otherwise faithful) segmentor's set_image
        # actually received -- a real Sam3Processor derives
        # state["original_width"/"height"] from exactly this, which every
        # later selection's pixel math is computed against.
        assert seg.images_set[-1].size == self.ORIGINAL_SIZE


class TestLamaMaskDilation:
    """Regression: a scrubbed object left a crisp, correctly-shaped ghost
    behind. Neither the frontend's segment list nor the engine's per-pass
    records were at fault (both clear correctly) — the outline was baked
    into LaMa's own output, because SAM3's mask stops at the object's
    interior and its ink contour sits just OUTSIDE the mask, so inpainting
    never touched it. The mask has to be grown before it reaches the model.
    """

    def _recording_inpainter(self):
        """Real LamaInpainter with the checkpoint-loading __init__ skipped,
        so the actual inpaint() path runs while `_lama` just records the
        mask it was handed."""
        inpainter = sfe.LamaInpainter.__new__(sfe.LamaInpainter)
        seen: dict = {}

        def fake_lama(image, mask_image):
            seen["mask"] = np.array(mask_image)
            return image

        inpainter._lama = fake_lama
        return inpainter, seen

    def test_mask_reaching_lama_is_grown_past_the_object_edge(self):
        inpainter, seen = self._recording_inpainter()
        mask = np.zeros((60, 60), dtype=bool)
        mask[20:40, 20:40] = True

        inpainter.inpaint(Image.new("RGB", (60, 60), (255, 255, 255)), mask)

        grown = seen["mask"] > 0
        r = sfe._MASK_DILATION_PX
        assert grown[mask].all(), "dilation must never drop any requested pixel"
        assert int(grown.sum()) > int(mask.sum())
        # Grown by exactly r in each direction — enough to clear the object's
        # own contour, and no further (an over-grown mask eats unselected art).
        assert grown[20 - r, 30] and grown[39 + r, 30]
        assert grown[30, 20 - r] and grown[30, 39 + r]
        assert not grown[20 - r - 1, 30]
        assert not grown[30, 20 - r - 1]

    def test_an_empty_mask_stays_empty(self):
        """Dilating nothing must not invent a region to inpaint — an empty
        held batch never reaches here (run_lama_pass skips the call), but a
        mask that unions to nothing must not become a scrub of the page."""
        inpainter, seen = self._recording_inpainter()

        inpainter.inpaint(Image.new("RGB", (30, 30), (0, 0, 0)), np.zeros((30, 30), dtype=bool))

        assert not (seen["mask"] > 0).any()


class TestMaskDilationIsBoundingBoxScoped:
    """Performance fix: the dilation above originally ran
    ImageFilter.MaxFilter over the WHOLE page regardless of mask size —
    fine at ~0.6s next to a ~29s CPU scrub, not fine next to a ~1.76s
    MPS-accelerated one (roughly a third of it). `_dilate_mask` scopes the
    filter to the mask's bounding box (+ the dilation radius as margin)
    instead. This is a pure performance change — every assertion here is
    about the two approaches producing IDENTICAL output, not new
    behaviour, mirroring the same real mask (a 1791x2298 page, ~19.5k px
    egg-shaped selection) the live investigation measured: full-page took
    576.8ms there, bbox-scoped took 5.9ms — 98x, both bit-for-bit equal.
    """

    @staticmethod
    def _full_page_dilate(mask_image: Image.Image, radius: int) -> Image.Image:
        """The ORIGINAL implementation, kept only as this test's oracle."""
        for _ in range(radius):
            mask_image = mask_image.filter(ImageFilter.MaxFilter(3))
        return mask_image

    def _realistic_page_mask(self) -> Image.Image:
        """Same scale and rough shape as the real mask this was measured
        against (an ellipse standing in for the egg SAM3 selected) — built
        synthetically so this test doesn't depend on a real image fixture,
        but at a page/mask size where bbox-scoping actually matters."""
        width, height = 1791, 2298
        mask = np.zeros((height, width), dtype=np.uint8)
        yy, xx = np.ogrid[:height, :width]
        cx, cy, rx, ry = 717, 1135, 95, 67  # centre/radii approximating the real egg
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
        mask[ellipse] = 255
        return Image.fromarray(mask, mode="L")

    def test_identical_to_full_page_dilation_on_a_realistic_mask(self):
        mask_image = self._realistic_page_mask()

        full = np.array(self._full_page_dilate(mask_image.copy(), sfe._MASK_DILATION_PX))
        bboxed = np.array(sfe._dilate_mask(mask_image.copy(), sfe._MASK_DILATION_PX))

        np.testing.assert_array_equal(bboxed, full)

    def test_identical_on_an_empty_mask(self):
        empty = Image.new("L", (400, 300), 0)

        full = np.array(self._full_page_dilate(empty.copy(), sfe._MASK_DILATION_PX))
        bboxed = np.array(sfe._dilate_mask(empty.copy(), sfe._MASK_DILATION_PX))

        np.testing.assert_array_equal(bboxed, full)
        assert not (bboxed > 0).any()

    def test_identical_on_a_mask_touching_the_image_edges(self):
        """The crop clamps to the image bounds when the object is near an
        edge — has to reproduce whatever edge behaviour full-page filtering
        hits there too, not just the interior case."""
        arr = np.zeros((200, 200), dtype=np.uint8)
        arr[0:20, 0:20] = 255       # touches the top-left corner
        arr[190:200, 190:200] = 255  # touches the bottom-right corner
        edge_mask = Image.fromarray(arr, mode="L")

        full = np.array(self._full_page_dilate(edge_mask.copy(), sfe._MASK_DILATION_PX))
        bboxed = np.array(sfe._dilate_mask(edge_mask.copy(), sfe._MASK_DILATION_PX))

        np.testing.assert_array_equal(bboxed, full)

    def test_meaningfully_faster_on_a_small_mask_over_a_large_page(self):
        """Not a hard perf gate (CI timing is noisy) — a wide, deliberately
        conservative margin (>=5x) against a measured 98x, just enough to
        catch someone accidentally reintroducing full-page filtering."""
        mask_image = self._realistic_page_mask()

        t0 = time.perf_counter()
        self._full_page_dilate(mask_image.copy(), sfe._MASK_DILATION_PX)
        full_page_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        sfe._dilate_mask(mask_image.copy(), sfe._MASK_DILATION_PX)
        bbox_scoped_seconds = time.perf_counter() - t0

        print(f"\n    full-page dilate:  {full_page_seconds*1000:7.1f} ms")
        print(f"    bbox-scoped dilate: {bbox_scoped_seconds*1000:7.1f} ms"
              f"  ({full_page_seconds/bbox_scoped_seconds:.1f}x)")
        assert bbox_scoped_seconds * 5 < full_page_seconds


# ---------------------------------------------------------------------------
# Export: keep-only, implicit discard
# ---------------------------------------------------------------------------

class TestExport:
    def test_held_and_scrubbed_but_never_captioned_is_not_exported(self, passthrough_bridge):
        """Implicit discard, proven rather than assumed: select with no
        caption -> hold -> scrub. Nothing marks it discarded; it must still
        be absent from the export, while a captioned keeper is present."""
        seg = FakeSam3()
        session = _session(seg)
        balloon, keeper = _select_many(session, seg, "things", [(6, 9, 6, 9), (0, 3, 0, 3)])
        session.set_held(balloon.mask_id, True)
        session.attach_caption(keeper.mask_id, "a red egg")

        session.run_lama_pass(NoOpInpainter())
        assert session.get_mask(balloon.mask_id).dataset_status == UNASSIGNED
        assert balloon.mask_id in session.last_scrub.mask_ids

        rows = _payload_rows(*sfe.build_sf_payload(session))

        assert f"sess-1:{balloon.mask_id}" not in rows
        keeper_row = rows[f"sess-1:{keeper.mask_id}"]
        assert keeper_row["caption"] == "a red egg"
        assert keeper_row["pass"] == "0"
        assert keeper_row["text_tag"] == "things"

    def test_keep_record_can_be_held_and_scrubbed_again_in_a_later_pass(self, passthrough_bridge):
        """Design doc §1.5/§3: something revealed and captioned in a later
        pass can itself be held and scrubbed to reveal what's behind it.
        Captioning must not make it scrub-ineligible, and the scrub must not
        cost it its caption or its place in the export."""
        seg = FakeSam3()
        session = _session(seg)
        front = _select_one(session, seg, "front", (0, 5, 0, 5))
        session.set_held(front.mask_id, True)
        session.run_lama_pass(NoOpInpainter())

        revealed = _select_one(session, seg, "revealed", (1, 4, 1, 4))
        assert revealed.pass_ == 1
        session.attach_caption(revealed.mask_id, "a lamp behind the balloon")
        session.set_held(revealed.mask_id, True)

        inpainter = PaintingInpainter()
        session.run_lama_pass(inpainter)

        np.testing.assert_array_equal(inpainter.masks[0], _geom((1, 4, 1, 4)))
        assert session.last_scrub == sfe.ScrubRecord(1, 2, (revealed.mask_id,))
        after = session.get_mask(revealed.mask_id)
        assert (after.dataset_status, after.caption, after.held) == (KEEP, "a lamp behind the balloon", False)

        rows = _payload_rows(*sfe.build_sf_payload(session))
        assert rows[f"sess-1:{revealed.mask_id}"]["caption"] == "a lamp behind the balloon"
        assert rows[f"sess-1:{revealed.mask_id}"]["pass"] == "1"

    def test_background_is_omitted_at_pass_zero_and_present_after_a_scrub(self, passthrough_bridge):
        seg = FakeSam3()
        session = _session(seg)

        rows = _payload_rows(*sfe.build_sf_payload(session))
        assert set(rows["sess-1:context"]) == {"original_image_bytes"}

        session.run_lama_pass(NoOpInpainter())
        rows = _payload_rows(*sfe.build_sf_payload(session))
        assert set(rows["sess-1:context"]) == {"original_image_bytes", "background_image_bytes"}

    def test_text_tag_is_omitted_when_a_box_had_no_text_substitute(self, passthrough_bridge):
        seg = FakeSam3()
        seg.geo_rects = [DRAWN]
        session = _session(seg)
        rec = session.add_box_selection(_norm_box(DRAWN), True)[0]
        session.attach_caption(rec.mask_id, "a widget")

        rows = _payload_rows(*sfe.build_sf_payload(session))

        assert set(rows[f"sess-1:{rec.mask_id}"]) == {"pass", "caption", "crop_png_bytes"}


class TestExportCropBytesNotMaskGeometry:
    """Regression: build_sf_payload used to export mask_png_bytes — a
    binary black/white silhouette of the mask geometry — instead of the
    segmented object itself. Real training/handoff data needs the object
    as it actually appears, cut out via the mask as an alpha channel,
    matching aa_persistence.py's own _crop_png_bytes (this module's
    _crop_png_bytes is a faithful port, not a shared import — see its own
    docstring for why the import can't go the other way).

    Assertions here check actual pixel VALUES against the source image, not
    just "the bytes changed" or "the key is present" — a geometry export
    also produces valid, differently-shaped PNG bytes under any key name,
    so only checking for non-mask-like content can catch a regression back
    to exporting geometry.
    """

    @staticmethod
    def _patterned_png(width: int, height: int) -> bytes:
        """A non-uniform image — each pixel's colour is a function of its
        own (x, y) — so a correctly-cropped region's pixel values can be
        checked against this image's ACTUAL content at those coordinates.
        A flat-colour test image couldn't distinguish "real pixels" from
        "a coincidentally-matching flat fill"."""
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                arr[y, x] = (x * 20 % 256, y * 25 % 256, (x + y) * 15 % 256)
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
        return buf.getvalue()

    def test_exported_crop_shows_the_objects_real_pixels(self, passthrough_bridge):
        seg = FakeSam3()
        seg.geo_rects = [DRAWN]
        original_bytes = self._patterned_png(SHAPE[1], SHAPE[0])
        session = sfe.SFSession(
            "sess-crop", sfe.ImmutableOriginal(original_bytes), seg, seg.set_image(None))
        rec = session.add_box_selection(_norm_box(DRAWN), True)[0]
        session.attach_caption(rec.mask_id, "a widget")

        rows = _payload_rows(*sfe.build_sf_payload(session))
        row = rows[f"sess-crop:{rec.mask_id}"]

        assert "crop_png_bytes" in row
        assert "mask_png_bytes" not in row  # old key must be gone, not kept alongside the new one

        crop = Image.open(io.BytesIO(base64.b64decode(row["crop_png_bytes"])))
        assert crop.mode == "RGBA"

        original = Image.open(io.BytesIO(original_bytes)).convert("RGBA")
        mask = rec.geometry.astype(bool)
        ys, xs = np.nonzero(mask)
        assert len(xs) > 0

        # Inside the mask: opaque, and colour matches the ORIGINAL's actual
        # pixels at that coordinate — not a flat fill a geometry export
        # (or a wrong-but-plausible constant colour) would produce.
        for y, x in zip(ys, xs):
            assert crop.getpixel((int(x), int(y))) == (*original.getpixel((int(x), int(y)))[:3], 255)

        # The masked region is genuinely non-constant (the pattern varies
        # over it) — a flat binary mask fill, or any single wrong colour,
        # could never produce this.
        masked_colors = {crop.getpixel((int(x), int(y)))[:3] for y, x in zip(ys, xs)}
        assert len(masked_colors) > 1

        # Outside the mask: fully transparent.
        ys_out, xs_out = np.nonzero(~mask)
        oy, ox = int(ys_out[0]), int(xs_out[0])
        assert crop.getpixel((ox, oy))[3] == 0

    def test_a_mask_kept_in_a_later_pass_is_cropped_from_that_passs_image(self, passthrough_bridge):
        """Not pass 0's original — a scrub between pass 0 and the mask's
        own pass can repaint pixels; a wrong-pass crop would silently
        export stale content. FakeSam3's masks aren't real SAM3 geometry
        (background_image_bytes-style pass images stand in fine here), so
        this checks the crop against session.image_for_pass(1) directly
        rather than needing meaningful patterned art in the scrubbed
        region."""
        seg = FakeSam3()
        session = _session(seg)
        front = _select_one(session, seg, "front", (0, 5, 0, 5))
        session.set_held(front.mask_id, True)
        # Paints pass 1's image red exactly where `front` was — pass 0
        # stays the flat black _tiny_png() original everywhere.
        session.run_lama_pass(PaintingInpainter(color=(255, 0, 0, 255)))

        # "revealed" sits at the same location `front` occupied — exactly
        # the recursive-peel case (design doc §1.5/§3): captioned in the
        # pass that now shows the scrub's repainted pixels there, not
        # pass 0's original ones.
        revealed = _select_one(session, seg, "revealed", (1, 4, 1, 4))
        assert revealed.pass_ == 1
        session.attach_caption(revealed.mask_id, "a lamp behind the balloon")

        rows = _payload_rows(*sfe.build_sf_payload(session))
        crop = Image.open(io.BytesIO(base64.b64decode(rows[f"sess-1:{revealed.mask_id}"]["crop_png_bytes"])))

        pass1_image = session.image_for_pass(1)
        ys, xs = np.nonzero(revealed.geometry)
        for y, x in zip(ys, xs):
            y, x = int(y), int(x)
            assert crop.getpixel((x, y)) == pass1_image.getpixel((x, y))
            # Confirms this is actually pinning something: pass 0's
            # original was flat black here, pass 1's is the scrub's red —
            # if build_sf_payload wrongly used pass 0, this would fail.
            assert crop.getpixel((x, y))[:3] == (255, 0, 0)
