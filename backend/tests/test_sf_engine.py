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

import io

import numpy as np
import pytest
from PIL import Image

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

        assert set(rows[f"sess-1:{rec.mask_id}"]) == {"pass", "caption", "mask_png_bytes"}
