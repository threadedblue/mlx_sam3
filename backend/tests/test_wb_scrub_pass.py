"""v4 §2: the WB-scrub result is identified by what produced it, not by its
pass number.

`run_lama_pass` is generic — it increments the pass whatever was scrubbed —
so pass 1 only means "WB-scrub result" if Balloon Scrub happened to run
before any ordinary S&S scrub. Nothing enforces that order, and nothing
should: a page with no balloons, or a user who segments something first, is
a legitimate workflow. So the pass a balloon scrub produces is RECORDED
(`SFSession.wb_scrub_pass`) rather than assumed.

Balloon-detected masks go through `add_box_selection` exactly like a
hand-drawn box; the only difference is `source=SOURCE_BALLOON`, which is what
lets the scrub recognise its own batch. There is no second scrub mechanism.
"""

from __future__ import annotations

import pytest

import sf_engine as sfe
from tests.test_sf_engine import (
    DRAWN,
    FakeSam3,
    PaintingInpainter,
    _norm_box,
    _select_one,
    _session,
)

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)


def _balloon_select(session, seg, rect=DRAWN):
    """What /segment/balloons does: an ordinary box selection, marked."""
    seg.geo_rects = [rect]
    return session.add_box_selection(
        _norm_box(rect), True, source=sfe.SOURCE_BALLOON
    )[0]


class TestWbScrubPassIdentification:
    def test_a_fresh_session_has_no_wb_scrub_pass(self):
        assert _session(FakeSam3()).wb_scrub_pass is None

    def test_balloon_scrub_first_records_pass_1(self):
        """The happy path the design doc's layer table assumes."""
        seg = FakeSam3()
        session = _session(seg)
        rec = _balloon_select(session, seg)
        session.set_held(rec.mask_id, True)
        session.run_lama_pass(PaintingInpainter(color=RED))

        assert session.pass_ == 1
        assert session.wb_scrub_pass == 1

    def test_balloon_scrub_after_an_ss_round_records_the_pass_it_actually_made(self):
        """Out of the design doc's intended order: an ordinary S&S scrub takes
        pass 1 first, so the WB-scrub lands at pass 2. Nothing may silently
        mislabel pass 1 as the WB result — the recorded pass must follow the
        balloons."""
        seg = FakeSam3()
        session = _session(seg)

        figure = _select_one(session, seg, "figure", (0, 3, 0, 3))
        session.set_held(figure.mask_id, True)
        session.run_lama_pass(PaintingInpainter(color=GREEN))
        assert session.pass_ == 1
        assert session.wb_scrub_pass is None, (
            "an ordinary S&S scrub must never claim to be the WB-scrub result"
        )

        balloon = _balloon_select(session, seg)
        session.set_held(balloon.mask_id, True)
        session.run_lama_pass(PaintingInpainter(color=RED))

        assert session.pass_ == 2
        assert session.wb_scrub_pass == 2

    def test_an_ss_round_after_the_wb_scrub_does_not_move_it(self):
        seg = FakeSam3()
        session = _session(seg)
        balloon = _balloon_select(session, seg)
        session.set_held(balloon.mask_id, True)
        session.run_lama_pass(PaintingInpainter(color=RED))
        assert session.wb_scrub_pass == 1

        figure = _select_one(session, seg, "figure", (0, 3, 0, 3))
        session.set_held(figure.mask_id, True)
        session.run_lama_pass(PaintingInpainter(color=GREEN))

        assert session.pass_ == 2
        assert session.wb_scrub_pass == 1, "the WB layer stays where it was made"

    def test_a_zero_balloon_page_grows_no_wb_layer(self):
        """Detection found nothing, so nothing balloon-sourced was ever
        selected — the session must not claim a WB-scrub layer that would
        really just be an ordinary pass."""
        seg = FakeSam3()
        session = _session(seg)

        figure = _select_one(session, seg, "figure", (0, 3, 0, 3))
        session.set_held(figure.mask_id, True)
        session.run_lama_pass(PaintingInpainter(color=GREEN))

        assert session.pass_ == 1
        assert session.wb_scrub_pass is None

    def test_balloons_unchecked_before_the_scrub_grow_no_wb_layer(self):
        """The opt-out checkbox list can empty the batch. A scrub that
        contained no balloon actually scrubbed none, so it is not the WB
        result."""
        seg = FakeSam3()
        session = _session(seg)
        balloon = _balloon_select(session, seg)
        session.set_held(balloon.mask_id, False)

        session.run_lama_pass(PaintingInpainter(color=RED))

        assert session.pass_ == 1
        assert session.wb_scrub_pass is None

    def test_a_balloon_mask_is_distinguishable_from_a_hand_drawn_box(self):
        """The whole mechanism rests on this: before v4 both were source
        'box' and genuinely indistinguishable."""
        seg = FakeSam3()
        session = _session(seg)
        balloon = _balloon_select(session, seg, (2, 6, 2, 6))
        seg.geo_rects = [(7, 9, 7, 9)]
        hand = session.add_box_selection(_norm_box((7, 9, 7, 9)), True)[0]

        assert balloon.source == sfe.SOURCE_BALLOON
        assert hand.source == sfe.SOURCE_BOX
        assert balloon.source != hand.source

    def test_balloon_selection_is_mechanically_an_ordinary_box_selection(self):
        """No second mechanism: same method, same reconciliation, same
        geometry — only `source` differs."""
        seg = FakeSam3()
        a = _session(seg)
        b = _session(FakeSam3())

        seg.geo_rects = [DRAWN]
        balloon = a.add_box_selection(
            _norm_box(DRAWN), True, source=sfe.SOURCE_BALLOON
        )[0]
        b.segmentor.geo_rects = [DRAWN]
        plain = b.add_box_selection(_norm_box(DRAWN), True)[0]

        assert balloon.geometry.tolist() == plain.geometry.tolist()
        assert balloon.pass_ == plain.pass_
        assert balloon.dataset_status == plain.dataset_status


class TestWbScrubPassSurvivesRestore:
    def test_restore_carries_the_wb_scrub_pass(self):
        """`MaskRecord.source` is not persisted (it reloads as 'loaded'), so
        the pass number itself has to come back — otherwise a resumed session
        silently loses its WB layer."""
        session = _session(FakeSam3())
        session.restore(pass_=3, pass_images={}, masks=[], wb_scrub_pass=2)

        assert session.wb_scrub_pass == 2

    def test_restore_without_one_leaves_it_absent(self):
        session = _session(FakeSam3())
        session.restore(pass_=1, pass_images={}, masks=[])

        assert session.wb_scrub_pass is None

    def test_a_reloaded_session_cannot_re_derive_it_from_masks(self):
        """Pins WHY the pass number is persisted rather than recomputed: the
        provenance that identifies a balloon mask does not survive a reload.
        If this ever stops being true, re-deriving becomes an option."""
        reloaded_mask = sfe.MaskRecord(
            mask_id="1:abc", pass_=1, geometry=None, source="loaded",
        )
        assert reloaded_mask.source != sfe.SOURCE_BALLOON
