"""Remove-a-mask and undo-last-scrub (segmentation overhaul, step 1)."""
import pytest

from tests.test_sf_engine import (
    FakeSam3,
    NoOpInpainter,
    PaintingInpainter,
    _norm_box,
    _pixel,
    _select_many,
    _select_one,
    _session,
    sfe,
)

A = (2, 5, 2, 5)
B = (6, 9, 6, 9)


class TestRemoveMask:
    def test_removes_a_current_pass_selection(self):
        seg = FakeSam3()
        s = _session(seg)
        a, b = _select_many(s, seg, "x", [A, B])
        removed = s.remove_mask(a.mask_id)
        assert removed.mask_id == a.mask_id
        assert [m.mask_id for m in s.masks] == [b.mask_id]

    def test_a_removed_mask_is_not_scrubbed(self):
        seg = FakeSam3()
        s = _session(seg)
        a, b = _select_many(s, seg, "x", [A, B])
        s.set_held(a.mask_id, True)
        s.set_held(b.mask_id, True)
        s.remove_mask(a.mask_id)
        painter = PaintingInpainter()
        s.run_lama_pass(painter)
        assert s.last_scrub.mask_ids == (b.mask_id,)
        assert painter.masks[0][2, 2] == 0

    def test_a_captioned_mask_can_be_removed(self):
        seg = FakeSam3()
        s = _session(seg)
        a = _select_one(s, seg, "x", A)
        s.attach_caption(a.mask_id, "boy")
        s.remove_mask(a.mask_id)
        assert s.masks == []

    def test_an_earlier_pass_mask_is_refused(self):
        seg = FakeSam3()
        s = _session(seg)
        a = _select_one(s, seg, "x", A)
        s.attach_caption(a.mask_id, "boy")
        s.run_lama_pass(NoOpInpainter())
        with pytest.raises(ValueError, match="pass 0"):
            s.remove_mask(a.mask_id)
        assert [m.mask_id for m in s.masks] == [a.mask_id]

    def test_unknown_mask_raises(self):
        s = _session(FakeSam3())
        with pytest.raises(sfe.UnknownMaskError):
            s.remove_mask("nope")


class TestUndoLastScrub:
    def test_nothing_to_undo(self):
        s = _session(FakeSam3())
        with pytest.raises(sfe.CannotUndoScrubError, match="nothing to undo"):
            s.undo_last_scrub()

    def test_restores_pass_image_and_held_flags(self):
        seg = FakeSam3()
        s = _session(seg)
        a = _select_one(s, seg, "x", A)
        s.set_held(a.mask_id, True)
        s.run_lama_pass(PaintingInpainter())
        assert s.pass_ == 1 and not s.get_mask(a.mask_id).held
        assert _pixel(s.image_for_pass(1), 3, 3) == (255, 0, 0, 255)

        s.undo_last_scrub()

        assert s.pass_ == 0
        assert s.get_mask(a.mask_id).held
        assert s.last_scrub is None and s.scrub_history == []
        with pytest.raises(ValueError):
            s.image_for_pass(1)
        # the segmentor was re-pointed at the restored image
        assert seg.images_set[-1].size == s.image_for_pass(0).size

    def test_scrub_again_after_undo_gives_the_same_result(self):
        seg = FakeSam3()
        s = _session(seg)
        a = _select_one(s, seg, "x", A)
        s.set_held(a.mask_id, True)
        s.run_lama_pass(PaintingInpainter())
        s.undo_last_scrub()
        s.run_lama_pass(PaintingInpainter())
        assert s.pass_ == 1
        assert _pixel(s.image_for_pass(1), 3, 3) == (255, 0, 0, 255)

    def test_undoes_one_scrub_at_a_time_down_to_the_start(self):
        seg = FakeSam3()
        s = _session(seg)
        a = _select_one(s, seg, "x", A)
        s.set_held(a.mask_id, True)
        s.run_lama_pass(NoOpInpainter())
        b = _select_one(s, seg, "y", B)
        s.set_held(b.mask_id, True)
        s.run_lama_pass(NoOpInpainter())
        assert s.pass_ == 2

        s.undo_last_scrub()
        assert s.pass_ == 1 and s.get_mask(b.mask_id).held
        assert s.last_scrub.to_pass == 1

        s.undo_last_scrub()
        assert s.pass_ == 0 and s.get_mask(a.mask_id).held

    def test_selections_made_after_the_scrub_are_discarded(self):
        seg = FakeSam3()
        s = _session(seg)
        a = _select_one(s, seg, "x", A)
        s.set_held(a.mask_id, True)
        s.run_lama_pass(NoOpInpainter())
        late = _select_one(s, seg, "y", B)
        assert late.pass_ == 1
        s.undo_last_scrub()
        assert [m.mask_id for m in s.masks] == [a.mask_id]

    def test_refused_when_it_would_destroy_captioned_work(self):
        seg = FakeSam3()
        s = _session(seg)
        a = _select_one(s, seg, "x", A)
        s.set_held(a.mask_id, True)
        s.run_lama_pass(NoOpInpainter())
        late = _select_one(s, seg, "y", B)
        s.attach_caption(late.mask_id, "keeper")

        with pytest.raises(sfe.CannotUndoScrubError, match="1 captioned"):
            s.undo_last_scrub()

        # nothing changed
        assert s.pass_ == 1
        assert s.get_mask(late.mask_id).caption == "keeper"
        assert not s.get_mask(a.mask_id).held

    def test_earlier_pass_keepers_survive_an_undo_of_a_later_scrub(self):
        seg = FakeSam3()
        s = _session(seg)
        a = _select_one(s, seg, "x", A)
        s.attach_caption(a.mask_id, "boy")
        s.set_held(a.mask_id, True)
        s.run_lama_pass(NoOpInpainter())
        s.undo_last_scrub()
        assert s.get_mask(a.mask_id).caption == "boy"

    def test_wb_scrub_pass_goes_back_to_what_it_was(self):
        seg = FakeSam3()
        s = _session(seg)
        a = _select_one(s, seg, "x", A)
        # a balloon-sourced selection makes the resulting pass the WB result
        seg.geo_rects = [B]
        s.add_box_selection(_norm_box(B), True, source=sfe.SOURCE_BALLOON)
        for m in list(s.masks):
            s.set_held(m.mask_id, True)
        s.run_lama_pass(NoOpInpainter())
        assert s.wb_scrub_pass == 1
        s.undo_last_scrub()
        assert s.wb_scrub_pass is None

    def test_a_reloaded_session_can_undo_one_level(self):
        seg = FakeSam3()
        s = _session(seg)
        a = _select_one(s, seg, "x", A)
        s.set_held(a.mask_id, True)
        s.run_lama_pass(NoOpInpainter())
        rec = s.last_scrub

        fresh = _session(FakeSam3())
        fresh.restore(
            1, {1: s.image_for_pass(1)}, list(s.masks), last_scrub=rec, wb_scrub_pass=None
        )
        fresh.undo_last_scrub()
        assert fresh.pass_ == 0
        with pytest.raises(sfe.CannotUndoScrubError):
            fresh.undo_last_scrub()
