"""Draft captions and the captioner runner.

The invariant under test: a model's draft NEVER makes a mask trainable.
`keep` + `caption` is what reaches the training set; only a person going
through `attach_caption` sets those.

Run with:  pytest backend/tests/test_captioner.py -v
"""

from __future__ import annotations

import threading
import time

import pytest

import captioner
import sf_engine as sfe
from tests.test_sf_engine import FakeSam3, _session




def _session_with_mask():
    seg = FakeSam3()
    seg.text_rects = {"egg": [(0, 3, 0, 3)]}
    session = _session(seg)
    return session, session.add_text_selection("egg")[0].mask_id


class TestDraftCaption:
    def test_draft_does_not_make_a_mask_trainable(self):
        session, mid = _session_with_mask()
        rec = session.set_draft_caption(mid, "  a red egg  ")
        assert rec.draft_caption == "a red egg"
        assert rec.caption is None
        assert rec.dataset_status is sfe.DatasetStatus.UNASSIGNED

    def test_empty_draft_is_refused(self):
        session, mid = _session_with_mask()
        with pytest.raises(ValueError):
            session.set_draft_caption(mid, "   ")

    def test_unknown_mask_raises(self):
        session, _ = _session_with_mask()
        with pytest.raises(sfe.UnknownMaskError):
            session.set_draft_caption("nope", "x")

    def test_approving_keeps_the_draft_so_edits_are_detectable(self):
        session, mid = _session_with_mask()
        session.set_draft_caption(mid, "a red egg")
        rec = session.attach_caption(mid, "a red egg, crosshatching")
        assert rec.dataset_status is sfe.DatasetStatus.KEEP
        assert rec.caption != rec.draft_caption
        assert rec.draft_caption == "a red egg"

    def test_a_later_draft_never_overwrites_a_reviewed_one(self):
        session, mid = _session_with_mask()
        session.set_draft_caption(mid, "first")
        session.attach_caption(mid, "reviewed")
        rec = session.set_draft_caption(mid, "second")
        assert rec.draft_caption == "first"
        assert rec.caption == "reviewed"


class TestCleanOutput:
    def test_plain_text_passes_through(self):
        assert captioner.clean_output("a red egg\n") == "a red egg"

    def test_banners_and_stats_are_dropped(self):
        raw = "==========\na red egg, ink\n==========\nPrompt: 12 tokens\nPeak memory: 5 GB\n"
        assert captioner.clean_output(raw) == "a red egg, ink"


class TestJob:
    def test_runs_every_item_and_survives_a_failure(self):
        seen = []

        def one(item):
            if item == "bad":
                raise RuntimeError("boom")
            seen.append(item)

        job = captioner.start(["a", "bad", "b"], one)
        deadline = time.time() + 5
        while job.state == "running" and time.time() < deadline:
            time.sleep(0.01)
        info = captioner.status(job.id)
        assert info["state"] == "done"
        assert (info["done"], info["total"], info["failed"]) == (3, 3, ["bad"])
        assert seen == ["a", "b"]

    def test_second_job_while_one_runs_is_refused(self):
        gate = threading.Event()
        job = captioner.start(["a"], lambda _item: gate.wait(5))
        try:
            with pytest.raises(captioner.CaptionerError):
                captioner.start(["b"], lambda _item: None)
        finally:
            gate.set()
            deadline = time.time() + 5
            while job.state == "running" and time.time() < deadline:
                time.sleep(0.01)

    def test_cancel_stops_before_the_next_item(self):
        gate = threading.Event()
        seen = []

        def one(item):
            seen.append(item)
            gate.wait(5)

        job = captioner.start(["a", "b", "c"], one)
        captioner.cancel(job.id)
        gate.set()
        deadline = time.time() + 5
        while job.state == "running" and time.time() < deadline:
            time.sleep(0.01)
        assert job.state == "cancelled"
        assert seen == ["a"]


class TestNeverNeeds:
    def test_marking_and_unmarking(self):
        session, mid = _session_with_mask()
        assert session.set_never(mid, True).dataset_status is sfe.DatasetStatus.DISCARD
        assert session.set_never(mid, False).dataset_status is sfe.DatasetStatus.UNASSIGNED

    def test_a_captioned_segment_cannot_be_marked(self):
        session, mid = _session_with_mask()
        session.attach_caption(mid, "a red egg")
        with pytest.raises(ValueError, match="captioned"):
            session.set_never(mid, True)
        assert session.get_mask(mid).caption == "a red egg"

    def test_unmarking_leaves_other_statuses_alone(self):
        session, mid = _session_with_mask()
        session.attach_caption(mid, "a red egg")
        assert session.set_never(mid, False).dataset_status is sfe.DatasetStatus.KEEP

    def test_captioning_a_never_segment_brings_it_back(self):
        session, mid = _session_with_mask()
        session.set_never(mid, True)
        assert session.attach_caption(mid, "a red egg").dataset_status is sfe.DatasetStatus.KEEP

    def test_a_run_skips_never_segments(self):
        class M:
            def __init__(self, mask_id, status):
                self.mask_id, self.dataset_status = mask_id, status
                self.caption = self.draft_caption = None
        masks = [M("a", sfe.DatasetStatus.UNASSIGNED), M("b", sfe.DatasetStatus.DISCARD)]
        assert captioner.masks_needing_draft(masks) == ["a"]

    def test_a_text_reground_cannot_evict_a_never_segment(self):
        seg = FakeSam3()
        seg.text_rects = {"egg": [(0, 3, 0, 3)], "rabbit": [(6, 9, 6, 9)]}
        session = _session(seg)
        egg = session.add_text_selection("egg")[0]
        session.set_never(egg.mask_id, True)
        session.add_text_selection("rabbit")
        assert session.get_mask(egg.mask_id).dataset_status is sfe.DatasetStatus.DISCARD


class TestClearCaption:
    def test_deleting_a_caption_returns_the_segment_to_uncaptioned(self):
        session, mid = _session_with_mask()
        session.set_draft_caption(mid, "an egg")
        session.attach_caption(mid, "a red egg")
        rec = session.clear_caption(mid)
        assert rec.dataset_status is sfe.DatasetStatus.UNASSIGNED
        assert rec.caption is None and rec.draft_caption is None

    def test_clearing_an_uncaptioned_segment_is_a_noop(self):
        session, mid = _session_with_mask()
        session.set_never(mid, True)
        assert session.clear_caption(mid).dataset_status is sfe.DatasetStatus.DISCARD

    def test_a_cleared_segment_can_be_captioned_again(self):
        session, mid = _session_with_mask()
        session.attach_caption(mid, "a red egg")
        session.clear_caption(mid)
        assert session.attach_caption(mid, "a blue egg").caption == "a blue egg"
