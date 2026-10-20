"""Re-entering a half-captioned session.

Sessions saved before the captioner existed have hand-written captions and no
drafts. A session saved mid-review has all three kinds of segment. Either must
reload intact, and a captioner run must pick up only what is still blank.

Uses the same stubbed-bridge save/restart/load shape as
test_resume_scrubbed_pass.py.

Run with:  pytest backend/tests/test_captioner_reentry.py -v
"""

from __future__ import annotations

import pytest

import aa_persistence
import captioner
import sf_engine as sfe
from services import SegmentationService
from tests.test_resume_scrubbed_pass import (
    SID, FakeBridge, FakeSam3, _upload, _restart, _norm_box_100,
)

RECTS = [(10, 30, 10, 30), (40, 60, 40, 60), (70, 90, 70, 90)]


@pytest.fixture()
def bridge(tmp_path, monkeypatch):
    fake = FakeBridge()
    monkeypatch.setattr(aa_persistence, "bridge", fake)
    monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf")
    return fake


@pytest.fixture()
def service(bridge, tmp_path):
    return SegmentationService(tmp_path / "sessions", FakeSam3())


def _three_masks(service):
    """A hand-captioned, B draft only, C blank."""
    sf = _upload(service)
    ids = []
    for rect in RECTS:
        service.processor.geo_rects = [rect]
        ids.append(sf.add_box_selection(_norm_box_100(rect), True)[0].mask_id)
    a, b, c = ids
    sf.attach_caption(a, "small man, knight armor")
    sf.set_draft_caption(b, "clown, sitting")
    service.sessions[SID]["state"] = sf.state
    return a, b, c


def _reload(service):
    service.save_session_to_disk(SID)
    _restart(service)
    service.load_session_from_disk(SID)
    return service.sf_sessions[SID]


def test_captions_and_drafts_survive_save_and_reload(service):
    a, b, c = _three_masks(service)
    sf = _reload(service)
    ra, rb, rc = sf.get_mask(a), sf.get_mask(b), sf.get_mask(c)
    assert (ra.dataset_status, ra.caption, ra.draft_caption) == (sfe.DatasetStatus.KEEP, "small man, knight armor", None)
    assert (rb.dataset_status, rb.caption, rb.draft_caption) == (sfe.DatasetStatus.UNASSIGNED, None, "clown, sitting")
    assert (rc.caption, rc.draft_caption) == (None, None)


def test_a_run_after_reentry_picks_up_only_the_blank_segment(service):
    a, b, c = _three_masks(service)
    sf = _reload(service)
    assert captioner.masks_needing_draft(sf.masks) == [c]


def test_a_reviewed_draft_keeps_both_texts_across_reload(service):
    a, b, c = _three_masks(service)
    sf = service.sf_sessions[SID]
    sf.attach_caption(b, "clown, sitting, crosshatching")
    sf = _reload(service)
    rb = sf.get_mask(b)
    assert rb.dataset_status is sfe.DatasetStatus.KEEP
    assert rb.caption == "clown, sitting, crosshatching"
    assert rb.draft_caption == "clown, sitting"  # still tells "corrected" from "approved"


def test_a_session_saved_before_drafts_existed_loads_with_none(service, bridge):
    a, b, c = _three_masks(service)
    service.save_session_to_disk(SID)
    # Strip every draft_caption triple, as an older save would never have had them.
    for path, (rows, cols, vals) in list(bridge.files.items()):
        keep = [i for i, col in enumerate(cols) if col != "draft_caption"]
        bridge.files[path] = ([rows[i] for i in keep], [cols[i] for i in keep], [vals[i] for i in keep])
    _restart(service)
    service.load_session_from_disk(SID)
    sf = service.sf_sessions[SID]
    assert sf.get_mask(a).caption == "small man, knight armor"
    assert all(m.draft_caption is None for m in sf.masks)
    # Old work is already reviewed; only the two uncaptioned segments are blank.
    assert captioner.masks_needing_draft(sf.masks) == [b, c]


def test_a_resumed_sessions_ledger_carries_thumbnails_and_drafts(service):
    """The Captions tab reads `all_masks` straight off /loadSession. That block
    is built separately from the live endpoints', and once shipped without
    `thumb_png_bytes` (every picture was a broken-image icon) and without
    `draft_captions` (no review queue after reopening a session)."""
    a, b, c = _three_masks(service)
    service.save_session_to_disk(SID)
    _restart(service)
    ledger = service.load_session_from_disk(SID)["all_masks"]
    by_id = dict(zip(ledger["mask_ids"], zip(ledger["thumb_png_bytes"], ledger["draft_captions"])))
    assert set(by_id) == {a, b, c}
    assert all(thumb for thumb, _ in by_id.values()), "every segment needs a thumbnail"
    assert by_id[b][1] == "clown, sitting"
    assert by_id[a][1] is None and by_id[c][1] is None


def test_never_needs_survives_a_reload_and_is_skipped_by_a_run(service):
    a, b, c = _three_masks(service)
    service.sf_sessions[SID].set_never(c, True)
    service.save_session_to_disk(SID)
    _restart(service)
    ledger = service.load_session_from_disk(SID)["all_masks"]
    status = dict(zip(ledger["mask_ids"], ledger["dataset_statuses"]))
    assert status[c] == "discard"
    sf = service.sf_sessions[SID]
    assert sf.get_mask(c).dataset_status is sfe.DatasetStatus.DISCARD
    # a: captioned by hand, b: draft only, c: never needs -> nothing left to caption.
    assert captioner.masks_needing_draft(sf.masks) == []
