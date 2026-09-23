"""Unit tests for services.py's process-lifetime crop cache
(`_cached_crop_png_b64`/`_invalidate_stale_crops`, wired into
`serialize_sf_masks` as the `crop_png_bytes` field).

Reuses `test_sf_engine.py`'s `FakeSam3`/session helpers rather than
duplicating them — this is exercising services.py's caching layer on top
of the same engine behaviour those tests already pin, not re-testing the
engine itself.

Run with:  pytest backend/tests/test_services.py -v
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

import sf_engine as sfe
import services
from tests.test_sf_engine import (
    DRAWN,
    FakeSam3,
    PaintingInpainter,
    _norm_box,
    _norm_point,
    _select_one,
    _session,
)

REFINED = (2, 6, 2, 5)  # DRAWN shrunk by one column — a real, IoU-high refinement


@pytest.fixture(autouse=True)
def _clear_crop_cache():
    """The cache is deliberately process-lifetime, not per-test — but tests
    still need a clean slate so an earlier test's entries (or lack of
    invalidation) can't make this test's assertions pass for the wrong
    reason. mask_ids are uuid4-derived, so no cross-test collision is even
    possible; this is purely hygiene."""
    services._crop_cache.clear()
    yield
    services._crop_cache.clear()


class TestCropCacheReuse:
    def test_repeated_serialize_reuses_the_cached_crop_without_recomputing(self, monkeypatch):
        seg = FakeSam3()
        seg.geo_rects = [DRAWN]
        session = _session(seg)
        session.add_box_selection(_norm_box(DRAWN), True)

        real_crop = sfe._crop_png_bytes
        calls = {"n": 0}

        def counting_crop(image, mask):
            calls["n"] += 1
            return real_crop(image, mask)

        monkeypatch.setattr(sfe, "_crop_png_bytes", counting_crop)

        first = services.serialize_sf_masks(session, session.state)
        second = services.serialize_sf_masks(session, session.state)

        assert calls["n"] == 1, "second serialize must reuse the cache, not recompute"
        assert first["crop_png_bytes"] == second["crop_png_bytes"]


class TestCropCacheInvalidation:
    def test_avoid_refinement_invalidates_the_cached_crop(self):
        seg = FakeSam3()
        seg.geo_rects = [DRAWN]
        session = _session(seg)
        rec = session.add_box_selection(_norm_box(DRAWN), True)[0]
        mask_id = rec.mask_id

        before = services.serialize_sf_masks(session, session.state)
        crop_before = before["crop_png_bytes"][before["mask_ids"].index(mask_id)]

        # Avoid prompt re-grounds and refines the SAME object's boundary in
        # place (see TestAvoidRefinesRecordedGeometry in test_sf_engine.py) —
        # same mask_id, different geometry.
        seg.geo_rects = [REFINED]
        session.add_point_selection(_norm_point(4, 3), False, "avoid this part")
        assert session.get_mask(mask_id).mask_id == mask_id  # still the same record

        after = services.serialize_sf_masks(session, session.state)
        crop_after = after["crop_png_bytes"][after["mask_ids"].index(mask_id)]

        assert crop_after != crop_before

        # And the refined crop is now itself cached, not recomputed forever.
        again = services.serialize_sf_masks(session, session.state)
        crop_again = again["crop_png_bytes"][again["mask_ids"].index(mask_id)]
        assert crop_again == crop_after

    def test_caption_and_hold_never_invalidate_the_cached_crop(self, monkeypatch):
        """The whole point of caching: rebuilding the AA preview after a
        caption or hold change must not recompute crops for masks whose
        geometry never moved."""
        seg = FakeSam3()
        seg.geo_rects = [DRAWN]
        session = _session(seg)
        rec = session.add_box_selection(_norm_box(DRAWN), True)[0]

        real_crop = sfe._crop_png_bytes
        calls = {"n": 0}

        def counting_crop(image, mask):
            calls["n"] += 1
            return real_crop(image, mask)

        monkeypatch.setattr(sfe, "_crop_png_bytes", counting_crop)

        services.serialize_sf_masks(session, session.state)  # populates the cache
        session.attach_caption(rec.mask_id, "a red egg")
        session.set_held(rec.mask_id, True)
        services.serialize_sf_masks(session, session.state)

        assert calls["n"] == 1


class TestCropCacheIsPassAware:
    def test_a_mask_kept_in_a_later_pass_is_cropped_from_that_passs_image(self):
        """Mirrors test_sf_engine.py's
        test_a_mask_kept_in_a_later_pass_is_cropped_from_that_passs_image,
        but through the cache/serialize_sf_masks path rather than
        build_sf_payload: pass 0 stays the flat original, pass 1 is
        repainted red where `front` was, and a mask selected fresh in pass 1
        must cache pass 1's content, not pass 0's."""
        seg = FakeSam3()
        session = _session(seg)
        front = _select_one(session, seg, "front", (0, 5, 0, 5))
        session.set_held(front.mask_id, True)
        session.run_lama_pass(PaintingInpainter(color=(255, 0, 0, 255)))

        revealed = _select_one(session, seg, "revealed", (1, 4, 1, 4))
        assert revealed.pass_ == 1

        result = services.serialize_sf_masks(session, session.state)
        idx = result["mask_ids"].index(revealed.mask_id)
        crop = Image.open(io.BytesIO(base64.b64decode(result["crop_png_bytes"][idx])))

        pass1_image = session.image_for_pass(1)
        ys, xs = np.nonzero(revealed.geometry)
        assert len(xs) > 0
        for y, x in zip(ys, xs):
            y, x = int(y), int(x)
            assert crop.getpixel((x, y)) == pass1_image.getpixel((x, y))
            # Pins that this is really pass 1's content, not pass 0's flat
            # original: pass 0 was black here, pass 1 is the scrub's red.
            assert crop.getpixel((x, y))[:3] == (255, 0, 0)
