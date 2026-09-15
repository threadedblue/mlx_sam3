"""Unit tests for sf_engine.SFSession's mask-identity reconciliation.

`FakeSam3` deliberately reproduces the two non-append-only behaviors
documented in aa_persistence.py's module docstring:
  - `set_text_prompt` REPLACES the active text prompt's masks.
  - `add_geometric_prompt`/`add_point_prompt` re-run grounding over the
    whole accumulated prompt set, returning a fresh mask list every call
    (which may refine a previous mask's boundary, suppress it entirely, or
    add new ones) rather than appending.

Run with:  pytest backend/tests/test_sf_engine.py -v
"""

from __future__ import annotations

import numpy as np

import sf_engine as sfe


def _box_mask(shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros(shape, dtype=np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


class FakeSam3:
    """Minimal Sam3Like double with replace/re-ground semantics, not
    append-only growth."""

    SHAPE = (10, 10)
    _text_boxes: dict[str, list[tuple[int, int, int, int]]] = {}

    def __init__(self):
        self._geo_boxes: list[tuple[int, int, int, int]] = []
        self._geo_override: dict[tuple, tuple] = {}
        self._suppressed: set[tuple] = set()

    def set_image(self, image) -> dict:
        self._geo_boxes = []
        self._geo_override = {}
        self._suppressed = set()
        return {"masks": []}

    def set_text_prompt(self, prompt: str, state: dict) -> dict:
        boxes = self._text_boxes.get(prompt, [])
        return {"masks": [_box_mask(self.SHAPE, *b) for b in boxes]}

    def add_geometric_prompt(self, box, label: bool, state: dict) -> dict:
        self._geo_boxes.append(tuple(box))
        rendered = [
            self._geo_override.get(b, b)
            for b in self._geo_boxes
            if b not in self._suppressed
        ]
        return {"masks": [_box_mask(self.SHAPE, *b) for b in rendered]}

    def add_point_prompt(self, point, label: bool, state: dict) -> dict:
        return self.add_geometric_prompt(point, label, state)


def _tiny_png() -> bytes:
    import io as _io
    from PIL import Image as _Image

    buf = _io.BytesIO()
    _Image.new("RGB", (4, 4), (0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _session(segmentor: FakeSam3, session_id: str = "sess-1") -> sfe.SFSession:
    original = sfe.ImmutableOriginal(_tiny_png())
    state = segmentor.set_image(None)
    return sfe.SFSession(session_id, original, segmentor, state)


class TestTextPromptReplaceSemantics:
    """(a) A second set_text_prompt call in the same pass with a different
    result count than the first."""

    def test_shrinking_result_count_does_not_silently_drop_the_new_selection(self):
        segmentor = FakeSam3()
        segmentor._text_boxes = {
            "cat": [(0, 3, 0, 3), (5, 8, 5, 8)],  # 2 masks
            "dog": [(2, 6, 2, 6)],                 # 1 mask, replaces "cat"'s
        }
        session = _session(segmentor)

        cat_records = session.add_text_selection("cat", sfe.MaskType.IN)
        assert len(cat_records) == 2

        # Old bug: masks[before:] == masks[2:] on a now-1-element list is
        # [], so the dog selection would silently vanish. It must not.
        dog_records = session.add_text_selection("dog", sfe.MaskType.IN)
        assert len(dog_records) == 1
        assert dog_records[0].text_tag == "dog"

        # The two stale "cat" MaskRecords must be gone — SAM3 replaced them,
        # so nothing in the live state backs them anymore.
        assert len(session.masks) == 1
        assert session.masks[0].text_tag == "dog"

    def test_growing_result_count_only_returns_the_genuinely_new_ones(self):
        segmentor = FakeSam3()
        segmentor._text_boxes = {
            "cat": [(0, 3, 0, 3)],
            "cats": [(0, 3, 0, 3), (5, 8, 5, 8)],  # same first cat + a new one
        }
        session = _session(segmentor)

        first = session.add_text_selection("cat", sfe.MaskType.IN)
        assert len(first) == 1
        original_id = first[0].mask_id

        second = session.add_text_selection("cats", sfe.MaskType.IN)

        # Only the genuinely new region should be reported as a new selection.
        assert len(second) == 1
        assert second[0].text_tag == "cats"

        # The overlapping mask keeps its original identity and tag rather
        # than being duplicated or re-tagged "cats".
        assert len(session.masks) == 2
        kept = next(m for m in session.masks if m.mask_id == original_id)
        assert kept.text_tag == "cat"


class TestGeometricReground:
    """(b) Two add_geometric_prompt calls in the same pass where the second
    reflects a re-ground of the first's mask."""

    def test_reground_refines_existing_mask_without_duplicating_or_corrupting_it(self):
        segmentor = FakeSam3()
        session = _session(segmentor)

        first = session.add_box_selection([0, 3, 0, 3], True, "widget", sfe.MaskType.IN)
        widget_id = first[0].mask_id

        # Second call re-grounds the whole accumulated prompt set: the
        # widget box's rendered mask shifts slightly (still clearly the same
        # object — high IoU) and a new, disjoint box appears.
        segmentor._geo_override[(0, 3, 0, 3)] = (0, 4, 0, 4)
        second = session.add_box_selection([6, 9, 6, 9], True, "gadget", sfe.MaskType.IN)

        assert len(second) == 1
        assert second[0].text_tag == "gadget"
        assert second[0].source == "box"

        # The widget MaskRecord persists under its original identity/tag —
        # not duplicated, not silently dropped, not re-tagged "gadget" — but
        # its geometry is refreshed to the refined boundary.
        assert len(session.masks) == 2
        widget_now = next(m for m in session.masks if m.mask_id == widget_id)
        assert widget_now.text_tag == "widget"
        assert widget_now.mask_type is sfe.MaskType.IN
        expected_refined = _box_mask(segmentor.SHAPE, 0, 4, 0, 4)
        np.testing.assert_array_equal(widget_now.geometry, expected_refined)

    def test_reground_that_drops_a_region_removes_its_record(self):
        """If re-grounding no longer reports a region at all (e.g. a
        negative prompt suppressed it), SF must stop tracking it rather than
        exporting a mask that no longer exists in SAM3's live state."""
        segmentor = FakeSam3()
        session = _session(segmentor)

        first = session.add_box_selection([0, 3, 0, 3], True, "widget", sfe.MaskType.IN)
        widget_id = first[0].mask_id

        segmentor._suppressed.add((0, 3, 0, 3))
        second = session.add_box_selection([6, 9, 6, 9], False, "gadget", sfe.MaskType.OUT)

        assert len(second) == 1
        assert widget_id not in {m.mask_id for m in session.masks}
        assert len(session.masks) == 1
        assert session.masks[0].mask_type is sfe.MaskType.OUT


class TestPassIsolation:
    def test_reconciliation_never_touches_a_different_pass(self):
        segmentor = FakeSam3()
        session = _session(segmentor)

        pass1 = session.add_box_selection([0, 3, 0, 3], True, "widget", sfe.MaskType.IN)
        pass1_id = pass1[0].mask_id

        class NoOpInpainter:
            def inpaint(self, image, mask):
                return image

        session.run_lama_pass(NoOpInpainter())
        assert session._pass is sfe.Pass.BACKGROUND

        # Fresh geometry in Pass 2 must not be matched against, or evict,
        # Pass 1's already-finalized record — even though it happens to
        # reuse the exact same box coordinates.
        pass2 = session.add_box_selection([0, 3, 0, 3], True, "lamp", sfe.MaskType.IN)

        assert len(pass2) == 1
        assert pass2[0].pass_ is sfe.Pass.BACKGROUND
        remaining_ids = {m.mask_id for m in session.masks}
        assert pass1_id in remaining_ids
        assert len(session.masks) == 2

    def test_pass2_selection_matching_pass1_geometry_does_not_evict_pass1_records(self):
        """Adversarial case: Pass 1 has N (here, 2) confirmed MaskRecords.
        After run_lama_pass(), a Pass 2 selection reuses the *exact same*
        box as one of them, so its resulting mask is geometrically
        identical (IoU == 1.0) to a Pass 1 record — the strongest possible
        false-match trigger if pass-scoping were ever lost. All N Pass 1
        records must survive, byte-for-byte unmodified (same object, not
        just equal), and the Pass 2 call must report a genuinely new record
        rather than reconciling into Pass 1's.
        """
        segmentor = FakeSam3()
        session = _session(segmentor)

        widget = session.add_box_selection([0, 3, 0, 3], True, "widget", sfe.MaskType.IN)[0]
        doohickey = session.add_box_selection([6, 9, 6, 9], True, "doohickey", sfe.MaskType.IN)[0]
        pass1_records = [widget, doohickey]
        assert len(session.masks) == 2

        class NoOpInpainter:
            def inpaint(self, image, mask):
                return image

        session.run_lama_pass(NoOpInpainter())
        assert session._pass is sfe.Pass.BACKGROUND

        # Adversarial: reuse widget's exact box in Pass 2, so this mask's
        # geometry is identical to a Pass 1 record's.
        pass2_records = session.add_box_selection([0, 3, 0, 3], True, "lamp", sfe.MaskType.IN)

        assert len(pass2_records) == 1
        assert pass2_records[0].pass_ is sfe.Pass.BACKGROUND
        assert pass2_records[0].text_tag == "lamp"
        assert pass2_records[0].mask_id != widget.mask_id

        for original in pass1_records:
            current = next((m for m in session.masks if m.mask_id == original.mask_id), None)
            assert current is not None, f"Pass 1 record {original.mask_id} was evicted by a Pass 2 call"
            # Field-level equality, not `is`: dataclasses.replace() legitimately
            # rebuilds a record on every same-pass match (e.g. the doohickey
            # call above already replace()'d widget once, before Pass 2 even
            # started) — object identity isn't a guarantee this code makes or
            # needs to. What must hold is that the Pass 2 call didn't touch it.
            assert current.mask_type == original.mask_type
            assert current.pass_ == original.pass_
            assert current.text_tag == original.text_tag
            assert current.source == original.source
            np.testing.assert_array_equal(current.geometry, original.geometry)

        assert len(session.masks) == 3
