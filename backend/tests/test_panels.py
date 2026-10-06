"""Manual panel-bounds tool (v4 §4).

A panel is a LAYOUT REGION, not an object. The property every test here
exists to pin is that authoring one never goes near SAM3 and never creates a
MaskRecord — panel detection was validated zero-shot first (manga 14/25,
comic 17/25, ray-cast snap 11/25 at IoU 0.90) and none of it was good enough
to build on, so these bounds are authored by hand and must stay structurally
separate from the selection pipeline.

Fixtures are COLOURED and NON-UNIFORM throughout. A black or flat fixture
would let a crop-from-the-wrong-pass bug pass silently: two passes of a flat
image are byte-identical, so "this crop came from pass 1, not pass 2" is
unprovable against one.

The bridge is stubbed rather than hitting real Julia parquet I/O —
test_sf_wiring.py, which does, hangs in this environment.
"""

from __future__ import annotations

import base64
import io
import json
import pathlib

import numpy as np
import pytest
from PIL import Image

import aa_persistence
import sf_engine as sfe
from services import SegmentationService

SID = "panel-session"
IMG = 200


def _colour_png(size=IMG, tint=0) -> bytes:
    """Non-uniform, non-black, and distinguishable per `tint` — so a crop
    taken from the wrong pass is visibly the wrong bytes, not a coincidence."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            arr[y, x] = (
                (20 + x * 3 + tint * 60) % 256,
                (70 + y * 2 + tint * 25) % 256,
                (130 + (x + y) + tint * 90) % 256,
            )
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class FakeBridge:
    def __init__(self):
        self.files: dict[str, tuple] = {}

    def save_parquet(self, path, rows, cols, vals):
        self.files[str(path)] = (list(rows), list(cols), list(vals))
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).touch()

    def load_parquet(self, path):
        if str(path) not in self.files:
            raise FileNotFoundError(path)
        return self.files[str(path)]

    def build_triples(self, rows, cols, vals):
        return (rows, cols, vals)


class CountingSam3:
    """Records every call, so "SAM3 was never touched" is asserted against
    the fake rather than assumed from reading the code."""

    def __init__(self, size=IMG):
        self.size = size
        self.geo_rects: list[tuple] = []
        self.calls: list[str] = []

    def _state(self, rects):
        return {
            "masks": [self._rect_mask(r) for r in rects],
            "boxes": [[float(x0), float(y0), float(x1), float(y1)]
                      for (y0, y1, x0, x1) in rects],
            "scores": [0.9] * len(rects),
            "original_width": self.size,
            "original_height": self.size,
        }

    def _rect_mask(self, rect):
        y0, y1, x0, x1 = rect
        m = np.zeros((self.size, self.size), dtype=np.float32)
        m[y0:y1, x0:x1] = 1.0
        return m

    def set_image(self, image):
        self.calls.append("set_image")
        return self._state([])

    def set_text_prompt(self, prompt, state):
        self.calls.append("set_text_prompt")
        return self._state([])

    def add_geometric_prompt(self, box, label, state):
        self.calls.append("add_geometric_prompt")
        return self._state(self.geo_rects)

    def add_point_prompt(self, point, label, state):
        self.calls.append("add_point_prompt")
        return self._state(self.geo_rects)


@pytest.fixture()
def bridge(tmp_path, monkeypatch):
    fake = FakeBridge()
    monkeypatch.setattr(aa_persistence, "bridge", fake)
    monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf")
    return fake


@pytest.fixture()
def service(bridge, tmp_path):
    return SegmentationService(tmp_path / "sessions", CountingSam3())


def _upload(service, contents=None):
    contents = contents or _colour_png()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    state = service.processor.set_image(image)
    service.register_session_data(SID, {
        "name": "panels page", "description": "", "image_url": "",
        "state": state, "original_image_bytes": contents,
        "image_size": image.size,
    })
    return service.sf_sessions[SID]


def _restart(service):
    service.sessions.clear()
    service.sf_sessions.clear()


def _norm_box(rect):
    y0, y1, x0, x1 = rect
    return [((x0 + x1) / 2) / IMG, ((y0 + y1) / 2) / IMG,
            (x1 - x0) / IMG, (y1 - y0) / IMG]


# ── Panels are not objects ───────────────────────────────────────────────────


class TestDrawingAPanelCreatesNoMask:
    def test_adding_a_panel_creates_no_mask_record(self, service):
        sf = _upload(service)
        assert sf.masks == []

        sf.add_panel(10, 10, 90, 90)

        assert sf.masks == [], "a panel must never become a MaskRecord"
        assert len(sf.panels) == 1

    def test_adding_a_panel_never_calls_sam3(self, service):
        sf = _upload(service)
        service.processor.calls.clear()

        sf.add_panel(10, 10, 90, 90)
        sf.add_panel(100, 100, 180, 180)
        sf.delete_panel(sf.panels[0].panel_id)

        assert service.processor.calls == [], (
            "panel authoring is pure geometry capture — it must not reach "
            "the segmentor at all"
        )

    def test_results_are_unchanged_by_a_panel(self, service):
        from services import serialize_sf_masks, serialize_sf_masks_all_passes
        sf = _upload(service)
        before = serialize_sf_masks(sf, sf.state)
        before_all = serialize_sf_masks_all_passes(sf)

        sf.add_panel(10, 10, 90, 90)

        assert serialize_sf_masks(sf, sf.state) == before
        assert serialize_sf_masks_all_passes(sf) == before_all

    def test_a_panel_does_not_disturb_an_existing_selection(self, service):
        sf = _upload(service)
        service.processor.geo_rects = [(20, 60, 20, 60)]
        rec = sf.add_box_selection(_norm_box((20, 60, 20, 60)), True)[0]
        service.processor.calls.clear()

        before = sf.get_mask(rec.mask_id)

        sf.add_panel(0, 0, 100, 100)

        after = sf.get_mask(rec.mask_id)
        assert [m.mask_id for m in sf.masks] == [rec.mask_id]
        # Every axis the selection pipeline owns, untouched. (Auto-hold on
        # selection is the /segment/* endpoint's doing, not the engine's, so
        # `held` is False here either way — compared rather than asserted
        # absolutely so this stays a "nothing changed" test.)
        assert (after.pass_, after.dataset_status, after.held, after.caption) == (
            before.pass_, before.dataset_status, before.held, before.caption
        )
        assert service.processor.calls == []

    def test_panels_have_no_pass_number(self, service):
        """Passes preserve dimensions, so a panel is valid against all of
        them — a pass field could only ever go stale."""
        sf = _upload(service)
        panel = sf.add_panel(10, 10, 90, 90)
        assert not hasattr(panel, "pass_")


# ── Validation ───────────────────────────────────────────────────────────────


class TestPanelValidation:
    def test_inverted_coordinates_are_normalised_not_rejected(self, service):
        sf = _upload(service)
        panel = sf.add_panel(90, 90, 10, 10)
        assert panel.as_box() == (10, 10, 90, 90)

    def test_a_box_below_the_minimum_side_is_refused(self, service):
        sf = _upload(service)
        with pytest.raises(ValueError, match="at least"):
            sf.add_panel(10, 10, 10 + sfe.MIN_PANEL_SIDE - 1, 90)
        assert sf.panels == []

    def test_a_degenerate_box_is_refused(self, service):
        sf = _upload(service)
        with pytest.raises(ValueError):
            sf.add_panel(50, 50, 50, 50)
        assert sf.panels == []

    def test_coordinates_are_clamped_to_the_image(self, service):
        sf = _upload(service)
        panel = sf.add_panel(-40, -40, IMG + 500, IMG + 500)
        assert panel.as_box() == (0, 0, IMG, IMG)

    def test_a_box_entirely_outside_the_image_is_refused_after_clamping(self, service):
        """Clamping collapses it to zero area, which must refuse rather than
        silently record a sliver on the edge."""
        sf = _upload(service)
        with pytest.raises(ValueError):
            sf.add_panel(IMG + 10, IMG + 10, IMG + 90, IMG + 90)

    def test_deleting_an_unknown_panel_raises(self, service):
        sf = _upload(service)
        with pytest.raises(sfe.UnknownPanelError):
            sf.delete_panel("no-such-panel")

    def test_delete_removes_only_the_named_panel(self, service):
        sf = _upload(service)
        a = sf.add_panel(0, 0, 50, 50)
        b = sf.add_panel(60, 60, 120, 120)
        c = sf.add_panel(130, 130, 190, 190)

        sf.delete_panel(b.panel_id)

        assert [p.panel_id for p in sf.panels] == [a.panel_id, c.panel_id]


# ── assign_panel (pure) ──────────────────────────────────────────────────────


class TestAssignPanel:
    HOST = sfe.PanelRecord("host", 0, 0, 100, 100)
    INSET = sfe.PanelRecord("inset", 10, 10, 50, 50)
    LEFT = sfe.PanelRecord("left", 0, 0, 50, 100)
    RIGHT = sfe.PanelRecord("right", 50, 0, 100, 100)

    def test_largest_share_wins(self):
        # 15 of 20 px of width fall in LEFT.
        got = sfe.assign_panel([35, 0, 55, 10], [self.LEFT, self.RIGHT])
        assert got is self.LEFT

    def test_largest_share_wins_the_other_way(self):
        got = sfe.assign_panel([45, 0, 65, 10], [self.LEFT, self.RIGHT])
        assert got is self.RIGHT

    def test_nested_smallest_containing_panel_wins(self):
        got = sfe.assign_panel([20, 20, 30, 30], [self.HOST, self.INSET])
        assert got is self.INSET

    def test_nesting_is_order_independent(self):
        """A list-order-dependent answer would make the result depend on
        authoring order, which the user never sees."""
        got = sfe.assign_panel([20, 20, 30, 30], [self.INSET, self.HOST])
        assert got is self.INSET

    def test_straddling_two_panels_picks_the_bigger_share(self):
        got = sfe.assign_panel([40, 0, 90, 100], [self.LEFT, self.RIGHT])
        assert got is self.RIGHT  # 40px in RIGHT vs 10px in LEFT

    def test_no_overlap_returns_none(self):
        assert sfe.assign_panel([150, 150, 180, 180], [self.HOST]) is None

    def test_no_panels_returns_none(self):
        assert sfe.assign_panel([1, 1, 9, 9], []) is None

    def test_degenerate_bbox_falls_back_to_containment(self):
        got = sfe.assign_panel([20, 20, 20, 20], [self.HOST, self.INSET])
        assert got is self.INSET

    def test_degenerate_bbox_outside_every_panel_returns_none(self):
        assert sfe.assign_panel([150, 150, 150, 150], [self.HOST]) is None

    def test_a_mask_touching_only_an_edge_does_not_count_as_overlap(self):
        """Half-open bounds: sharing an edge is zero area, not a match."""
        assert sfe.assign_panel([100, 0, 120, 20], [self.HOST]) is None


# ── Persistence and the REAL resume sequence ─────────────────────────────────


def _panel_row_cols(bridge, session_id=SID):
    """Every column written on the `_panels` sentinel row, or None if the row
    was not written at all. Absent-means-none is the contract, so the two
    cases must stay distinguishable here."""
    for path, (rows, cols, vals) in bridge.files.items():
        if not path.endswith("segment.parquet"):
            continue
        target = f"{session_id}:{aa_persistence._PANELS_TARGET}"
        found = {c: v for r, c, v in zip(rows, cols, vals) if r == target}
        return found or None
    return None


class TestPanelPersistence:
    def test_the_row_format_is_one_row_keyed_by_index_and_id(self, service, bridge):
        sf = _upload(service)
        a = sf.add_panel(10, 20, 110, 120)
        b = sf.add_panel(120, 130, 190, 195)
        service.save_session_to_disk(SID)

        cols = _panel_row_cols(bridge)
        assert cols is not None
        assert set(cols) == {f"0000:{a.panel_id}", f"0001:{b.panel_id}"}
        assert json.loads(cols[f"0000:{a.panel_id}"]) == [10, 20, 110, 120]
        assert json.loads(cols[f"0001:{b.panel_id}"]) == [120, 130, 190, 195]

    def test_panels_survive_save_restart_and_reconstruction(self, service):
        sf = _upload(service)
        a = sf.add_panel(10, 20, 110, 120)
        b = sf.add_panel(120, 130, 190, 195)
        service.save_session_to_disk(SID)
        _restart(service)

        reloaded = service.get_session(SID)
        assert reloaded is not None
        restored = service.sf_sessions[SID]
        assert [(p.panel_id, p.as_box()) for p in restored.panels] == [
            (a.panel_id, (10, 20, 110, 120)),
            (b.panel_id, (120, 130, 190, 195)),
        ]

    def test_authoring_order_survives_the_round_trip(self, service):
        """D4M.jl sorts columns, so without the index prefix the list would
        come back shuffled into uuid order."""
        sf = _upload(service)
        drawn = [sf.add_panel(i * 20, 0, i * 20 + 18, 40) for i in range(8)]
        service.save_session_to_disk(SID)
        _restart(service)
        service.get_session(SID)

        assert [p.panel_id for p in service.sf_sessions[SID].panels] == [
            p.panel_id for p in drawn
        ]

    def test_deleting_the_last_panel_removes_the_row_entirely(self, service, bridge):
        sf = _upload(service)
        panel = sf.add_panel(10, 10, 90, 90)
        service.save_session_to_disk(SID)
        assert _panel_row_cols(bridge) is not None

        sf.delete_panel(panel.panel_id)
        service.save_session_to_disk(SID)

        assert _panel_row_cols(bridge) is None, (
            "absent row means no panels — an empty row must never be written"
        )

    def test_a_session_that_never_had_panels_writes_no_row(self, service, bridge):
        _upload(service)
        service.save_session_to_disk(SID)
        assert _panel_row_cols(bridge) is None

    def test_deleting_the_last_panel_restores_as_none(self, service):
        sf = _upload(service)
        panel = sf.add_panel(10, 10, 90, 90)
        service.save_session_to_disk(SID)
        sf.delete_panel(panel.panel_id)
        service.save_session_to_disk(SID)
        _restart(service)

        service.get_session(SID)
        assert service.sf_sessions[SID].panels == []

    def test_a_panels_only_session_with_no_masks_round_trips(self, service):
        """Panels must not depend on a mask existing to be saved or read —
        segment.parquet otherwise has only sentinel rows."""
        sf = _upload(service)
        assert sf.masks == []
        panel = sf.add_panel(25, 35, 125, 135)
        service.save_session_to_disk(SID)
        _restart(service)

        response = service.load_session_from_disk(SID)

        assert response["panels"] == [
            {"panel_id": panel.panel_id, "x0": 25, "y0": 35, "x1": 125, "y1": 135}
        ]
        assert service.sf_sessions[SID].masks == []
        assert response["results"]["mask_ids"] == []


class TestPanelsThroughTheRealResumeSequence:
    """Not just get_session: the actual order the frontend uses — GET
    /loadSession, then the follow-up /upload that re-registers the session's
    image (main.dart's `_loadImageFromUrl`). That second call is what wiped
    sessions before register_session_data's restore-before-merge guard
    existed, so panels have to survive BOTH steps, in that order."""

    def test_panels_survive_loadsession_then_the_upload_follow_up(self, service):
        sf = _upload(service)
        panel = sf.add_panel(30, 40, 150, 160)
        contents = sf.original.bytes
        service.save_session_to_disk(SID)
        _restart(service)

        # 1. GET /loadSession
        response = service.load_session_from_disk(SID)
        assert response["panels"] == [
            {"panel_id": panel.panel_id, "x0": 30, "y0": 40, "x1": 150, "y1": 160}
        ]

        # 2. the frontend's follow-up /upload for the same session
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        service.register_session_data(SID, {
            "name": "panels page", "description": "", "image_url": "",
            "state": service.processor.set_image(image),
            "original_image_bytes": contents, "image_size": image.size,
        })

        assert [p.as_box() for p in service.sf_sessions[SID].panels] == [
            (30, 40, 150, 160)
        ], "the /upload follow-up must not replace the restored SFSession"

    def test_the_upload_follow_up_alone_cannot_wipe_panels(self, service):
        """The dangerous ordering: /upload reaching a cold cache for a
        session that is genuinely on disk. register_session_data warms from
        disk FIRST, so the panels must come back rather than be blanked."""
        sf = _upload(service)
        sf.add_panel(30, 40, 150, 160)
        contents = sf.original.bytes
        service.save_session_to_disk(SID)
        _restart(service)  # cold cache, no /loadSession first

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        service.register_session_data(SID, {
            "name": "panels page", "description": "", "image_url": "",
            "state": service.processor.set_image(image),
            "original_image_bytes": contents, "image_size": image.size,
        })

        assert [p.as_box() for p in service.sf_sessions[SID].panels] == [
            (30, 40, 150, 160)
        ]

    def test_a_broken_session_refuses_rather_than_blanking_its_panels(
        self, service, monkeypatch
    ):
        """The other guard: if reconstruction THREW, register_session_data
        must raise instead of merging onto a blank session — which would
        drop the panels on disk."""
        from services import SessionLoadError
        sf = _upload(service)
        sf.add_panel(30, 40, 150, 160)
        contents = sf.original.bytes
        service.save_session_to_disk(SID)
        _restart(service)

        def boom(image):
            raise ValueError("simulated reconstruction failure")

        monkeypatch.setattr(service.processor, "set_image", boom)
        service.get_session(SID)
        assert SID in service._broken_sessions

        with pytest.raises(SessionLoadError):
            service.register_session_data(SID, {
                "name": "", "description": "", "image_url": "",
                "state": {}, "original_image_bytes": contents,
                "image_size": (IMG, IMG),
            })

    def test_loadsession_reports_panels_for_a_session_with_masks_too(self, service):
        sf = _upload(service)
        service.processor.geo_rects = [(20, 60, 20, 60)]
        sf.add_box_selection(_norm_box((20, 60, 20, 60)), True)
        panel = sf.add_panel(10, 10, 120, 120)
        service.sessions[SID]["state"] = sf.state
        service.save_session_to_disk(SID)
        _restart(service)

        response = service.load_session_from_disk(SID)

        assert [p["panel_id"] for p in response["panels"]] == [panel.panel_id]
        # Panels live in their own field; the mask ledger is untouched by them.
        assert len(response["all_masks"]["mask_ids"]) == 1
        assert "panel_id" not in response["all_masks"]


# ── Panel crops and export ───────────────────────────────────────────────────


class PaintingInpainter:
    """Repaints the masked region a flat colour, so a pass's image differs
    from the one before it in a way a crop comparison can actually see."""

    def __init__(self, color):
        self.color = color

    def inpaint(self, image, mask):
        out = image.convert("RGBA").copy()
        mask_img = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
        painted = Image.new("RGBA", out.size, self.color)
        out.paste(painted, (0, 0), mask_img)
        return out


def _payload_rows(rows, cols, vals):
    out: dict[str, dict[str, str]] = {}
    for r, c, v in zip(rows, cols, vals):
        out.setdefault(r, {})[c] = v
    return out


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")


class TestPanelCropProvenance:
    def test_panel_crop_comes_from_the_masks_own_pass_image(self, service):
        """Two passes with genuinely different pixels. A keeper captioned in
        pass 0 must carry pass 0's panel context, NOT the current pass's —
        the scrub that produced pass 1 repainted the very area its panel
        covers."""
        sf = _upload(service)
        sf.add_panel(0, 0, 120, 120)

        # A pass-0 selection, captioned, then scrubbed -> produces pass 1.
        service.processor.geo_rects = [(10, 50, 10, 50)]
        early = sf.add_box_selection(_norm_box((10, 50, 10, 50)), True)[0]
        sf.attach_caption(early.mask_id, "a thing in pass 0")
        sf.set_held(early.mask_id, True)
        sf.run_lama_pass(PaintingInpainter(color=(255, 0, 255, 255)))
        assert sf.pass_ == 1

        rows = _payload_rows(*sfe.build_sf_payload(sf))
        exported = rows[f"{SID}:{early.mask_id}"]
        crop = _decode(exported["panel_crop_png_bytes"])

        pass0_expected = _decode(
            base64.b64encode(
                sfe.panel_crop_png_bytes(sf.image_for_pass(0), (0, 0, 120, 120))
            ).decode()
        )
        pass1_other = _decode(
            base64.b64encode(
                sfe.panel_crop_png_bytes(sf.image_for_pass(1), (0, 0, 120, 120))
            ).decode()
        )

        assert list(crop.getdata()) == list(pass0_expected.getdata())
        assert list(crop.getdata()) != list(pass1_other.getdata()), (
            "the two passes must genuinely differ, or this proves nothing"
        )

    def test_a_later_pass_keeper_uses_that_later_passs_panel_image(self, service):
        sf = _upload(service)
        sf.add_panel(0, 0, 120, 120)
        service.processor.geo_rects = [(10, 40, 10, 40)]
        first = sf.add_box_selection(_norm_box((10, 40, 10, 40)), True)[0]
        sf.set_held(first.mask_id, True)
        sf.run_lama_pass(PaintingInpainter(color=(0, 255, 0, 255)))

        service.processor.geo_rects = [(60, 100, 60, 100)]
        later = sf.add_box_selection(_norm_box((60, 100, 60, 100)), True)[0]
        sf.attach_caption(later.mask_id, "kept in pass 1")
        assert later.pass_ == 1

        rows = _payload_rows(*sfe.build_sf_payload(sf))
        crop = _decode(rows[f"{SID}:{later.mask_id}"]["panel_crop_png_bytes"])
        expected = sfe.panel_crop_png_bytes(sf.image_for_pass(1), (0, 0, 120, 120))

        assert list(crop.getdata()) == list(_decode(base64.b64encode(expected).decode()).getdata())

    def test_a_mask_with_no_panel_falls_back_to_the_full_page_and_says_so(self, service):
        sf = _upload(service)
        sf.add_panel(150, 150, 195, 195)  # nowhere near the mask
        service.processor.geo_rects = [(10, 50, 10, 50)]
        rec = sf.add_box_selection(_norm_box((10, 50, 10, 50)), True)[0]
        sf.attach_caption(rec.mask_id, "unpanelled")

        rows = _payload_rows(*sfe.build_sf_payload(sf))
        exported = rows[f"{SID}:{rec.mask_id}"]

        assert exported["panel_source"] == sfe.PANEL_SOURCE_FULL_PAGE
        crop = _decode(exported["panel_crop_png_bytes"])
        assert crop.size == (IMG, IMG), "the fallback is the whole page"

    def test_an_assigned_mask_names_its_panel(self, service):
        sf = _upload(service)
        panel = sf.add_panel(0, 0, 120, 120)
        service.processor.geo_rects = [(10, 50, 10, 50)]
        rec = sf.add_box_selection(_norm_box((10, 50, 10, 50)), True)[0]
        sf.attach_caption(rec.mask_id, "panelled")

        rows = _payload_rows(*sfe.build_sf_payload(sf))
        exported = rows[f"{SID}:{rec.mask_id}"]

        assert exported["panel_source"] == panel.panel_id
        assert _decode(exported["panel_crop_png_bytes"]).size == (120, 120)

    def test_a_session_with_no_panels_at_all_still_exports_the_fallback(self, service):
        """Panels are optional; export must not become conditional on them."""
        sf = _upload(service)
        service.processor.geo_rects = [(10, 50, 10, 50)]
        rec = sf.add_box_selection(_norm_box((10, 50, 10, 50)), True)[0]
        sf.attach_caption(rec.mask_id, "no panels anywhere")

        rows = _payload_rows(*sfe.build_sf_payload(sf))
        exported = rows[f"{SID}:{rec.mask_id}"]

        assert exported["panel_source"] == sfe.PANEL_SOURCE_FULL_PAGE
        assert "panel_crop_png_bytes" in exported

    def test_panel_crop_is_a_rectangle_not_an_alpha_cutout(self, service):
        """Unlike a mask crop, a panel crop has no alpha-masking step — it is
        a layout region, so every pixel in the rectangle is opaque."""
        sf = _upload(service)
        crop = _decode(base64.b64encode(
            sfe.panel_crop_png_bytes(sf.image_for_pass(0), (10, 10, 60, 60))
        ).decode())
        assert crop.size == (50, 50)
        assert all(px[3] == 255 for px in crop.getdata())

    def test_panel_crop_clamps_rather_than_raising(self, service):
        sf = _upload(service)
        crop = _decode(base64.b64encode(
            sfe.panel_crop_png_bytes(sf.image_for_pass(0), (-50, -50, IMG + 50, IMG + 50))
        ).decode())
        assert crop.size == (IMG, IMG)

    def test_an_empty_intersection_yields_a_1x1_rather_than_raising(self, service):
        sf = _upload(service)
        crop = _decode(base64.b64encode(
            sfe.panel_crop_png_bytes(sf.image_for_pass(0), (IMG + 10, IMG + 10, IMG + 20, IMG + 20))
        ).decode())
        assert crop.size == (1, 1)


# ── Endpoints ────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(service, monkeypatch):
    """TestClient without a `with` block so lifespan never runs and no real
    SAM3 model loads — same convention test_source_staleness.py uses."""
    from fastapi.testclient import TestClient
    import main
    monkeypatch.setattr(main, "service", service)
    monkeypatch.setattr(main, "processor", service.processor)
    monkeypatch.setattr(main, "model", object())
    return TestClient(main.app)


class TestPanelEndpoints:
    def test_add_list_and_delete_round_trip(self, client, service):
        _upload(service)

        added = client.post("/panel/add", json={
            "session_id": SID, "x0": 10, "y0": 20, "x1": 110, "y1": 120,
        })
        assert added.status_code == 200
        panel_id = added.json()["panel_id"]
        assert added.json()["panels"] == [
            {"panel_id": panel_id, "x0": 10, "y0": 20, "x1": 110, "y1": 120}
        ]

        listed = client.post("/panel/list", json={"session_id": SID})
        assert listed.status_code == 200
        assert listed.json()["panels"] == added.json()["panels"]

        removed = client.post("/panel/delete", json={
            "session_id": SID, "panel_id": panel_id,
        })
        assert removed.status_code == 200
        assert removed.json()["panels"] == []

    def test_add_rejects_a_box_below_the_minimum(self, client, service):
        _upload(service)
        r = client.post("/panel/add", json={
            "session_id": SID, "x0": 10, "y0": 10, "x1": 12, "y1": 90,
        })
        assert r.status_code == 422

    def test_add_rejects_an_inverted_degenerate_box(self, client, service):
        _upload(service)
        r = client.post("/panel/add", json={
            "session_id": SID, "x0": 50, "y0": 50, "x1": 50, "y1": 50,
        })
        assert r.status_code == 422

    def test_add_clamps_to_the_image(self, client, service):
        _upload(service)
        r = client.post("/panel/add", json={
            "session_id": SID, "x0": -30, "y0": -30, "x1": 900, "y1": 900,
        })
        assert r.status_code == 200
        assert r.json()["panels"][0] == {
            "panel_id": r.json()["panel_id"], "x0": 0, "y0": 0, "x1": IMG, "y1": IMG,
        }

    def test_unknown_session_is_404_on_every_panel_endpoint(self, client):
        for path, body in (
            ("/panel/add", {"session_id": "nope", "x0": 0, "y0": 0, "x1": 50, "y1": 50}),
            ("/panel/delete", {"session_id": "nope", "panel_id": "x"}),
            ("/panel/list", {"session_id": "nope"}),
        ):
            assert client.post(path, json=body).status_code == 404, path

    def test_deleting_an_unknown_panel_is_404(self, client, service):
        _upload(service)
        r = client.post("/panel/delete", json={
            "session_id": SID, "panel_id": "no-such-panel",
        })
        assert r.status_code == 404

    def test_the_add_response_carries_no_mask_fields(self, client, service):
        """Structural: a panel response must not look like a selection
        response, or the frontend could route it into the mask path."""
        _upload(service)
        body = client.post("/panel/add", json={
            "session_id": SID, "x0": 10, "y0": 10, "x1": 90, "y1": 90,
        }).json()
        assert set(body) == {"session_id", "panel_id", "panels"}
        assert "results" not in body and "all_masks" not in body

    def test_adding_a_panel_via_the_endpoint_calls_no_sam3(self, client, service):
        _upload(service)
        service.processor.calls.clear()
        client.post("/panel/add", json={
            "session_id": SID, "x0": 10, "y0": 10, "x1": 90, "y1": 90,
        })
        assert service.processor.calls == []
