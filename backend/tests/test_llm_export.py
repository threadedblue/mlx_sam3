"""The read-only LLM export (v1) — POST /export/llm-payload.

A SEPARATE export path alongside build_sf_payload, not a replacement:
build_sf_payload's own output and tests are unchanged (confirmed by a
dedicated test here that imports and re-runs its own byte-for-byte
expectations unperturbed). This one exists for a different consumer (a DN
LLM node, via vision-API submissions) with different needs: a shared
macro-context page (Artifact 1, frozen to a stable file path), a panel crop
cut from THAT page rather than the mask's own pass (Artifact 2), and a
TIGHT, alpha-free segment crop composited onto mid-grey (Artifact 3) instead
of build_sf_payload's full-canvas RGBA crop.

Fixtures are COLOURED and NON-UNIFORM throughout, and masks are deliberately
NON-RECTANGULAR (an L-shape) where the grey-compositing property is under
test — a solid rectangular mask fills its own bounding box exactly, so
"outside mask but inside bbox" would never be exercised by one.

The bridge is stubbed rather than hitting real Julia parquet I/O —
test_sf_wiring.py, which does, hangs in this environment.
"""

from __future__ import annotations

import base64
import io
import pathlib

import numpy as np
import pytest
from PIL import Image

import aa_persistence
import sf_engine as sfe
from services import SegmentationService

SID = "llm-export-session"
IMG = 200


def _colour_png(size=IMG, tint=0) -> bytes:
    """Non-uniform, non-black, and distinguishable per `tint` — so a crop
    taken from the wrong pass or wrong page is visibly the wrong bytes."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            arr[y, x] = ((20 + x * 3 + tint * 60) % 256,
                         (70 + y * 2 + tint * 25) % 256,
                         (130 + (x + y) + tint * 90) % 256)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _colour_png_rect(w: int, h: int, tint=0) -> bytes:
    """Same formula as `_colour_png`, but non-square — for the page-
    downscale tests, which need something bigger than 1536 on one side."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x] = ((20 + x * 3 + tint * 60) % 256,
                         (70 + y * 2 + tint * 25) % 256,
                         (130 + (x + y) + tint * 90) % 256)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class FakeBridge:
    def __init__(self):
        self.files: dict[str, tuple] = {}
        # Counts calls, not just content -- a write that re-persists
        # UNCHANGED content is invisible to a before/after bytes diff
        # (idempotent), but it is still a write that must not happen during
        # a read-only export. See TestReadOnly's dedicated test.
        self.save_parquet_calls = 0

    def save_parquet(self, path, rows, cols, vals):
        self.save_parquet_calls += 1
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
    """Records every call, so "never reaches SAM3" is asserted against the
    fake rather than inferred from reading the code. `geo_rects` describes
    an L-shape (list of (y0,y1,x0,x1) bars unioned together) when
    `l_shaped=True` is set on the next selection, otherwise a solid box."""

    def __init__(self, size=IMG):
        self.size = size
        self.geo_rects: list[tuple] = []
        self.l_shaped = False
        self.calls: list[str] = []

    def _state(self, rects):
        return {"masks": [self._mask_for(r) for r in rects],
                "boxes": [[float(x0), float(y0), float(x1), float(y1)]
                          for (y0, y1, x0, x1) in rects],
                "scores": [0.9] * len(rects),
                "original_width": self.size, "original_height": self.size}

    def _mask_for(self, rect):
        y0, y1, x0, x1 = rect
        m = np.zeros((self.size, self.size), dtype=np.float32)
        if self.l_shaped:
            # Left bar + bottom bar: an L within the same bounding box the
            # solid rect would have, so bbox-but-outside-mask pixels exist.
            bar = max(1, (x1 - x0) // 3)
            foot = max(1, (y1 - y0) // 3)
            m[y0:y1, x0:x0 + bar] = 1.0
            m[y1 - foot:y1, x0:x1] = 1.0
        else:
            m[y0:y1, x0:x1] = 1.0
        return m

    def set_image(self, image):
        self.calls.append("set_image"); return self._state([])

    def set_text_prompt(self, prompt, state):
        self.calls.append("set_text_prompt"); return self._state([])

    def add_geometric_prompt(self, box, label, state):
        self.calls.append("add_geometric_prompt"); return self._state(self.geo_rects)

    def add_point_prompt(self, point, label, state):
        self.calls.append("add_point_prompt"); return self._state(self.geo_rects)


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
        "name": "llm export page", "description": "", "image_url": "",
        "state": state, "original_image_bytes": contents,
        "image_size": image.size,
    })
    return service.sf_sessions[SID]


def _restart(service):
    service.sessions.clear()
    service.sf_sessions.clear()


def _norm_box(rect, size=IMG):
    y0, y1, x0, x1 = rect
    return [((x0 + x1) / 2) / size, ((y0 + y1) / 2) / size,
            (x1 - x0) / size, (y1 - y0) / size]


def _select(service, sf, rect, l_shaped=False, **kw):
    service.processor.geo_rects = [rect]
    service.processor.l_shaped = l_shaped
    rec = sf.add_box_selection(_norm_box(rect), True, **kw)[0]
    service.processor.l_shaped = False
    return rec


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


def _decode(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))


# ── resolve_artifact1_pass (pure) ────────────────────────────────────────────


class TestResolveArtifact1Pass:
    def test_none_falls_back_to_pass_0_original(self):
        assert sfe.resolve_artifact1_pass(None) == (0, sfe.PAGE_RULE_ORIGINAL)

    def test_wb_scrub_pass_1(self):
        assert sfe.resolve_artifact1_pass(1) == (1, sfe.PAGE_RULE_WB_SCRUB)

    def test_wb_scrub_pass_2_balloon_scrub_ran_after_an_ordinary_round(self):
        assert sfe.resolve_artifact1_pass(2) == (2, sfe.PAGE_RULE_WB_SCRUB)


# ── segment_crop_png_bytes: tight, grey, RGB (pure) ──────────────────────────


class TestSegmentCropTightAndGrey:
    def test_dimensions_equal_the_bbox(self):
        img = Image.new("RGBA", (IMG, IMG), (10, 20, 30, 255))
        mask = np.zeros((IMG, IMG), dtype=np.uint8)
        mask[10:20, 5:8] = 1
        mask[17:20, 5:15] = 1  # an L, bbox (5,10,15,20) -> 10x10
        x0, y0, x1, y1 = sfe._geometry_bbox(mask)

        crop = _decode(sfe.segment_crop_png_bytes(img, mask))

        assert crop.size == (x1 - x0, y1 - y0) == (10, 10)

    def test_outside_mask_pixels_are_exactly_mid_grey(self):
        img = Image.new("RGBA", (IMG, IMG), (10, 20, 30, 255))
        mask = np.zeros((IMG, IMG), dtype=np.uint8)
        mask[10:20, 5:8] = 1
        mask[17:20, 5:15] = 1  # L-shape: top-right of its own bbox is empty

        crop = np.array(_decode(sfe.segment_crop_png_bytes(img, mask)))

        assert tuple(crop[0, 9]) == (128, 128, 128), "outside the L, inside the bbox"

    def test_mode_is_rgb_no_alpha_channel(self):
        img = Image.new("RGBA", (IMG, IMG), (10, 20, 30, 255))
        mask = np.zeros((IMG, IMG), dtype=np.uint8)
        mask[10:20, 5:15] = 1

        crop = _decode(sfe.segment_crop_png_bytes(img, mask))

        assert crop.mode == "RGB"

    def test_subject_pixels_are_unchanged(self):
        img = Image.open(io.BytesIO(_colour_png())).convert("RGBA")
        mask = np.zeros((IMG, IMG), dtype=np.uint8)
        mask[10:20, 5:8] = 1
        mask[17:20, 5:15] = 1

        crop = np.array(_decode(sfe.segment_crop_png_bytes(img, mask)))
        src = np.array(img.convert("RGB"))

        # Inside the left bar of the L (rows 10-19, cols 5-7): the crop's
        # own-coordinate pixel must equal the source image's pixel there,
        # byte for byte — no blending, no recolouring.
        for ry in range(10):   # crop row = source row - y0(10)
            assert tuple(crop[ry, 0]) == tuple(src[10 + ry, 5])

    def test_an_all_zero_mask_returns_a_1x1_grey_pixel_not_a_crash(self):
        img = Image.new("RGBA", (IMG, IMG), (10, 20, 30, 255))
        mask = np.zeros((IMG, IMG), dtype=np.uint8)

        crop = _decode(sfe.segment_crop_png_bytes(img, mask))

        assert crop.size == (1, 1)
        assert tuple(np.array(crop)[0, 0]) == (128, 128, 128)


# ── Pass-aware sourcing: segment from its own pass, panel from Artifact 1 ────


class TestPassAwareSourcing:
    def test_a_mask_kept_in_pass_2_is_cut_from_pass_2_while_panel_comes_from_artifact1(
        self, service
    ):
        """Two passes with genuinely different pixels, and no balloon scrub
        at all (wb_scrub_pass stays None -> Artifact 1 is pass 0). The kept
        mask's OWN region is repainted during pass0->1 and never touched
        again, so pass 2's pixels there differ from pass 0's — the only way
        to prove which pass a crop actually came from."""
        sf = _upload(service)
        panel = sf.add_panel(0, 0, 120, 120)

        REPAINTED = (10, 50, 10, 50)
        first = _select(service, sf, REPAINTED)
        sf.set_held(first.mask_id, True)
        sf.run_lama_pass(PaintingInpainter(color=(255, 0, 255, 255)))  # pass0->1

        ELSEWHERE = (130, 170, 130, 170)
        second = _select(service, sf, ELSEWHERE)
        sf.set_held(second.mask_id, True)
        sf.run_lama_pass(PaintingInpainter(color=(0, 255, 0, 255)))  # pass1->2

        assert sf.pass_ == 2
        assert sf.wb_scrub_pass is None

        # Select the SAME rect again at pass 2 (now magenta) and keep it.
        kept = _select(service, sf, REPAINTED)
        sf.attach_caption(kept.mask_id, "kept at pass 2")
        assert kept.pass_ == 2

        payload = sfe.build_llm_export_payload(sf, include="keep")
        seg = next(s for s in payload["segments"] if s["mask_id"] == kept.mask_id)
        assert seg["pass"] == 2
        assert seg["panel_id"] == panel.panel_id

        kept_geometry = sf.get_mask(kept.mask_id).geometry
        own_pass_crop = sfe.segment_crop_png_bytes(sf.image_for_pass(2), kept_geometry)
        wrong_pass_crop = sfe.segment_crop_png_bytes(sf.image_for_pass(0), kept_geometry)

        assert seg["crop_png_bytes"] == own_pass_crop
        assert seg["crop_png_bytes"] != wrong_pass_crop, (
            "the two passes must genuinely differ at this mask's own region, "
            "or this proves nothing"
        )

        panel_row = next(p for p in payload["panels"] if p["panel_id"] == panel.panel_id)
        from_artifact1 = sfe._panel_crop_jpeg_bytes(
            sf.image_for_pass(0).convert("RGB"), panel.as_box()
        )
        from_masks_pass = sfe._panel_crop_jpeg_bytes(
            sf.image_for_pass(2).convert("RGB"), panel.as_box()
        )
        assert panel_row["crop_jpeg_bytes"] == from_artifact1
        assert panel_row["crop_jpeg_bytes"] != from_masks_pass, (
            "the panel region overlaps the repainted area, so pass 0 and "
            "pass 2 must differ there too — otherwise this proves nothing"
        )


# ── No panel contains the mask ────────────────────────────────────────────────


class TestNoPanelFallback:
    def test_panel_id_is_null_and_no_panel_crop_is_emitted(self, service):
        sf = _upload(service)
        sf.add_panel(150, 150, 195, 195)  # nowhere near the mask
        rec = _select(service, sf, (10, 50, 10, 50))
        sf.attach_caption(rec.mask_id, "unpanelled")

        payload = sfe.build_llm_export_payload(sf, include="keep")

        seg = payload["segments"][0]
        assert seg["panel_id"] is None
        assert payload["panels"] == []

    def test_a_session_with_no_panels_at_all_still_exports_the_segment(self, service):
        sf = _upload(service)
        rec = _select(service, sf, (10, 50, 10, 50))
        sf.attach_caption(rec.mask_id, "no panels anywhere")

        payload = sfe.build_llm_export_payload(sf, include="keep")

        assert payload["segments"][0]["panel_id"] is None
        assert payload["panels"] == []

    def test_a_panel_referenced_by_two_masks_is_cropped_only_once(self, service):
        """'once per panel' — not once per mask that happens to sit inside
        it. The dict-dedup is the whole point; this pins it against a
        future change that re-crops per row."""
        sf = _upload(service)
        panel = sf.add_panel(0, 0, 120, 120)
        a = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(a.mask_id, "first")
        b = _select(service, sf, (60, 90, 60, 90))
        sf.attach_caption(b.mask_id, "second")

        payload = sfe.build_llm_export_payload(sf, include="keep")

        assert len(payload["segments"]) == 2
        assert all(s["panel_id"] == panel.panel_id for s in payload["segments"])
        assert len(payload["panels"]) == 1, "one panel row, however many masks reference it"


# ── include="keep" vs "held_or_keep", pre_wb, stable ids ─────────────────────


class TestIncludeFilterAndPreWb:
    def test_keep_only_excludes_held_but_uncaptioned(self, service):
        sf = _upload(service)
        kept = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(kept.mask_id, "captioned")
        held_only = _select(service, sf, (60, 90, 60, 90))
        sf.set_held(held_only.mask_id, True)
        untouched = _select(service, sf, (110, 140, 110, 140))

        payload = sfe.build_llm_export_payload(sf, include="keep")

        ids = {s["mask_id"] for s in payload["segments"]}
        assert ids == {kept.mask_id}

    def test_held_or_keep_includes_both(self, service):
        sf = _upload(service)
        kept = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(kept.mask_id, "captioned")
        held_only = _select(service, sf, (60, 90, 60, 90))
        sf.set_held(held_only.mask_id, True)
        untouched = _select(service, sf, (110, 140, 110, 140))

        payload = sfe.build_llm_export_payload(sf, include="held_or_keep")

        ids = {s["mask_id"] for s in payload["segments"]}
        assert ids == {kept.mask_id, held_only.mask_id}
        assert untouched.mask_id not in ids

    def test_an_unknown_include_value_raises(self, service):
        sf = _upload(service)
        with pytest.raises(ValueError, match="include"):
            sfe.build_llm_export_payload(sf, include="everything")

    def test_caption_defaults_to_empty_string_not_none(self, service):
        """held_or_keep can include a record with dataset_status still
        UNASSIGNED and caption=None — the wire shape promises a string."""
        sf = _upload(service)
        held_only = _select(service, sf, (60, 90, 60, 90))
        sf.set_held(held_only.mask_id, True)

        payload = sfe.build_llm_export_payload(sf, include="held_or_keep")

        assert payload["segments"][0]["caption"] == ""

    def test_pre_wb_true_before_the_wb_scrub_pass_false_at_or_after(self, service):
        sf = _upload(service)

        # An ordinary (non-balloon) scrub first: pass0 -> pass1.
        early = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(early.mask_id, "captioned before any scrub")
        ordinary = _select(service, sf, (40, 60, 40, 60))
        sf.set_held(ordinary.mask_id, True)
        sf.run_lama_pass(PaintingInpainter(color=(200, 0, 0, 255)))
        assert sf.pass_ == 1

        # A balloon scrub next: pass1 -> pass2, wb_scrub_pass becomes 2.
        balloon = _select(service, sf, (70, 90, 70, 90), source=sfe.SOURCE_BALLOON)
        sf.set_held(balloon.mask_id, True)
        sf.run_lama_pass(PaintingInpainter(color=(0, 200, 0, 255)))
        assert sf.pass_ == 2
        assert sf.wb_scrub_pass == 2

        at_wb = _select(service, sf, (110, 130, 110, 130))
        sf.attach_caption(at_wb.mask_id, "captioned at the wb-scrub pass")

        payload = sfe.build_llm_export_payload(sf, include="keep")
        by_id = {s["mask_id"]: s for s in payload["segments"]}

        assert by_id[early.mask_id]["pass"] == 0
        assert by_id[early.mask_id]["pre_wb"] is True
        assert by_id[at_wb.mask_id]["pass"] == 2
        assert by_id[at_wb.mask_id]["pre_wb"] is False, "pass 2 is NOT below wb_scrub_pass 2"

    def test_no_wb_scrub_at_all_means_pre_wb_is_always_false(self, service):
        sf = _upload(service)
        rec = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(rec.mask_id, "no balloon scrub ever ran")

        payload = sfe.build_llm_export_payload(sf, include="keep")

        assert payload["segments"][0]["pre_wb"] is False


class TestIdsStable:
    def test_mask_ids_are_unaffected_by_list_reordering(self, service):
        sf = _upload(service)
        a = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(a.mask_id, "a")
        b = _select(service, sf, (60, 90, 60, 90))
        sf.attach_caption(b.mask_id, "b")

        before = {s["mask_id"] for s in sfe.build_llm_export_payload(sf, include="keep")["segments"]}

        sf.masks = list(reversed(sf.masks))
        after = {s["mask_id"] for s in sfe.build_llm_export_payload(sf, include="keep")["segments"]}

        assert before == after == {a.mask_id, b.mask_id}

    def test_mask_ids_survive_a_save_restart_reload_round_trip(self, service):
        sf = _upload(service)
        rec = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(rec.mask_id, "stable across resume")
        original_id = rec.mask_id

        service.sessions[SID]["state"] = sf.state
        service.save_session_to_disk(SID)
        _restart(service)
        service.load_session_from_disk(SID)  # the real frontend's own first call

        reloaded = service.sf_sessions[SID]
        payload = sfe.build_llm_export_payload(reloaded, include="keep")

        assert payload["segments"][0]["mask_id"] == original_id


# ── Page downscale ────────────────────────────────────────────────────────────


class TestPageDownscale:
    # No mask is selected in these: the page/Artifact-1 block is computed
    # unconditionally before build_llm_export_payload's mask loop even
    # starts, so an empty `segments: []` exercises it just as well — and
    # sidesteps CountingSam3's fixed 200x200 mask-array size, which has no
    # way to match a deliberately oversized/non-square page image.

    def test_an_oversized_page_is_capped_to_1536_on_its_long_side(self, service):
        contents = _colour_png_rect(2000, 1000)
        sf = _upload(service, contents)

        payload = sfe.build_llm_export_payload(sf, include="keep")
        page = _decode(payload["page_jpeg_bytes"])

        assert max(page.size) <= 1536
        # Native width/height, reported separately from the downscaled file,
        # stay the ORIGINAL dimensions -- bbox/panel coordinates are native.
        assert payload["width"] == 2000 and payload["height"] == 1000
        # Aspect ratio preserved (within rounding).
        assert abs(page.size[0] / page.size[1] - 2000 / 1000) < 0.01

    def test_a_page_already_under_the_cap_is_not_upscaled(self, service):
        sf = _upload(service)  # 200x200
        payload = sfe.build_llm_export_payload(sf, include="keep")
        page = _decode(payload["page_jpeg_bytes"])

        assert page.size == (IMG, IMG)

    def test_the_page_jpeg_has_no_alpha_channel(self, service):
        sf = _upload(service)
        payload = sfe.build_llm_export_payload(sf, include="keep")

        assert _decode(payload["page_jpeg_bytes"]).mode == "RGB"


# ── Read-only: nothing in the registry or parquet changes ────────────────────


class TestReadOnly:
    def test_build_sf_payload_is_unperturbed_by_this_module_existing(self, service):
        """The scope limit, pinned directly: build_sf_payload's OWN output
        for a captioned mask is identical to what it always was — no new
        columns, no altered crop bytes."""
        sf = _upload(service)
        rec = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(rec.mask_id, "unchanged by the new export path")

        rows, cols, vals = sfe.build_sf_payload(sf)
        by_row: dict[str, dict[str, object]] = {}
        for r, c, v in zip(rows, cols, vals):
            by_row.setdefault(r, {})[c] = v
        exported = by_row[f"{SID}:{rec.mask_id}"]

        assert set(exported) == {
            "pass", "caption", "crop_png_bytes", "panel_crop_png_bytes", "panel_source",
        }
        assert exported["panel_source"] == sfe.PANEL_SOURCE_FULL_PAGE

    def test_two_exports_leave_registry_and_parquet_bytes_identical(self, service, bridge):
        sf = _upload(service)
        sf.add_panel(0, 0, 100, 100)
        rec = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(rec.mask_id, "x")
        service.sessions[SID]["state"] = sf.state
        service.save_session_to_disk(SID)

        snapshot = {
            path: (list(r), list(c), list(v)) for path, (r, c, v) in bridge.files.items()
        }
        calls_before = bridge.save_parquet_calls
        masks_before = list(sf.masks)
        panels_before = list(sf.panels)

        service.export_llm_payload(SID, sf, "keep")
        service.export_llm_payload(SID, sf, "keep")

        assert bridge.files == snapshot, "no parquet write may happen during export"
        assert bridge.save_parquet_calls == calls_before, (
            "not just unchanged CONTENT -- no write call at all. An export "
            "that re-persists identical content is idempotent and invisible "
            "to the snapshot diff above, but it is still a write a read-only "
            "endpoint must never make."
        )
        assert sf.masks == masks_before
        assert sf.panels == panels_before
        assert service.sessions[SID]["name"] == "llm export page"

    def test_export_never_calls_sam3(self, service):
        sf = _upload(service)
        sf.add_panel(0, 0, 100, 100)
        rec = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(rec.mask_id, "x")
        service.processor.calls.clear()

        sfe.build_llm_export_payload(sf, include="keep")

        assert service.processor.calls == []


# ── The HTTP surface ─────────────────────────────────────────────────────────


@pytest.fixture()
def client(service, monkeypatch):
    """TestClient without a `with` block so lifespan never runs and no real
    SAM3 model loads — same convention test_prompt_template.py/
    test_source_staleness.py use."""
    from fastapi.testclient import TestClient
    import main
    monkeypatch.setattr(main, "service", service)
    monkeypatch.setattr(main, "processor", service.processor)
    monkeypatch.setattr(main, "model", object())
    return TestClient(main.app)


class TestExportEndpoint:
    def test_the_response_shape_matches_the_contract(self, service, client):
        sf = _upload(service)
        panel = sf.add_panel(0, 0, 120, 120)
        rec = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(rec.mask_id, "a caption")

        r = client.post("/export/llm-payload", json={"session_id": SID})

        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"schema", "session_id", "page", "panels", "segments"}
        assert body["schema"] == "sf.llm_payload/v1"
        assert set(body["page"]) == {
            "page_b64", "page_path", "page_pass", "page_rule", "width", "height",
        }
        # the page ships as data, and is exactly the bytes written to disk
        assert base64.b64decode(body["page"]["page_b64"]) == pathlib.Path(
            body["page"]["page_path"]
        ).read_bytes()
        assert body["page"]["page_rule"] == "original"
        assert body["page"]["page_pass"] == 0
        seg = body["segments"][0]
        assert set(seg) == {
            "mask_id", "caption", "pass", "bbox", "panel_id", "segment_crop", "pre_wb",
        }
        assert seg["mask_id"] == rec.mask_id
        assert seg["panel_id"] == panel.panel_id
        pr = body["panels"][0]
        assert set(pr) == {"panel_id", "bbox", "panel_crop"}
        # base64-decodable
        base64.b64decode(seg["segment_crop"])
        base64.b64decode(pr["panel_crop"])

    def test_artifact1_is_written_at_the_stable_session_path(self, service, client, tmp_path):
        _upload(service)
        rec_sf = service.sf_sessions[SID]
        rec = _select(service, rec_sf, (10, 30, 10, 30))
        rec_sf.attach_caption(rec.mask_id, "x")

        r = client.post("/export/llm-payload", json={"session_id": SID})

        page_path = pathlib.Path(r.json()["page"]["page_path"])
        assert page_path.name == "artifact1.jpg"
        assert page_path.parent == aa_persistence.session_dir(SID)
        assert page_path.exists()

    def test_overwrites_on_every_call_rather_than_accumulating(self, service, client):
        sf = _upload(service)
        rec = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(rec.mask_id, "x")

        r1 = client.post("/export/llm-payload", json={"session_id": SID})
        page_path = pathlib.Path(r1.json()["page"]["page_path"])
        first_bytes = page_path.read_bytes()
        r2 = client.post("/export/llm-payload", json={"session_id": SID})
        second_bytes = page_path.read_bytes()

        assert first_bytes == second_bytes
        assert r1.json()["page"]["page_path"] == r2.json()["page"]["page_path"]

    def test_a_cold_cache_404s_like_every_other_session_endpoint(self, service, client):
        """Same contract as /mask/hold, /mask/prompt-draft, /panel/*:
        get_or_create_sf_session only reads the in-memory cache, so a
        request before any /loadSession 404s."""
        sf = _upload(service)
        rec = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(rec.mask_id, "x")
        service.sessions[SID]["state"] = sf.state
        service.save_session_to_disk(SID)
        _restart(service)

        r = client.post("/export/llm-payload", json={"session_id": SID})
        assert r.status_code == 404

    def test_it_works_through_the_real_resume_sequence(self, service, client):
        sf = _upload(service)
        rec = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(rec.mask_id, "resumed caption")
        service.sessions[SID]["state"] = sf.state
        service.save_session_to_disk(SID)
        _restart(service)

        service.load_session_from_disk(SID)  # the frontend's own first call
        r = client.post("/export/llm-payload", json={"session_id": SID})

        assert r.status_code == 200
        assert r.json()["segments"][0]["caption"] == "resumed caption"

    def test_unknown_session_is_404(self, client):
        r = client.post("/export/llm-payload", json={"session_id": "no-such-session"})
        assert r.status_code == 404

    def test_an_unknown_include_value_is_422(self, service, client):
        _upload(service)
        r = client.post("/export/llm-payload", json={"session_id": SID, "include": "bogus"})
        assert r.status_code == 422

    def test_include_defaults_to_keep(self, service, client):
        sf = _upload(service)
        held_only = _select(service, sf, (10, 30, 10, 30))
        sf.set_held(held_only.mask_id, True)

        r = client.post("/export/llm-payload", json={"session_id": SID})

        assert r.json()["segments"] == []

    def test_held_or_keep_via_the_endpoint(self, service, client):
        sf = _upload(service)
        held_only = _select(service, sf, (10, 30, 10, 30))
        sf.set_held(held_only.mask_id, True)

        r = client.post("/export/llm-payload", json={"session_id": SID, "include": "held_or_keep"})

        assert [s["mask_id"] for s in r.json()["segments"]] == [held_only.mask_id]

    def test_the_endpoint_never_calls_sam3(self, service, client):
        sf = _upload(service)
        sf.add_panel(0, 0, 100, 100)
        rec = _select(service, sf, (10, 30, 10, 30))
        sf.attach_caption(rec.mask_id, "x")
        service.processor.calls.clear()

        client.post("/export/llm-payload", json={"session_id": SID})

        assert service.processor.calls == []
