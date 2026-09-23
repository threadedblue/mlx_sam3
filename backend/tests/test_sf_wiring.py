"""End-to-end tests for the v2 model (sf-model-v2-design.md) wired into the
running backend: select/hold/caption/scrub over the real HTTP endpoints,
aa_persistence.py's schema (pass/dataset_status/held/caption), and the
training-set compile step.

Drives real HTTP endpoints via FastAPI's TestClient against `main.app`,
with `main.model`/`main.processor`/`main.service` swapped for fakes so no
real SAM3/LaMa model ever loads — `TestClient(app)` used without a `with`
block does not run `lifespan`, confirmed separately.

Persistence is explicit-Save-only (design doc §5): nothing here relies on
an endpoint silently autosaving — every test that checks disk state calls
/saveSession (or `service.save_session_to_disk`) itself.

Run with:  pytest backend/tests/test_sf_wiring.py -v
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import aa_persistence
import compile_training_set
import d4m_juliacall_bridge as bridge
import main
import sf_engine
from services import SegmentationService

IMG_SIZE = 100


def _mask_for_px_box(px_box: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = px_box
    m = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    m[int(y0):int(y1), int(x0):int(x1)] = 1.0
    return m


class FakeProcessor:
    """Minimal Sam3Processor double with the same non-append-only semantics
    documented in aa_persistence.py's module docstring: `set_text_prompt`
    replaces, `add_geometric_prompt`/`add_point_prompt` re-ground the whole
    accumulated box/point history every call."""

    def __init__(self):
        self.text_boxes: dict[str, list[tuple[float, float, float, float]]] = {}
        self._geo_boxes: list[tuple] = []
        # Opt-in, single-shot: makes the NEXT add_geometric_prompt/
        # add_point_prompt call report zero instances, without touching
        # accumulated history -- simulates a click that finds nothing (v3
        # spec §3 Fix 2), which this double's normal box-around-the-point
        # construction can never produce on its own (the point is always
        # trivially inside its own derived box).
        self.next_geometric_prompt_finds_nothing = False

    def set_image(self, image):
        w, h = image.size
        self._geo_boxes = []
        return {"masks": [], "boxes": [], "scores": [], "original_width": w, "original_height": h}

    def set_text_prompt(self, prompt, state):
        px_boxes = self.text_boxes.get(prompt, [])
        masks = [_mask_for_px_box(b) for b in px_boxes]
        return {**state, "masks": masks, "boxes": [list(b) for b in px_boxes], "scores": [0.9] * len(masks)}

    def add_geometric_prompt(self, box, label, state):
        if self.next_geometric_prompt_finds_nothing:
            self.next_geometric_prompt_finds_nothing = False
            return {**state, "masks": [], "boxes": [], "scores": []}
        self._geo_boxes.append(tuple(box))
        px_boxes = [self._to_px(b) for b in self._geo_boxes]
        masks = [_mask_for_px_box(b) for b in px_boxes]
        return {**state, "masks": masks, "boxes": [list(b) for b in px_boxes], "scores": [0.9] * len(masks)}

    def add_point_prompt(self, point, label, state):
        x, y = point
        return self.add_geometric_prompt([x, y, 0.05, 0.05], label, state)

    def reset_all_prompts(self, state):
        """Mirrors the real Sam3Processor.reset_all_prompts: drops the
        accumulated geometric-prompt history and the last grounding result.
        Was missing entirely until /reset's own test coverage needed it —
        every prior test exercising /reset went through it via `_reset`
        only implicitly, none actually asserted on its response."""
        self._geo_boxes = []
        for key in ("geometric_prompt", "boxes", "masks", "masks_logits", "scores"):
            state.pop(key, None)

    @staticmethod
    def _to_px(cxcywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        cx, cy, w, h = cxcywh
        return (
            (cx - w / 2) * IMG_SIZE, (cy - h / 2) * IMG_SIZE,
            (cx + w / 2) * IMG_SIZE, (cy + h / 2) * IMG_SIZE,
        )


class FakeInpainter:
    def inpaint(self, image, mask):
        return image  # no-op scrub — orchestration is what's under test, not pixels


class RecordingInpainter:
    """Like FakeInpainter, but keeps every union mask it was called with —
    for tests that need to prove WHICH geometry actually reached LaMa
    (e.g. that an opted-out mask's region is excluded), not just which
    mask_ids the response says were scrubbed."""

    def __init__(self):
        self.masks: list[np.ndarray] = []

    def inpaint(self, image, mask):
        self.masks.append(mask)
        return image


def _upload_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (IMG_SIZE, IMG_SIZE), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    fake_processor = FakeProcessor()
    fake_service = SegmentationService(tmp_path / "sessions", fake_processor)

    monkeypatch.setattr(main, "model", object())
    monkeypatch.setattr(main, "processor", fake_processor)
    monkeypatch.setattr(main, "service", fake_service)
    monkeypatch.setattr(main, "_lama_inpainter", FakeInpainter())
    monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf_storage")

    return TestClient(main.app)


def _upload(client) -> str:
    files = {"file": ("test.png", _upload_png_bytes(), "image/png")}
    r = client.post("/upload", files=files)
    assert r.status_code == 200
    return r.json()["session_id"]


def _box_select(client, session_id, box, text_substitute=None) -> str:
    """Selects via /segment/box and returns the mask id THIS call touched.

    Not `results["mask_ids"][0]`/length: `results` is the full current-pass
    list (serialize_sf_masks is not a diff — see its docstring), so it
    already has every prior selection in it too once more than one exists.
    `selected_mask_id` is the field that exists specifically to answer
    "which one was this call about."
    """
    r = client.post("/segment/box", json={
        "session_id": session_id, "box": box, "label": True,
        **({"text_substitute": text_substitute} if text_substitute else {}),
    })
    assert r.status_code == 200
    mask_id = r.json()["selected_mask_id"]
    assert mask_id is not None, f"expected a selection, got {r.json()}"
    return mask_id


def _hold(client, session_id, mask_id, held=True):
    r = client.post("/mask/hold", json={"session_id": session_id, "mask_id": mask_id, "held": held})
    assert r.status_code == 200, r.text
    return r.json()


def _caption(client, session_id, mask_id, caption):
    r = client.post("/mask/caption", json={"session_id": session_id, "mask_id": mask_id, "caption": caption})
    assert r.status_code == 200, r.text
    return r.json()


class TestHttpSelectHoldCaptionScrub:
    """The actual acceptance test from the prompt: select (no caption) ->
    hold -> caption via the second mask's Prompt-card-equivalent call ->
    hold a second, uncaptioned mask -> scrub -> confirm both scrubbed."""

    def test_full_flow(self, client):
        session_id = _upload(client)

        keeper_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        balloon_id = _box_select(client, session_id, [0.7, 0.7, 0.2, 0.2], text_substitute="word balloon")
        assert keeper_id != balloon_id

        # Selected, and already held by default (v3 spec §5: Scrub is
        # gated on selection alone now, so every newly selected mask is
        # auto-held the instant /segment/* creates it) — but neither
        # captioned yet.
        sf_session = main.service.sf_sessions[session_id]
        assert all(m.held and m.dataset_status == sf_engine.DatasetStatus.UNASSIGNED for m in sf_session.masks)

        # Redundant now that selection auto-holds (kept as an explicit,
        # harmless no-op — it's still what a real Prompt-card checkbox
        # click sends when a row starts checked and the user leaves it
        # checked): caption still requires its own explicit step either way.
        _hold(client, session_id, keeper_id, True)
        caption_resp = _caption(client, session_id, keeper_id, "a red egg")
        assert caption_resp["dataset_status"] == "keep"
        assert sf_session.get_mask(keeper_id).held is True  # captioning didn't touch held

        _hold(client, session_id, balloon_id, True)
        assert sf_session.get_mask(balloon_id).dataset_status == sf_engine.DatasetStatus.UNASSIGNED  # never captioned

        r = client.post("/lama/scrub", json={"session_id": session_id})
        assert r.status_code == 200
        body = r.json()
        assert body["from_pass"] == 0 and body["to_pass"] == 1
        assert set(body["scrubbed_mask_ids"]) == {keeper_id, balloon_id}

        assert sf_session.pass_ == 1
        assert sf_session.get_mask(keeper_id).held is False
        assert sf_session.get_mask(balloon_id).held is False
        # Scrubbing didn't touch dataset_status either direction.
        assert sf_session.get_mask(keeper_id).dataset_status == sf_engine.DatasetStatus.KEEP
        assert sf_session.get_mask(balloon_id).dataset_status == sf_engine.DatasetStatus.UNASSIGNED

    def test_results_include_the_current_pass_number(self, client):
        """Fix: serialize_sf_masks never returned a pass number at all —
        every record in a response is implicitly "whatever pass the
        session is currently at", but nothing surfaced that number itself.
        Needed by SegForge/frontend's AA preview adapter, which has no
        other way to label which pass a row belongs to."""
        session_id = _upload(client)

        r = client.post("/segment/box", json={"session_id": session_id, "box": [0.2, 0.2, 0.2, 0.2], "label": True})
        assert r.json()["results"]["passes"] == [0]

        _box_select(client, session_id, [0.7, 0.7, 0.2, 0.2])
        r = client.post("/segment/box", json={"session_id": session_id, "box": [0.2, 0.2, 0.2, 0.2], "label": True})
        # Both current-pass records report the same pass — a constant
        # repeated once per row, not per-record data — matching
        # serialize_sf_masks' own current-pass-only filter.
        assert r.json()["results"]["passes"] == [0, 0]

        client.post("/lama/scrub", json={"session_id": session_id})
        r = client.post("/segment/box", json={"session_id": session_id, "box": [0.5, 0.5, 0.1, 0.1], "label": True})
        assert r.json()["results"]["passes"] == [1]

    def test_all_masks_is_a_session_wide_ledger_that_survives_a_scrub(self, client):
        """Fix: reported live -- "I did a selection with a prompt... I did
        the scrub. AA preview shows nothing" -- reproduced and confirmed:
        a captioned/kept mask vanished from every endpoint's `results` the
        moment its pass was scrubbed past. Correct for `results` (current-
        pass-only, matches what the canvas is currently painting — see
        serialize_sf_masks' own docstring), but it left the AA Preview tab
        with no way to ever show the data it exists to preview, since the
        moment something is scrubbed is exactly when it stops being "the
        current pass". `all_masks` is the fix: present on every mutation
        endpoint's response, spans every pass, never resets on a scrub."""
        session_id = _upload(client)
        keeper_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        _hold(client, session_id, keeper_id, True)
        cap = _caption(client, session_id, keeper_id, "a grey egg")
        assert "all_masks" in cap
        assert keeper_id in cap["all_masks"]["mask_ids"]

        r = client.post("/lama/scrub", json={"session_id": session_id})
        assert r.status_code == 200
        body = r.json()
        assert "all_masks" in body
        idx = body["all_masks"]["mask_ids"].index(keeper_id)
        assert body["all_masks"]["dataset_statuses"][idx] == "keep"
        assert body["all_masks"]["captions"][idx] == "a grey egg"
        # The pass it was traced against, not the new current pass (1).
        assert body["all_masks"]["passes"][idx] == 0

        # A LATER, unrelated action at the NEW pass must still report it
        # too — not just the scrub response itself.
        r = client.post("/segment/point", json={"session_id": session_id, "point": [0.9, 0.05], "label": False})
        assert keeper_id in r.json()["all_masks"]["mask_ids"]
        # But NOT in the current-pass-only `results` — that field's scope
        # is deliberately unchanged by this fix.
        assert keeper_id not in r.json()["results"]["mask_ids"]

    def test_all_masks_shows_an_uncaptioned_scrubbed_mask_as_unassigned_not_absent(self, client):
        """The flip side: implicit discard (held + scrubbed, never
        captioned) is real and intentional (sf_engine.py's TestExport) —
        but it's an EXPORT-time filter (build_sf_payload's keep-only pass),
        never a deletion from SFSession.masks. The record stays forever as
        an audit trail (confirmed directly: DatasetStatus has no discard
        value, only unassigned/keep), so all_masks — which mirrors
        SFSession.masks with no dataset_status filter of its own — must
        keep showing it, just correctly as unassigned/unheld, not silently
        drop it. Dropping records `build_sf_payload` will never export
        would defeat the point of a session-wide ledger just as much as
        the original bug did."""
        session_id = _upload(client)
        balloon_id = _box_select(client, session_id, [0.7, 0.7, 0.2, 0.2])
        _hold(client, session_id, balloon_id, True)

        r = client.post("/lama/scrub", json={"session_id": session_id})
        all_masks = r.json()["all_masks"]
        assert balloon_id in all_masks["mask_ids"]
        idx = all_masks["mask_ids"].index(balloon_id)
        assert all_masks["dataset_statuses"][idx] == "unassigned"
        assert all_masks["held_flags"][idx] is False  # cleared by the scrub
        assert all_masks["captions"][idx] is None

    def test_scrub_is_repeatable_no_two_pass_cap(self, client):
        session_id = _upload(client)
        for _ in range(3):
            r = client.post("/lama/scrub", json={"session_id": session_id})
            assert r.status_code == 200
        assert main.service.sf_sessions[session_id].pass_ == 3

    def test_hold_unknown_mask_is_404_not_500(self, client):
        session_id = _upload(client)
        r = client.post("/mask/hold", json={"session_id": session_id, "mask_id": "0:nope", "held": True})
        assert r.status_code == 404

    def test_caption_unknown_mask_is_404_and_empty_caption_is_422(self, client):
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])

        r = client.post("/mask/caption", json={"session_id": session_id, "mask_id": "0:nope", "caption": "x"})
        assert r.status_code == 404

        r = client.post("/mask/caption", json={"session_id": session_id, "mask_id": mask_id, "caption": "   "})
        assert r.status_code == 422

    def test_holding_a_record_from_an_earlier_pass_is_422(self, client):
        session_id = _upload(client)
        old_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        client.post("/lama/scrub", json={"session_id": session_id})  # advances to pass 1

        r = client.post("/mask/hold", json={"session_id": session_id, "mask_id": old_id, "held": True})
        assert r.status_code == 422

    def test_delete_session_clears_the_sf_session_zombie(self, client):
        """/mask/hold and /mask/caption call get_or_create_sf_session
        directly — they never call get_session() first — and that method
        checks the sf_sessions cache before anything else. If DELETE
        /session/{id} only cleared `service.sessions`, both endpoints would
        keep succeeding against the deleted session's stale SFSession
        object indefinitely (only a process restart would actually clear
        it). This is the exact zombie this test pins.
        """
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        assert session_id in main.service.sf_sessions

        r = client.delete(f"/session/{session_id}")
        assert r.status_code == 200

        assert session_id not in main.service.sessions
        assert session_id not in main.service.sf_sessions

        r = client.post("/mask/hold", json={"session_id": session_id, "mask_id": mask_id, "held": True})
        assert r.status_code == 404

        r = client.post("/mask/caption", json={"session_id": session_id, "mask_id": mask_id, "caption": "x"})
        assert r.status_code == 404

    def test_delete_unknown_session_is_404(self, client):
        r = client.delete("/session/never-existed")
        assert r.status_code == 404

    def test_reset_clears_selected_mask_id_and_the_deleted_mask_404s(self, client):
        """Confirmed live: after /reset, selected_mask_id kept echoing the
        id of a mask reset had just discarded, in every later /segment/*
        response, until the backend restarted -- because /reset discarded
        the record but never cleared sf_session.last_touched_mask_id, the
        field those responses echo.

        Explicitly unchecked below: v3 spec §5 auto-holds every new
        selection (see TestSelectionAutoHoldsForScrub), and a held record
        now survives /reset (test_reset_now_preserves_a_freshly_selected_
        still_checked_mask) -- this test is about last_touched_mask_id
        specifically, which needs the record to actually be discarded to
        prove it isn't just the top-level field being blanked, so it has to
        reach that state explicitly now rather than getting it for free.
        """
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        _hold(client, session_id, mask_id, False)

        r = client.post("/reset", json={"session_id": session_id})
        assert r.status_code == 200
        assert r.json().get("selected_mask_id") is None

        # The record itself is gone -- confirms this isn't just the
        # top-level field being blanked while the object underneath survives.
        r = client.post("/mask/hold", json={"session_id": session_id, "mask_id": mask_id, "held": True})
        assert r.status_code == 404

        r = client.post("/mask/caption", json={"session_id": session_id, "mask_id": mask_id, "caption": "x"})
        assert r.status_code == 404

        # A later no-op call (Avoid with nothing to match) must not revive
        # the stale id either -- this is the exact case F/G replay from the
        # investigation that originally surfaced it.
        r = client.post("/segment/point", json={"session_id": session_id, "point": [0.9, 0.05], "label": False})
        assert r.status_code == 200
        assert r.json().get("selected_mask_id") is None

    def test_reset_preserves_a_captioned_masks_geometry_and_status_in_its_own_response(self, client):
        """Confirmed live: Clear Prompts was making captioned/kept masks
        vanish from the canvas, because /reset discarded EVERY current-pass
        MaskRecord unconditionally -- committed work along with the
        uncommitted, now-ungrounded selections it actually needs to clear.
        /reset's job is to reset SAM3's own grounding/prompt state; losing
        visibility into already-committed work is a different thing and
        must not happen. `results` here uses the exact same
        serialize_sf_masks shape /segment/* responses do, so a UI reading
        it the same way sees the mask reappear immediately, not just on
        the next unrelated mutation."""
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        _caption(client, session_id, mask_id, "a red balloon")

        r = client.post("/reset", json={"session_id": session_id})
        assert r.status_code == 200
        results = r.json()["results"]

        assert mask_id in results["mask_ids"]
        idx = results["mask_ids"].index(mask_id)
        assert results["dataset_statuses"][idx] == "keep"
        assert results["captions"][idx] == "a red balloon"

        # The stale-selected_mask_id fix is a separate concern from this
        # one and must be unaffected by it: reset still clears focus to
        # null even though the record itself now survives.
        assert r.json().get("selected_mask_id") is None

        # The surviving record is still a live, addressable mask, not a
        # read-only echo -- Hold must still work on it post-reset.
        r = client.post("/mask/hold", json={"session_id": session_id, "mask_id": mask_id, "held": True})
        assert r.status_code == 200

    def test_reset_still_discards_an_uncaptioned_and_unchecked_selection(self, client):
        """The other half of the same fix, pinned explicitly: a selection
        with nothing backing it once SAM3's grounding is cleared -- never
        captioned, and explicitly unchecked out of the next scrub batch --
        must still be discarded by /reset. Otherwise this fix would
        silently turn every ungrounded selection permanent instead of only
        protecting genuinely committed work.

        Selection alone no longer implies unheld (v3 spec §5: every new
        selection is auto-held -- see TestSelectionAutoHoldsForScrub), so
        reaching the "uncaptioned, unheld" state this test is named for now
        needs the explicit uncheck below -- the real UI's equivalent of a
        user excluding this row via ObjectsSelectedCard before doing
        anything else with it. See
        test_reset_now_preserves_a_freshly_selected_still_checked_mask for
        what happens WITHOUT that uncheck -- a discovered side effect of
        §5, not something this test is about.
        """
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        _hold(client, session_id, mask_id, False)

        r = client.post("/reset", json={"session_id": session_id})
        assert r.status_code == 200

        assert mask_id not in r.json()["results"]["mask_ids"]

        r = client.post("/mask/hold", json={"session_id": session_id, "mask_id": mask_id, "held": True})
        assert r.status_code == 404

    def test_reset_now_preserves_a_freshly_selected_still_checked_mask(self, client):
        """v3 spec §5 side effect, discovered while implementing it, not
        itself requested by §5: since every new selection starts held (see
        TestSelectionAutoHoldsForScrub), /reset's existing
        carries-user-state check (`dataset_status == keep OR held`, added
        for §2) now treats "selected and not yet explicitly unchecked" as
        committed-enough-to-survive too -- which used to require a
        deliberate Hold click before this change. Whether that's the right
        call for Clear Prompts under the new default, or whether
        `_carries_user_state` should narrow back to `dataset_status ==
        keep` only now that `held` alone no longer signals a deliberate
        action, is an open follow-up -- this pins the actual CURRENT
        behavior, not a recommendation of what it should be.
        """
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])

        r = client.post("/reset", json={"session_id": session_id})
        assert r.status_code == 200

        assert mask_id in r.json()["results"]["mask_ids"]


class TestSelectionAutoHoldsForScrub:
    """v3 spec §5: Scrub Selected Regions is gated on selection alone, not
    a separate Hold step -- every newly selected mask starts held the
    instant /segment/* creates it, across all three selection modes, so
    the frontend's per-object checkbox list can default to checked."""

    def test_text_prompt_selection_is_held_immediately(self, client):
        session_id = _upload(client)
        main.processor.text_boxes["egg"] = [(10, 10, 30, 30)]

        r = client.post("/segment/text", json={"session_id": session_id, "prompt": "egg"})
        assert r.status_code == 200
        mask_id = r.json()["results"]["mask_ids"][0]

        assert main.service.sf_sessions[session_id].get_mask(mask_id).held is True

    def test_box_selection_is_held_immediately(self, client):
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])

        assert main.service.sf_sessions[session_id].get_mask(mask_id).held is True

    def test_point_selection_is_held_immediately(self, client):
        session_id = _upload(client)

        r = client.post("/segment/point", json={"session_id": session_id, "point": [0.2, 0.2], "label": True})
        assert r.status_code == 200
        mask_id = r.json()["selected_mask_id"]
        assert mask_id is not None

        assert main.service.sf_sessions[session_id].get_mask(mask_id).held is True

    def test_a_re_grounded_not_newly_created_match_is_not_re_held(self, client):
        """Only genuinely NEW records get auto-held -- an existing record
        that a later call merely re-grounds/refines (same object, refined
        geometry) is untouched by this, so a user's earlier uncheck of it
        survives a later call that happens to re-match its geometry."""
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        _hold(client, session_id, mask_id, False)  # user explicitly unchecked it

        # A second box draw over unrelated geometry re-grounds the whole
        # accumulated prompt history (FakeProcessor.add_geometric_prompt),
        # which re-returns the first box's mask unchanged -- reconciled by
        # IoU against the existing record, not recreated.
        _box_select(client, session_id, [0.7, 0.7, 0.1, 0.1])

        assert main.service.sf_sessions[session_id].get_mask(mask_id).held is False

    def test_unchecking_a_row_excludes_its_geometry_from_the_union_sent_to_lama(self, client, monkeypatch):
        """The actual behavior the opt-out has to prove: not just that
        `held` flips to False and the mask_id is absent from
        `scrubbed_mask_ids`, but that its GEOMETRY never reaches LaMa's
        union mask at all."""
        recording = RecordingInpainter()
        monkeypatch.setattr(main, "_lama_inpainter", recording)

        session_id = _upload(client)
        keeper_id = _box_select(client, session_id, [0.1, 0.1, 0.2, 0.2])   # px (0,0)-(20,20)
        excluded_id = _box_select(client, session_id, [0.7, 0.1, 0.2, 0.2])  # px (60,0)-(80,20)
        other_id = _box_select(client, session_id, [0.1, 0.7, 0.2, 0.2])    # px (0,60)-(20,80)

        # Both start checked by default (§5) -- only opt OUT the middle one.
        _hold(client, session_id, excluded_id, False)

        r = client.post("/lama/scrub", json={"session_id": session_id})
        assert r.status_code == 200
        body = r.json()

        assert set(body["scrubbed_mask_ids"]) == {keeper_id, other_id}
        assert excluded_id not in body["scrubbed_mask_ids"]

        assert len(recording.masks) == 1
        union = recording.masks[0].astype(bool)
        assert union[5, 5], "keeper's region must be in the union"
        assert union[65, 5], "other's region must be in the union"
        assert not union[5, 65], "excluded's region must NOT be in the union"


class TestPointSelectionFailureIsSurfaced:
    """v3 spec §3 Fix 2, at the HTTP layer: a point click that finds
    nothing must come back as `selected_mask_id: null` -- not omitted,
    not echoing a stale id from an earlier, unrelated call -- so the
    frontend can tell "this click found nothing" from "this click
    re-touched something already selected". `_pick_for_point`'s own
    fallback removal is unit-tested directly in test_sf_engine.py; this
    confirms the full round trip through /segment/point actually surfaces
    it, which is the thing a real client observes.
    """

    def test_a_click_that_finds_nothing_reports_a_null_selected_mask_id(self, client):
        session_id = _upload(client)
        main.processor.next_geometric_prompt_finds_nothing = True

        r = client.post("/segment/point", json={"session_id": session_id, "point": [0.5, 0.5], "label": True})

        assert r.status_code == 200
        body = r.json()
        assert body["selected_mask_id"] is None
        assert body["results"]["mask_ids"] == []

    def test_a_failed_click_does_not_overwrite_an_earlier_selection_with_a_stale_echo(self, client):
        """The specific hijack this closes: previously, `last_touched_
        mask_id` was left at whatever an EARLIER call set it to, so a
        failed click's response would echo that id as if THIS click had
        found it. Confirm a genuinely failed click reports null even
        though something else is already selected and focused in the
        session."""
        session_id = _upload(client)
        first_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])

        main.processor.next_geometric_prompt_finds_nothing = True
        r = client.post("/segment/point", json={"session_id": session_id, "point": [0.8, 0.8], "label": True})

        assert r.status_code == 200
        body = r.json()
        assert body["selected_mask_id"] is None, (
            "must not echo the earlier box selection's id as if this "
            "failed point click had touched it"
        )
        # The earlier selection itself must survive untouched.
        assert first_id in body["results"]["mask_ids"]

    def test_a_click_that_matches_something_still_reports_it_normally(self, client):
        """Control: the surfacing fix must not make a genuinely successful
        click look like a failure."""
        session_id = _upload(client)

        r = client.post("/segment/point", json={"session_id": session_id, "point": [0.2, 0.2], "label": True})

        assert r.status_code == 200
        body = r.json()
        assert body["selected_mask_id"] is not None
        assert body["selected_mask_id"] in body["results"]["mask_ids"]


class TestSaveReloadRoundTrip:
    """dataset_status, held, caption, and pass must survive a full
    save_session -> read_session_raw cycle unchanged — a correctness bug,
    not a nice-to-have, per the design doc (§6): a failure here means
    captioned work silently never reaches the training set."""

    def _build_session(self, client) -> tuple[str, str, str]:
        session_id = _upload(client)
        keeper_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        balloon_id = _box_select(client, session_id, [0.7, 0.7, 0.2, 0.2], text_substitute="word balloon")
        _hold(client, session_id, keeper_id, True)
        _caption(client, session_id, keeper_id, "a red egg")
        _hold(client, session_id, balloon_id, True)
        client.post("/lama/scrub", json={"session_id": session_id})  # pass 0 -> 1

        lamp_id = _box_select(client, session_id, [0.4, 0.4, 0.1, 0.1], text_substitute="lamp")
        _caption(client, session_id, lamp_id, "a brass lamp")
        return session_id, keeper_id, lamp_id

    def test_fields_survive_direct_save_and_read(self, client):
        session_id, keeper_id, lamp_id = self._build_session(client)
        r = client.post("/saveSession", json={"session_id": session_id})
        assert r.status_code == 200

        raw = aa_persistence.read_session_raw(session_id)
        assert raw is not None
        assert set(raw["pass_images"].keys()) == {1}  # pass 0 has no scrub image; only pass 1 does

        by_id = {seg["segment_id"]: seg for seg in raw["segments"]}

        keeper = by_id[keeper_id]
        assert keeper["pass"] == "0"
        assert keeper["dataset_status"] == "keep"
        assert keeper["caption"] == "a red egg"
        assert keeper["held"] is False  # cleared by the scrub

        lamp = by_id[lamp_id]
        assert lamp["pass"] == "1"
        assert lamp["dataset_status"] == "keep"
        assert lamp["caption"] == "a brass lamp"

        # The uncaptioned balloon has no caption column at all (None, not "").
        balloon = next(s for s in raw["segments"] if s["text_tag"] == "word balloon")
        assert balloon["dataset_status"] == "unassigned"
        assert balloon["caption"] is None
        assert balloon["held"] is False

    def test_full_reload_reconstructs_sf_session_across_passes(self, client):
        session_id, keeper_id, lamp_id = self._build_session(client)
        client.post("/saveSession", json={"session_id": session_id})

        # Simulate a process restart: evict the in-memory cache entirely.
        del main.service.sessions[session_id]
        del main.service.sf_sessions[session_id]

        assert main.service.get_session(session_id) is not None
        reloaded = main.service.sf_sessions[session_id]

        assert reloaded.pass_ == 1
        assert reloaded.working_copy is not None
        assert reloaded.image_for_pass(0) is not None
        assert reloaded.image_for_pass(1) is not None

        keeper = reloaded.get_mask(keeper_id)
        assert (keeper.pass_, keeper.dataset_status, keeper.caption, keeper.held) == (0, sf_engine.DatasetStatus.KEEP, "a red egg", False)
        lamp = reloaded.get_mask(lamp_id)
        assert (lamp.pass_, lamp.dataset_status, lamp.caption) == (1, sf_engine.DatasetStatus.KEEP, "a brass lamp")

    def test_held_record_survives_a_save_reload_cycle(self, client):
        """A record still IN the pending batch (not yet scrubbed) must
        reload as held — this is the state a user leaves mid-session."""
        session_id = _upload(client)
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        _hold(client, session_id, mask_id, True)
        client.post("/saveSession", json={"session_id": session_id})

        del main.service.sessions[session_id]
        del main.service.sf_sessions[session_id]
        main.service.get_session(session_id)

        assert main.service.sf_sessions[session_id].get_mask(mask_id).held is True


class TestRealResumeSequenceDoesNotDestroyData:
    """The actual product sequence on every app relaunch, reproduced
    precisely — NOT `TestSaveReloadRoundTrip`'s `get_session()` shortcut,
    which the real app never calls on resume and which is why that class's
    tests pass regardless of this bug.

    Confirmed live: GET /loadSession never warmed `self.sessions`/
    `self.sf_sessions`, so the frontend's follow-up POST /upload (which
    always fires once bytes come back from /loadSession — main.dart's
    `_maybeAutoLoadImage` -> `_loadImageFromUrl`, pre-Fix-3) found nothing
    cached to merge into, rebuilt the session from scratch via
    `register_session_data`, and that same call auto-saves at its own end
    — so a session's name, description, prompts and every mask were
    silently overwritten with near-nothing on disk before the user touched
    anything. This class is the test that would have caught it.

    Fix 3 (frontend) additionally skips the /upload call entirely once
    /loadSession has restored everything — not exercisable from a backend
    test. So this deliberately still drives /upload afterward, proving
    Fix 1+2 protect the data on the backend even if a client calls it
    anyway (today's pre-Fix-3 frontend, or any future caller).
    """

    def _build_named_session(self, client) -> tuple[str, str, str, str]:
        """Mirrors the real flow precisely: DoubleNaught calls /initSession
        with a real name/description BEFORE the SF app ever uploads
        anything — `_upload(client)` alone (used elsewhere in this file)
        skips that step and so can't prove name/description survive."""
        session_id = "resume-real-flow"
        r = client.post("/initSession", json={
            "session_id": session_id, "name": "My Session", "description": "a real session",
            "image_url": "https://example.com/original.png",
        })
        assert r.status_code == 200

        files = {"file": ("test.png", _upload_png_bytes(), "image/png")}
        r = client.post("/upload", files=files, data={"session_id": session_id})
        assert r.status_code == 200

        keeper_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        balloon_id = _box_select(client, session_id, [0.7, 0.7, 0.2, 0.2], text_substitute="word balloon")
        _hold(client, session_id, keeper_id, True)
        _caption(client, session_id, keeper_id, "a red egg")
        _hold(client, session_id, balloon_id, True)
        client.post("/lama/scrub", json={"session_id": session_id})  # pass 0 -> 1

        lamp_id = _box_select(client, session_id, [0.4, 0.4, 0.1, 0.1], text_substitute="lamp")
        _caption(client, session_id, lamp_id, "a brass lamp")
        return session_id, keeper_id, balloon_id, lamp_id

    def test_loadSession_then_upload_survives_the_real_resume_sequence(self, client):
        session_id, keeper_id, balloon_id, lamp_id = self._build_named_session(client)
        r = client.post("/saveSession", json={"session_id": session_id})
        assert r.status_code == 200

        # Simulate a backend restart: evict the in-memory caches entirely
        # — same method the investigation's own reproduction used.
        del main.service.sessions[session_id]
        del main.service.sf_sessions[session_id]

        # Step 1: GET /loadSession — exactly what _loadLaunchSession does
        # on every app launch.
        r = client.get(f"/loadSession/{session_id}")
        assert r.status_code == 200
        loaded = r.json()
        assert loaded["name"] == "My Session"
        image_b64 = loaded["image_b64"]
        assert image_b64

        # Step 2: POST /upload of those SAME bytes — exactly what
        # _maybeAutoLoadImage -> _loadImageFromUrl does with them
        # (pre-Fix-3; Fix 3 itself skips this call, but that's a frontend
        # change this backend test can't exercise).
        image_bytes = base64.b64decode(image_b64)
        files = {"file": ("test.png", image_bytes, "image/png")}
        r = client.post("/upload", files=files, data={"session_id": session_id})
        assert r.status_code == 200

        # /upload's own register_session_data call auto-saves at its end —
        # this is the exact moment the pre-fix bug silently destroyed the
        # session, with no explicit Save click anywhere in this sequence.
        raw = aa_persistence.read_session_raw(session_id)
        assert raw is not None
        assert raw["name"] == "My Session"
        assert raw["description"] == "a real session"
        assert len(raw["prompts"]) > 0, "prompt history must survive"

        by_id = {seg["segment_id"]: seg for seg in raw["segments"]}
        assert set(by_id.keys()) >= {keeper_id, balloon_id, lamp_id}, (
            f"expected all 3 masks, got {sorted(by_id.keys())}"
        )
        assert by_id[keeper_id]["dataset_status"] == "keep"
        assert by_id[keeper_id]["caption"] == "a red egg"
        assert by_id[lamp_id]["dataset_status"] == "keep"
        assert by_id[lamp_id]["caption"] == "a brass lamp"
        assert by_id[balloon_id]["dataset_status"] == "unassigned"

        # And the in-memory SFSession /segment/* would actually use is
        # equally intact, not just what got persisted.
        sf = main.service.sf_sessions[session_id]
        assert len(sf.masks) == 3
        assert sf.get_mask(keeper_id).caption == "a red egg"

        # Explicit Save afterward must not regress anything either.
        r = client.post("/saveSession", json={"session_id": session_id})
        assert r.status_code == 200
        raw = aa_persistence.read_session_raw(session_id)
        assert raw["name"] == "My Session"
        assert len(raw["segments"]) == 3

    def test_upload_without_loadSession_first_still_restores_from_disk(self, client):
        """Fix 2's own defense-in-depth claim, isolated: a caller that
        reaches /upload for an existing, saved session WITHOUT /loadSession
        ever having run — so nothing is cached in memory at all — must
        still restore from disk rather than starting from a blank slate.
        This is the scenario Fix 1 alone does NOT cover."""
        session_id, keeper_id, balloon_id, lamp_id = self._build_named_session(client)
        client.post("/saveSession", json={"session_id": session_id})

        del main.service.sessions[session_id]
        del main.service.sf_sessions[session_id]

        # No GET /loadSession at all — straight to /upload, exactly the
        # gap Fix 1 alone leaves open for any caller that skips it.
        files = {"file": ("test.png", _upload_png_bytes(), "image/png")}
        r = client.post("/upload", files=files, data={"session_id": session_id})
        assert r.status_code == 200

        raw = aa_persistence.read_session_raw(session_id)
        assert raw["name"] == "My Session"
        assert raw["description"] == "a real session"
        by_id = {seg["segment_id"]: seg for seg in raw["segments"]}
        assert set(by_id.keys()) >= {keeper_id, balloon_id, lamp_id}
        assert by_id[keeper_id]["caption"] == "a red egg"

        sf = main.service.sf_sessions[session_id]
        assert len(sf.masks) == 3

    def test_upload_for_a_genuinely_new_session_is_unaffected(self, client):
        """Fix 2 must not weaken first-time session creation: a session_id
        with nothing on disk yet (get_session returns None) still gets a
        fresh {} and a brand-new SFSession, exactly as before."""
        session_id = _upload(client)  # no prior /initSession, nothing on disk

        assert session_id in main.service.sessions
        assert main.service.sessions[session_id].get("name") is None
        sf = main.service.sf_sessions[session_id]
        assert sf.masks == []

        # And it's fully usable immediately — the normal new-session path.
        mask_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        assert mask_id is not None


class TestStateObjectUnifiedAfterResume:
    """Follow-up investigation into `_load_session_into_memory`: for any
    session resumed at pass > 0, `session["state"]` (the replayed pass-0
    state `_load_session_into_memory` builds) and `sf_session.state` (what
    `_reconstruct_sf_session` rebuilds via a fresh `set_image` on that
    pass's actual image) used to be two DIFFERENT objects — confirmed live
    (`session["state"] is sf_session.state` was False immediately after
    resume). Every live selection/scrub/save handler rebinds
    `state = sf_session.state` before doing anything real, so this never
    corrupted grounding, reconciliation, scrub, or persistence — confirmed
    live, all four stayed correct even with the desync present, self-
    healing it on first use. But `/reset` reads `session["state"]`
    directly and never re-aliases, so it silently acted on the stale
    object; and the very first prompted_boxes display marker after a
    pass>0 resume was appended to that same stale object, then never
    reached the response (which serializes from the correct one) — lost,
    not delayed. Fix: `_load_session_into_memory` now re-aliases
    `session["state"]` to `sf_session.state` once, right after
    reconstruction, restoring the single-object invariant the rest of the
    codebase already assumes.
    """

    def _build_pass1_session(self, client) -> tuple[str, str]:
        session_id = "state-identity-resume"
        r = client.post("/initSession", json={
            "session_id": session_id, "name": "My Session", "description": "a real session",
            "image_url": "https://example.com/original.png",
        })
        assert r.status_code == 200
        files = {"file": ("test.png", _upload_png_bytes(), "image/png")}
        r = client.post("/upload", files=files, data={"session_id": session_id})
        assert r.status_code == 200

        keeper_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        _hold(client, session_id, keeper_id, True)
        _caption(client, session_id, keeper_id, "a red egg")
        client.post("/lama/scrub", json={"session_id": session_id})  # pass 0 -> 1
        return session_id, keeper_id

    def _resume(self, client, session_id):
        client.post("/saveSession", json={"session_id": session_id})
        del main.service.sessions[session_id]
        del main.service.sf_sessions[session_id]
        r = client.get(f"/loadSession/{session_id}")
        assert r.status_code == 200

    def test_session_state_is_sf_session_state_immediately_after_resume(self, client):
        """The actual invariant being restored, asserted directly by object
        identity — not just through downstream symptoms."""
        session_id, keeper_id = self._build_pass1_session(client)
        self._resume(client, session_id)

        session = main.service.sessions[session_id]
        sf_session = main.service.sf_sessions[session_id]
        assert sf_session.pass_ == 1
        assert session["state"] is sf_session.state

    def test_prompted_boxes_marker_present_on_first_click_after_pass1_resume(self, client):
        """Reproduces the investigation's own repro precisely: before the
        fix, the FIRST box/point click after a pass>0 resume appended its
        display marker to the stale object and it never reached the
        response — count was 0 on the first click, 1 on the second. Now it
        must be present on the very first click."""
        session_id, keeper_id = self._build_pass1_session(client)
        self._resume(client, session_id)

        r = client.post("/segment/box", json={"session_id": session_id, "box": [0.7, 0.7, 0.2, 0.2], "label": True})
        assert r.status_code == 200
        body = r.json()
        assert "prompted_boxes" in body["results"], "marker lost on the first click after resume"
        assert len(body["results"]["prompted_boxes"]) == 1

        # And the second click still correctly accumulates to 2 — this
        # already worked pre-fix (via self-healing) and must keep working.
        r = client.post("/segment/box", json={"session_id": session_id, "box": [0.45, 0.45, 0.1, 0.1], "label": True})
        assert len(r.json()["results"]["prompted_boxes"]) == 2

    def test_reset_after_pass1_resume_uses_the_correct_state_object(self, client):
        """/reset reads session["state"] directly and never re-aliased it
        — before the fix, Clear Prompts immediately after a pass>0 resume
        silently cleared the wrong (stale, pass-0) object, and the desync
        it read from survived the call. Confirmed live this never actually
        lost committed work (nothing to clear on a fresh reconstruction),
        but it left the objects split, so the NEXT click's marker was lost
        too. After the fix, /reset operates on the unified object and the
        identity holds all the way through."""
        session_id, keeper_id = self._build_pass1_session(client)
        self._resume(client, session_id)

        r = client.post("/reset", json={"session_id": session_id})
        assert r.status_code == 200

        session = main.service.sessions[session_id]
        sf_session = main.service.sf_sessions[session_id]
        assert session["state"] is sf_session.state, "desync must not survive /reset"

        # The captioned pass-0 record is committed work — /reset must not
        # touch it regardless of which object it read.
        assert sf_session.get_mask(keeper_id).caption == "a red egg"

        # And the marker on the click right after reset — previously lost
        # too, since /reset left the split in place — must be present now.
        r = client.post("/segment/box", json={"session_id": session_id, "box": [0.7, 0.7, 0.2, 0.2], "label": True})
        assert "prompted_boxes" in r.json()["results"]

    def test_pass1_resume_selection_scrub_and_persistence_still_correct(self, client):
        """The regression check the investigation asked for: grounding,
        per-pass reconciliation, scrub, and persistence were all already
        confirmed correct on a pass>0 resume BEFORE this fix (the handlers
        self-healed the desync on first use). Confirm they're still
        correct now that the objects are unified from the start rather
        than healed after the fact — same assertions, fixed code."""
        session_id, keeper_id = self._build_pass1_session(client)
        self._resume(client, session_id)

        # A new selection at pass 1, drawn at the SAME normalized box as
        # the pass-0 keeper — must NOT cross-pass reconcile onto it.
        r = client.post("/segment/box", json={"session_id": session_id, "box": [0.2, 0.2, 0.2, 0.2], "label": True})
        assert r.status_code == 200
        new_id = r.json()["selected_mask_id"]
        assert new_id != keeper_id
        assert new_id.startswith("1:"), "must be recorded in pass 1, not reconciled into pass 0"

        sf_session = main.service.sf_sessions[session_id]
        assert sorted(m.pass_ for m in sf_session.masks) == [0, 1]
        assert sf_session.get_mask(keeper_id).caption == "a red egg", "pass-0 record untouched"

        # Scrub 1 -> 2 must still work correctly on the resumed session.
        _hold(client, session_id, new_id, True)
        r = client.post("/lama/scrub", json={"session_id": session_id})
        assert r.status_code == 200
        assert (r.json()["from_pass"], r.json()["to_pass"]) == (1, 2)
        assert sf_session.pass_ == 2
        assert sf_session.image_for_pass(2) is not None

        # And it all still persists correctly.
        r = client.post("/saveSession", json={"session_id": session_id})
        assert r.status_code == 200
        raw = aa_persistence.read_session_raw(session_id)
        assert raw["name"] == "My Session"
        by_id = {seg["segment_id"]: seg for seg in raw["segments"]}
        assert by_id[keeper_id]["caption"] == "a red egg"
        assert new_id in by_id
        assert set(raw["pass_images"].keys()) == {1, 2}


class TestImageUrlSaveRoundTrip:
    """Follow-up to sessionImageUrlFrom (the /loadSession READ side, already
    correct): closes the WRITE side. `original_filename` survived
    save/reload for free, riding along on /upload's multipart file field
    itself (`file.filename`) — nothing client-side had to separately
    declare it. `image_url` had no equivalent: confirmed directly, neither
    /upload nor /saveSession's request body ever carried it for a session
    created via SF's own standalone upload/file-dialog flow — it was
    never in the save request AT ALL, not received-but-dropped by the
    backend. /upload now accepts an optional `image_url` form field
    (main.dart's `_loadImageFromUrl` sends whatever `_imageUrl`/the Image
    URL field is currently showing — file:// or http(s):// alike, same
    code path either way), and only writes it when actually supplied so
    an unrelated /upload call can never blank out a URL /initSession
    already set for this session_id.
    """

    def test_upload_with_no_image_url_field_has_none_to_save(self, client):
        """The optional field really is optional: when the client sends
        no image_url (an older client, or genuinely nothing to report),
        /upload must still succeed and simply have nothing to persist —
        not a required-field regression. original_filename is unaffected
        either way, confirming this is specifically about image_url."""
        session_id = _upload(client)  # no image_url in this call

        r = client.post("/saveSession", json={"session_id": session_id})
        assert r.status_code == 200

        raw = aa_persistence.read_session_raw(session_id)
        assert raw["original_filename"] == "test.png"
        assert raw["image_url"] is None

    def test_file_dialog_upload_with_image_url_survives_save_and_reload(self, client):
        """The actual fix, and the actual bug report, reproduced end to
        end: a file-dialog-equivalent upload — /upload's multipart request
        now carries image_url, the file:// URI the frontend's Image URL
        field was already showing live — survives Save and comes back
        correctly on /loadSession, the exact response shape the frontend's
        sessionImageUrlFrom (already correct) parses."""
        file_url = "file:///Users/someone/Pictures/cat.png"
        files = {"file": ("cat.png", _upload_png_bytes(), "image/png")}
        r = client.post("/upload", files=files, data={"image_url": file_url})
        assert r.status_code == 200
        session_id = r.json()["session_id"]

        r = client.post("/saveSession", json={"session_id": session_id})
        assert r.status_code == 200

        raw = aa_persistence.read_session_raw(session_id)
        assert raw["original_filename"] == "cat.png"
        assert raw["image_url"] == file_url

        # Full round trip through the real endpoint the frontend calls on
        # resume, not just the persisted file.
        r = client.get(f"/loadSession/{session_id}")
        assert r.status_code == 200
        assert r.json()["image_url"] == file_url

    def test_http_url_upload_survives_save_and_reload_too(self, client):
        """Item 3: confirms this isn't file://-specific — an http(s)://
        URL supplied the same way (SF's standalone "Image URL" field
        entry, not DoubleNaught's separate /initSession path) round-trips
        identically. The gap was an absent field, not a URI-scheme
        parsing issue."""
        http_url = "https://example.com/dog.jpg"
        files = {"file": ("dog.jpg", _upload_png_bytes(), "image/png")}
        r = client.post("/upload", files=files, data={"image_url": http_url})
        assert r.status_code == 200
        session_id = r.json()["session_id"]

        r = client.post("/saveSession", json={"session_id": session_id})
        assert r.status_code == 200

        r = client.get(f"/loadSession/{session_id}")
        assert r.status_code == 200
        assert r.json()["image_url"] == http_url

    def test_initsession_then_upload_session_already_preserves_image_url(self, client):
        """The OTHER real path (DoubleNaught's "+ New Session" flow) —
        /initSession sets image_url server-side FIRST; a later /upload for
        the same session_id (SF's own auto-load re-registering it) merges
        into the existing in-memory session dict rather than replacing it
        (register_session_data does `.update()`), so the URL survives into
        the saved registry untouched. Confirmed this already worked before
        this fix and still does — the fix only had to cover /upload's own,
        separate omission."""
        session_id = "preinit-session"
        r = client.post("/initSession", json={
            "session_id": session_id, "name": "n", "description": "d",
            "image_url": "https://example.com/cat.png",
        })
        assert r.status_code == 200

        files = {"file": ("test.png", _upload_png_bytes(), "image/png")}
        r = client.post("/upload", files=files, data={"session_id": session_id})
        assert r.status_code == 200

        r = client.post("/saveSession", json={"session_id": session_id})
        assert r.status_code == 200

        raw = aa_persistence.read_session_raw(session_id)
        assert raw["image_url"] == "https://example.com/cat.png"

    def test_upload_image_url_never_clobbers_an_initsession_value_when_absent(self, client):
        """The defensive half of the fix: /upload only writes image_url
        when actually supplied. A later /upload call for the same session
        that sends no image_url (e.g. a re-segment-adjacent upload with
        stale/empty client state) must not blank out what /initSession
        already established."""
        session_id = "preinit-session-2"
        r = client.post("/initSession", json={
            "session_id": session_id, "name": "n", "description": "d",
            "image_url": "https://example.com/original.png",
        })
        assert r.status_code == 200

        files = {"file": ("test.png", _upload_png_bytes(), "image/png")}
        r = client.post("/upload", files=files, data={"session_id": session_id})  # no image_url
        assert r.status_code == 200

        r = client.post("/saveSession", json={"session_id": session_id})
        assert r.status_code == 200

        raw = aa_persistence.read_session_raw(session_id)
        assert raw["image_url"] == "https://example.com/original.png"


class TestOldSessionMigration:
    """v1 (`MaskType.IN`/`OUT`, two-pass) rows read back with defined v2
    defaults, not undefined/missing behavior."""

    def test_pre_v2_segment_migrates_to_unassigned_unheld_pass_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf_storage")
        session_id = "legacy-session"
        d = aa_persistence.session_dir(session_id)

        # Hand-write registry.parquet + a v1-format segment.parquet: bare
        # uuid4 row key, mask_type column, "foreground" pass — no
        # dataset_status/held/caption columns at all.
        bridge.save_parquet(str(d / "registry.parquet"), [session_id], ["name"], ["Legacy"])
        old_mask_id = "old-mask-uuid"
        row = f"{session_id}:{old_mask_id}"
        tiny_png = _upload_png_bytes()
        bridge.save_parquet(
            str(d / "segment.parquet"),
            [f"{session_id}:_source", row, row, row, row],
            ["image_bytes", "crop_bytes", "mask_bytes", "mask_type", "pass"],
            [aa_persistence._b64(tiny_png), aa_persistence._b64(tiny_png), aa_persistence._b64(tiny_png),
             "in", "foreground"],
        )

        raw = aa_persistence.read_session_raw(session_id)
        assert raw is not None
        assert len(raw["segments"]) == 1
        seg = raw["segments"][0]
        assert seg["segment_id"] == old_mask_id
        assert seg["pass"] == "0"
        assert seg["dataset_status"] == "unassigned"
        assert seg["held"] is False
        assert seg["caption"] is None
        assert raw["pass_images"] == {}

    def test_pre_v2_background_sentinel_migrates_to_pass_one_image(self, tmp_path, monkeypatch):
        monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf_storage")
        session_id = "legacy-session-bg"
        d = aa_persistence.session_dir(session_id)
        bridge.save_parquet(str(d / "registry.parquet"), [session_id], ["name"], ["Legacy"])
        tiny_png = _upload_png_bytes()
        bridge.save_parquet(
            str(d / "segment.parquet"),
            [f"{session_id}:_source", f"{session_id}:_background"],
            ["image_bytes", "background_image_bytes"],
            [aa_persistence._b64(tiny_png), aa_persistence._b64(tiny_png)],
        )

        raw = aa_persistence.read_session_raw(session_id)
        assert set(raw["pass_images"].keys()) == {1}

    def test_reconstructed_session_resumes_at_pass_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(aa_persistence, "STORAGE_ROOT", tmp_path / "sf_storage")
        fake_processor = FakeProcessor()
        fake_service = SegmentationService(tmp_path / "sessions", fake_processor)

        session_id = "legacy-session-2"
        d = aa_persistence.session_dir(session_id)
        bridge.save_parquet(str(d / "registry.parquet"), [session_id], ["name"], ["Legacy"])
        old_mask_id = "bare-uuid-no-pass-prefix"
        row = f"{session_id}:{old_mask_id}"
        tiny_png = _upload_png_bytes()
        bridge.save_parquet(
            str(d / "segment.parquet"),
            [f"{session_id}:_source", row, row],
            ["image_bytes", "crop_bytes", "mask_bytes"],
            [aa_persistence._b64(tiny_png)] * 3,
        )

        loaded = fake_service.get_session(session_id)
        assert loaded is not None
        sf_session = fake_service.sf_sessions[session_id]
        assert len(sf_session.masks) == 1
        migrated = sf_session.masks[0]
        assert migrated.mask_id == f"0:{old_mask_id}"
        assert migrated.pass_ == 0
        assert migrated.dataset_status == sf_engine.DatasetStatus.UNASSIGNED
        assert sf_session.pass_ == 0


class TestCompileTrainingSet:
    """Only dataset_status == keep AND captioned masks reach the compiled
    snapshot — the implicit-discard behavior, proven end to end through the
    real HTTP + compile pipeline, not just at the sf_engine unit level."""

    def test_only_captioned_keeper_is_compiled(self, client, tmp_path):
        session_id = _upload(client)
        keeper_id = _box_select(client, session_id, [0.2, 0.2, 0.2, 0.2])
        balloon_id = _box_select(client, session_id, [0.7, 0.7, 0.2, 0.2], text_substitute="word balloon")
        _hold(client, session_id, keeper_id, True)
        _caption(client, session_id, keeper_id, "a red egg")
        _hold(client, session_id, balloon_id, True)  # held, but never captioned

        client.post("/lama/scrub", json={"session_id": session_id})
        client.post("/saveSession", json={"session_id": session_id})

        out_dir = tmp_path / "compiled"
        r = client.post("/dataset/compile", json={"output_dir": str(out_dir)})
        assert r.status_code == 200
        result = r.json()
        assert result["entry_count"] == 1

        lines = (out_dir / "metadata.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        import json as _json
        entry = _json.loads(lines[0])
        assert entry["text"] == "a red egg"
        assert entry["segment_id"] == keeper_id
        assert (out_dir / entry["file_name"]).exists()

    def test_compile_is_idempotent_across_two_sessions(self, client, tmp_path):
        s1 = _upload(client)
        m1 = _box_select(client, s1, [0.2, 0.2, 0.2, 0.2])
        _caption(client, s1, m1, "first image's keeper")
        client.post("/saveSession", json={"session_id": s1})

        out_dir = tmp_path / "compiled"
        r1 = compile_training_set.compile_training_set(str(out_dir))
        assert r1["entry_count"] == 1

        s2 = _upload(client)
        m2 = _box_select(client, s2, [0.3, 0.3, 0.2, 0.2])
        _caption(client, s2, m2, "second image's keeper")
        client.post("/saveSession", json={"session_id": s2})

        r2 = compile_training_set.compile_training_set(str(out_dir))
        assert r2["entry_count"] == 2  # re-run picked up both, not appended onto the first run's file

        lines = (out_dir / "metadata.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
