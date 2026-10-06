"""Prompt templating v1 — the pure expander, registry persistence, and the
read-only draft endpoint.

The expanded text is NEVER persisted per mask: there is no MaskRecord field
for it. It is computed on demand, pre-fills the caption box, and becomes
permanent only if the user presses Save Caption, which is the existing
/mask/caption path.

Fixtures are COLOURED and NON-UNIFORM. A flat fixture would make a
crop-from-the-wrong-pass or wrong-panel bug invisible.

The bridge is stubbed rather than hitting real Julia parquet I/O —
test_sf_wiring.py, which does, hangs in this environment.
"""

from __future__ import annotations

import io
import pathlib

import numpy as np
import pytest
from PIL import Image

import aa_persistence
import prompt_template as pt
import sf_engine as sfe
from services import SegmentationService, SessionLoadError

SID = "template-session"
IMG = 200


def _colour_png(size=IMG, tint=0) -> bytes:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            arr[y, x] = ((20 + x * 3 + tint * 60) % 256,
                         (70 + y * 2 + tint * 25) % 256,
                         (130 + (x + y) + tint * 90) % 256)
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
    """Records every call, so "never reaches SAM3" is asserted against the
    fake rather than inferred from reading the code."""

    def __init__(self, size=IMG):
        self.size = size
        self.geo_rects: list[tuple] = []
        self.calls: list[str] = []

    def _state(self, rects):
        return {"masks": [self._m(r) for r in rects],
                "boxes": [[float(x0), float(y0), float(x1), float(y1)]
                          for (y0, y1, x0, x1) in rects],
                "scores": [0.9] * len(rects),
                "original_width": self.size, "original_height": self.size}

    def _m(self, r):
        y0, y1, x0, x1 = r
        m = np.zeros((self.size, self.size), dtype=np.float32)
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


def _upload(service, template=None):
    contents = _colour_png()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    state = service.processor.set_image(image)
    data = {"name": "template page", "description": "", "image_url": "",
            "state": state, "original_image_bytes": contents,
            "image_size": image.size}
    if template is not None:
        data["prompt_template"] = template
    service.register_session_data(SID, data)
    return service.sf_sessions[SID]


def _restart(service):
    service.sessions.clear()
    service.sf_sessions.clear()


def _norm_box(rect):
    y0, y1, x0, x1 = rect
    return [((x0 + x1) / 2) / IMG, ((y0 + y1) / 2) / IMG,
            (x1 - x0) / IMG, (y1 - y0) / IMG]


# ── The pure expander ────────────────────────────────────────────────────────


class TestExpandPromptTemplate:
    def test_each_placeholder_expands(self):
        got = pt.expand_prompt_template(
            "tag={{text_tag}} pass={{pass}} bbox={{bbox}} panel={{panel}}",
            text_tag="nemo", pass_=2, bbox=[10, 20, 110, 140],
            panel_label=pt.panel_label(3),
        )
        assert got == "tag=nemo pass=2 bbox=10,20,110,140 panel=panel 3"

    def test_whitespace_inside_the_braces_is_tolerated(self):
        assert pt.expand_prompt_template("{{ text_tag }}", text_tag="x") == "x"

    def test_an_unknown_name_is_left_literal(self):
        got = pt.expand_prompt_template("{{nope}} and {{text_tag}}", text_tag="x")
        assert got == "{{nope}} and x"

    def test_a_known_placeholder_with_no_value_renders_empty(self):
        """Not the literal {{text_tag}} — a mask with no tag genuinely has
        nothing to say there, and the raw placeholder would read as a bug."""
        assert pt.expand_prompt_template("[{{text_tag}}]", text_tag=None) == "[]"

    def test_stray_and_malformed_braces_never_raise(self):
        src = 'stray { and } and {{ and {{ }} and {{a b}} and {"json": 1}'
        assert pt.expand_prompt_template(src) == src

    def test_str_format_would_have_raised_on_that_input(self):
        """Pins WHY this is regex-based: the same string kills str.format."""
        with pytest.raises((KeyError, ValueError, IndexError)):
            'stray { and } and {"json": 1}'.format()

    def test_substituted_values_are_never_re_expanded(self):
        """A text_tag containing a placeholder must come out literally, or
        one field's data would leak into another's slot."""
        got = pt.expand_prompt_template("{{text_tag}}", text_tag="{{pass}}", pass_=7)
        assert got == "{{pass}}"

    def test_text_without_placeholders_is_returned_unchanged(self):
        src = "just a plain caption prompt, no braces at all"
        assert pt.expand_prompt_template(src) == src
        # Idempotent: expanding the result again changes nothing.
        assert pt.expand_prompt_template(pt.expand_prompt_template(src)) == src

    def test_an_empty_template_expands_to_empty(self):
        assert pt.expand_prompt_template("", text_tag="x") == ""

    def test_bbox_renders_as_integer_page_pixels(self):
        assert pt.format_bbox([1.0, 2.0, 3.9, 4.2]) == "1,2,3,4"
        assert pt.format_bbox(None) == ""

    def test_panel_label_is_one_based_or_full_page(self):
        assert pt.panel_label(1) == "panel 1"
        assert pt.panel_label(12) == "panel 12"
        assert pt.panel_label(None) == "full page"

    def test_the_whitelist_is_exactly_four_names(self):
        assert set(pt.KNOWN_PLACEHOLDERS) == {"text_tag", "pass", "bbox", "panel"}

    def test_repeated_placeholders_all_expand(self):
        got = pt.expand_prompt_template("{{pass}}/{{pass}}", pass_=1)
        assert got == "1/1"


# ── Registry persistence ─────────────────────────────────────────────────────


class TestTemplatePersistence:
    def test_init_session_persists_the_template(self, service):
        _upload(service, template="describe {{text_tag}}")
        service.save_session_to_disk(SID)
        _restart(service)

        raw = aa_persistence.read_session_raw(SID)
        assert raw["prompt_template"] == "describe {{text_tag}}"

    def test_round_trips_through_loadSession(self, service):
        _upload(service, template="a {{panel}} template")
        service.save_session_to_disk(SID)
        _restart(service)

        response = service.load_session_from_disk(SID)
        assert response["prompt_template"] == "a {{panel}} template"

    def test_a_session_with_no_template_reads_back_empty(self, service):
        _upload(service)
        service.save_session_to_disk(SID)
        _restart(service)

        assert service.load_session_from_disk(SID)["prompt_template"] == ""

    def test_an_empty_template_does_not_clobber_a_stored_one(self, service):
        """The guard's whole purpose: register_session_data MERGES, so a key
        present with "" would overwrite. DN sends "" on every call that is
        not creating a new session."""
        _upload(service, template="the real template")
        # A later merge that carries no template at all (the guard in
        # main.init_session is what omits the key).
        service.register_session_data(SID, {"name": "renamed"})

        assert service.sessions[SID]["prompt_template"] == "the real template"
        service.save_session_to_disk(SID)
        _restart(service)
        assert service.load_session_from_disk(SID)["prompt_template"] == "the real template"

    def test_a_resumed_session_warms_the_template_into_memory(self, service):
        """Both warm paths must carry it — register_session_data's merge AND
        _load_session_into_memory's own dict — or the draft endpoint would
        see nothing for a resumed session."""
        _upload(service, template="warmed {{pass}}")
        service.save_session_to_disk(SID)
        _restart(service)

        service.get_session(SID)
        assert service.sessions[SID]["prompt_template"] == "warmed {{pass}}"

    def test_a_staged_session_restore_keeps_the_template(self, service):
        """The restore-before-merge guard: /upload reaching a cold cache for
        a session genuinely on disk must restore the template rather than
        blank it. Being in _REGISTRY_FIELDS is what makes that work."""
        sf = _upload(service, template="survives the merge")
        contents = sf.original.bytes
        service.save_session_to_disk(SID)
        _restart(service)  # cold cache, no /loadSession first

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        service.register_session_data(SID, {
            "name": "template page", "description": "", "image_url": "",
            "state": service.processor.set_image(image),
            "original_image_bytes": contents, "image_size": image.size,
        })

        assert service.sessions[SID]["prompt_template"] == "survives the merge"

    def test_a_failed_restore_still_raises_rather_than_blanking(
        self, service, monkeypatch
    ):
        sf = _upload(service, template="must not be lost")
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

    def test_the_template_is_in_the_registry_fields_tuple(self):
        """That tuple is both the write list (_build_registry iterates it)
        and what register_session_data's cold-cache branch restores."""
        assert "prompt_template" in aa_persistence.REGISTRY_FIELDS


# ── The draft endpoint ───────────────────────────────────────────────────────


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


def _select(service, sf, rect):
    service.processor.geo_rects = [rect]
    return sf.add_box_selection(_norm_box(rect), True)[0]


class TestPromptDraftEndpoint:
    def test_it_expands_the_template_for_the_named_mask(self, client, service):
        sf = _upload(service, template="tag={{text_tag}} pass={{pass}} bbox={{bbox}} on {{panel}}")
        rec = _select(service, sf, (20, 60, 20, 60))

        r = client.post("/mask/prompt-draft", json={"session_id": SID, "mask_id": rec.mask_id})

        assert r.status_code == 200
        draft = r.json()["draft"]
        assert draft.startswith("tag=")
        assert "pass=0" in draft
        assert "bbox=20,20,60,60" in draft
        assert "on full page" in draft, "no panels authored -> full-page fallback"

    def test_it_uses_the_panel_assignment_when_panels_exist(self, client, service):
        sf = _upload(service, template="{{panel}}")
        sf.add_panel(0, 0, 100, 100)      # panel 1
        sf.add_panel(100, 0, 200, 100)    # panel 2
        rec = _select(service, sf, (20, 60, 120, 160))  # sits in panel 2

        r = client.post("/mask/prompt-draft", json={"session_id": SID, "mask_id": rec.mask_id})

        assert r.json()["draft"] == "panel 2", "1-based, matching the Panels card"

    def test_the_innermost_panel_wins_for_a_nested_pair(self, client, service):
        sf = _upload(service, template="{{panel}}")
        sf.add_panel(0, 0, 200, 200)      # host, panel 1
        sf.add_panel(10, 10, 90, 90)      # inset, panel 2
        rec = _select(service, sf, (20, 60, 20, 60))

        assert client.post("/mask/prompt-draft",
                           json={"session_id": SID, "mask_id": rec.mask_id}
                           ).json()["draft"] == "panel 2"

    def test_an_empty_draft_when_the_session_has_no_template(self, client, service):
        sf = _upload(service)  # no template at all
        rec = _select(service, sf, (20, 60, 20, 60))

        r = client.post("/mask/prompt-draft", json={"session_id": SID, "mask_id": rec.mask_id})

        assert r.status_code == 200
        assert r.json()["draft"] == "", "empty is what tells the UI to stand back"

    def test_it_never_calls_sam3(self, client, service):
        sf = _upload(service, template="{{text_tag}} {{pass}} {{bbox}} {{panel}}")
        sf.add_panel(0, 0, 100, 100)
        rec = _select(service, sf, (20, 60, 20, 60))
        service.processor.calls.clear()

        client.post("/mask/prompt-draft", json={"session_id": SID, "mask_id": rec.mask_id})

        assert service.processor.calls == [], (
            "building a draft is pure string work — it must not reach the "
            "segmentor at all"
        )

    def test_it_writes_nothing_and_changes_no_record(self, client, service):
        sf = _upload(service, template="{{text_tag}}")
        rec = _select(service, sf, (20, 60, 20, 60))
        before = sf.get_mask(rec.mask_id)
        before_fields = (before.pass_, before.dataset_status, before.held,
                         before.caption, before.text_tag)
        n_masks, n_panels = len(sf.masks), len(sf.panels)

        client.post("/mask/prompt-draft", json={"session_id": SID, "mask_id": rec.mask_id})

        after = sf.get_mask(rec.mask_id)
        assert (after.pass_, after.dataset_status, after.held,
                after.caption, after.text_tag) == before_fields
        assert len(sf.masks) == n_masks and len(sf.panels) == n_panels
        # And nothing was persisted: no caption appeared anywhere.
        assert after.caption is None

    def test_the_draft_is_not_stored_on_the_record(self, client, service):
        """Scope limit: there is NO MaskRecord field for this. It becomes
        persistent only via the existing Save Caption path."""
        sf = _upload(service, template="a draft")
        rec = _select(service, sf, (20, 60, 20, 60))
        client.post("/mask/prompt-draft", json={"session_id": SID, "mask_id": rec.mask_id})

        assert not hasattr(sf.get_mask(rec.mask_id), "prompt_draft")
        assert not hasattr(sf.get_mask(rec.mask_id), "llm_prompt")

    def test_404_for_an_unknown_session(self, client):
        r = client.post("/mask/prompt-draft",
                        json={"session_id": "nope", "mask_id": "0:x"})
        assert r.status_code == 404

    def test_404_for_an_unknown_mask(self, client, service):
        _upload(service, template="t")
        r = client.post("/mask/prompt-draft",
                        json={"session_id": SID, "mask_id": "0:no-such-mask"})
        assert r.status_code == 404

    def test_a_text_tag_containing_a_placeholder_is_not_re_expanded(
        self, client, service
    ):
        """End to end through the endpoint, not just the pure function."""
        sf = _upload(service, template="[{{text_tag}}]")
        service.processor.geo_rects = [(20, 60, 20, 60)]
        rec = sf.add_box_selection(_norm_box((20, 60, 20, 60)), True,
                                   text_substitute="{{pass}}")[0]

        draft = client.post("/mask/prompt-draft",
                            json={"session_id": SID, "mask_id": rec.mask_id}
                            ).json()["draft"]

        assert draft == "[{{pass}}]"

    def test_it_works_through_the_real_resume_sequence(self, client, service):
        """GET /loadSession first, exactly as the frontend does on resume —
        that call is what warms both caches. `get_or_create_sf_session`
        (shared with /mask/hold, /mask/caption and /panel/*) reads only the
        in-memory cache and never loads from disk itself, so a request
        arriving before any /loadSession 404s. That is pre-existing behaviour
        of every one of those endpoints, not something templating changed."""
        sf = _upload(service, template="resumed {{pass}}")
        rec = _select(service, sf, (20, 60, 20, 60))
        service.sessions[SID]["state"] = sf.state
        service.save_session_to_disk(SID)
        _restart(service)

        service.load_session_from_disk(SID)   # the frontend's own first call
        draft = client.post("/mask/prompt-draft",
                            json={"session_id": SID, "mask_id": rec.mask_id}
                            ).json()["draft"]

        assert draft == "resumed 0"

    def test_a_cold_cache_404s_like_every_other_mask_endpoint(
        self, client, service
    ):
        """Pins the shared contract rather than leaving it undiscovered."""
        sf = _upload(service, template="t")
        rec = _select(service, sf, (20, 60, 20, 60))
        service.sessions[SID]["state"] = sf.state
        service.save_session_to_disk(SID)
        _restart(service)

        for path, body in (
            ("/mask/prompt-draft", {"session_id": SID, "mask_id": rec.mask_id}),
            ("/mask/hold", {"session_id": SID, "mask_id": rec.mask_id, "held": True}),
        ):
            assert client.post(path, json=body).status_code == 404, path

    def test_the_endpoint_guard_stops_an_empty_template_clobbering(
        self, client, service
    ):
        """Through the real POST /initSession, not register_session_data —
        the guard lives in the endpoint, so only an endpoint-level test can
        catch its removal. DN sends prompt_template:"" on every call that is
        not creating a new session."""
        _upload(service, template="the stored template")

        r = client.post("/initSession", json={
            "session_id": SID, "name": "n", "description": "d",
            "image_url": "", "mode": "", "prompt_template": "",
        })

        assert r.status_code == 200
        assert service.sessions[SID]["prompt_template"] == "the stored template"

    def test_the_endpoint_does_store_a_supplied_template(self, client, service):
        """The other half: a non-empty template must actually be written, or
        the guard would be trivially satisfiable by storing nothing."""
        _upload(service)

        client.post("/initSession", json={
            "session_id": SID, "name": "n", "description": "d",
            "image_url": "", "mode": "", "prompt_template": "a new template",
        })

        assert service.sessions[SID]["prompt_template"] == "a new template"
