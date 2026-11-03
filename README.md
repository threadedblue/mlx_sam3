# SegForge

SegForge (SF) is a Flutter macOS app with a FastAPI backend for curating image segments and captions. It runs SAM 3 on Apple Silicon (MLX), lets you review the masks it finds, and exports approved segments with captions for downstream use. It is the segmentation stage of the DoubleNaught (DN) pipeline.

> Status: work in progress. Not recommended for general use yet.

## What it does

- **Segments** an image with SAM 3 from text prompts.
- **Curates** masks. Each mask has two independent axes: `dataset_status` (`unassigned` or `keep`) and `held`.
- **Passes**: an unbounded chain of passes, where foreground objects are removed with LaMa inpainting and the next pass segments what is left.
- **Captions** each mask with a local Qwen2-VL-2B captioner. A human approves or edits the draft, and the result becomes the segment's `caption`.
- **Scrubs** speech balloons (Balloon Scrub) and handles comic panels (Panels card).
- **Exports** segments with captions in the `sf.llm_payload/v1` schema.

In DN, the export feeds the LLM Documenter, which produces `better_caption` and `best_caption` under the ontology guardrails. SF stops at `caption`; it does not run the Documenter or train any LoRA.

## Modes

The app has three modes: Comix, Poster and Cine. Only **Comix** is built. Poster and Cine are visible but have no toolset yet.

## Requirements

- Apple Silicon Mac (MLX)
- Python 3.11 or newer
- Flutter with macOS desktop support
- `d4m_juliacall_bridge` (the existing backend tests import it)

## Run

```bash
./run.sh BE [--clear-storage]            # backend on http://127.0.0.1:8401 (API docs at /docs)
./run.sh FE [--url=URL] [--session=ID]   # Flutter macOS app
./run.sh STOP                            # stop the backend
```

`BE` activates `.venv`, installs `backend/requirements.txt`, and starts uvicorn with `--reload`. `FE` runs `flutter clean`, `flutter pub get`, then `flutter run -d macos`.

Environment variables:

| Variable | Purpose |
|---|---|
| `SEGFORGE_PORT` | Backend port (default 8401) |
| `SEGFORGE_BACKEND_URL` | Backend URL the app talks to (default `http://127.0.0.1:<port>`) |
| `SEGFORGE_IMAGE_URL` | Image to load at launch (same as `--url=`) |
| `SEGFORGE_SESSION_ID` | Session to open (same as `--session=`); if unset, the backend allocates one on upload |
| `SF_CAPTIONER_PROMPT_FILE` | Prompt template for the Qwen captioner |

## Captioner prompt

The captioner reads its prompt template on every run. It uses `SF_CAPTIONER_PROMPT_FILE` if set, otherwise DN's `storage/prompts/qwen_caption.md`. If the file is missing or empty it falls back to `Describe this image.`. The template is expanded per mask with its `text_tag` and pass number, so the mask's name is supplied by the tag rather than typed by hand.

## Web build

```bash
./bld.sh
```

Runs `flutter build web --debug --base-href /web/` and writes to `frontend/build/web`.

## Tests

```bash
cd backend
pytest tests/
```

Known issue: the full suite hangs at `test_sf_wiring.py::TestHttpSelectHoldCaptionScrub::test_full_flow`. It hangs without the recent captioner changes too, and the cause is unknown. That test passes when run alone, so run files individually, or deselect it, until it is fixed. The two `test_lora_train` tests have also failed and have not been investigated.

## Layout

```
backend/            FastAPI app (main.py), sf_engine.py (masks, passes), captioner.py, tests/
frontend/           Flutter app (lib/sf_mode.dart defines modes and cards)
cloud_run/          Separate FLUX.1-dev inference service (not part of the segmentation flow)
run.sh, bld.sh      Run and build scripts
main1.py, make_captions.py   Early standalone scripts, kept for reference
```

## Related

DoubleNaught (DN) hosts SF's output in its workflow and owns the ontologies and prompts. SF's scope is segmentation and `caption` only; caption improvement and ontology validation belong to DN.
