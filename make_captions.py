"""
AI-powered panel captioning for Little Nemo LoRA dataset.
Uses a local Ollama vision model to generate content-descriptive captions.

Usage:
  # List available models first
  python make_captions.py --list-models

  # Run captioning
  python make_captions.py panels/ metadata.jsonl

  # Specify model or host explicitly
  python make_captions.py panels/ metadata.jsonl --model llava:13b
  python make_captions.py panels/ metadata.jsonl --host http://localhost:11434

Requires:  pip install opencv-python requests
Resume-safe: already-captioned entries in the output jsonl are skipped,
so you can Ctrl-C and restart freely.
"""

import argparse
import base64
import json
import os
import glob
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import requests

# ── configuration ─────────────────────────────────────────────────────────────

TRIGGER       = "WINSORMCCAY"
STYLE_SUFFIX  = ("Winsor McCay art, art nouveau illustration, "
                 "vintage newspaper comic strip, "
                 "pen and ink with watercolor, early 1900s")
DEFAULT_HOST  = "http://localhost:11434"
DEFAULT_MODEL = "llava"       # change to llava:13b, llava-phi3, moondream, etc.
MAX_WORKERS   = 2             # Ollama is single-threaded; 2 is usually the sweet spot
MAX_IMG_DIM   = 672           # resize before sending — keeps it fast

VISION_PROMPT = """\
You are writing image captions for a LoRA training dataset of Winsor McCay's \
"Little Nemo in Slumberland" comic strip panels (early 1900s newspaper comics).

Describe what you see concisely and specifically. Cover:
- Main subjects: characters, creatures, figures, objects
- Setting or environment: palace, forest, ocean, dreamscape, etc.
- Action or mood
- Any notable visual or architectural details

Rules:
- Do NOT quote or describe speech bubble text
- Do NOT start with "This panel shows", "This image", or similar preamble
- Do NOT mention panel numbers or borders
- Be specific — e.g. "young boy in white nightgown", "giant ornate archway", \
"anthropomorphic rabbit pulling chariot"
- 30–55 words maximum
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def list_models(host: str):
    """Print available models from the Ollama instance."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json().get("models", [])
        if not models:
            print("No models found.")
            return
        print(f"Models available at {host}:")
        for m in models:
            size = m.get("details", {}).get("parameter_size", "?")
            print(f"  {m['name']:<35} {size}")
    except Exception as e:
        print(f"Could not reach Ollama at {host}: {e}")


def load_done(output_path: str) -> set:
    """Return filenames already present in the output jsonl."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["file_name"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def encode_image(img_path: str, max_dim: int = MAX_IMG_DIM) -> str | None:
    """Read, resize, and base64-encode a panel image as JPEG."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def composition_hint(img_path: str) -> str:
    img = cv2.imread(img_path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    ratio = w / h
    stem = Path(img_path).stem
    panel_idx = int(stem.rsplit("_p", 1)[1])
    if panel_idx == 0 and ratio > 3.0:
        return "This is the decorative title banner for 'Little Nemo in Slumberland'."
    if ratio > 3.0:
        return "This is a wide panoramic comic strip panel."
    return "This is a single comic strip panel."


# ── captioning ────────────────────────────────────────────────────────────────

def caption_one(host: str,
                model: str,
                img_path: str,
                write_lock: threading.Lock,
                out_file,
                max_img_dim: int) -> tuple[str, str | None, str | None]:
    """
    Caption one panel via Ollama. Returns (filename, caption, error_or_None).
    Writes the JSONL record immediately on success.
    """
    fname = os.path.basename(img_path)

    img_b64 = encode_image(img_path, max_dim=max_img_dim)
    if img_b64 is None:
        return fname, None, "unreadable"

    hint   = composition_hint(img_path)
    prompt = f"{hint}\n\n{VISION_PROMPT}".strip()

    payload = {
        "model":  model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.3,   # low temp → consistent captions
            "num_predict": 120,
        },
    }

    for attempt in range(3):
        try:
            r = requests.post(
                f"{host}/api/generate",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            description = r.json().get("response", "").strip()
            if not description:
                return fname, None, "empty response"

            caption = f"{TRIGGER}, {description}, {STYLE_SUFFIX}"
            record  = {"file_name": fname, "caption": caption}

            with write_lock:
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()

            return fname, caption, None

        except requests.exceptions.Timeout:
            wait = 10 * (attempt + 1)
            print(f"    timeout on {fname}, waiting {wait}s …")
            time.sleep(wait)
        except requests.exceptions.RequestException as e:
            return fname, None, str(e)

    return fname, None, "timeout — max retries exceeded"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Caption Little Nemo panels with a local Ollama vision model"
    )
    parser.add_argument("panels_dir", nargs="?",
                        help="Directory containing panel PNG files")
    parser.add_argument("output", nargs="?",
                        help="Output metadata.jsonl path")
    parser.add_argument("--host",  default=DEFAULT_HOST,
                        help=f"Ollama host URL (default: {DEFAULT_HOST})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama vision model (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Concurrent requests (default: {MAX_WORKERS})")
    parser.add_argument("--max-image-dim", type=int, default=MAX_IMG_DIM,
                        help=f"Max image dimension before sending (default: {MAX_IMG_DIM}px)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models from Ollama and exit")
    args = parser.parse_args()

    if args.list_models:
        list_models(args.host)
        return

    if not args.panels_dir or not args.output:
        parser.error("panels_dir and output are required (unless using --list-models)")

    # ── verify Ollama is reachable and model exists ──
    try:
        r = requests.get(f"{args.host}/api/tags", timeout=5)
        r.raise_for_status()
        available = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama at {args.host}\n  {e}")
        return

    # Accept partial name match (e.g. "llava" matches "llava:latest")
    matched = next((m for m in available
                    if m == args.model or m.startswith(args.model + ":")), None)
    if not matched:
        print(f"ERROR: Model '{args.model}' not found at {args.host}")
        print(f"Available: {', '.join(available) or 'none'}")
        print("Pull a vision model with:  docker exec <container> ollama pull llava")
        return
    model = matched

    all_panels = sorted(glob.glob(os.path.join(args.panels_dir, "*.png")))
    done       = load_done(args.output)
    pending    = [p for p in all_panels if os.path.basename(p) not in done]

    print(f"Ollama host   : {args.host}")
    print(f"Model         : {model}")
    print(f"Panels total  : {len(all_panels)}")
    print(f"Already done  : {len(done)}")
    print(f"To caption    : {len(pending)}")
    print(f"Concurrency   : {args.workers}")
    print()

    if not pending:
        print("All panels already captioned — nothing to do.")
        return

    write_lock = threading.Lock()
    errors     = []
    done_count = len(done)
    total      = len(all_panels)

    with open(args.output, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    caption_one,
                    args.host, model, p,
                    write_lock, out_f, args.max_image_dim
                ): p
                for p in pending
            }
            for future in as_completed(futures):
                fname, caption, err = future.result()
                done_count += 1
                if err:
                    errors.append((fname, err))
                    print(f"  [{done_count:>5}/{total}] ERROR  {fname} — {err}")
                else:
                    snippet = caption[:90] + "…" if len(caption) > 90 else caption
                    print(f"  [{done_count:>5}/{total}] {fname}  →  {snippet}")

    success = done_count - len(errors)
    print(f"\nFinished. {success} captioned, {len(errors)} errors.")
    if errors:
        print("\nFailed files:")
        for fname, err in errors:
            print(f"  {fname}: {err}")
    print(f"\nOutput → {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
