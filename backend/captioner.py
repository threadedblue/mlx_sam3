"""Run the trained captioner (mlx-vlm) over SF segments.

The captioner is a Qwen2-VL model plus a small LoRA adapter, run through
`python -m mlx_vlm generate` — the same call DoubleNaught's Caption node
makes. It lives in whatever Python has mlx-vlm installed (DN's backend venv
today), so SF runs it as a SUBPROCESS of that interpreter rather than
importing it; SF's own environment needs nothing new.

Configuration (environment, read per call so a restart is not needed):
    SF_CAPTIONER_PYTHON   interpreter with mlx-vlm installed. Default: DN's
                          backend venv next to this repo, if it exists.
    SF_CAPTIONER_MODEL    base model id. Default Qwen2-VL-2B-Instruct-bf16.
    SF_CAPTIONER_ADAPTER  adapter .safetensors path. Empty = plain base model.

A run is a background job over a list of masks: one subprocess per mask
(model reload each time — slow but simple), each result written to the
session as a DRAFT the moment it arrives, so the table fills in live.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

DEFAULT_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-bf16"
PROMPT = "Describe this image."
_DN_VENV_PYTHON = Path(__file__).resolve().parents[2] / "DoubleNaught" / "backend" / ".venv" / "bin" / "python"


def masks_needing_draft(masks) -> List[str]:
    """Mask ids a run should caption: nothing from a person (caption), no
    earlier model attempt (draft), and not marked "never needs" (`discard`).
    This is what makes re-entering a half-captioned session pick up where it
    left off instead of redoing work."""
    return [
        m.mask_id for m in masks
        if not (m.caption and m.caption.strip()) and not m.draft_caption
        and getattr(m.dataset_status, "value", m.dataset_status) != "discard"
    ]


class CaptionerError(RuntimeError):
    pass


def config() -> Dict[str, str]:
    python = os.environ.get("SF_CAPTIONER_PYTHON", "")
    if not python and _DN_VENV_PYTHON.is_file():
        python = str(_DN_VENV_PYTHON)
    return {
        "python": python,
        "model": os.environ.get("SF_CAPTIONER_MODEL", DEFAULT_MODEL),
        "adapter": os.environ.get("SF_CAPTIONER_ADAPTER", ""),
    }


def clean_output(text: str) -> str:
    """The generated text only. `mlx_vlm generate` can wrap it in `=====`
    banners and trail token/speed stats; keep what is between them."""
    kept: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("=========="):
            if kept:
                break  # closing banner: the stats follow
            continue  # opening banner
        kept.append(line)
    return "\n".join(kept).strip()


def caption_file(image_path: str, timeout: float = 600) -> str:
    cfg = config()
    if not cfg["python"]:
        raise CaptionerError("no captioner interpreter: set SF_CAPTIONER_PYTHON to a Python with mlx-vlm installed")
    if cfg["adapter"] and not Path(cfg["adapter"]).is_file():
        raise CaptionerError(f"adapter not found: {cfg['adapter']}")
    cmd = [
        cfg["python"], "-m", "mlx_vlm", "generate",
        "--model", cfg["model"], "--image", image_path, "--prompt", PROMPT,
        "--max-tokens", "200", "--temperature", "0",
    ]
    if cfg["adapter"]:
        cmd += ["--adapter-path", cfg["adapter"]]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise CaptionerError(f"captioning timed out after {int(timeout)}s")
    if done.returncode != 0:
        tail = (done.stderr or "").strip().splitlines()[-1:] or ["no error output"]
        raise CaptionerError(f"caption failed: {tail[0]}")
    text = clean_output(done.stdout)
    if not text:
        raise CaptionerError("the model returned no text")
    return text


def caption_png(png_bytes: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "segment.png")
        with open(path, "wb") as fh:
            fh.write(png_bytes)
        return caption_file(path)


class _Job:
    def __init__(self, total: int) -> None:
        self.id = uuid.uuid4().hex
        self.total = total
        self.done = 0
        self.failed: List[str] = []
        self.state = "running"  # running | done | cancelled
        self.cancel = threading.Event()
        self.last_error: Optional[str] = None


_jobs: Dict[str, _Job] = {}
_lock = threading.Lock()


def start(items: List[str], caption_one: Callable[[str], None]) -> _Job:
    """Run `caption_one(item)` for each item on a background thread.

    One job at a time: the model is the scarce resource, and two jobs
    interleaving would just halve each one's speed.
    """
    with _lock:
        if any(j.state == "running" for j in _jobs.values()):
            raise CaptionerError("a captioner run is already in progress")
        job = _Job(len(items))
        _jobs[job.id] = job

    def work() -> None:
        for item in items:
            if job.cancel.is_set():
                job.state = "cancelled"
                return
            try:
                caption_one(item)
            except Exception as exc:  # one bad crop must not stop the batch
                job.failed.append(item)
                job.last_error = str(exc)
            job.done += 1
        job.state = "done"

    threading.Thread(target=work, daemon=True).start()
    return job


def status(job_id: str) -> Optional[Dict[str, object]]:
    job = _jobs.get(job_id)
    if job is None:
        return None
    return {
        "job_id": job.id, "state": job.state, "total": job.total, "done": job.done,
        "failed": list(job.failed), "last_error": job.last_error,
    }


def cancel(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if job is None:
        return False
    job.cancel.set()
    return True
