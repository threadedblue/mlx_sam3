"""Whether this running backend is serving its own current source.

A long-lived uvicorn keeps serving whatever it imported at startup. Nothing in
the HTTP surface distinguishes that from a freshly-started one: `/health`
answers 200 identically either way, so a launcher that reuses "anything
healthy" will reuse a process running days-old code indefinitely. That is not
hypothetical — a SegForge backend started Sept 30 served pre-fix code through
Oct 2, surviving every restart attempt, because the only question anyone asked
it was "are you listening?".

So the backend answers a second question about itself: "is the source on disk
still the source I imported?" It is the right party to answer it — it knows
its own source tree without a caller having to guess paths, and the answer is
meaningful even for a standalone launch with no DoubleNaught involved.

Acting on the answer necessarily falls to someone else: a stale process cannot
restart itself into new code. `/health` therefore reports the fact and the
launcher decides (see DoubleNaught's `_ensureBackendHealthy`).

CONTENT HASH, NOT MTIME. The obvious implementation — compare the newest source
mtime against process start time — false-positives on any touch that doesn't
change content: an editor save with no edit, a `git checkout` that rewrites an
identical file, a build tool stamping files. Each would force a pointless
restart mid-session. Hashing the bytes cannot do that, and cannot false-negative
on coarse mtime resolution either. The cost is trivial at this size (~23 files,
~470 KB, low single-digit milliseconds) against a health endpoint polled every
10 seconds.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent

# Compiled bytecode mirrors the .py files it came from, so hashing it would
# double-count; it also churns on import, which would read as a source change.
_EXCLUDED_DIRS = {"__pycache__", ".venv", ".pytest_cache", "tests"}


def _source_files() -> list[Path]:
    """Every .py file this backend actually runs from, in stable order.

    `tests/` is excluded deliberately: editing a test cannot change what the
    running server serves, and restarting a loaded SAM3 model (tens of seconds)
    for a test edit would be a real cost for no behavioural difference.
    """
    out: list[Path] = []
    for path in BACKEND_DIR.rglob("*.py"):
        if any(part in _EXCLUDED_DIRS for part in path.relative_to(BACKEND_DIR).parts):
            continue
        out.append(path)
    return sorted(out)


def compute_digest() -> str:
    """SHA-256 over the backend's source, path-sensitive.

    Paths are hashed alongside contents so that renaming, adding or deleting a
    file registers as a change even when the surviving bytes are identical.
    An unreadable file (deleted mid-walk) is folded in as a distinct marker
    rather than raising: a partially-readable tree is itself a change worth
    reporting, and `/health` must not 500 because of a transient.
    """
    h = hashlib.sha256()
    for path in _source_files():
        h.update(str(path.relative_to(BACKEND_DIR)).encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(path.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()


# Captured once, at import — i.e. the digest of the source this process
# actually loaded. Everything after this point compares against it.
STARTED_AT: float = time.time()
DIGEST_AT_START: str = compute_digest()


def status() -> dict:
    """The staleness facts, for `/health` to report.

    `stale` is the whole point: True means the files on disk no longer match
    what this process imported, so it is serving code that no longer exists.
    """
    current = compute_digest()
    return {
        "started_at": STARTED_AT,
        "source_digest_at_start": DIGEST_AT_START,
        "source_digest_now": current,
        "stale": current != DIGEST_AT_START,
    }
