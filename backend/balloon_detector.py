"""Comic speech-balloon detection via ogkalu/comic-speech-bubble-detector-yolov8m.

Weights are downloaded on first use and cached by huggingface-hub, NOT vendored
into the repo — a deliberate choice: the checkpoint is 52MB, `huggingface-hub`
is already a dependency, and vendoring it would put a binary that large into git
history permanently. The cost is that the first call needs network access; every
call after that is offline off the HF cache.

Validated zero-shot against Little Nemo pages 0026/0042/0122: 96% recall of text
regions at conf>=0.05, and zero false positives on artwork at every threshold
tested -- including the dense Art Nouveau lily field and the pale ovoid eggs on
0026, both of which were expected lookalike risks and produced nothing.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from PIL import Image

HF_REPO = "ogkalu/comic-speech-bubble-detector-yolov8m"
HF_FILE = "comic-speech-bubble-detector.pt"

# Part 1's finding: both classes fire on real balloons, and on some pages the
# only detection for a real balloon is `text_free`. Filtering to text_bubble
# would silently drop them, so both classes are accepted.
ACCEPTED_CLASSES = ("text_bubble", "text_free")

_model: Optional[Any] = None


def _get_model():
    """Process-wide YOLO instance — loading it reads a 52MB checkpoint, so this
    must not happen fresh on every request (mirrors `_get_lama_inpainter`)."""
    global _model
    if _model is None:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        _model = YOLO(hf_hub_download(HF_REPO, HF_FILE))
    return _model


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedup(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Collapse the two classes' near-identical boxes into one region each.

    Both classes routinely fire on the same balloon with boxes differing by a
    pixel or two; without this, the same balloon is selected twice.
    """
    kept: list[dict] = []
    for d in sorted(detections, key=lambda x: -x["confidence"]):
        if all(_iou(d["box"], k["box"]) <= iou_threshold for k in kept):
            kept.append(d)
    return kept


def detect(image: Image.Image, confidence: float = 0.25) -> list[dict]:
    """Detected text regions, deduped, ordered most-confident first.

    Each entry: `box` as pixel xyxy, `confidence`, `cls`.
    """
    model = _get_model()
    result = model.predict(image, conf=confidence, verbose=False)[0]

    detections = []
    for b in result.boxes:
        name = model.names[int(b.cls[0])]
        if name not in ACCEPTED_CLASSES:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        detections.append(
            {
                "box": [x1, y1, x2, y2],
                "confidence": round(float(b.conf[0]), 4),
                "cls": name,
            }
        )
    return dedup(detections)
