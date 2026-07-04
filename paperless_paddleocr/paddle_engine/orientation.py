"""Page orientation detection for ocrmypdf's --rotate-pages hook.

Wraps PaddleOCR's PP-LCNet document orientation classifier. The classifier
is a ~7 MB model cached under ~/.paddlex like the OCR models; the instance
is cached at module level because ocrmypdf calls the hook once per page.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from ocrmypdf.pluginspec import OrientationConfidence

log = logging.getLogger("paperless.paddleocr.orientation")

#: Below this probability a 4-way classifier is guessing (uniform prior is
#: 0.25); report "no rotation" so paperless's rotate_pages_threshold, which
#: was tuned for Tesseract's confidence scale, cannot act on noise.
MIN_PROBABILITY = 0.5

#: Classifier label -> ocrmypdf angle (degrees the content is rotated
#: clockwise; ocrmypdf applies the counterclockwise correction). Verified
#: 2026-07-03 against paddleocr 3.7.0 (PP-LCNet_x1_0_doc_ori): an upright
#: page labels "0", 90 degrees clockwise labels "90", 180 labels "180", 90
#: degrees counterclockwise labels "270". The label is the clockwise rotation
#: of the content, so the identity mapping is correct.
_LABEL_TO_ANGLE = {"0": 0, "90": 90, "180": 180, "270": 270}

_MODEL: Any = None
_MODEL_LOCK = threading.Lock()
_PREDICT_LOCK = threading.Lock()


def _load_model() -> Any:
    from paddleocr import DocImgOrientationClassification

    return DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")


def _get_model() -> Any:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = _load_model()
    return _MODEL


def get_orientation(input_file: Path) -> OrientationConfidence:
    """Classify the page's rotation; degrade to 'upright' on any failure."""
    try:
        model = _get_model()
        with _PREDICT_LOCK:
            result = model.predict(str(input_file))
        data = result[0]
        labels = data.get("label_names") or []
        scores = data.get("scores") or []
        label = str(labels[0]) if labels else "0"
        probability = float(scores[0]) if scores else 0.0
    except Exception:
        log.exception("Orientation detection failed for %s; assuming upright.", input_file)
        return OrientationConfidence(angle=0, confidence=0.0)

    angle = _LABEL_TO_ANGLE.get(label, 0)
    if probability < MIN_PROBABILITY:
        return OrientationConfidence(angle=0, confidence=0.0)
    return OrientationConfidence(angle=angle, confidence=probability * 100.0)
