"""Page orientation detection for ocrmypdf's --rotate-pages hook.

Runs PaddleOCR's PP-LCNet document orientation classifier in the paddle
worker. The classifier is a ~7 MB model cached under ~/.paddlex like the
OCR models.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ocrmypdf.pluginspec import OrientationConfidence

from paperless_paddleocr.paddle_engine import worker

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


def get_orientation(input_file: Path) -> OrientationConfidence:
    """Classify the page's rotation; degrade to 'upright' on any failure."""
    try:
        result = worker.call({"op": "orientation", "image": str(input_file)})
        label = result["label"]
        probability = result["score"]
    except Exception:
        log.exception("Orientation detection failed for %s; assuming upright.", input_file)
        return OrientationConfidence(angle=0, confidence=0.0)

    angle = _LABEL_TO_ANGLE.get(label, 0)
    if probability < MIN_PROBABILITY:
        return OrientationConfidence(angle=0, confidence=0.0)
    return OrientationConfidence(angle=angle, confidence=probability * 100.0)
