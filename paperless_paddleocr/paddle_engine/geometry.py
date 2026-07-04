"""Pure polygon geometry for turning PaddleOCR detections into hOCR boxes.

PaddleOCR emits text regions and words as polygons -- usually a 4-point
quad in top-left, top-right, bottom-right, bottom-left order, either as a
plain nested list or a numpy array, so the point accessor stays
duck-typed. These functions are stateless and carry no layout or
reading-order logic; line and column reconstruction lives in
:mod:`paperless_paddleocr.paddle_engine.layout`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: ``(x0, y0, x1, y1)`` axis-aligned box in image pixels.
BBox = tuple[int, int, int, int]

#: Weight of one inter-word gap relative to a single glyph when
#: :func:`estimate_word_boxes` partitions a line -- a space is narrower
#: than an average glyph.
_GAP_WEIGHT = 0.5


def poly_to_bbox(poly: Any) -> BBox:
    """Return the exact axis-aligned envelope of a detection polygon.

    The smallest axis-aligned box that contains every polygon point:
    ``min`` / ``max`` over the xs and ys, cast to ``int``. This is the most
    accurate axis-aligned representation of the detection. Baseline skew is
    handled separately by :mod:`paperless_paddleocr.paddle_engine.layout`,
    which reads the polygon corners directly, so no corner-averaging is
    done here. Duck-typed over nested lists and numpy arrays.
    """
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def estimate_word_boxes(words: Sequence[str], box: BBox) -> list[BBox]:
    """Spread ``words`` across a line ``box`` by a length-weighted partition.

    Fallback only -- used when an engine returns a line transcription but
    no per-word boxes (the VL ``ocr`` response). The invisible text layer
    still needs *a* position per word, so the line box width is divided up:

    * a word's weight is ``max(1, len(word))``;
    * each inter-word gap weighs ``0.5`` (a space is narrower than a glyph);
    * each width is proportional to its weight over the total.

    The last word's right edge is snapped to the ``box`` right edge so
    rounding never leaves a gap. Every word inherits the line ``box``'s
    ``y0`` / ``y1``. A single word fills the whole box; empty input returns
    ``[]``. Rough by design -- it backs the invisible layer, not layout
    analysis.
    """
    if not words:
        return []
    x0, y0, x1, y1 = box
    if len(words) == 1:
        return [(x0, y0, x1, y1)]

    line_width = x1 - x0
    weights = [max(1, len(w)) for w in words]
    total = sum(weights) + _GAP_WEIGHT * (len(words) - 1)
    gap_width = round(line_width * _GAP_WEIGHT / total) if total > 0 else 0

    boxes: list[BBox] = []
    cursor = x0
    last = len(words) - 1
    for i, weight in enumerate(weights):
        width = round(line_width * weight / total) if total > 0 else 0
        right = x1 if i == last else cursor + width
        boxes.append((cursor, y0, right, y1))
        cursor = right + gap_width
    return boxes
