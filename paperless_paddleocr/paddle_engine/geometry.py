"""Pure polygon geometry for turning PaddleOCR detections into hOCR boxes.

PaddleOCR emits text regions and words as polygons -- usually a 4-point
quad in top-left, top-right, bottom-right, bottom-left order, either as a
plain nested list or a numpy array, so the point accessor stays
duck-typed. These functions are stateless and carry no layout or
reading-order logic; line and column reconstruction lives in
:mod:`paperless_paddleocr.paddle_engine.layout`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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


def union_boxes(boxes: Iterable[BBox]) -> BBox:
    """Smallest axis-aligned box containing every box of the non-empty ``boxes``."""
    xs0, ys0, xs1, ys1 = zip(*boxes, strict=True)
    return min(xs0), min(ys0), max(xs1), max(ys1)


def estimate_word_boxes(words: Sequence[str], box: BBox) -> list[BBox]:
    """Spread ``words`` across a line ``box`` by a length-weighted partition.

    Fallback only -- used when an engine returns a line transcription but
    no per-word boxes (the VL ``ocr`` response). The invisible text layer
    still needs *a* position per word, so the line box width is divided up:

    * a word's weight is ``max(1, len(word))``;
    * each inter-word gap weighs ``0.5`` (a space is narrower than a glyph);
    * each width is proportional to its weight over the total.

    Each edge lies at the rounded share of the box width that the weight to its
    left takes, so no rounding error adds up and the last word ends at the
    right edge of ``box``. Every word inherits the ``y0`` / ``y1`` of the line
    ``box``. A single word fills the whole box, and empty input returns ``[]``.
    Rough by design: it backs the invisible layer, not layout analysis.
    """
    if not words:
        return []
    x0, y0, x1, y1 = box
    if len(words) == 1:
        return [(x0, y0, x1, y1)]

    line_width = x1 - x0
    weights = [max(1, len(w)) for w in words]
    total = sum(weights) + _GAP_WEIGHT * (len(words) - 1)

    boxes: list[BBox] = []
    weight_to_left = 0.0
    for weight in weights:
        left = x0 + round(line_width * weight_to_left / total)
        right = x0 + round(line_width * (weight_to_left + weight) / total)
        boxes.append((left, y0, right, y1))
        weight_to_left += weight + _GAP_WEIGHT
    return boxes
