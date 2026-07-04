"""Unit tests for paddle_engine.geometry pure polygon math."""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.geometry import (
    estimate_word_boxes,
    poly_to_bbox,
)


def test_poly_to_bbox_axis_aligned_quad():
    poly = [[10, 20], [110, 20], [110, 60], [10, 60]]
    assert poly_to_bbox(poly) == (10, 20, 110, 60)


def test_poly_to_bbox_skewed_quad_uses_min_max_envelope():
    # both edges tilt; the envelope is the outermost corner on each side
    poly = [[10, 22], [110, 18], [112, 58], [8, 62]]
    assert poly_to_bbox(poly) == (8, 18, 112, 62)


def test_poly_to_bbox_non_quad_polygon():
    poly = [[5, 5], [40, 2], [60, 30], [25, 50], [0, 25]]
    assert poly_to_bbox(poly) == (0, 2, 60, 50)


def test_estimate_word_boxes_single_word_fills_box():
    assert estimate_word_boxes(["hello"], (10, 5, 210, 45)) == [(10, 5, 210, 45)]


def test_estimate_word_boxes_empty_input():
    assert estimate_word_boxes([], (0, 0, 100, 20)) == []


def test_estimate_word_boxes_partitions_and_snaps_last_word():
    boxes = estimate_word_boxes(["aa", "bbbb"], (0, 0, 120, 10))
    # every word inherits the line's y0 / y1
    assert all(b[1] == 0 and b[3] == 10 for b in boxes)
    # words run left-to-right starting at the box left edge
    assert boxes[0][0] == 0
    assert boxes[0][2] <= boxes[1][0]
    # the longer word is wider than the shorter one
    assert (boxes[1][2] - boxes[1][0]) > (boxes[0][2] - boxes[0][0])
    # the last word's right edge is snapped to the box right edge
    assert boxes[-1][2] == 120
