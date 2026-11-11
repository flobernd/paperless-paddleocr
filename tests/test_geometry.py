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


def test_estimate_word_boxes_partitions_the_line():
    boxes = estimate_word_boxes(["aa", "bbbb"], (0, 0, 120, 10))
    # every word inherits the line's y0 / y1
    assert all(b[1] == 0 and b[3] == 10 for b in boxes)
    # words run left-to-right starting at the box left edge
    assert boxes[0][0] == 0
    assert boxes[0][2] <= boxes[1][0]
    # the longer word is wider than the shorter one
    assert (boxes[1][2] - boxes[1][0]) > (boxes[0][2] - boxes[0][0])
    # the last word ends at the box right edge
    assert boxes[-1][2] == 120


def test_estimate_word_boxes_keeps_the_last_word_inside_the_line():
    # Total weight 8.5 on 9 px: the last word starts at round(9 * 7.5 / 8.5) = 8.
    assert estimate_word_boxes(list("abcdef"), (0, 0, 9, 10))[-1] == (8, 0, 9, 10)


def test_estimate_word_boxes_start_at_the_left_edge_of_the_line():
    # Total weight 4.5 on 90 px: the edges lie at 0, 40, 50 and 90 px from the left edge.
    boxes = estimate_word_boxes(["aa", "bb"], (100, 0, 190, 10))
    assert boxes == [(100, 0, 140, 10), (150, 0, 190, 10)]
