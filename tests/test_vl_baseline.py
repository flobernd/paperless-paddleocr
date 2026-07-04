"""Unit test for vl._poly_baseline_y bottom-edge baseline extraction."""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.vl import _poly_baseline_y


def test_poly_baseline_y_quad_uses_bottom_corner_mean():
    # top-left, top-right, bottom-right, bottom-left; bottom ys are 40 and 44
    poly = [[0, 0], [100, 0], [100, 40], [0, 44]]
    assert _poly_baseline_y(poly) == 42.0


def test_poly_baseline_y_non_quad_uses_lowest_point():
    poly = [[0, 0], [50, 5], [25, 60]]
    assert _poly_baseline_y(poly) == 60.0
