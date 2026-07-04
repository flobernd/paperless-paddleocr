"""estimate_skew must recover small page rotations from text-line structure.

Synthetic pages use horizontal black bars as text-line stand-ins; the
projection-profile score peaks when the bars are horizontal again.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from paperless_paddleocr.paddle_engine.deskew import estimate_skew


def _bars_image(tmp_path, rotate_by: float):
    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    for y in range(60, 560, 40):
        draw.rectangle([40, y, 760, y + 12], fill=0)
    if rotate_by:
        img = img.rotate(rotate_by, expand=True, fillcolor=255)
    path = tmp_path / "page.png"
    img.save(path)
    return path


def test_straight_page_reports_near_zero(tmp_path):
    assert abs(estimate_skew(_bars_image(tmp_path, 0.0))) <= 0.2


def test_clockwise_skew_needs_positive_correction(tmp_path):
    # rotate_by=-2 turns the content 2 degrees clockwise; the correction that
    # ocrmypdf applies via PIL (positive = counterclockwise) must be about +2.
    angle = estimate_skew(_bars_image(tmp_path, -2.0))
    assert 1.5 <= angle <= 2.5


def test_counterclockwise_skew_needs_negative_correction(tmp_path):
    angle = estimate_skew(_bars_image(tmp_path, 2.0))
    assert -2.5 <= angle <= -1.5


def test_angle_beyond_search_range_is_treated_as_not_skew(tmp_path):
    # A true angle just outside the range peaks at the search boundary; the
    # estimator must report 0 rather than rotating by the clamped amount.
    assert estimate_skew(_bars_image(tmp_path, 6.0)) == 0.0
