"""estimate_skew must recover small page rotations from text-line structure.

Synthetic pages use horizontal black bars as text-line stand-ins; the
projection-profile score peaks when the bars are horizontal again.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from paperless_paddleocr.paddle_engine.deskew import estimate_skew


def _bars_image(tmp_path, rotate_by: float, size=(800, 600)):
    width, height = size
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    for y in range(60, height - 40, 40):
        draw.rectangle([40, y, width - 40, y + 12], fill=0)
    if rotate_by:
        img = img.rotate(rotate_by, expand=True, fillcolor=255)
    path = tmp_path / "page.png"
    img.save(path)
    return path


def test_straight_page_reports_zero(tmp_path):
    assert estimate_skew(_bars_image(tmp_path, 0.0)) == 0.0


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


def test_fine_pass_finds_the_true_angle_inside_the_coarse_cell(tmp_path):
    # The correction 1.3 lies between the coarse candidates 1.0 and 1.5, so the
    # fine pass has to score every tenth of that cell to find it. A pass that
    # skips tenths still reaches 1.3 on the default 800x600 page by luck, so this
    # test uses a larger page.
    page = _bars_image(tmp_path, -1.3, size=(2500, 1800))
    assert estimate_skew(page) == pytest.approx(1.3, abs=0.05)


def test_blank_page_reports_zero(tmp_path):
    path = tmp_path / "blank.png"
    Image.new("L", (800, 600), 255).save(path)
    assert estimate_skew(path) == 0.0
