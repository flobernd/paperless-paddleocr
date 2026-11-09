"""build_page must reuse one PaddleOCR instance per configuration.

Constructing PaddleOCR reloads model weights (seconds, hundreds of MB), and
ocrmypdf calls generate_hocr once per page from a thread pool, so without a
cache an N-page document pays N model loads and races construction.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest
from PIL import Image

from paperless_paddleocr.paddle_engine import classic


class _CountingPaddleOCR:
    instances = 0

    def __init__(self, **kwargs: Any) -> None:
        _CountingPaddleOCR.instances += 1
        self.kwargs = kwargs

    def predict(self, path: str, **kwargs: Any) -> list:
        return []


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(classic, "PaddleOCR", _CountingPaddleOCR)
    _CountingPaddleOCR.instances = 0
    classic._ENGINE_CACHE.clear()
    yield
    classic._ENGINE_CACHE.clear()


def _paddleocr_returning(result: dict[str, Any]) -> type[_CountingPaddleOCR]:
    class _ResultPaddleOCR(_CountingPaddleOCR):
        def predict(self, path: str, **kwargs: Any) -> list:
            return [result]

    return _ResultPaddleOCR


def _opts(lang: str = "en") -> types.SimpleNamespace:
    return types.SimpleNamespace(languages=[lang], paddle_engine="classic-cpu")


def _png(tmp_path):
    img = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(img)
    return img


def test_same_options_reuse_one_engine(tmp_path):
    img = _png(tmp_path)
    classic.build_page(img, _opts())
    classic.build_page(img, _opts())
    assert _CountingPaddleOCR.instances == 1


def test_each_language_gets_its_own_engine(tmp_path):
    img = _png(tmp_path)
    classic.build_page(img, _opts("en"))
    classic.build_page(img, _opts("german"))
    assert _CountingPaddleOCR.instances == 2


def test_custom_model_dirs_are_part_of_the_key(tmp_path):
    img = _png(tmp_path)
    classic.build_page(img, _opts())
    with_dir = _opts()
    with_dir.paddle_det_model_dir = "/models/det"
    classic.build_page(img, with_dir)
    assert _CountingPaddleOCR.instances == 2


def test_custom_model_names_are_part_of_the_key(tmp_path):
    img = _png(tmp_path)
    classic.build_page(img, _opts())
    with_name = _opts()
    with_name.paddle_rec_model_name = "PP-OCRv6_small_rec"
    classic.build_page(img, with_name)
    assert _CountingPaddleOCR.instances == 2


def test_build_page_accepts_numpy_result_arrays(tmp_path, monkeypatch):
    result = {
        "rec_texts": ["one", "two"],
        "rec_scores": [0.9, 0.8],
        "rec_polys": np.array(
            [[[0, 0], [50, 0], [50, 20], [0, 20]], [[0, 30], [50, 30], [50, 50], [0, 50]]]
        ),
        "text_word": [["one"], ["two"]],
        "text_word_region": np.array(
            [[[[5, 2], [45, 2], [45, 18], [5, 18]]], [[[5, 32], [45, 32], [45, 48], [5, 48]]]]
        ),
    }
    monkeypatch.setattr(classic, "PaddleOCR", _paddleocr_returning(result))
    page = classic.build_page(_png(tmp_path), _opts())
    assert [b.lines[0].text for b in page.blocks] == ["one", "two"]
    assert [w.box for b in page.blocks for w in b.lines[0].words] == [
        (5, 2, 45, 18),
        (5, 32, 45, 48),
    ]


def test_build_page_splits_line_text_when_a_word_item_is_none(tmp_path, monkeypatch):
    result = {
        "rec_texts": ["one", "two"],
        "rec_scores": [0.9, 0.8],
        "rec_polys": [[[0, 0], [50, 0], [50, 20], [0, 20]], [[0, 30], [50, 30], [50, 50], [0, 50]]],
        "text_word": [None, ["two"]],
        "text_word_region": [[[[5, 2], [45, 2], [45, 18], [5, 18]]], None],
    }
    monkeypatch.setattr(classic, "PaddleOCR", _paddleocr_returning(result))
    page = classic.build_page(_png(tmp_path), _opts())
    assert [w.box for b in page.blocks for w in b.lines[0].words] == [
        (0, 0, 50, 20),
        (0, 30, 50, 50),
    ]
