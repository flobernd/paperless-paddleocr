"""build_page must reuse one PaddleOCR instance per configuration.

Constructing PaddleOCR reloads model weights (seconds, hundreds of MB), and
ocrmypdf calls generate_hocr once per page from a thread pool, so without a
cache an N-page document pays N model loads and races construction.
"""

from __future__ import annotations

import types
from typing import Any

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
