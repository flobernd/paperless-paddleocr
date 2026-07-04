"""build_page must reuse one PaddleOCRVL pipeline per configuration.

PaddleOCRVL construction loads local preprocessing models and builds the
remote client; per-page construction wastes seconds and RAM on every page.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from PIL import Image

from paperless_paddleocr.paddle_engine import vl


class _CountingPipeline:
    instances = 0

    def __init__(self, **kwargs: Any) -> None:
        _CountingPipeline.instances += 1
        self.kwargs = kwargs
        self.pipeline_version = kwargs.get("pipeline_version")

    def predict(self, path: str, **kwargs: Any) -> list:
        return []


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    # vl._build_pipeline does `import paddle` and calls set_device; provide a
    # stub so the tests run without paddlepaddle installed.
    fake_paddle = types.ModuleType("paddle")
    fake_paddle.set_device = lambda device: None
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    monkeypatch.setattr(vl, "PaddleOCRVL", _CountingPipeline)
    _CountingPipeline.instances = 0
    vl._PIPELINE_CACHE.clear()
    yield
    vl._PIPELINE_CACHE.clear()


def _opts(url: str = "http://gpu:8118") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        languages=["en"],
        paddle_engine="vl-remote",
        paddle_vl_server_url=url,
        paddle_vl_model_name="PaddleOCR-VL-1.5-0.9B",
        paddle_vl_api_key="",
    )


def _png(tmp_path):
    img = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(img)
    return img


def test_same_options_reuse_one_pipeline(tmp_path):
    img = _png(tmp_path)
    vl.build_page(img, _opts())
    vl.build_page(img, _opts())
    assert _CountingPipeline.instances == 1


def test_different_server_gets_its_own_pipeline(tmp_path):
    img = _png(tmp_path)
    vl.build_page(img, _opts("http://gpu-a:8118"))
    vl.build_page(img, _opts("http://gpu-b:8118"))
    assert _CountingPipeline.instances == 2
