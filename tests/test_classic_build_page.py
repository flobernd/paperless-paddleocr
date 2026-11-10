"""build_page has the paddle worker OCR the page and builds a Page from its answer.

The worker sends each field of the classic result as a plain list and the
size of the image the pipeline detected on as ``preprocessed_shape``. A fake
``worker.call`` records the request and answers with such a result.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from paperless_paddleocr.paddle_engine import classic, worker


def _opts(lang: str = "en", engine: str = "classic-cpu") -> types.SimpleNamespace:
    return types.SimpleNamespace(languages=[lang], paddle_engine=engine)


def _png(tmp_path: Path) -> Path:
    img = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(img)
    return img


def _result(**fields: Any) -> dict[str, Any]:
    return {
        "rec_polys": [],
        "rec_scores": [],
        "rec_texts": [],
        "text_word": [],
        "text_word_region": [],
        "preprocessed_shape": None,
        **fields,
    }


def _answer(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    def call(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        return result

    monkeypatch.setattr(worker, "call", call)
    return requests


def _one_word_result(preprocessed_shape: list[int]) -> dict[str, Any]:
    return _result(
        rec_texts=["one"],
        rec_scores=[0.9],
        rec_polys=[[[0, 0], [100, 0], [100, 15], [0, 15]]],
        text_word=[["one"]],
        text_word_region=[[[[10, 4], [50, 4], [50, 10], [10, 10]]]],
        preprocessed_shape=preprocessed_shape,
    )


def test_build_page_sends_the_image_path_and_the_engine_kwargs(tmp_path, monkeypatch):
    png = _png(tmp_path)
    options = _opts("german", "classic-gpu")
    requests = _answer(monkeypatch, _result())
    classic.build_page(png, options)
    assert requests == [
        {"op": "classic", "image": str(png), "kwargs": classic.engine_kwargs(options)},
    ]


def test_build_page_shapes_each_region_into_a_line_with_its_word_boxes(tmp_path, monkeypatch):
    result = _result(
        rec_texts=["one", "two"],
        rec_scores=[0.9, 0.8],
        rec_polys=[[[0, 0], [50, 0], [50, 20], [0, 20]], [[0, 30], [50, 30], [50, 50], [0, 50]]],
        text_word=[["one"], ["two"]],
        text_word_region=[
            [[[5, 2], [45, 2], [45, 18], [5, 18]]],
            [[[5, 32], [45, 32], [45, 48], [5, 48]]],
        ],
    )
    _answer(monkeypatch, result)
    page = classic.build_page(_png(tmp_path), _opts())
    assert [b.lines[0].text for b in page.blocks] == ["one", "two"]
    assert [w.box for b in page.blocks for w in b.lines[0].words] == [
        (5, 2, 45, 18),
        (5, 32, 45, 48),
    ]


def test_build_page_scales_boxes_from_the_preprocessed_height_and_width(tmp_path, monkeypatch):
    # The page is 60 wide and 40 high, and the pipeline detected on an image
    # 20 high and 120 wide, so x scales by 0.5 and y by 2.
    _answer(monkeypatch, _one_word_result([20, 120]))
    block = classic.build_page(_png(tmp_path), _opts()).blocks[0]
    assert (block.box, block.lines[0].words[0].box) == ((0, 0, 50, 30), (5, 8, 25, 20))


def test_build_page_keeps_the_boxes_for_a_zero_size_preprocessed_shape(tmp_path, monkeypatch):
    _answer(monkeypatch, _one_word_result([0, 120]))
    block = classic.build_page(_png(tmp_path), _opts()).blocks[0]
    assert (block.box, block.lines[0].words[0].box) == ((0, 0, 100, 15), (10, 4, 50, 10))


def test_build_page_splits_line_text_when_a_word_item_is_none(tmp_path, monkeypatch):
    result = _result(
        rec_texts=["one", "two"],
        rec_scores=[0.9, 0.8],
        rec_polys=[[[0, 0], [50, 0], [50, 20], [0, 20]], [[0, 30], [50, 30], [50, 50], [0, 50]]],
        text_word=[None, ["two"]],
        text_word_region=[[[[5, 2], [45, 2], [45, 18], [5, 18]]], None],
    )
    _answer(monkeypatch, result)
    page = classic.build_page(_png(tmp_path), _opts())
    assert [w.box for b in page.blocks for w in b.lines[0].words] == [
        (0, 0, 50, 20),
        (0, 30, 50, 50),
    ]
