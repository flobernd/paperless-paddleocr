"""build_page has the paddle worker run the VL pipeline and builds a Page from its answer.

The worker sends the spotted words as ``spotting`` (null when the page result
has none) and each layout block with the keys ``label``, ``content`` and
``bbox``, where ``bbox`` can be null. A fake ``worker.call`` records the
request and answers with such a result.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from paperless_paddleocr.paddle_engine import vl, worker
from paperless_paddleocr.paddle_engine.hocr import Page


def _opts(**extra: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        languages=["en"],
        paddle_engine="vl-remote",
        paddle_vl_server_url="http://gpu:8118",
        paddle_vl_api_key="",
        **extra,
    )


def _png(tmp_path: Path) -> Path:
    img = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(img)
    return img


def _result(spotting: dict[str, Any] | None, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"spotting_capable": True, "spotting": spotting, "blocks": blocks}


def _answer(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    def call(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        return result

    monkeypatch.setattr(worker, "call", call)
    return requests


def _line_texts(page: Page) -> list[str]:
    return [line.text for block in page.blocks for line in block.lines]


def test_build_page_sends_the_image_path_the_kwargs_and_the_pipeline_version(tmp_path, monkeypatch):
    monkeypatch.delenv("PAPERLESS_PADDLEOCR_VL_API_KEY", raising=False)
    png = _png(tmp_path)
    options = _opts(
        paddle_vl_model_name="PaddleOCR-VL-1.6-0.9B",
        paddle_vl_pipeline_version="v1.6",
    )
    requests = _answer(monkeypatch, _result(None, []))
    vl.build_page(png, options)
    # The constructor arguments carry no pipeline_version, which the worker
    # adds to each pipeline it builds.
    assert requests == [
        {
            "op": "vl",
            "image": str(png),
            "kwargs": {
                "device": "cpu",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_layout_detection": False,
                "vl_rec_api_model_name": "PaddleOCR-VL-1.6-0.9B",
                "vl_rec_backend": "vllm-server",
                "vl_rec_server_url": "http://gpu:8118/v1",
            },
            "pipeline_version": "v1.6",
        },
    ]


def test_build_page_places_the_spotted_words(tmp_path, monkeypatch):
    spotting = {
        "rec_texts": ["Hello", "world"],
        "rec_polys": [
            [[10, 20], [60, 20], [60, 40], [10, 40]],
            [[70, 20], [120, 20], [120, 40], [70, 40]],
        ],
    }
    # The spotting prompt also returns the whole transcription as one block.
    blocks = [{"label": "spotting", "content": "Hello world", "bbox": [0, 0, 200, 100]}]
    _answer(monkeypatch, _result(spotting, blocks))
    page = vl.build_page(_png(tmp_path), _opts())
    words = [(w.text, w.box) for block in page.blocks for line in block.lines for w in line.words]
    assert words == [("Hello", (10, 20, 60, 40)), ("world", (70, 20, 120, 40))]


@pytest.mark.parametrize("spotting", [{"rec_texts": [], "rec_polys": []}])
def test_build_page_reads_the_blocks_when_no_word_was_spotted(tmp_path, monkeypatch, spotting):
    blocks = [{"label": "text", "content": "one two", "bbox": [10, 10, 110, 30]}]
    _answer(monkeypatch, _result(spotting, blocks))
    page = vl.build_page(_png(tmp_path), _opts())
    assert [(block.box, _line_texts(page)) for block in page.blocks] == [
        ((10, 10, 110, 30), ["one two"]),
    ]


def test_build_page_skips_a_block_without_a_bbox(tmp_path, monkeypatch):
    blocks = [
        {"label": "text", "content": "lost", "bbox": None},
        {"label": "text", "content": "kept", "bbox": [10, 50, 110, 70]},
    ]
    _answer(monkeypatch, _result(None, blocks))
    page = vl.build_page(_png(tmp_path), _opts())
    assert _line_texts(page) == ["kept"]
