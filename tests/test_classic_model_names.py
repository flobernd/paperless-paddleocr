"""Model-name options for the classic pipeline.

``add_options`` must register the ``--paddle-*-model-name`` args so
``ocrmypdf.ocr(paddle_det_model_name=...)`` accepts them as kwargs (paperless
forwards ``PAPERLESS_OCR_USER_ARGS`` into that call unchanged). Without a name
kwarg, paddleocr pairs a custom model dir with the default model name from the
base pipeline config and paddlex rejects the dir with "Model name mismatch",
so model names are the only way to select non-default official models.
"""

from __future__ import annotations

import argparse
import types
from typing import Any

from paperless_paddleocr import ocrmypdf_plugin
from paperless_paddleocr.paddle_engine import classic


def test_add_options_registers_model_name_args():
    parser = argparse.ArgumentParser()
    ocrmypdf_plugin.add_options(parser)
    ns = parser.parse_args(
        [
            "--paddle-det-model-name",
            "PP-OCRv6_small_det",
            "--paddle-rec-model-name",
            "PP-OCRv6_small_rec",
        ]
    )
    assert ns.paddle_det_model_name == "PP-OCRv6_small_det"
    assert ns.paddle_rec_model_name == "PP-OCRv6_small_rec"


def test_model_name_args_default_to_none():
    parser = argparse.ArgumentParser()
    ocrmypdf_plugin.add_options(parser)
    ns = parser.parse_args([])
    assert ns.paddle_det_model_name is None
    assert ns.paddle_rec_model_name is None


class _PaddleOCRStub:
    """Records the kwargs passed to PaddleOCR(...) for assertion."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        _PaddleOCRStub.last_kwargs = dict(kwargs)


def _opts(**extra: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(languages=["en"], paddle_engine="classic-cpu", **extra)


def test_build_engine_forwards_model_names(monkeypatch):
    monkeypatch.setattr(classic, "PaddleOCR", _PaddleOCRStub)
    classic._build_engine(
        _opts(
            paddle_det_model_name="PP-OCRv6_small_det",
            paddle_rec_model_name="PP-OCRv6_small_rec",
        )
    )
    kwargs = _PaddleOCRStub.last_kwargs
    assert kwargs["text_detection_model_name"] == "PP-OCRv6_small_det"
    assert kwargs["text_recognition_model_name"] == "PP-OCRv6_small_rec"


def test_build_engine_omits_unset_model_names(monkeypatch):
    monkeypatch.setattr(classic, "PaddleOCR", _PaddleOCRStub)
    classic._build_engine(_opts())
    kwargs = _PaddleOCRStub.last_kwargs
    assert "text_detection_model_name" not in kwargs
    assert "text_recognition_model_name" not in kwargs
