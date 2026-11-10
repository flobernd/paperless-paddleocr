"""Model-name and model-directory options for the classic pipeline.

``add_options`` registers the ``--paddle-*-model-name`` args on the ocrmypdf
command line. paperless forwards ``PAPERLESS_OCR_USER_ARGS`` into
``ocrmypdf.ocr(...)`` unchanged, and ocrmypdf 17 keeps a keyword that has no
option field, such as ``paddle_det_model_name``, as an attribute of the
options. Without a name kwarg, paddleocr pairs a custom model dir with the
default model name from the base pipeline config and paddlex rejects the dir
with "Model name mismatch", so model names are the only way to select
non-default official models.
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


def _opts(**extra: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(languages=["en"], paddle_engine="classic-cpu", **extra)


def test_engine_kwargs_forwards_model_names():
    kwargs = classic.engine_kwargs(
        _opts(
            paddle_det_model_name="PP-OCRv6_small_det",
            paddle_rec_model_name="PP-OCRv6_small_rec",
        )
    )
    assert kwargs["text_detection_model_name"] == "PP-OCRv6_small_det"
    assert kwargs["text_recognition_model_name"] == "PP-OCRv6_small_rec"


def test_engine_kwargs_forwards_model_dirs():
    kwargs = classic.engine_kwargs(
        _opts(paddle_det_model_dir="/models/det", paddle_rec_model_dir="/models/rec"),
    )
    assert kwargs["text_detection_model_dir"] == "/models/det"
    assert kwargs["text_recognition_model_dir"] == "/models/rec"


def test_engine_kwargs_omits_unset_model_names():
    kwargs = classic.engine_kwargs(_opts())
    assert "text_detection_model_name" not in kwargs
    assert "text_recognition_model_name" not in kwargs
