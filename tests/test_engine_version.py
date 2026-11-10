"""The engine reports the paddleocr version of the worker interpreter.

The version comes from the ping that check_options sends. ocrmypdf can ask
for it before check_options ran, and then the version must not start the
worker itself. A fake ``worker.call`` answers the ping.
"""

from __future__ import annotations

import types

import pytest

from paperless_paddleocr import ocrmypdf_plugin
from paperless_paddleocr.paddle_engine import PaddleOCREngine, worker

_PING = {
    "cuda_devices": 0,
    "paddle": "3.2.1",
    "paddleocr": "3.4.0",
    "paddleocr_vl": True,
    "python": "3.13.5",
}


@pytest.fixture(autouse=True)
def _answer_the_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "_ping_result", None)
    monkeypatch.setattr(worker, "call", lambda request: _PING)


def test_creator_tag_names_the_paddleocr_version_of_the_ping():
    options = types.SimpleNamespace(paddle_engine="classic-cpu")
    ocrmypdf_plugin.check_options(options)
    assert PaddleOCREngine.creator_tag(options) == "PaddleOCR 3.4.0"


def test_version_before_a_ping_is_unknown():
    assert PaddleOCREngine.version() == "unknown"
