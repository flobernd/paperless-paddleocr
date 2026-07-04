from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from paperless_paddleocr.paddle_engine import vl


class _CapturingPipeline:
    last_kwargs: dict[str, Any] = {}

    # pipeline_version is declared explicitly (not swallowed by **kwargs)
    # because _build_pipeline feature-detects it via inspect.signature.
    def __init__(self, pipeline_version: str = "v1.6", **kwargs: Any) -> None:
        _CapturingPipeline.last_kwargs = {"pipeline_version": pipeline_version, **kwargs}
        self.pipeline_version = pipeline_version


@pytest.fixture(autouse=True)
def _stub_paddle(monkeypatch):
    fake_paddle = types.ModuleType("paddle")
    fake_paddle.set_device = lambda device: None
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    monkeypatch.setattr(vl, "PaddleOCRVL", _CapturingPipeline)


def _opts(version: str | None) -> types.SimpleNamespace:
    opts = types.SimpleNamespace(
        languages=["en"],
        paddle_engine="vl-remote",
        paddle_vl_server_url="http://gpu:8118",
        paddle_vl_model_name="PaddleOCR-VL-1.5-0.9B",
        paddle_vl_api_key="",
    )
    if version is not None:
        opts.paddle_vl_pipeline_version = version
    return opts


def test_default_pipeline_version_is_v15():
    vl._build_pipeline(_opts(None))
    assert _CapturingPipeline.last_kwargs["pipeline_version"] == "v1.5"


def test_configured_pipeline_version_is_passed_through():
    vl._build_pipeline(_opts("v1.6"))
    assert _CapturingPipeline.last_kwargs["pipeline_version"] == "v1.6"


def test_v1_disables_spotting_prompt():
    pipeline = _CapturingPipeline(pipeline_version="v1")
    assert vl._spotting_capable(pipeline) is False
    assert vl._spotting_capable(_CapturingPipeline(pipeline_version="v1.6")) is True


def test_missing_doc_parser_dependency_gets_install_hint(monkeypatch):
    class DependencyError(Exception):
        pass

    class _Exploding:
        def __init__(self, **kwargs: Any) -> None:
            raise DependencyError("paddlex says no")

    monkeypatch.setattr(vl, "PaddleOCRVL", _Exploding)
    with pytest.raises(RuntimeError, match=r"paperless-paddleocr\[vl\]"):
        vl._build_pipeline(_opts(None))
