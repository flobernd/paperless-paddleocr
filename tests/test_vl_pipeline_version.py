"""The VL request carries the pipeline version and the constructor kwargs.

The paddle worker builds the ``PaddleOCRVL`` pipeline from these values. The
tests of the install hint replace ``worker.call`` with a function that raises
the error the worker client reports.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
from PIL import Image

from paperless_paddleocr.paddle_engine import vl, worker


def _opts(version: str | None = None) -> types.SimpleNamespace:
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


def _png(tmp_path: Path) -> Path:
    img = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(img)
    return img


def _fail_with(monkeypatch: pytest.MonkeyPatch, error: worker.PaddleWorkerError) -> None:
    def call(request: dict) -> dict:
        raise error

    monkeypatch.setattr(worker, "call", call)


def test_default_pipeline_version_is_v15():
    assert vl.pipeline_version(_opts()) == "v1.5"


def test_configured_pipeline_version_is_passed_through():
    assert vl.pipeline_version(_opts("v1.6")) == "v1.6"


def test_pipeline_kwargs_point_the_client_at_the_v1_api_of_the_server():
    # The OpenAI client appends /chat/completions to the URL as it is, and the
    # server answers only under /v1.
    assert vl.pipeline_kwargs(_opts())["vl_rec_server_url"] == "http://gpu:8118/v1"


def test_api_key_from_the_environment_reaches_the_pipeline_kwargs(monkeypatch):
    # The paperless parser never passes the key as an option.
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_VL_API_KEY", "sekrit")
    assert vl.pipeline_kwargs(_opts()).get("vl_rec_api_key") == "sekrit"


def test_missing_doc_parser_dependency_names_the_worker_interpreter(tmp_path, monkeypatch):
    # The doc-parser packages belong in the worker interpreter, not in the one
    # that runs paperless.
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_PYTHON", "/srv/paddle-venv/bin/python")
    _fail_with(
        monkeypatch, worker.PaddleWorkerError("paddlex says no", error_type="DependencyError")
    )
    with pytest.raises(RuntimeError, match="/srv/paddle-venv/bin/python"):
        vl.build_page(_png(tmp_path), _opts())


def test_other_worker_errors_keep_their_type_and_message(tmp_path, monkeypatch):
    _fail_with(monkeypatch, worker.PaddleWorkerError("Place(undefined:0)", error_type="TypeError"))
    with pytest.raises(worker.PaddleWorkerError, match=r"Place\(undefined:0\)"):
        vl.build_page(_png(tmp_path), _opts())
