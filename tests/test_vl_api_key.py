from __future__ import annotations

import contextlib
import io
import logging
import types

import pytest
from PIL import Image

from paperless_paddleocr import ocrmypdf_plugin
from paperless_paddleocr.paddle_engine import vl, worker
from paperless_paddleocr.paddle_engine.vl import resolve_vl_api_key


def test_api_key_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_VL_API_KEY", " sekrit ")
    assert resolve_vl_api_key(types.SimpleNamespace(paddle_vl_api_key="")) == "sekrit"


def test_option_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_VL_API_KEY", "env")
    assert resolve_vl_api_key(types.SimpleNamespace(paddle_vl_api_key="cli")) == "cli"


def test_preflight_sends_the_key_from_the_environment(monkeypatch):
    # The paperless parser never passes the key as an option, so the probe
    # sees it only through the environment.
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_VL_API_KEY", "sekrit")
    monkeypatch.setattr(ocrmypdf_plugin, "_PROBED_SERVERS", set())
    monkeypatch.setattr(worker, "_ping_result", None)
    monkeypatch.setattr(
        worker,
        "call",
        lambda request: {
            "cuda_devices": 0,
            "paddle": "3.2.1",
            "paddleocr": "3.4.0",
            "paddleocr_vl": True,
            "python": "3.13.5",
        },
    )
    sent: list[str | None] = []

    @contextlib.contextmanager
    def capture(request, timeout):
        sent.append(request.get_header("Authorization"))
        yield io.BytesIO(b"{}")

    monkeypatch.setattr(ocrmypdf_plugin, "_urlopen", capture)
    ocrmypdf_plugin.check_options(
        types.SimpleNamespace(
            paddle_engine="vl-remote",
            paddle_vl_server_url="http://gpu:8118",
            paddle_vl_api_key="",
        ),
    )
    assert sent == ["Bearer sekrit"]


def test_api_key_stays_out_of_the_log_and_the_error(tmp_path, monkeypatch, caplog):
    # The key travels to the worker inside the request, and paperless writes
    # its debug log to a file.
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_VL_API_KEY", "secret-api-key")

    def call(request):
        raise worker.PaddleWorkerError(
            "paddlex says no",
            error_chain=("RuntimeError", "DependencyError"),
        )

    monkeypatch.setattr(worker, "call", call)
    png = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(png)
    caplog.set_level(logging.DEBUG)
    options = types.SimpleNamespace(
        languages=["en"],
        paddle_engine="vl-remote",
        paddle_vl_server_url="http://gpu:8118",
        paddle_vl_api_key="",
    )
    with pytest.raises(RuntimeError) as info:
        vl.build_page(png, options)
    assert "secret-api-key" not in caplog.text
    assert "secret-api-key" not in str(info.value)
