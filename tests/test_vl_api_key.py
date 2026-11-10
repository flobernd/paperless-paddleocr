from __future__ import annotations

import contextlib
import io
import sys
import types

from paperless_paddleocr import ocrmypdf_plugin
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
    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCRVL = object
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)
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
