"""The vl-remote preflight must turn config mistakes into clear errors.

urlopen is stubbed at the plugin module level; no network is touched.
"""

from __future__ import annotations

import contextlib
import http.client
import io
import sys
import types
import urllib.error

import pytest
from ocrmypdf.exceptions import MissingDependencyError

from paperless_paddleocr import ocrmypdf_plugin


@pytest.fixture(autouse=True)
def _fresh_probe_cache(monkeypatch):
    ocrmypdf_plugin._PROBED_SERVERS.clear()
    # check_options imports PaddleOCRVL first; satisfy it with a stub.
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCRVL = object
    fake.PaddleOCR = object
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    yield
    ocrmypdf_plugin._PROBED_SERVERS.clear()


def _opts() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        paddle_engine="vl-remote",
        paddle_vl_server_url="http://gpu:8118",
        paddle_vl_api_key="secret",
    )


def _ok_urlopen(calls):
    @contextlib.contextmanager
    def fake(request, timeout=0):
        calls.append(request.full_url)
        yield io.BytesIO(b"{}")

    return fake


def test_reachable_server_passes_and_is_cached(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(ocrmypdf_plugin, "_urlopen", _ok_urlopen(calls))
    ocrmypdf_plugin.check_options(_opts())
    ocrmypdf_plugin.check_options(_opts())
    assert calls == ["http://gpu:8118/v1/models"]


def test_unreachable_server_raises_actionable_error(monkeypatch):
    def fail(request, timeout=0):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ocrmypdf_plugin, "_urlopen", fail)
    with pytest.raises(MissingDependencyError, match="not reachable"):
        ocrmypdf_plugin.check_options(_opts())


def test_socket_error_without_reason_raises_actionable_error(monkeypatch):
    # socket.timeout and ConnectionResetError are OSErrors that are not URLErrors.
    def fail(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(ocrmypdf_plugin, "_urlopen", fail)
    with pytest.raises(MissingDependencyError, match="not reachable"):
        ocrmypdf_plugin._probe_vl_server("http://gpu:8118", "")


def test_url_without_scheme_raises_actionable_error():
    with pytest.raises(MissingDependencyError, match="not reachable"):
        ocrmypdf_plugin._probe_vl_server("gpuhost", "")


def test_non_http_answer_raises_actionable_error(monkeypatch):
    # urllib lets http.client's BadStatusLine through unwrapped, and it is no OSError.
    def ssh_banner(request, timeout):
        raise http.client.BadStatusLine("SSH-2.0-OpenSSH_9.6\r\n")

    monkeypatch.setattr(ocrmypdf_plugin, "_urlopen", ssh_banner)
    with pytest.raises(MissingDependencyError, match="not reachable") as info:
        ocrmypdf_plugin._probe_vl_server("http://gpu:22", "")
    assert "\r" not in str(info.value) and "\n" not in str(info.value)


def test_auth_failure_raises_actionable_error(monkeypatch):
    def unauthorized(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(ocrmypdf_plugin, "_urlopen", unauthorized)
    with pytest.raises(MissingDependencyError, match="rejected the API key"):
        ocrmypdf_plugin.check_options(_opts())


def test_other_http_errors_do_not_block(monkeypatch):
    # A 404 on /v1/models means the server is up but shaped differently;
    # blocking OCR on that would be a false negative.
    def not_found(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 404, "Nope", hdrs=None, fp=None)

    monkeypatch.setattr(ocrmypdf_plugin, "_urlopen", not_found)
    ocrmypdf_plugin.check_options(_opts())
