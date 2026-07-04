from __future__ import annotations

import pytest

from paperless_paddleocr.paddle_engine.vl import normalize_server_url


def test_appends_v1_when_missing():
    assert normalize_server_url("http://gpu:8118") == "http://gpu:8118/v1"


def test_keeps_existing_version_suffix():
    assert normalize_server_url("http://gpu:8118/v1") == "http://gpu:8118/v1"
    assert normalize_server_url("http://gpu:8118/v2/") == "http://gpu:8118/v2"


def test_empty_url_raises():
    with pytest.raises(RuntimeError):
        normalize_server_url("   ")
