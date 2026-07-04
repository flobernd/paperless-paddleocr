"""Unit tests for the classic-cpu / classic-gpu device selection.

``classic._build_engine`` forwards an explicit ``device=`` kwarg to
``paddleocr.PaddleOCR`` derived from ``options.paddle_engine``:

* ``classic-cpu`` (or any non-gpu value) → ``device="cpu"`` and the OneDNN
  workaround (``enable_mkldnn=False``) is applied - see ``classic.py`` for
  why the workaround is CPU-only.
* ``classic-gpu`` → ``device="gpu"`` and the OneDNN workaround is *not*
  applied (it's a CPU-path execution detail).

The PaddleOCR import is monkeypatched to a capturing stub so the tests
run on any host, with or without paddle installed.
"""

from __future__ import annotations

import types
from typing import Any

from paperless_paddleocr.paddle_engine import classic


class _PaddleOCRStub:
    """Records the kwargs passed to PaddleOCR(...) for assertion."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        _PaddleOCRStub.last_kwargs = dict(kwargs)


def _opts(engine: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(languages=["en"], paddle_engine=engine)


def test_resolve_device_defaults_to_cpu():
    assert classic._resolve_device(_opts("classic-cpu")) == "cpu"
    assert classic._resolve_device(types.SimpleNamespace()) == "cpu"
    assert classic._resolve_device(_opts("vl-remote")) == "cpu"


def test_resolve_device_returns_gpu_for_classic_gpu():
    assert classic._resolve_device(_opts("classic-gpu")) == "gpu"


def test_build_engine_classic_cpu_uses_cpu_and_mkldnn_workaround(monkeypatch):
    monkeypatch.setattr(classic, "PaddleOCR", _PaddleOCRStub)
    classic._build_engine(_opts("classic-cpu"))
    kwargs = _PaddleOCRStub.last_kwargs
    assert kwargs["device"] == "cpu"
    # CPU-only OneDNN/PIR crash workaround must be applied on CPU.
    assert kwargs.get("enable_mkldnn") is False


def test_build_engine_classic_gpu_uses_gpu_and_skips_mkldnn_workaround(monkeypatch):
    monkeypatch.setattr(classic, "PaddleOCR", _PaddleOCRStub)
    classic._build_engine(_opts("classic-gpu"))
    kwargs = _PaddleOCRStub.last_kwargs
    assert kwargs["device"] == "gpu"
    # The OneDNN workaround is a CPU-path execution detail - must not be
    # injected into the GPU kwargs (where the key is meaningless and might
    # confuse a future PaddleOCR release).
    assert "enable_mkldnn" not in kwargs


def test_build_engine_ml_omits_lang_kwarg(monkeypatch):
    monkeypatch.setattr(classic, "PaddleOCR", _PaddleOCRStub)
    classic._build_engine(types.SimpleNamespace(languages=["ml"], paddle_engine="classic-cpu"))
    assert "lang" not in _PaddleOCRStub.last_kwargs


def test_build_engine_bundle_code_is_translated(monkeypatch):
    monkeypatch.setattr(classic, "PaddleOCR", _PaddleOCRStub)
    classic._build_engine(
        types.SimpleNamespace(languages=["latin"], paddle_engine="classic-cpu"),
    )
    assert _PaddleOCRStub.last_kwargs["lang"] == "la"
