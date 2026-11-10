"""Unit tests for the classic-cpu / classic-gpu device selection.

``classic.engine_kwargs`` gives the ``paddleocr.PaddleOCR`` constructor an
explicit ``device=`` kwarg derived from ``options.paddle_engine``:

* ``classic-cpu`` (or any non-gpu value) → ``device="cpu"`` and the OneDNN
  workaround (``enable_mkldnn=False``) is applied - see ``classic.py`` for
  why the workaround is CPU-only.
* ``classic-gpu`` → ``device="gpu"`` and the OneDNN workaround is *not*
  applied (it's a CPU-path execution detail).
"""

from __future__ import annotations

import types

from paperless_paddleocr.paddle_engine import classic


def _opts(engine: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(languages=["en"], paddle_engine=engine)


def test_resolve_device_defaults_to_cpu():
    assert classic._resolve_device(_opts("classic-cpu")) == "cpu"
    assert classic._resolve_device(types.SimpleNamespace()) == "cpu"
    assert classic._resolve_device(_opts("vl-remote")) == "cpu"


def test_resolve_device_returns_gpu_for_classic_gpu():
    assert classic._resolve_device(_opts("classic-gpu")) == "gpu"


def test_engine_kwargs_classic_cpu_uses_cpu_and_mkldnn_workaround():
    kwargs = classic.engine_kwargs(_opts("classic-cpu"))
    assert kwargs["device"] == "cpu"
    # CPU-only OneDNN/PIR crash workaround must be applied on CPU.
    assert kwargs.get("enable_mkldnn") is False


def test_engine_kwargs_classic_gpu_uses_gpu_and_skips_mkldnn_workaround():
    kwargs = classic.engine_kwargs(_opts("classic-gpu"))
    assert kwargs["device"] == "gpu"
    # The OneDNN workaround is a CPU-path execution detail - must not be
    # injected into the GPU kwargs (where the key is meaningless and might
    # confuse a future PaddleOCR release).
    assert "enable_mkldnn" not in kwargs


def test_engine_kwargs_ml_omits_lang_kwarg():
    kwargs = classic.engine_kwargs(
        types.SimpleNamespace(languages=["ml"], paddle_engine="classic-cpu"),
    )
    assert "lang" not in kwargs


def test_engine_kwargs_bundle_code_is_translated():
    kwargs = classic.engine_kwargs(
        types.SimpleNamespace(languages=["latin"], paddle_engine="classic-cpu"),
    )
    assert kwargs["lang"] == "la"
