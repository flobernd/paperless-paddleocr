"""Unit tests for the ``classic-gpu`` validation in ``check_options``.

The hook must fail fast with ``MissingDependencyError`` when the runtime
can't actually deliver a GPU - either because paddlepaddle-gpu isn't
installed, or because the host has no visible CUDA device. The classic-cpu
path stays untouched (no CUDA probe).
"""

from __future__ import annotations

import sys
import types

import pytest
from ocrmypdf.exceptions import MissingDependencyError

from paperless_paddleocr import ocrmypdf_plugin


def _opts(engine: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(paddle_engine=engine)


def _install_fake_paddleocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a ``paddleocr`` import that satisfies the CPU-side check.

    ``check_options`` does ``from paddleocr import PaddleOCR`` before the
    GPU-specific probe, so the test environment needs ``paddleocr``
    importable even though the real package isn't a CI dependency.
    """
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCR = object  # symbol presence is all the import needs
    monkeypatch.setitem(sys.modules, "paddleocr", fake)


def _install_fake_paddle(monkeypatch: pytest.MonkeyPatch, *, device_count: int) -> None:
    """Provide a ``paddle`` import whose ``device.cuda.device_count()`` is fixed."""
    fake_paddle = types.ModuleType("paddle")
    fake_cuda = types.SimpleNamespace(device_count=lambda: device_count)
    fake_paddle.device = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)


def test_check_options_classic_gpu_passes_with_visible_cuda_device(monkeypatch):
    _install_fake_paddleocr(monkeypatch)
    _install_fake_paddle(monkeypatch, device_count=1)
    # Should not raise.
    ocrmypdf_plugin.check_options(_opts("classic-gpu"))


def test_check_options_classic_gpu_fails_when_no_device_visible(monkeypatch):
    _install_fake_paddleocr(monkeypatch)
    _install_fake_paddle(monkeypatch, device_count=0)
    with pytest.raises(MissingDependencyError, match="no CUDA device is visible"):
        ocrmypdf_plugin.check_options(_opts("classic-gpu"))


def test_check_options_classic_cpu_does_not_probe_cuda(monkeypatch):
    _install_fake_paddleocr(monkeypatch)
    # Deliberately do NOT install fake paddle: a CUDA probe would crash on
    # the missing import. classic-cpu must skip the probe entirely.
    monkeypatch.delitem(sys.modules, "paddle", raising=False)
    ocrmypdf_plugin.check_options(_opts("classic-cpu"))


def test_check_options_unknown_engine_raises_value_error(monkeypatch):
    _install_fake_paddleocr(monkeypatch)
    with pytest.raises(ValueError, match="Unknown paddle_engine"):
        ocrmypdf_plugin.check_options(_opts("classic"))  # old value, removed
