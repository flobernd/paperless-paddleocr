"""check_options checks the worker interpreter through a ping before the first page.

The worker's ping reports the paddle and paddleocr versions (null for a
package that does not import, with the reason beside), whether
``PaddleOCRVL`` imports and how many CUDA devices paddle sees. A fake
``worker.call`` answers it.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
from ocrmypdf.exceptions import MissingDependencyError

from paperless_paddleocr import ocrmypdf_plugin
from paperless_paddleocr.paddle_engine import worker

INTERPRETER = "/srv/paddle-venv/bin/python"
NO_PADDLE = {"paddle": None, "paddle_error": "ModuleNotFoundError: No module named 'paddle'"}
NO_PADDLEOCR = {
    "paddleocr": None,
    "paddleocr_error": "ModuleNotFoundError: No module named 'paddleocr'",
    "paddleocr_vl": False,
}


@pytest.fixture(autouse=True)
def _fresh_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_PYTHON", INTERPRETER)
    monkeypatch.setattr(worker, "_ping_result", None)


def _opts(engine: str) -> types.SimpleNamespace:
    # No server URL, so a vl-remote check that gets past the dependency checks
    # fails on the URL instead of probing the network.
    return types.SimpleNamespace(paddle_engine=engine)


def _ping(**fields: Any) -> dict[str, Any]:
    return {
        "cuda_devices": 1,
        "paddle": "3.2.1",
        "paddle_error": None,
        "paddleocr": "3.4.0",
        "paddleocr_error": None,
        "paddleocr_vl": True,
        "python": "3.13.5",
        **fields,
    }


def _answer(monkeypatch: pytest.MonkeyPatch, *answers: Any) -> list[dict[str, Any]]:
    """Answer each request with the next of ``answers``, raising it when it is an exception."""
    requests: list[dict[str, Any]] = []
    pending = list(answers)

    def call(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        if not pending:
            raise AssertionError("the worker was asked again")
        answer = pending.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(worker, "call", call)
    return requests


def test_check_options_classic_gpu_passes_with_visible_cuda_device(monkeypatch):
    _answer(monkeypatch, _ping(cuda_devices=1))
    ocrmypdf_plugin.check_options(_opts("classic-gpu"))


def test_check_options_classic_gpu_fails_when_no_device_visible(monkeypatch):
    _answer(monkeypatch, _ping(cuda_devices=0))
    with pytest.raises(MissingDependencyError, match="no CUDA device is visible"):
        ocrmypdf_plugin.check_options(_opts("classic-gpu"))


def test_check_options_classic_cpu_needs_no_cuda_device(monkeypatch):
    _answer(monkeypatch, _ping(cuda_devices=0))
    ocrmypdf_plugin.check_options(_opts("classic-cpu"))


@pytest.mark.parametrize("engine", ["classic-cpu", "vl-remote"])
def test_missing_paddleocr_names_the_worker_interpreter(monkeypatch, engine):
    _answer(monkeypatch, _ping(**NO_PADDLEOCR))
    with pytest.raises(MissingDependencyError) as info:
        ocrmypdf_plugin.check_options(_opts(engine))
    assert INTERPRETER in str(info.value)


@pytest.mark.parametrize("engine", ["classic-cpu", "vl-remote"])
def test_missing_paddle_names_paddlepaddle_and_the_worker_interpreter(monkeypatch, engine):
    _answer(monkeypatch, _ping(**NO_PADDLE, cuda_devices=0))
    with pytest.raises(MissingDependencyError) as info:
        ocrmypdf_plugin.check_options(_opts(engine))
    assert "paddlepaddle" in str(info.value)
    assert INTERPRETER in str(info.value)


def test_paddleocr_that_does_not_import_names_the_reason(monkeypatch):
    reason = "PermissionError: [Errno 13] Permission denied: '/home/paperless/.paddlex'"
    _answer(monkeypatch, _ping(**{**NO_PADDLEOCR, "paddleocr_error": reason}))
    with pytest.raises(MissingDependencyError) as info:
        ocrmypdf_plugin.check_options(_opts("classic-cpu"))
    assert reason in str(info.value)


def test_paddle_that_does_not_import_names_the_reason(monkeypatch):
    reason = "PermissionError: [Errno 13] Permission denied: '/home/paperless/.cache/paddle'"
    _answer(monkeypatch, _ping(**{**NO_PADDLE, "paddle_error": reason}, cuda_devices=0))
    with pytest.raises(MissingDependencyError) as info:
        ocrmypdf_plugin.check_options(_opts("classic-cpu"))
    assert reason in str(info.value)


def test_failed_ping_names_the_worker_interpreter(monkeypatch):
    error = worker.PaddleWorkerError("The paddle worker exited before it answered the ping request")
    _answer(monkeypatch, error)
    with pytest.raises(MissingDependencyError) as info:
        ocrmypdf_plugin.check_options(_opts("classic-cpu"))
    assert INTERPRETER in str(info.value)


def test_check_options_unknown_engine_raises_value_error_before_the_ping(monkeypatch):
    _answer(monkeypatch, worker.PaddleWorkerError("The paddle worker exited"))
    with pytest.raises(ValueError, match="Unknown paddle_engine"):
        ocrmypdf_plugin.check_options(_opts("classic"))  # old value, removed


def test_a_complete_ping_is_kept_for_the_next_document(monkeypatch):
    requests = _answer(monkeypatch, _ping())
    ocrmypdf_plugin.check_options(_opts("classic-cpu"))
    ocrmypdf_plugin.check_options(_opts("classic-cpu"))
    assert requests == [{"op": "ping"}]


def test_a_ping_that_reports_a_missing_package_is_asked_again(monkeypatch):
    # The operator can repair the worker venv while this process runs.
    _answer(monkeypatch, _ping(**NO_PADDLEOCR), _ping())
    with pytest.raises(MissingDependencyError):
        ocrmypdf_plugin.check_options(_opts("classic-cpu"))
    ocrmypdf_plugin.check_options(_opts("classic-cpu"))
