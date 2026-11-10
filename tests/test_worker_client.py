"""The worker client runs requests in the paddle worker and survives its failures.

The tests that call the worker spawn the real script under this interpreter,
with the fake ``paddle`` and ``paddleocr`` packages of ``tests/fakes`` on its
``PYTHONPATH``.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from paperless_paddleocr.paddle_engine import worker

FAKES = Path(__file__).resolve().parent / "fakes"


@pytest.fixture
def fake_worker_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(FAKES))
    # A worker that stops answering fails the test after this many seconds
    # instead of stalling the run for the default ten minutes.
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_WORKER_TIMEOUT", "30")
    monkeypatch.setattr(worker, "_worker", None)
    yield
    # monkeypatch puts the old value back after this teardown, which would
    # orphan the worker the test spawned.
    spawned = worker._worker
    if spawned is not None:
        spawned.stop()


@pytest.fixture
def png(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(path)
    return path


def _classic(png: Path, **kwargs: Any) -> dict[str, Any]:
    return {"op": "classic", "image": str(png), "kwargs": {"device": "cpu", **kwargs}}


def test_call_returns_the_result(fake_worker_env):
    assert worker.call({"op": "ping"})["paddleocr_vl"] is True


def test_error_response_raises_with_the_type(fake_worker_env):
    with pytest.raises(worker.PaddleWorkerError) as info:
        worker.call({"op": "dance"})
    assert info.value.error_type == "ValueError"


def test_error_response_raises_with_the_chain(fake_worker_env, png):
    # The fake raises a RuntimeError from a DependencyError, as paddlex does.
    request = {
        "op": "vl",
        "image": str(png),
        "kwargs": {"device": "cpu", "dependency_error": "cause"},
        "pipeline_version": "v1.5",
    }
    with pytest.raises(worker.PaddleWorkerError) as info:
        worker.call(request)
    assert info.value.error_chain == ("RuntimeError", "DependencyError")


def test_error_response_without_a_chain_raises_the_worker_error(
    fake_worker_env, monkeypatch, tmp_path
):
    script = tmp_path / "chainless_worker.py"
    script.write_text(
        "import json, sys\n"
        "request = json.loads(sys.stdin.readline())\n"
        "error = {'type': 'RuntimeError', 'message': 'no chain here', 'traceback': ''}\n"
        "print(json.dumps({'id': request['id'], 'ok': False, 'error': error}), flush=True)\n"
        "sys.stdin.read()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(worker, "WORKER_SCRIPT", script)
    with pytest.raises(worker.PaddleWorkerError, match="no chain here"):
        worker.call({"op": "ping"})


def test_crash_raises_with_stderr_and_next_call_respawns(fake_worker_env, png, caplog):
    with pytest.raises(worker.PaddleWorkerError, match="exploded"):
        worker.call(_classic(png, explode=True))
    assert worker.call({"op": "ping"})["paddleocr"] is not None
    # The error reported the crash, so the respawn must not report it again
    # as a worker that died between requests.
    assert caplog.records == []


def test_hung_worker_is_killed_and_the_next_call_respawns(fake_worker_env, monkeypatch, png):
    worker.call({"op": "ping"})
    hung = worker._worker.process
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_WORKER_TIMEOUT", "1")
    with pytest.raises(worker.PaddleWorkerError, match="PAPERLESS_PADDLEOCR_WORKER_TIMEOUT"):
        worker.call(_classic(png, hang=True))
    assert hung.returncode == -signal.SIGKILL
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_WORKER_TIMEOUT", "60")
    assert worker.call({"op": "ping"})["paddleocr"] is not None


def test_worker_that_died_between_calls_is_replaced_with_a_warning(fake_worker_env, caplog):
    # As when the OOM killer picks the idle worker, the largest process.
    worker.call({"op": "ping"})
    dead = worker._worker.process
    dead.kill()
    dead.wait()
    assert worker.call({"op": "ping"})["paddleocr"] == "3.4.0"
    assert [record.levelname for record in caplog.records] == ["WARNING"]


def test_worker_that_stops_reading_after_the_check_fails_the_call_with_the_worker_error(
    fake_worker_env, monkeypatch, tmp_path
):
    # The worker closes its stdin before its first answer and keeps running,
    # so the second call passes the liveness check and its write fails, as
    # when a worker dies between the check and the write.
    script = tmp_path / "deaf_worker.py"
    script.write_text(
        "import json, os, sys, time\n"
        "request = json.loads(sys.stdin.readline())\n"
        "os.close(0)\n"
        "print(json.dumps({'id': request['id'], 'ok': True, 'result': {}}), flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(worker, "WORKER_SCRIPT", script)
    worker.call({"op": "ping"})
    with pytest.raises(worker.PaddleWorkerError):
        worker.call({"op": "ping"})


@pytest.mark.parametrize("value", ["0", "-1"])
def test_timeout_below_one_second_falls_back_to_the_default(
    fake_worker_env, monkeypatch, caplog, value
):
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_WORKER_TIMEOUT", value)
    assert worker.call({"op": "ping"})["paddleocr"] == "3.4.0"
    assert "PAPERLESS_PADDLEOCR_WORKER_TIMEOUT" in caplog.text


@pytest.mark.parametrize("line", ["not json", '{"id": null, "ok": false, "error": {}}'])
def test_line_that_is_not_the_response_fails_the_call_at_once(
    fake_worker_env, monkeypatch, tmp_path, line
):
    script = tmp_path / "stray_worker.py"
    script.write_text(
        f"import sys\nsys.stdin.readline()\nprint({line!r}, flush=True)\nsys.stdin.read()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(worker, "WORKER_SCRIPT", script)
    with pytest.raises(worker.PaddleWorkerError) as info:
        worker.call({"op": "ping"})
    assert "PAPERLESS_PADDLEOCR_WORKER_TIMEOUT" not in str(info.value)


def test_concurrent_calls_each_get_their_own_response(fake_worker_env, png):
    # The two pipeline versions answer with a different spotting_capable, so
    # a response that reaches a thread of the other version shows. Eight
    # threads, because with fewer a client without the lock can hand each
    # thread its own response by luck.
    versions = ("v1", "v1.5") * 4
    barrier = threading.Barrier(len(versions))
    results: dict[int, set[bool]] = {}

    def run(index: int, version: str) -> None:
        request = {
            "op": "vl",
            "image": str(png),
            "kwargs": {"device": "cpu"},
            "pipeline_version": version,
        }
        barrier.wait()
        results[index] = {worker.call(request)["spotting_capable"] for _ in range(100)}

    threads = [threading.Thread(target=run, args=item) for item in enumerate(versions)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == {
        0: {False},
        1: {True},
        2: {False},
        3: {True},
        4: {False},
        5: {True},
        6: {False},
        7: {True},
    }


def test_worker_that_floods_stderr_during_a_request_still_answers(fake_worker_env, png):
    assert worker.call(_classic(png, flood_stderr=True))["rec_texts"] == ["hello"]


def test_request_stays_out_of_the_log_and_the_error(fake_worker_env, png, caplog):
    caplog.set_level(logging.DEBUG)
    with pytest.raises(worker.PaddleWorkerError) as info:
        worker.call(_classic(png, explode=True, vl_rec_api_key="secret-api-key"))
    assert "secret-api-key" not in caplog.text
    assert "secret-api-key" not in str(info.value)


def test_missing_interpreter_is_a_worker_error(fake_worker_env, monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_PYTHON", str(tmp_path / "missing" / "python"))
    with pytest.raises(worker.PaddleWorkerError):
        worker.call({"op": "ping"})


def test_environment_drops_the_omp_thread_limit(monkeypatch):
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    assert "OMP_THREAD_LIMIT" not in worker.worker_environment()


def test_interpreter_prefers_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_PYTHON", str(tmp_path / "py"))
    assert worker.worker_interpreter() == str(tmp_path / "py")


def test_interpreter_uses_the_default_when_it_exists(monkeypatch, tmp_path):
    default = tmp_path / "python"
    default.touch()
    monkeypatch.delenv("PAPERLESS_PADDLEOCR_PYTHON", raising=False)
    monkeypatch.setattr(worker, "DEFAULT_WORKER_PYTHON", str(default))
    assert worker.worker_interpreter() == str(default)


def test_interpreter_falls_back_to_sys_executable_when_default_is_missing(monkeypatch):
    monkeypatch.delenv("PAPERLESS_PADDLEOCR_PYTHON", raising=False)
    monkeypatch.setattr(worker, "DEFAULT_WORKER_PYTHON", "/nonexistent/python")
    assert worker.worker_interpreter() == sys.executable
