"""The worker client runs requests in the paddle worker and survives its failures.

The tests that call the worker spawn the real script under this interpreter,
with the fake ``paddle`` and ``paddleocr`` packages of ``tests/fakes`` on its
``PYTHONPATH``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import select
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from paperless_paddleocr.paddle_engine import worker

FAKES = Path(__file__).resolve().parent / "fakes"
LATE_INTERPRETER = FAKES / "late_interpreter.py"


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


def test_error_response_raises_with_the_type(fake_worker_env):
    with pytest.raises(worker.PaddleWorkerError) as info:
        worker.call({"op": "dance"})
    assert info.value.error_chain == ("ValueError",)


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
        "error = {'message': 'no chain here'}\n"
        "print(json.dumps({'id': request['id'], 'ok': False, 'error': error}), flush=True)\n"
        "sys.stdin.read()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(worker, "WORKER_SCRIPT", script)
    with pytest.raises(worker.PaddleWorkerError, match="no chain here"):
        worker.call({"op": "ping"})


@pytest.mark.parametrize(
    "envelope",
    [
        "{'result': {}}",
        "{'ok': True, 'result': None}",
        "{'ok': False, 'error': 'boom'}",
        "{'ok': False, 'error': {'type': 'RuntimeError', 'traceback': ''}}",
    ],
)
def test_response_off_protocol_fails_the_call_with_the_op(
    fake_worker_env, monkeypatch, tmp_path, envelope
):
    script = tmp_path / "off_protocol_worker.py"
    script.write_text(
        "import json, sys\n"
        "request = json.loads(sys.stdin.readline())\n"
        f"print(json.dumps({{'id': request['id'], **{envelope}}}), flush=True)\n"
        "sys.stdin.read()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(worker, "WORKER_SCRIPT", script)
    with pytest.raises(worker.PaddleWorkerError, match="ping request"):
        worker.call({"op": "ping"})


def test_worker_that_answered_off_protocol_is_replaced(fake_worker_env, monkeypatch, tmp_path):
    # Only the first worker answers off protocol, so the second call answers
    # only if it went to a fresh worker.
    marker = tmp_path / "started"
    script = tmp_path / "off_protocol_once_worker.py"
    script.write_text(
        "import json, pathlib, sys\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "first = not marker.exists()\n"
        "marker.touch()\n"
        "for line in sys.stdin:\n"
        "    result = None if first else {'fresh': True}\n"
        "    answer = {'id': json.loads(line)['id'], 'ok': True, 'result': result}\n"
        "    print(json.dumps(answer), flush=True)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(worker, "WORKER_SCRIPT", script)
    with pytest.raises(worker.PaddleWorkerError):
        worker.call({"op": "ping"})
    assert worker.call({"op": "ping"}) == {"fresh": True}


def test_crash_raises_with_stderr_and_next_call_respawns(fake_worker_env, png, caplog):
    with pytest.raises(worker.PaddleWorkerError, match="exploded"):
        worker.call(_classic(png, explode=True))
    assert worker.call({"op": "ping"})["paddleocr"] is not None
    # The error reported the crash, so the respawn must not report it again
    # as a worker that died between requests.
    assert [r for r in caplog.records if r.name == "paperless.paddleocr.worker"] == []


def test_native_crash_leaves_the_python_stack_in_the_error(fake_worker_env, png):
    with pytest.raises(worker.PaddleWorkerError) as info:
        worker.call(_classic(png, segfault=True))
    assert "Fatal Python error" in str(info.value) and "in predict" in str(info.value)


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
    assert [
        record.levelname for record in caplog.records if record.name == "paperless.paddleocr.worker"
    ] == ["WARNING"]


def test_ping_after_a_repaired_import_runs_in_a_fresh_worker(fake_worker_env, monkeypatch):
    monkeypatch.setattr(worker, "_ping_result", None)
    monkeypatch.setenv("FAKE_PADDLEPADDLE_FAIL_IMPORT", "1")
    assert worker.ping()["paddle"] is None
    monkeypatch.delenv("FAKE_PADDLEPADDLE_FAIL_IMPORT")
    assert worker.ping()["paddle"] == "3.2.1"


def test_ping_that_keeps_no_answer_stops_its_worker(fake_worker_env, monkeypatch):
    monkeypatch.setattr(worker, "_ping_result", None)
    monkeypatch.setenv("FAKE_PADDLEPADDLE_FAIL_IMPORT", "1")
    worker.call({"op": "ping"})
    stale = worker._worker.process
    assert worker.ping()["paddle"] is None
    assert stale.returncode == -signal.SIGKILL


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


def test_worker_start_that_fails_after_the_process_started_leaves_no_process(
    fake_worker_env, monkeypatch
):
    started: list[subprocess.Popen[str]] = []
    real_popen = subprocess.Popen
    real_start = threading.Thread.start

    def popen(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        started.append(real_popen(*args, **kwargs))
        return started[-1]

    def start(thread: threading.Thread) -> None:
        if thread.name == "paddle-worker-stdout":
            raise RuntimeError("can't start new thread")
        real_start(thread)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        with pytest.raises(RuntimeError, match="can't start new thread"):
            worker.call({"op": "ping"})
        assert started[0].wait(timeout=10) == -signal.SIGKILL
    finally:
        for process in started:
            process.kill()


def test_worker_start_interrupted_after_the_parent_thread_runs_leaves_no_process(
    fake_worker_env, monkeypatch
):
    started: list[subprocess.Popen[str]] = []
    spawned = threading.Event()
    real_popen = subprocess.Popen
    real_start = threading.Thread.start

    def popen(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        started.append(real_popen(*args, **kwargs))
        spawned.set()
        return started[-1]

    def start(thread: threading.Thread) -> None:
        real_start(thread)
        if thread.name == "paddle-worker-parent":
            raise RuntimeError("interrupted after the start")

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        with pytest.raises(RuntimeError, match="interrupted after the start"):
            worker.call({"op": "ping"})
        # The parent thread can start the process after the exception has
        # reached the caller.
        assert spawned.wait(timeout=10)
        assert started[0].wait(timeout=10) == -signal.SIGKILL
    finally:
        for process in started:
            process.kill()


def test_worker_started_from_a_thread_that_ended_keeps_its_models(
    fake_worker_env, monkeypatch, png, tmp_path
):
    construction_log = tmp_path / "constructions.log"
    monkeypatch.setenv("FAKE_PADDLE_CONSTRUCTION_LOG", str(construction_log))
    # Like an ocrmypdf page thread, which ends with its document.
    page_thread = threading.Thread(target=worker.call, args=(_classic(png),))
    page_thread.start()
    page_thread.join()
    worker.call(_classic(png))
    assert construction_log.read_text(encoding="utf-8").splitlines() == ["PaddleOCR"]


#: Sends one classic request, which the fake answers only after 30 s.
_HANGING_PARENT = (
    "import sys\n"
    "from paperless_paddleocr.paddle_engine import worker\n"
    "kwargs = {'device': 'cpu', 'hang': True}\n"
    "worker.call({'op': 'classic', 'image': sys.argv[1], 'kwargs': kwargs})\n"
)


def _pid_when_written(path: Path, parent: subprocess.Popen[bytes]) -> int:
    deadline = time.monotonic() + 30
    while not path.exists():
        status = parent.poll()
        assert status is None, f"the parent exited with status {status} before {path} was written"
        assert time.monotonic() < deadline, f"no pid in {path} within 30 s"
        time.sleep(0.05)
    return int(path.read_text(encoding="utf-8"))


def _kill_if_running(pidfd: int) -> None:
    # A worker that outlives its parent sleeps on as an orphan.
    with contextlib.suppress(ProcessLookupError):
        signal.pidfd_send_signal(pidfd, signal.SIGKILL)


def _worker_exits_after_its_parent_is_killed(png: Path, pid_file: Path) -> bool:
    """Whether the worker exits within 10 s after the kill of the process that called it.

    The parent sends a classic request that hangs, and a fake writes the
    worker's pid to ``pid_file`` before the parent is killed with SIGKILL.
    """
    with contextlib.ExitStack() as cleanup:
        parent = cleanup.enter_context(
            subprocess.Popen([sys.executable, "-c", _HANGING_PARENT, str(png)])
        )
        # The kill runs before Popen.__exit__ reaps the parent.
        cleanup.callback(parent.kill)
        pidfd = os.pidfd_open(_pid_when_written(pid_file, parent))
        cleanup.callback(os.close, pidfd)
        cleanup.callback(_kill_if_running, pidfd)
        parent.send_signal(signal.SIGKILL)
        # The pidfd turns readable once the worker exits, whether or not its
        # new parent has reaped it.
        ready, _, _ = select.select([pidfd], [], [], 10)
        return bool(ready)


@pytest.mark.skipif(sys.platform != "linux", reason="the parent-death signal exists on Linux only")
def test_worker_in_the_paddle_import_dies_with_its_killed_parent(
    fake_worker_env, monkeypatch, png, tmp_path
):
    # The paddle import at the start of the worker is the first step that can
    # take long, and the worker blocks there before it reads the request.
    pid_file = tmp_path / "worker.pid"
    monkeypatch.setenv("FAKE_PADDLEPADDLE_HANG_IMPORT", str(pid_file))
    assert _worker_exits_after_its_parent_is_killed(png, pid_file), "the worker outlived its parent"


@pytest.mark.skipif(sys.platform != "linux", reason="the parent-death signal exists on Linux only")
def test_worker_whose_parent_died_before_its_start_exits_at_once(
    fake_worker_env, monkeypatch, png, tmp_path
):
    # The kernel sends no signal for a parent that died before the worker
    # asked for one, and the worker of this interpreter asks only after the
    # parent's death, with the parent's request waiting in its stdin.
    interpreter = tmp_path / "python"
    interpreter.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(LATE_INTERPRETER))} "$@"\n',
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    pid_file = tmp_path / "worker.pid"
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_PYTHON", str(interpreter))
    monkeypatch.setenv("FAKE_INTERPRETER_PID_FILE", str(pid_file))
    assert _worker_exits_after_its_parent_is_killed(png, pid_file), "the worker ran on as an orphan"


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
