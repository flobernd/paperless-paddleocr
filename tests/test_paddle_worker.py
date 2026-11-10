"""The paddle worker answers the line protocol of its module docstring.

Each test runs the real script in a child process with the fake ``paddle``
and ``paddleocr`` packages of ``tests/fakes`` on ``PYTHONPATH``.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

WORKER_PATH = Path(__file__).resolve().parent.parent / "paperless_paddleocr" / "paddle_worker.py"
FAKES = Path(__file__).resolve().parent / "fakes"

#: A hung worker fails the test after this many seconds instead of stalling the run.
RESPONSE_TIMEOUT = 30

Spawn = Callable[..., subprocess.Popen[str]]


@pytest.fixture
def spawn_worker() -> Iterator[Spawn]:
    with contextlib.ExitStack() as stack:

        def spawn(**env: str) -> subprocess.Popen[str]:
            proc = stack.enter_context(
                subprocess.Popen(
                    [sys.executable, str(WORKER_PATH)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    env={**os.environ, "PYTHONPATH": str(FAKES), **env},
                ),
            )
            # The kill runs before Popen.__exit__ closes the pipes and reaps
            # the child, so a test that fails mid-protocol leaves no process.
            stack.callback(proc.kill)
            return proc

        yield spawn


@pytest.fixture
def worker(spawn_worker: Spawn) -> subprocess.Popen[str]:
    return spawn_worker()


@pytest.fixture
def worker_without_paddleocr(spawn_worker: Spawn) -> subprocess.Popen[str]:
    return spawn_worker(FAKE_PADDLE_FAIL_IMPORT="1")


@pytest.fixture
def png(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(path)
    return path


def _send(proc: subprocess.Popen[str], line: str) -> dict[str, Any]:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(line + "\n")
    proc.stdin.flush()
    ready, _, _ = select.select([proc.stdout], [], [], RESPONSE_TIMEOUT)
    assert ready, f"no response within {RESPONSE_TIMEOUT} s"
    response_line = proc.stdout.readline()
    assert response_line, f"the worker closed its stdout (exit code {proc.wait(timeout=10)})"
    return json.loads(response_line)


def _call(proc: subprocess.Popen[str], request: dict[str, Any]) -> dict[str, Any]:
    response = _send(proc, json.dumps(request))
    assert response["id"] == request["id"]
    return response


def test_ping_reports_versions_and_devices(spawn_worker):
    worker = spawn_worker(FAKE_CUDA_DEVICES="2")
    result = _call(worker, {"id": 1, "op": "ping"})["result"]
    reported = (
        result["paddle"],
        result["paddleocr"],
        result["paddleocr_vl"],
        result["cuda_devices"],
    )
    assert reported == ("3.2.1", "3.4.0", True, 2)


def test_ping_reports_missing_paddleocr_as_null(worker_without_paddleocr):
    result = _call(worker_without_paddleocr, {"id": 1, "op": "ping"})["result"]
    assert result["paddleocr"] is None


def test_classic_returns_plain_lists(worker, png):
    request = {"id": 2, "op": "classic", "image": str(png), "kwargs": {"device": "cpu"}}
    assert _call(worker, request)["result"] == {
        "rec_texts": ["hello"],
        "rec_scores": [pytest.approx(0.9)],
        "rec_polys": [[[0, 0], [40, 0], [40, 10], [0, 10]]],
        "text_word": [["hel", " ", "lo"]],
        "text_word_region": [
            [
                [[0, 0], [15, 0], [15, 10], [0, 10]],
                [[15, 0], [20, 0], [20, 10], [15, 10]],
                [[20, 0], [40, 0], [40, 10], [20, 10]],
            ],
        ],
        "preprocessed_shape": [20, 40],
    }


def test_classic_reuses_one_engine_per_kwargs(worker, png):
    for _ in range(2):
        _call(worker, {"id": 3, "op": "classic", "image": str(png), "kwargs": {"device": "cpu"}})
    _call(
        worker,
        {"id": 4, "op": "classic", "image": str(png), "kwargs": {"device": "cpu", "lang": "de"}},
    )
    assert _call(worker, {"id": 5, "op": "ping"})["result"]["classic_engines"] == 2


def test_vl_normalizes_blocks_and_reports_spotting(worker, png):
    request = {
        "id": 6,
        "op": "vl",
        "image": str(png),
        "kwargs": {"device": "cpu"},
        "pipeline_version": "v1.5",
    }
    assert _call(worker, request)["result"] == {
        "spotting_capable": True,
        "spotting": {
            "rec_texts": ["Hi", "there"],
            "rec_polys": [
                [[0, 0], [20, 0], [20, 10], [0, 10]],
                [[30, 0], [80, 0], [80, 10], [30, 10]],
            ],
        },
        "blocks": [
            {"label": "text", "content": "Hi there", "bbox": [0, 0, 100, 20]},
            {"label": "header", "content": "Page 1", "bbox": [0, 30, 100, 40]},
        ],
    }


def test_vl_v1_pipeline_gets_the_ocr_prompt(worker, png):
    request = {
        "id": 7,
        "op": "vl",
        "image": str(png),
        "kwargs": {"device": "cpu"},
        "pipeline_version": "v1",
    }
    result = _call(worker, request)["result"]
    assert (result["spotting_capable"], result["spotting"]) == (False, None)


def test_vl_pipeline_without_the_version_parameter_runs_in_ocr_mode(spawn_worker, png):
    worker = spawn_worker(FAKE_PADDLEOCR_VL_LEGACY="1")
    request = {
        "id": 12,
        "op": "vl",
        "image": str(png),
        "kwargs": {"device": "cpu"},
        "pipeline_version": "v1.5",
    }
    response = _call(worker, request)
    assert response["ok"] is True, response["error"]["message"]
    assert response["result"]["spotting_capable"] is False


def test_orientation_returns_the_top_label_and_score(worker, png):
    result = _call(worker, {"id": 8, "op": "orientation", "image": str(png)})["result"]
    assert result == {"label": "90", "score": pytest.approx(0.97)}


def test_unknown_op_is_an_error_response(worker):
    response = _call(worker, {"id": 9, "op": "dance"})
    assert response["ok"] is False and response["error"]["type"] == "ValueError"


@pytest.mark.parametrize("line", ["not json", '{"op": "ping"}'])
def test_request_without_an_id_is_answered_with_a_null_id(worker, line):
    response = _send(worker, line)
    assert (response["id"], response["ok"]) == (None, False)
    assert _call(worker, {"id": 10, "op": "ping"})["ok"] is True


def test_stdout_writes_of_paddle_libraries_miss_the_protocol(worker, png):
    kwargs = {"device": "cpu", "noisy": True}
    request = {"id": 11, "op": "classic", "image": str(png), "kwargs": kwargs}
    assert _call(worker, request)["result"]["rec_texts"] == ["hello"]


def test_worker_exits_on_stdin_eof(worker):
    worker.stdin.close()
    assert worker.wait(timeout=10) == 0
