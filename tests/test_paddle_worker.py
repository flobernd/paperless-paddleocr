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
def construction_log(tmp_path: Path) -> Path:
    return tmp_path / "constructions.log"


@pytest.fixture
def spawn_worker(construction_log: Path) -> Iterator[Spawn]:
    with contextlib.ExitStack() as stack:

        def spawn(*args: str, **env: str) -> subprocess.Popen[str]:
            proc = stack.enter_context(
                subprocess.Popen(
                    [sys.executable, str(WORKER_PATH), *args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    env={
                        **os.environ,
                        "FAKE_PADDLE_CONSTRUCTION_LOG": str(construction_log),
                        "PYTHONPATH": str(FAKES),
                        **env,
                    },
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


def _constructions(log: Path, name: str) -> int:
    return log.read_text(encoding="utf-8").splitlines().count(name)


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


def test_ping_reports_the_error_of_a_failed_cuda_query(spawn_worker):
    worker = spawn_worker(FAKE_CUDA_QUERY_ERROR="CUDA driver not found")
    result = _call(worker, {"id": 1, "op": "ping"})["result"]
    assert (result["cuda_devices"], result["cuda_error"]) == (
        0,
        "RuntimeError: CUDA driver not found",
    )


def test_ping_reports_no_paddleocr_vl_without_the_pipeline_version_parameter(spawn_worker):
    worker = spawn_worker(FAKE_PADDLEOCR_VL_LEGACY="1")
    assert _call(worker, {"id": 1, "op": "ping"})["result"]["paddleocr_vl"] is False


def test_ping_reports_missing_paddleocr_as_null(worker_without_paddleocr):
    result = _call(worker_without_paddleocr, {"id": 1, "op": "ping"})["result"]
    assert result["paddleocr"] is None


def test_ping_reports_why_paddleocr_does_not_import(worker_without_paddleocr):
    result = _call(worker_without_paddleocr, {"id": 1, "op": "ping"})["result"]
    assert "No module named 'paddleocr'" in result["paddleocr_error"]


def test_ping_reports_missing_paddle_as_null_beside_paddleocr(spawn_worker):
    worker = spawn_worker(FAKE_PADDLEPADDLE_FAIL_IMPORT="1")
    result = _call(worker, {"id": 1, "op": "ping"})["result"]
    assert (result["paddle"], result["paddleocr"]) == (None, "3.4.0")


def test_ping_reports_the_first_paddle_import_error(spawn_worker):
    # The worker imports paddle at its start, so the import of the ping is
    # the second one, and the fake fails it with an AttributeError.
    worker = spawn_worker(FAKE_PADDLEPADDLE_FAIL_IMPORT="permission")
    error = _call(worker, {"id": 1, "op": "ping"})["result"]["paddle_error"]
    assert "PermissionError" in error
    assert "/home/paperless/.cache/paddle" in error


def test_ping_answers_when_the_paddle_import_error_cannot_be_printed(spawn_worker):
    worker = spawn_worker(FAKE_PADDLEPADDLE_FAIL_IMPORT="unprintable")
    result = _call(worker, {"id": 1, "op": "ping"})["result"]
    assert result["paddle"] is None and "UnprintableImportError" in result["paddle_error"]


def test_ping_imports_paddleocr_again_after_a_failed_import(spawn_worker):
    worker = spawn_worker(FAKE_PADDLE_FAIL_IMPORT="once")
    _call(worker, {"id": 1, "op": "ping"})
    result = _call(worker, {"id": 2, "op": "ping"})["result"]
    assert (result["paddleocr"], result["paddleocr_error"]) == ("3.4.0", None)


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


def test_classic_result_of_an_unexpected_type_is_an_error(worker, png):
    kwargs = {"device": "cpu", "unexpected_result": True}
    request = {"id": 13, "op": "classic", "image": str(png), "kwargs": kwargs}
    assert _call(worker, request)["ok"] is False


def _raw_classic(request_id: int, png: Path, raw_results: list[Any]) -> dict[str, Any]:
    kwargs = {"device": "cpu", "raw_results": raw_results}
    return {"id": request_id, "op": "classic", "image": str(png), "kwargs": kwargs}


def test_classic_blank_page_is_an_empty_result(worker, png):
    # The shape paddleocr 3.7 returns for a page without text.
    page = {
        "rec_polys": [],
        "rec_scores": [],
        "rec_texts": [],
        "text_word": [],
        "text_word_region": [],
    }
    assert _call(worker, _raw_classic(23, png, [page]))["result"] == {
        "rec_polys": [],
        "rec_scores": [],
        "rec_texts": [],
        "text_word": [],
        "text_word_region": [],
        "preprocessed_shape": None,
    }


def test_classic_prediction_without_a_page_result_is_an_error(worker, png):
    message = _call(worker, _raw_classic(24, png, []))["error"]["message"]
    assert "paddleocr 3.4.0" in message


@pytest.mark.parametrize(
    ("page", "missing"),
    [
        ({"rec_scores": [], "rec_texts": []}, "rec_polys"),
        ({"rec_polys": [], "rec_texts": []}, "rec_scores"),
        ({"rec_polys": [], "rec_scores": []}, "rec_texts"),
        ({"rec_polys": [], "rec_scores": [], "rec_texts": None}, "rec_texts"),
    ],
)
def test_classic_page_result_without_a_region_field_is_an_error(worker, png, page, missing):
    message = _call(worker, _raw_classic(25, png, [page]))["error"]["message"]
    assert missing in message
    assert "paddleocr 3.4.0" in message


def test_classic_region_fields_of_unequal_length_are_an_error(worker, png):
    poly = [[0, 0], [40, 0], [40, 10], [0, 10]]
    page = {"rec_polys": [poly, poly], "rec_scores": [0.9], "rec_texts": ["one", "two"]}
    message = _call(worker, _raw_classic(26, png, [page]))["error"]["message"]
    assert "paddleocr 3.4.0" in message


def test_classic_reuses_one_engine_per_kwargs(worker, png, construction_log):
    for _ in range(2):
        _call(worker, {"id": 3, "op": "classic", "image": str(png), "kwargs": {"device": "cpu"}})
    _call(
        worker,
        {"id": 4, "op": "classic", "image": str(png), "kwargs": {"device": "cpu", "lang": "de"}},
    )
    assert _constructions(construction_log, "PaddleOCR") == 2


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


def _raw_vl(
    request_id: int, png: Path, raw_results: list[Any], **fake_kwargs: Any
) -> dict[str, Any]:
    return {
        "id": request_id,
        "op": "vl",
        "image": str(png),
        "kwargs": {"device": "cpu", "raw_results": raw_results, **fake_kwargs},
        "pipeline_version": "v1",
    }


def test_vl_prediction_without_a_page_result_is_an_error(worker, png):
    message = _call(worker, _raw_vl(27, png, []))["error"]["message"]
    assert "paddleocr 3.4.0" in message


@pytest.mark.parametrize("page", [{"spotting_res": {}}, {"parsing_res_list": None}])
def test_vl_page_result_without_parsing_res_list_is_an_error(worker, png, page):
    message = _call(worker, _raw_vl(28, png, [page]))["error"]["message"]
    assert "parsing_res_list" in message
    assert "paddleocr 3.4.0" in message


def test_vl_empty_spotting_result_gives_the_blocks(worker, png):
    # The v1 pipeline answers each request with an empty spotting result.
    block = {"label": "ocr", "content": "Hi", "bbox": [0, 0, 40, 20]}
    page = {"spotting_res": {}, "parsing_res_list": [block]}
    assert _call(worker, _raw_vl(29, png, [page], block_objects=True))["result"] == {
        "spotting_capable": False,
        "spotting": {"rec_texts": [], "rec_polys": []},
        "blocks": [{"label": "ocr", "content": "Hi", "bbox": [0, 0, 40, 20]}],
    }


def test_vl_spotting_fields_of_unequal_length_are_an_error(worker, png):
    page = {"spotting_res": {"rec_texts": ["Hi"]}, "parsing_res_list": []}
    message = _call(worker, _raw_vl(30, png, [page]))["error"]["message"]
    assert "rec_polys" in message
    assert "paddleocr 3.4.0" in message


@pytest.mark.parametrize(
    ("spotting", "block"),
    [
        pytest.param({}, "Hi", id="v1-block-without-members"),
        pytest.param(
            {"rec_texts": []},
            {"label": "ocr", "content": "Hi", "bbox": [0, 0, 40, 20]},
            id="spotting-without-rec_polys",
        ),
        pytest.param(
            {"rec_polys": [], "rec_texts": None},
            {"label": "ocr", "content": "Hi", "bbox": [0, 0, 40, 20]},
            id="spotting-with-rec_texts-none",
        ),
    ],
)
def test_vl_block_of_an_unknown_shape_is_an_error_without_a_complete_spotting_result(
    worker, png, spotting, block
):
    page = {"spotting_res": spotting, "parsing_res_list": [block]}
    message = _call(worker, _raw_vl(31, png, [page]))["error"]["message"]
    assert "layout block" in message
    assert "paddleocr 3.4.0" in message


def test_vl_block_of_an_unknown_shape_passes_beside_a_complete_blank_spotting_result(worker, png):
    block = {"label": "spotting", "content": "", "bbox": [0, 0, 40, 20]}
    page = {"spotting_res": {"rec_polys": [], "rec_texts": []}, "parsing_res_list": [block]}
    response = _call(worker, _raw_vl(32, png, [page]))
    assert response["ok"] is True, response["error"]["message"]


def test_vl_block_without_content_on_a_blank_v1_page_is_a_result(worker, png):
    block = {"label": "ocr", "content": "", "bbox": [0, 0, 40, 20]}
    page = {"spotting_res": {}, "parsing_res_list": [block]}
    response = _call(worker, _raw_vl(33, png, [page], block_objects=True))
    assert response["ok"] is True, response["error"]["message"]
    assert response["result"]["blocks"] == [{"label": "ocr", "content": "", "bbox": [0, 0, 40, 20]}]


def test_vl_reuses_one_pipeline_per_configuration(worker, png, construction_log):
    request = {
        "op": "vl",
        "image": str(png),
        "kwargs": {"device": "cpu"},
        "pipeline_version": "v1.5",
    }
    for request_id in (14, 15):
        _call(worker, {**request, "id": request_id})
    other_server = {"device": "cpu", "vl_rec_server_url": "http://other:8118/v1"}
    _call(worker, {**request, "id": 16, "kwargs": other_server})
    _call(worker, {**request, "id": 17, "pipeline_version": "v1"})
    assert _constructions(construction_log, "PaddleOCRVL") == 3


def test_orientation_returns_the_top_label_and_score(worker, png):
    result = _call(worker, {"id": 8, "op": "orientation", "image": str(png)})["result"]
    assert result == {"label": "90", "score": pytest.approx(0.97)}


@pytest.mark.parametrize(
    ("raw", "missing"),
    [('[{"scores": [0.97]}]', "label_names"), ('[{"label_names": ["90"]}]', "scores")],
)
def test_orientation_result_without_a_field_is_an_error(spawn_worker, png, raw, missing):
    worker = spawn_worker(FAKE_ORIENTATION_RESULT=raw)
    message = _call(worker, {"id": 34, "op": "orientation", "image": str(png)})["error"]["message"]
    assert missing in message
    assert "paddleocr 3.4.0" in message


def test_orientation_reuses_one_model(worker, png, construction_log):
    for request_id in (18, 19):
        _call(worker, {"id": request_id, "op": "orientation", "image": str(png)})
    assert _constructions(construction_log, "DocImgOrientationClassification") == 1


def test_unknown_op_is_an_error_response(worker):
    response = _call(worker, {"id": 9, "op": "dance"})
    assert response["ok"] is False and response["error"]["chain"][0] == "ValueError"


def _vl_failing_on_dependencies(request_id: int, how: str, png: Path) -> dict[str, Any]:
    return {
        "id": request_id,
        "op": "vl",
        "image": str(png),
        "kwargs": {"device": "cpu", "dependency_error": how},
        "pipeline_version": "v1.5",
    }


@pytest.mark.parametrize(
    ("how", "chain"),
    [
        ("cause", ["RuntimeError", "DependencyError"]),
        ("context", ["RuntimeError", "DependencyError"]),
        ("suppressed", ["RuntimeError"]),
    ],
)
def test_error_response_names_each_exception_the_error_came_from(worker, png, how, chain):
    error = _call(worker, _vl_failing_on_dependencies(21, how, png))["error"]
    assert error["chain"] == chain


def test_error_response_arrives_for_exceptions_that_cause_each_other(worker, png):
    error = _call(worker, _vl_failing_on_dependencies(22, "cycle", png))["error"]
    assert error["chain"][:2] == ["RuntimeError", "DependencyError"]


def test_exception_that_cannot_be_printed_is_an_error_response(worker, png):
    kwargs = {"device": "cpu", "unprintable_error": True}
    response = _call(worker, {"id": 20, "op": "classic", "image": str(png), "kwargs": kwargs})
    assert response["error"]["chain"][0] == "UnprintableError"


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


@pytest.mark.skipif(sys.platform != "linux", reason="the parent-death signal exists on Linux only")
def test_worker_whose_parent_is_not_the_given_process_exits_at_once(spawn_worker):
    # The parent of this test process is not the worker's parent, as for a
    # worker whose parent died before the worker asked for the signal.
    worker = spawn_worker("--parent-pid", str(os.getppid()))
    assert worker.wait(timeout=10) != 0


@pytest.mark.skipif(sys.platform != "linux", reason="the parent-death signal exists on Linux only")
def test_parent_check_exits_with_status_1_when_nobody_reads_stderr():
    # A parent that died held the only reader of the worker's stderr. Without
    # PYTHONUNBUFFERED, text left in the buffer of sys.stderr fails the flush
    # at exit, and the status becomes 120.
    reader, writer = os.pipe()
    os.close(reader)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONUNBUFFERED"}
    try:
        worker = subprocess.run(
            [sys.executable, str(WORKER_PATH), "--parent-pid", str(os.getppid())],
            stdin=subprocess.DEVNULL,
            stderr=writer,
            env=env,
            timeout=10,
        )
    finally:
        os.close(writer)
    assert worker.returncode == 1
