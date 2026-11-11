"""Stand-in for paddleocr in the worker tests.

Importable only when ``tests/fakes`` is on ``PYTHONPATH``, which the tests
set for the worker process alone. The results use the containers of the
real pipelines: numpy arrays nested inside Python lists, numpy scalars in
score lists, and objects for layout blocks.
"""

import json
import os
import resource
import signal
import sys
import time

import numpy as np

# Fails the paddleocr import. FAKE_PADDLEPADDLE_FAIL_IMPORT fails the paddle
# import.
if os.environ.get("FAKE_PADDLE_FAIL_IMPORT") == "1":
    raise ImportError("No module named 'paddleocr'")

# Fails the first import only, as for a package the operator installs while
# the worker runs.
if os.environ.get("FAKE_PADDLE_FAIL_IMPORT") == "once":
    os.environ["FAKE_PADDLE_FAIL_IMPORT"] = ""
    raise ImportError("No module named 'paddleocr'")

__version__ = "3.4.0"


def _record_construction(name):
    # The fakes live in the worker process, so the tests see which models the
    # worker reuses only through this file.
    log = os.environ.get("FAKE_PADDLE_CONSTRUCTION_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as stream:
            stream.write(name + "\n")


class UnprintableError(Exception):
    def __str__(self):
        raise RuntimeError("__str__ failed")


class PaddleOCR:
    def __init__(self, **kwargs):
        _record_construction("PaddleOCR")
        if kwargs.get("unprintable_error"):
            raise UnprintableError()
        if kwargs.get("explode"):
            # A worker that dies in the middle of a request, as on an OOM in
            # PaddlePaddle, after a last line on stderr.
            print("exploded", file=sys.stderr)
            raise SystemExit(1)
        self.kwargs = kwargs

    def predict(self, image, return_word_box=False):
        if self.kwargs.get("unexpected_result"):
            return ["not a page result"]
        # The raw prediction a test chose, for the shapes a paddleocr release
        # other than the tested one may return.
        if "raw_results" in self.kwargs:
            return self.kwargs["raw_results"]
        if self.kwargs.get("hang"):
            # Far past the one-second timeout of the hang test and the
            # deadline of the test of a worker that starts after its parent
            # died, so the sleep ends in a kill.
            time.sleep(30)
        if self.kwargs.get("segfault"):
            # A worker that dies in native code in the middle of a request, as
            # on a segfault in PaddlePaddle. The crash is on purpose, and its
            # core file would land in the working directory of the test run.
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            os.kill(os.getpid(), signal.SIGSEGV)
        if self.kwargs.get("flood_stderr"):
            # More than a pipe buffer holds, so a parent that does not read
            # stderr while it waits for the response stalls the worker.
            sys.stderr.write(("x" * 1023 + "\n") * 1024)
        if self.kwargs.get("noisy"):
            # One write from Python and one from native code, and neither may
            # reach the worker's protocol lines.
            print("noise from python")
            os.write(1, b"noise from native code\n")
        result = {
            "rec_texts": ["hello"],
            "rec_scores": [np.float32(0.9)],
            "rec_polys": [np.array([[0, 0], [40, 0], [40, 10], [0, 10]])],
            "doc_preprocessor_res": {"output_img": np.zeros((20, 40, 3))},
        }
        # The real pipeline reports sub-word tokens only on request.
        if return_word_box:
            result["text_word"] = [["hel", " ", "lo"]]
            result["text_word_region"] = [
                [
                    np.array([[0, 0], [15, 0], [15, 10], [0, 10]]),
                    np.array([[15, 0], [20, 0], [20, 10], [15, 10]]),
                    np.array([[20, 0], [40, 0], [40, 10], [20, 10]]),
                ],
            ]
        return [result]


class _VLBlock:
    def __init__(self, label, content, bbox):
        self.label = label
        self.content = content
        self.bbox = bbox


def _require_device():
    import paddle

    # The real constructor probes bfloat16 support on the active device and
    # fails while none is set.
    if paddle._device is None:
        raise TypeError("Place(undefined:0)")


class DependencyError(Exception):
    """Stands in for paddlex.utils.deps.DependencyError, which the worker reports by name."""


def _fail_on_dependencies(how):
    # paddlex raises a RuntimeError from its DependencyError when it creates a
    # pipeline whose packages are missing. The other shapes cover the rest of
    # the rules by which Python links one exception to the next.
    message = "A dependency error occurred during pipeline creation."
    try:
        raise DependencyError("`PaddleOCR-VL-1.5` requires additional dependencies.")
    except DependencyError as missing:
        if how == "cause":
            raise RuntimeError(message) from missing
        if how == "context":
            raise RuntimeError(message)  # noqa: B904
        if how == "suppressed":
            raise RuntimeError(message) from None
        if how == "cycle":
            error = RuntimeError(message)
            missing.__cause__ = error
            raise error from missing
        raise


class PaddleOCRVL:
    def __init__(self, pipeline_version=None, **kwargs):
        _record_construction("PaddleOCRVL")
        if kwargs.get("dependency_error"):
            _fail_on_dependencies(kwargs["dependency_error"])
        _require_device()
        self.pipeline_version = pipeline_version
        self.raw_results = kwargs.get("raw_results")
        # The request is JSON, so a test that wants the block objects of
        # paddlex passes the attributes of each block as a dict.
        if kwargs.get("block_objects"):
            self.raw_results = [
                {**page, "parsing_res_list": [_VLBlock(**b) for b in page["parsing_res_list"]]}
                for page in self.raw_results
            ]

    def predict(self, image, use_layout_detection=None, use_queues=None, prompt_label=None):
        # The vl adapter runs no local layout model and no prediction queues,
        # and the worker must keep both off.
        if use_layout_detection is not False or use_queues is not False:
            raise ValueError("use_layout_detection and use_queues must be False")
        if self.raw_results is not None:
            return self.raw_results
        result = {
            "parsing_res_list": [
                _VLBlock("text", "Hi there", [0, 0, 100, 20]),
                _VLBlock("header", "Page 1", np.array([0, 30, 100, 40])),
            ],
        }
        if prompt_label == "spotting":
            result["spotting_res"] = {
                "rec_texts": ["Hi", "there"],
                "rec_polys": np.array(
                    [[[0, 0], [20, 0], [20, 10], [0, 10]], [[30, 0], [80, 0], [80, 10], [30, 10]]],
                ),
            }
        return [result]


class _PaddleOCRVLWithoutPipelineVersion(PaddleOCRVL):
    """The releases before the pipeline_version parameter, which reject it."""

    def __init__(self, device=None):
        _record_construction("PaddleOCRVL")
        _require_device()
        self.raw_results = None


if os.environ.get("FAKE_PADDLEOCR_VL_LEGACY") == "1":
    PaddleOCRVL = _PaddleOCRVLWithoutPipelineVersion


class DocImgOrientationClassification:
    def __init__(self, model_name=None):
        _record_construction("DocImgOrientationClassification")

    def predict(self, image):
        # The orientation op takes no model arguments, so the raw prediction a
        # test chooses comes through the environment.
        if "FAKE_ORIENTATION_RESULT" in os.environ:
            return json.loads(os.environ["FAKE_ORIENTATION_RESULT"])
        return [{"label_names": ["90"], "scores": [0.97]}]
