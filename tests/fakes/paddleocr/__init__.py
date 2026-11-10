"""Stand-in for paddleocr in the worker tests.

Importable only when ``tests/fakes`` is on ``PYTHONPATH``, which the tests
set for the worker process alone. The results use the containers of the
real pipelines: numpy arrays nested inside Python lists, numpy scalars in
score lists, and dicts or objects for layout blocks.
"""

import os

import numpy as np

if os.environ.get("FAKE_PADDLE_FAIL_IMPORT") == "1":
    raise ImportError("No module named 'paddleocr'")

__version__ = "3.4.0"


class PaddleOCR:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, image, return_word_box=False):
        if self.kwargs.get("noisy"):
            # Paddle libraries write to standard output from Python and from
            # native code, and neither may reach the worker's protocol lines.
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


class PaddleOCRVL:
    def __init__(self, pipeline_version=None, **kwargs):
        _require_device()
        self.pipeline_version = pipeline_version

    def predict(self, image, use_layout_detection=None, use_queues=None, prompt_label=None):
        result = {
            # Older paddleocr releases return dicts, current paddlex returns
            # block objects.
            "parsing_res_list": [
                {"block_label": "text", "block_content": "Hi there", "block_bbox": [0, 0, 100, 20]},
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
        _require_device()


if os.environ.get("FAKE_PADDLEOCR_VL_LEGACY") == "1":
    PaddleOCRVL = _PaddleOCRVLWithoutPipelineVersion


class DocImgOrientationClassification:
    def __init__(self, model_name=None):
        pass

    def predict(self, image):
        return [{"label_names": ["90"], "scores": [0.97]}]
