"""get_orientation maps the worker's classifier result to ocrmypdf's OrientationConfidence.

The real classifier needs paddle and a downloaded model in the paddle
worker. These tests replace ``worker.call`` and pin the request, the
mapping, the scaling and the failure behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from paperless_paddleocr.paddle_engine import orientation, worker


def _answer(monkeypatch: pytest.MonkeyPatch, label: str, score: float) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    def call(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        return {"label": label, "score": score}

    monkeypatch.setattr(worker, "call", call)
    return requests


def test_request_names_the_orientation_op_and_the_image_path(monkeypatch):
    requests = _answer(monkeypatch, "0", 0.9)
    orientation.get_orientation(Path("page.png"))
    assert requests == [{"op": "orientation", "image": "page.png"}]


def test_confident_rotation_is_reported(monkeypatch):
    _answer(monkeypatch, "90", 0.97)
    oc = orientation.get_orientation(Path("x.png"))
    assert oc.angle == 90
    assert round(oc.confidence) == 97


def test_low_probability_is_suppressed(monkeypatch):
    # A 4-way classifier below 0.5 is guessing; paperless's default
    # rotate_pages_threshold (12) would otherwise act on noise.
    _answer(monkeypatch, "180", 0.4)
    oc = orientation.get_orientation(Path("x.png"))
    assert (oc.angle, oc.confidence) == (0, 0.0)


def test_worker_failure_degrades_to_no_rotation_with_a_logged_error(monkeypatch, caplog):
    def fail(request: dict[str, Any]) -> dict[str, Any]:
        raise worker.PaddleWorkerError("The paddle worker exited before it answered")

    monkeypatch.setattr(worker, "call", fail)
    oc = orientation.get_orientation(Path("x.png"))
    assert (oc.angle, oc.confidence) == (0, 0.0)
    assert [(r.name, r.levelname) for r in caplog.records] == [
        ("paperless.paddleocr.orientation", "ERROR"),
    ]
