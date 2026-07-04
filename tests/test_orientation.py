"""get_orientation maps classifier output to ocrmypdf's OrientationConfidence.

The real classifier needs paddle + a downloaded model; these tests stub the
model loader and pin the mapping, scaling and failure behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paperless_paddleocr.paddle_engine import orientation


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    orientation._MODEL = None
    yield
    orientation._MODEL = None


class _FakeModel:
    def __init__(self, label: str, score: float) -> None:
        self._label, self._score = label, score

    def predict(self, path: str) -> list:
        return [{"label_names": [self._label], "scores": [self._score]}]


def test_confident_rotation_is_reported(monkeypatch):
    monkeypatch.setattr(orientation, "_load_model", lambda: _FakeModel("90", 0.97))
    oc = orientation.get_orientation(Path("x.png"))
    assert oc.angle == 90
    assert round(oc.confidence) == 97


def test_low_probability_is_suppressed(monkeypatch):
    # A 4-way classifier below 0.5 is guessing; paperless's default
    # rotate_pages_threshold (12) would otherwise act on noise.
    monkeypatch.setattr(orientation, "_load_model", lambda: _FakeModel("180", 0.4))
    oc = orientation.get_orientation(Path("x.png"))
    assert (oc.angle, oc.confidence) == (0, 0.0)


def test_model_failure_degrades_to_no_rotation(monkeypatch):
    def boom():
        raise RuntimeError("no paddle here")

    monkeypatch.setattr(orientation, "_load_model", boom)
    oc = orientation.get_orientation(Path("x.png"))
    assert (oc.angle, oc.confidence) == (0, 0.0)
