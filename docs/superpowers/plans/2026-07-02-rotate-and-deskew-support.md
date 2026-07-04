# Rotate and Deskew Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `PAPERLESS_OCR_ROTATE_PAGES` and `PAPERLESS_OCR_DESKEW` actually work: implement `get_orientation` with
PaddleOCR's document orientation classifier and `get_deskew` with a projection-profile estimator.

**Architecture:** Two new modules under `paddle_engine/`: `orientation.py` (wraps
`paddleocr.DocImgOrientationClassification`, cached instance, defensive fallback to angle 0 / confidence 0) and
`deskew.py` (pure PIL + numpy, no paddle dependency, fully unit-testable in CI). `engine.py`'s
`get_orientation`/`get_deskew` delegate to them. ocrmypdf drives everything else: it calls `get_orientation` only when
`--rotate-pages` is set and rotates when `confidence >= rotate_pages_threshold` (verified ocrmypdf 17.8.0
`_pipeline.py:485-495`); it rotates the raster by `get_deskew(...)` degrees, positive = counterclockwise, via
`PIL.Image.rotate` (verified `_pipeline.py:645-663`).

**Tech Stack:** Python 3.12, numpy, Pillow, paddleocr (runtime only; stubbed in tests).

## Global Constraints

- Python `>=3.12`; `ruff check .`, `ruff format --check .`, `mypy --ignore-missing-imports paperless_paddleocr` must pass.
- CI tests must run without paddlepaddle/paddleocr; `orientation.py` is tested with a stub
  model, `deskew.py` runs for real (numpy must be added to the CI test deps, Task 3).
- Any failure inside these hooks must degrade to the current behaviour (no rotation), never
  break OCR of the page.
- No em-dashes in prose or comments; comments explain WHY only.

---

### Task 1: Deskew estimator (pure, CI-testable)

**Files:**

- Create: `paperless_paddleocr/paddle_engine/deskew.py`
- Modify: `paperless_paddleocr/paddle_engine/engine.py:139-141`
- Test: `tests/test_deskew.py` (create)

**Interfaces:**

- Produces: `deskew.estimate_skew(input_file: Path) -> float` returning degrees to rotate
  counterclockwise (matches `PIL.Image.rotate` and therefore ocrmypdf's contract).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_deskew.py`:

```python
"""estimate_skew must recover small page rotations from text-line structure.

Synthetic pages use horizontal black bars as text-line stand-ins; the
projection-profile score peaks when the bars are horizontal again.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from paperless_paddleocr.paddle_engine.deskew import estimate_skew


def _bars_image(tmp_path, rotate_by: float):
    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    for y in range(60, 560, 40):
        draw.rectangle([40, y, 760, y + 12], fill=0)
    if rotate_by:
        img = img.rotate(rotate_by, expand=True, fillcolor=255)
    path = tmp_path / "page.png"
    img.save(path)
    return path


def test_straight_page_reports_near_zero(tmp_path):
    assert abs(estimate_skew(_bars_image(tmp_path, 0.0))) <= 0.2


def test_clockwise_skew_needs_positive_correction(tmp_path):
    # rotate_by=-2 turns the content 2 degrees clockwise; the correction that
    # ocrmypdf applies via PIL (positive = counterclockwise) must be about +2.
    angle = estimate_skew(_bars_image(tmp_path, -2.0))
    assert 1.5 <= angle <= 2.5


def test_counterclockwise_skew_needs_negative_correction(tmp_path):
    angle = estimate_skew(_bars_image(tmp_path, 2.0))
    assert -2.5 <= angle <= -1.5


def test_angle_beyond_search_range_is_treated_as_not_skew(tmp_path):
    # A true angle just outside the range peaks at the search boundary; the
    # estimator must report 0 rather than rotating by the clamped amount.
    assert estimate_skew(_bars_image(tmp_path, 6.0)) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_deskew.py -v`
Expected: FAIL with `ModuleNotFoundError: ... deskew`

- [ ] **Step 3: Implement `deskew.py`**

```python
"""Projection-profile skew estimation for ocrmypdf's deskew hook.

The classic Postl method: rotate a downscaled binarised page through
candidate angles and score how sharply ink concentrates into rows (sum of
squared differences of adjacent row-ink counts). Text lines give a strong
peak at the deskewed angle. Pure PIL + numpy so the estimator is exact to
test in CI without paddle installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

#: Scans are rarely skewed beyond a few degrees; a wider search would start
#: locking onto rotated tables and figures instead of the text body.
MAX_SKEW_DEGREES = 5.0
_COARSE_STEP = 0.5
_FINE_STEP = 0.1
#: Skew is a global property; full resolution adds cost, not signal.
_TARGET_WIDTH = 1200


def _score(img: Image.Image, angle: float) -> float:
    rotated = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)
    ink = 255 - np.asarray(rotated, dtype=np.int64)
    profile = ink.sum(axis=1)
    diff = profile[1:] - profile[:-1]
    return float((diff * diff).sum())


def estimate_skew(input_file: Path) -> float:
    """Degrees to rotate counterclockwise so text lines run horizontally."""
    with Image.open(input_file) as raw:
        gray = raw.convert("L")
        if gray.width > _TARGET_WIDTH:
            height = max(1, round(gray.height * _TARGET_WIDTH / gray.width))
            gray = gray.resize((_TARGET_WIDTH, height))
        # A global mean threshold is enough: the score only needs ink rows to
        # dominate background rows, not a clean segmentation.
        arr = np.asarray(gray)
        binary = Image.fromarray(np.where(arr < arr.mean(), 0, 255).astype(np.uint8))

    best_angle = 0.0
    best_score = _score(binary, 0.0)
    steps = int(MAX_SKEW_DEGREES / _COARSE_STEP)
    for i in range(-steps, steps + 1):
        angle = i * _COARSE_STEP
        s = _score(binary, angle)
        if s > best_score:
            best_angle, best_score = angle, s

    fine_span = int(_COARSE_STEP / _FINE_STEP)
    for i in range(-fine_span, fine_span + 1):
        angle = best_angle + i * _FINE_STEP
        if abs(angle) > MAX_SKEW_DEGREES:
            continue
        s = _score(binary, angle)
        if s > best_score:
            best_angle, best_score = angle, s

    # A best fit at the search boundary means the true angle is outside the
    # range; rotating by the clamped value would make things worse.
    if abs(best_angle) >= MAX_SKEW_DEGREES:
        return 0.0
    return best_angle
```

- [ ] **Step 4: Wire `engine.py` and run the tests**

In `engine.py`, replace:

```python
    @staticmethod
    def get_deskew(input_file: Path, options: Any) -> float:
        return 0.0
```

with:

```python
    @staticmethod
    def get_deskew(input_file: Path, options: Any) -> float:
        from paperless_paddleocr.paddle_engine import deskew

        try:
            return deskew.estimate_skew(input_file)
        except Exception:
            log.exception("Deskew estimation failed for %s; skipping deskew.", input_file)
            return 0.0
```

Add at module level in `engine.py` (it currently has no logger):

```python
import logging

log = logging.getLogger("paperless.paddleocr.engine")
```

Run: `pytest tests/test_deskew.py -v` - expected: all PASS (allow a few seconds; the
estimator rotates a 1.2 MP image about 30 times per case).

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/paddle_engine/deskew.py paperless_paddleocr/paddle_engine/engine.py tests/test_deskew.py
git commit -m "Implement projection-profile deskew for ocrmypdf"
```

---

### Task 2: Orientation classifier wrapper

**Files:**

- Create: `paperless_paddleocr/paddle_engine/orientation.py`
- Modify: `paperless_paddleocr/paddle_engine/engine.py:134-137`
- Test: `tests/test_orientation.py` (create)

**Interfaces:**

- Produces: `orientation.get_orientation(input_file: Path) -> OrientationConfidence`.
- Consumes: `paddleocr.DocImgOrientationClassification` (verified exported by
  paddleocr 3.7.0 `paddleocr/__init__.py`); its `predict()` returns paddlex
  classification results supporting `.get("label_names")` / `.get("scores")` with labels
  `"0" | "90" | "180" | "270"`.

**Label direction caveat (resolve during Step 4):** ocrmypdf treats `angle` as "how far
the page content is rotated clockwise" and applies the counterclockwise correction
(`_pipeline.py:467-495`). Whether the paddle label means "content is rotated N degrees
clockwise" or "rotate N degrees clockwise to fix" must be confirmed empirically once with
a real model; the `_LABEL_TO_ANGLE` table below is the single place to encode the result.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_orientation.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orientation.py -v`
Expected: FAIL with `ModuleNotFoundError: ... orientation`

- [ ] **Step 3: Implement `orientation.py`**

```python
"""Page orientation detection for ocrmypdf's --rotate-pages hook.

Wraps PaddleOCR's PP-LCNet document orientation classifier. The classifier
is a ~7 MB model cached under ~/.paddlex like the OCR models; the instance
is cached at module level because ocrmypdf calls the hook once per page.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from ocrmypdf.pluginspec import OrientationConfidence

log = logging.getLogger("paperless.paddleocr.orientation")

#: Below this probability a 4-way classifier is guessing (uniform prior is
#: 0.25); report "no rotation" so paperless's rotate_pages_threshold, which
#: was tuned for Tesseract's confidence scale, cannot act on noise.
MIN_PROBABILITY = 0.5

#: Classifier label -> ocrmypdf angle (degrees the content is rotated
#: clockwise; ocrmypdf applies the counterclockwise correction). Verify once
#: against a real model (see plan step) and adjust here if the label turns
#: out to mean the correction instead of the rotation.
_LABEL_TO_ANGLE = {"0": 0, "90": 90, "180": 180, "270": 270}

_MODEL: Any = None
_MODEL_LOCK = threading.Lock()
_PREDICT_LOCK = threading.Lock()


def _load_model() -> Any:
    from paddleocr import DocImgOrientationClassification

    return DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")


def _get_model() -> Any:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = _load_model()
    return _MODEL


def get_orientation(input_file: Path) -> OrientationConfidence:
    """Classify the page's rotation; degrade to 'upright' on any failure."""
    try:
        model = _get_model()
        with _PREDICT_LOCK:
            result = model.predict(str(input_file))
        data = result[0]
        labels = data.get("label_names") or []
        scores = data.get("scores") or []
        label = str(labels[0]) if labels else "0"
        probability = float(scores[0]) if scores else 0.0
    except Exception:
        log.exception("Orientation detection failed for %s; assuming upright.", input_file)
        return OrientationConfidence(angle=0, confidence=0.0)

    angle = _LABEL_TO_ANGLE.get(label, 0)
    if probability < MIN_PROBABILITY:
        return OrientationConfidence(angle=0, confidence=0.0)
    return OrientationConfidence(angle=angle, confidence=probability * 100.0)
```

In `engine.py`, replace:

```python
    @staticmethod
    def get_orientation(input_file: Path, options: Any) -> OrientationConfidence:
        # PaddleOCR and ocrmypdf handle rotation/deskew themselves.
        return OrientationConfidence(angle=0, confidence=0.0)
```

with:

```python
    @staticmethod
    def get_orientation(input_file: Path, options: Any) -> OrientationConfidence:
        from paperless_paddleocr.paddle_engine import orientation

        return orientation.get_orientation(input_file)
```

Run: `pytest tests/test_orientation.py -v` - expected: all PASS.

- [ ] **Step 4: Verify the label direction against the real model (required)**

On a machine with paddlepaddle + paddleocr installed, run this snippet:

```python
from PIL import Image, ImageDraw
from paddleocr import DocImgOrientationClassification

img = Image.new("L", (600, 800), 255)
d = ImageDraw.Draw(img)
for y in range(80, 720, 50):
    d.text((60, y), "The quick brown fox jumps over the lazy dog", fill=0)
img.rotate(-90, expand=True, fillcolor=255).save("cw90.png")  # content rotated 90 CW

m = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
print(m.predict("cw90.png")[0])
```

If the printed label is `"90"`, `_LABEL_TO_ANGLE` is correct as written. If it is
`"270"`, swap the 90/270 entries and update the table's comment with the observed
convention and the date. Then feed a real rotated scan through
`ocrmypdf --plugin paperless_paddleocr.ocrmypdf_plugin --rotate-pages in.pdf out.pdf`
and confirm the output page is upright.

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/paddle_engine/orientation.py paperless_paddleocr/paddle_engine/engine.py tests/test_orientation.py
git commit -m "Implement page orientation detection for rotate-pages"
```

---

### Task 3: CI dependency and README truth

**Files:**

- Modify: `.github/workflows/ci.yml:125-129` (test deps)
- Modify: `README.md` (honoured-settings table)

- [ ] **Step 1: Add numpy to the CI test install**

In the `tests` job of `.github/workflows/ci.yml`, change:

```yaml
          python -m pip install "ocrmypdf>=17.4" "lxml>=4.9" pillow pytest
```

to:

```yaml
          python -m pip install "ocrmypdf>=17.4" "lxml>=4.9" pillow numpy pytest
```

(numpy is a transitive runtime dependency via paddleocr, so this adds nothing to
production installs; the CI job skips dependency resolution with `--no-deps`.)

- [ ] **Step 2: Update the README honoured-settings rows**

Replace the effect cells of the two rows:

- `PAPERLESS_OCR_DESKEW` row: "Pre-OCR deskew, estimated by a projection-profile
  analysis of the page raster (range ±5°)."
- `PAPERLESS_OCR_ROTATE_PAGES`, `PAPERLESS_OCR_ROTATE_PAGES_THRESHOLD` row: "Pre-OCR page
  rotation via PaddleOCR's document orientation classifier. The threshold compares
  against a probability percentage (0-100); raise it to rotate more conservatively."

- [ ] **Step 3: Run everything and commit**

Run: `pytest tests/ -q` - expected: all PASS.

```bash
git add .github/workflows/ci.yml README.md
git commit -m "Add numpy to CI test deps and document rotate/deskew support"
```
