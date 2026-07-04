# VL Remote Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fail fast on unreachable/misauthenticated VL servers, make the pipeline version configurable (paddleocr 3.7
supports v1/v1.5/v1.6), move the heavy `doc-parser` extra behind an optional `[vl]` extra, and surface actionable errors
when VL dependencies are missing.

**Architecture:** The preflight probe lives in `ocrmypdf_plugin.check_options` (stdlib `urllib`, result cached per
process). URL normalisation moves from a private options-taking helper to a string-taking `normalize_server_url` in
`vl.py` so both call sites share it. The pipeline version flows exactly like the other VL settings: env var in
`parser.py` -> ocrmypdf kwarg -> `add_options` argparse arg -> `vl._build_pipeline`.

**Tech Stack:** Python 3.12, `urllib.request`, pytest.

## Global Constraints

- Python `>=3.12`; `ruff check .`, `ruff format --check .`, `mypy --ignore-missing-imports paperless_paddleocr` must pass.
- Tests must run without paddlepaddle/paddleocr and without network access.
- No em-dashes in prose or comments; comments explain WHY only.

---

### Task 1: Public server-URL normaliser

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/vl.py:84-101`
- Test: `tests/test_vl_server_url.py` (create)

**Interfaces:**

- Produces: `vl.normalize_server_url(raw: str) -> str` (appends `/v1` unless a version
  suffix is present; strips trailing slashes). `vl._server_url(options)` becomes a thin
  wrapper and keeps its behaviour.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vl_server_url.py`:

```python
from __future__ import annotations

import pytest

from paperless_paddleocr.paddle_engine.vl import normalize_server_url


def test_appends_v1_when_missing():
    assert normalize_server_url("http://gpu:8118") == "http://gpu:8118/v1"


def test_keeps_existing_version_suffix():
    assert normalize_server_url("http://gpu:8118/v1") == "http://gpu:8118/v1"
    assert normalize_server_url("http://gpu:8118/v2/") == "http://gpu:8118/v2"


def test_empty_url_raises():
    with pytest.raises(RuntimeError):
        normalize_server_url("   ")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vl_server_url.py -v`
Expected: FAIL with `ImportError: cannot import name 'normalize_server_url'`

- [ ] **Step 3: Implement**

In `vl.py`, replace the body of `_server_url` with a wrapper and add the public function
(the docstring moves over verbatim, minus the options plumbing):

```python
def normalize_server_url(raw: str) -> str:
    """Normalised inference-server URL.

    PaddleOCR-VL forwards the URL verbatim to an OpenAI-style client that
    appends ``/chat/completions``; the genai-vllm-server only serves under
    the standard ``/v1`` prefix. Accept the URL with or without it.
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        raise RuntimeError(
            "vl-remote engine requires PAPERLESS_PADDLEOCR_VL_SERVER_URL "
            "(or --paddle-vl-server-url) to point at a PaddleOCR-VL "
            "inference server.",
        )
    url = cleaned.rstrip("/")
    if not re.search(r"/v\d+$", url):
        url = f"{url}/v1"
    return url


def _server_url(options: Any) -> str:
    return normalize_server_url(getattr(options, "paddle_vl_server_url", "") or "")
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_vl_server_url.py tests/ -q` - expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/paddle_engine/vl.py tests/test_vl_server_url.py
git commit -m "Extract a public VL server URL normaliser"
```

---

### Task 2: Connectivity preflight in check_options

**Files:**

- Modify: `paperless_paddleocr/ocrmypdf_plugin.py:536-551` (vl-remote branch)
- Test: `tests/test_vl_preflight.py` (create)

**Interfaces:**

- Produces: `ocrmypdf_plugin._probe_vl_server(server_url: str, api_key: str) -> None`
  raising `ocrmypdf.exceptions.MissingDependencyError` on unreachable server or auth
  failure; module-level `_PROBED_SERVERS: set[tuple[str, str]]` caches successes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vl_preflight.py`:

```python
"""The vl-remote preflight must turn config mistakes into clear errors.

urlopen is stubbed at the plugin module level; no network is touched.
"""

from __future__ import annotations

import contextlib
import io
import sys
import types
import urllib.error

import pytest
from ocrmypdf.exceptions import MissingDependencyError

from paperless_paddleocr import ocrmypdf_plugin


@pytest.fixture(autouse=True)
def _fresh_probe_cache(monkeypatch):
    ocrmypdf_plugin._PROBED_SERVERS.clear()
    # check_options imports PaddleOCRVL first; satisfy it with a stub.
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCRVL = object
    fake.PaddleOCR = object
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    yield
    ocrmypdf_plugin._PROBED_SERVERS.clear()


def _opts() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        paddle_engine="vl-remote",
        paddle_vl_server_url="http://gpu:8118",
        paddle_vl_api_key="secret",
    )


def _ok_urlopen(calls):
    @contextlib.contextmanager
    def fake(request, timeout=0):
        calls.append(request.full_url)
        yield io.BytesIO(b"{}")

    return fake


def test_reachable_server_passes_and_is_cached(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(ocrmypdf_plugin, "_urlopen", _ok_urlopen(calls))
    ocrmypdf_plugin.check_options(_opts())
    ocrmypdf_plugin.check_options(_opts())
    assert calls == ["http://gpu:8118/v1/models"]


def test_unreachable_server_raises_actionable_error(monkeypatch):
    def fail(request, timeout=0):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ocrmypdf_plugin, "_urlopen", fail)
    with pytest.raises(MissingDependencyError, match="not reachable"):
        ocrmypdf_plugin.check_options(_opts())


def test_auth_failure_raises_actionable_error(monkeypatch):
    def unauthorized(request, timeout=0):
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", hdrs=None, fp=None
        )

    monkeypatch.setattr(ocrmypdf_plugin, "_urlopen", unauthorized)
    with pytest.raises(MissingDependencyError, match="rejected the API key"):
        ocrmypdf_plugin.check_options(_opts())


def test_other_http_errors_do_not_block(monkeypatch):
    # A 404 on /v1/models means the server is up but shaped differently;
    # blocking OCR on that would be a false negative.
    def not_found(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 404, "Nope", hdrs=None, fp=None)

    monkeypatch.setattr(ocrmypdf_plugin, "_urlopen", not_found)
    ocrmypdf_plugin.check_options(_opts())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vl_preflight.py -v`
Expected: FAIL with `AttributeError: ... has no attribute '_PROBED_SERVERS'`

- [ ] **Step 3: Implement**

In `ocrmypdf_plugin.py`, add near the other module-level definitions:

```python
from urllib.request import Request, urlopen as _urlopen

#: Successful probes per (url, api_key); check_options runs once per
#: document, and one probe per worker process is enough.
_PROBED_SERVERS: set[tuple[str, str]] = set()


def _probe_vl_server(server_url: str, api_key: str) -> None:
    """Fail fast when the VL server is unreachable or rejects the API key.

    Without this, a wrong URL or key surfaces only at first predict, deep
    inside a celery task, after the local pipeline is fully constructed.
    """
    from urllib.error import HTTPError, URLError

    from ocrmypdf.exceptions import MissingDependencyError

    from paperless_paddleocr.paddle_engine.vl import normalize_server_url

    base = normalize_server_url(server_url)
    cache_key = (base, api_key)
    if cache_key in _PROBED_SERVERS:
        return

    request = Request(f"{base}/models")  # noqa: S310 - operator-configured URL
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with _urlopen(request, timeout=5):  # noqa: S310
            pass
    except HTTPError as e:
        if e.code in (401, 403):
            raise MissingDependencyError(
                f"The PaddleOCR-VL server at {base} rejected the API key "
                f"(HTTP {e.code}). Check PAPERLESS_PADDLEOCR_VL_API_KEY against "
                "the api-key configured in the server's vllm_config.yaml.",
            ) from e
        log.warning(
            "PaddleOCR-VL server preflight got HTTP %d from %s/models; continuing.",
            e.code,
            base,
        )
    except URLError as e:
        raise MissingDependencyError(
            f"The PaddleOCR-VL server at {base} is not reachable ({e.reason}). "
            "Check PAPERLESS_PADDLEOCR_VL_SERVER_URL and that the "
            "paddleocr-genai-vllm-server container is running.",
        ) from e
    _PROBED_SERVERS.add(cache_key)
```

In `check_options`, at the end of the vl-remote branch (after the empty-`server_url`
check, replacing the bare `return`):

```python
        _probe_vl_server(server_url, (getattr(options, "paddle_vl_api_key", "") or "").strip())
        return
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_vl_preflight.py tests/test_check_options_gpu.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/ocrmypdf_plugin.py tests/test_vl_preflight.py
git commit -m "Preflight the VL server before starting OCR"
```

---

### Task 3: Configurable pipeline version

**Files:**

- Modify: `paperless_paddleocr/parser.py` (env read + ocrmypdf kwarg)
- Modify: `paperless_paddleocr/ocrmypdf_plugin.py` (`add_options`)
- Modify: `paperless_paddleocr/paddle_engine/vl.py:104-146, 267` (`_build_pipeline`,
  spotting detection)
- Test: `tests/test_vl_pipeline_version.py` (create)
- Modify: `README.md` (new env var section)

**Interfaces:**

- Produces: env `PAPERLESS_PADDLEOCR_VL_PIPELINE_VERSION` (default `v1.5`), ocrmypdf kwarg
  `paddle_vl_pipeline_version`. paddleocr 3.7 accepts `v1`, `v1.5`, `v1.6` and raises
  `ValueError` for anything else, which is the desired failure mode for typos.
- Note: `v1.5` stays the default because the documented server image serves
  `PaddleOCR-VL-1.5-0.9B`; the local pipeline version must match the served model family.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vl_pipeline_version.py`:

```python
from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from paperless_paddleocr.paddle_engine import vl


class _CapturingPipeline:
    last_kwargs: dict[str, Any] = {}

    # pipeline_version is declared explicitly (not swallowed by **kwargs)
    # because _build_pipeline feature-detects it via inspect.signature.
    def __init__(self, pipeline_version: str = "v1.6", **kwargs: Any) -> None:
        _CapturingPipeline.last_kwargs = {"pipeline_version": pipeline_version, **kwargs}
        self.pipeline_version = pipeline_version


@pytest.fixture(autouse=True)
def _stub_paddle(monkeypatch):
    fake_paddle = types.ModuleType("paddle")
    fake_paddle.set_device = lambda device: None
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    monkeypatch.setattr(vl, "PaddleOCRVL", _CapturingPipeline)


def _opts(version: str | None) -> types.SimpleNamespace:
    opts = types.SimpleNamespace(
        languages=["en"],
        paddle_engine="vl-remote",
        paddle_vl_server_url="http://gpu:8118",
        paddle_vl_model_name="PaddleOCR-VL-1.5-0.9B",
        paddle_vl_api_key="",
    )
    if version is not None:
        opts.paddle_vl_pipeline_version = version
    return opts


def test_default_pipeline_version_is_v15():
    vl._build_pipeline(_opts(None))
    assert _CapturingPipeline.last_kwargs["pipeline_version"] == "v1.5"


def test_configured_pipeline_version_is_passed_through():
    vl._build_pipeline(_opts("v1.6"))
    assert _CapturingPipeline.last_kwargs["pipeline_version"] == "v1.6"


def test_v1_disables_spotting_prompt():
    pipeline = _CapturingPipeline(pipeline_version="v1")
    assert vl._spotting_capable(pipeline) is False
    assert vl._spotting_capable(_CapturingPipeline(pipeline_version="v1.6")) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vl_pipeline_version.py -v`
Expected: FAIL (`pipeline_version` key absent / `_spotting_capable` missing).

- [ ] **Step 3: Implement**

`vl.py` - in `_build_pipeline`, replace the feature-detection block:

```python
    # pipeline_version is a newer constructor parameter; request the
    # word-spotting pipeline only when the installed paddleocr accepts it.
    if "pipeline_version" in inspect.signature(PaddleOCRVL.__init__).parameters:
        kwargs["pipeline_version"] = "v1.5"
        log.debug("PaddleOCR-VL: pipeline_version=v1.5 (spotting capable)")
    else:
        log.debug("PaddleOCR-VL: pipeline_version unsupported (ocr mode only)")
```

with:

```python
    version = (
        getattr(options, "paddle_vl_pipeline_version", "") or ""
    ).strip() or "v1.5"
    # pipeline_version is a newer constructor parameter; older paddleocr
    # falls back to its only pipeline (ocr mode, no word spotting).
    if "pipeline_version" in inspect.signature(PaddleOCRVL.__init__).parameters:
        kwargs["pipeline_version"] = version
        log.debug("PaddleOCR-VL: pipeline_version=%s", version)
    else:
        log.debug("PaddleOCR-VL: pipeline_version unsupported (ocr mode only)")
```

Add near `_server_url`:

```python
def _spotting_capable(pipeline: Any) -> bool:
    """Word spotting shipped with the v1.5 pipeline; v1 predates it.

    Anything newer than v1 is assumed capable; build_page already falls back
    to ocr mode when a response carries no spotting result.
    """
    version = getattr(pipeline, "pipeline_version", None)
    return version is not None and version != "v1"
```

In `build_page`, replace:

```python
    has_spotting = getattr(pipeline, "pipeline_version", None) == "v1.5"
```

with:

```python
    has_spotting = _spotting_capable(pipeline)
```

If `docs/superpowers/plans/2026-07-02-engine-instance-caching.md` has landed, the version
must also become part of the pipeline cache key. In `vl._pipeline_key`, add as the last
tuple element:

```python
        (getattr(options, "paddle_vl_pipeline_version", "") or "").strip() or "v1.5",
```

`ocrmypdf_plugin.py` - in `add_options`, after the `--paddle-vl-api-key` argument:

```python
    paddle.add_argument(
        "--paddle-vl-pipeline-version",
        default="v1.5",
        dest="paddle_vl_pipeline_version",
        metavar="VERSION",
        help=(
            "vl-remote: PaddleOCR-VL pipeline version (v1, v1.5, v1.6). Must "
            "match the model family the inference server serves "
            "(default: v1.5)."
        ),
    )
```

`parser.py` - in `__init__`, after `self._vl_api_key`:

```python
        self._vl_pipeline_version: str = (
            os.environ.get("PAPERLESS_PADDLEOCR_VL_PIPELINE_VERSION", "") or ""
        ).strip() or "v1.5"
```

and in `construct_ocrmypdf_parameters`, after `"paddle_vl_api_key": self._vl_api_key,`:

```python
            "paddle_vl_pipeline_version": self._vl_pipeline_version,
```

`README.md` - add after the `PAPERLESS_PADDLEOCR_VL_API_KEY` section:

```markdown
### `PAPERLESS_PADDLEOCR_VL_PIPELINE_VERSION`

PaddleOCR-VL pipeline version used on the paperless side (`v1`, `v1.5`, `v1.6`). Must
match the model family the inference server serves: keep `v1.5` for the documented
`PaddleOCR-VL-1.5-0.9B` server; set `v1.6` together with
`PAPERLESS_PADDLEOCR_VL_MODEL_NAME` when the server runs a 1.6 model.

Default: `v1.5`.
```

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -q` - expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/parser.py paperless_paddleocr/ocrmypdf_plugin.py \
    paperless_paddleocr/paddle_engine/vl.py tests/test_vl_pipeline_version.py README.md
git commit -m "Make the PaddleOCR-VL pipeline version configurable"
```

---

### Task 4: Move doc-parser behind a [vl] extra and map init errors

**Files:**

- Modify: `pyproject.toml:39-47`
- Modify: `examples/Dockerfile.vl-remote:38-41`
- Modify: `paperless_paddleocr/paddle_engine/vl.py` (`_build_pipeline` error mapping)
- Modify: `README.md` (install instructions)
- Test: `tests/test_vl_pipeline_cache.py` or `tests/test_vl_pipeline_version.py` (extend)

- [ ] **Step 1: Change pyproject dependencies**

Replace the `dependencies` list (keeping the explanatory comment about paddle wheels):

```toml
dependencies = [
    "ocrmypdf>=17.4",
    "lxml>=4.9",
    "paddleocr>=3.0",
]

[project.optional-dependencies]
# The doc-parser extra pulls paddlex[genai-client,ocr], which only the
# vl-remote engine needs. classic-cpu/classic-gpu images stay lean without it.
vl = ["paddleocr[doc-parser]>=3.0"]
```

- [ ] **Step 2: Update the vl-remote Dockerfile**

In `examples/Dockerfile.vl-remote`, replace:

```dockerfile
RUN pip install --no-cache-dir "paddlepaddle>=3.0" \
 && pip install --no-cache-dir /tmp/plugin/*.whl \
 && rm -rf /tmp/plugin
```

with:

```dockerfile
RUN pip install --no-cache-dir "paddlepaddle>=3.0" \
 && WHEEL=$(ls /tmp/plugin/*.whl) \
 && pip install --no-cache-dir "${WHEEL}[vl]" \
 && rm -rf /tmp/plugin
```

- [ ] **Step 3: Map opaque init failures to an install hint**

`PaddleOCRVL` imports fine without the extra; the missing dependencies surface as a
paddlex `DependencyError` at construction. In `vl.py`, wrap the constructor call at the
end of `_build_pipeline`. Replace:

```python
    return PaddleOCRVL(**kwargs)
```

with:

```python
    try:
        return PaddleOCRVL(**kwargs)
    except Exception as e:
        # paddlex raises its own DependencyError when the doc-parser extra is
        # missing; matching by name avoids importing paddlex here.
        if type(e).__name__ == "DependencyError":
            raise RuntimeError(
                "PaddleOCR-VL dependencies are missing. Install the plugin "
                "with the vl extra: pip install 'paperless-paddleocr[vl]' "
                "(or pip install 'paddleocr[doc-parser]').",
            ) from e
        raise
```

Add a test (append to `tests/test_vl_pipeline_version.py`):

```python
def test_missing_doc_parser_dependency_gets_install_hint(monkeypatch):
    class DependencyError(Exception):
        pass

    class _Exploding:
        def __init__(self, **kwargs: Any) -> None:
            raise DependencyError("paddlex says no")

    monkeypatch.setattr(vl, "PaddleOCRVL", _Exploding)
    with pytest.raises(RuntimeError, match=r"paperless-paddleocr\[vl\]"):
        vl._build_pipeline(_opts(None))
```

- [ ] **Step 4: Update README install notes**

In "C. Non-Docker host install" step 3, add after the existing pip line:

````markdown
For the `vl-remote` engine install the `vl` extra instead:

```bash
pip install "paperless-paddleocr[vl] @ git+https://github.com/flobernd/paperless-paddleocr.git@v0.1.0"
```
````

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/ -q` and `python -m build --wheel` (confirms the extra is declared
correctly; requires `pip install build`).
Expected: tests PASS; the wheel builds and `paperless_paddleocr-*.whl` metadata contains
`Provides-Extra: vl`.

```bash
git add pyproject.toml examples/Dockerfile.vl-remote paperless_paddleocr/paddle_engine/vl.py tests/test_vl_pipeline_version.py README.md
git commit -m "Move VL dependencies behind an optional vl extra"
```

---

### Task 5: Warn when spotting coordinates leave the page

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/vl.py` (`build_page`)
- Test: `tests/test_vl_ocr_page.py` (extend; file created by the layout-fixes plan, create
  it here with just this test if that plan has not landed)

Rationale: the classic adapter rescales coordinates from PaddleOCR's preprocessed space
(`classic._preprocess_scale`); the VL adapter trusts the pipeline to return original-image
coordinates. If a future paddleocr resizes before spotting, every box lands wrong with no
signal. A cheap bound check turns that silent corruption into a log line.

- [ ] **Step 1: Write the failing test**

```python
def test_out_of_bounds_word_boxes_are_logged(caplog):
    import logging

    from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, Word
    from paperless_paddleocr.paddle_engine.vl import _warn_out_of_bounds

    page = Page(width=100, height=50, lang="en", ocr_system="test")
    word = Word("x", (0, 0, 300, 40), 95)  # x1 far beyond width=100
    page.blocks.append(Block(box=word.box, lines=[Line(word.box, 95, "x", [word])]))
    with caplog.at_level(logging.WARNING, logger="paperless.paddleocr.vl"):
        _warn_out_of_bounds(page)
    assert "outside the 100x50 page" in caplog.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vl_ocr_page.py -v`
Expected: FAIL with `ImportError: cannot import name '_warn_out_of_bounds'`

- [ ] **Step 3: Implement**

Add to `vl.py`:

```python
def _warn_out_of_bounds(page: Page) -> None:
    """Flag boxes outside the page: a symptom of mis-scaled VL coordinates.

    The VL pipeline is trusted to report original-image coordinates (the
    classic adapter rescales, this one cannot); if that contract breaks in a
    paddleocr update, the text layer silently lands in the wrong place.
    2% slack allows detector overshoot at the margins.
    """
    limit_x, limit_y = page.width * 1.02, page.height * 1.02
    bad = sum(
        1
        for block in page.blocks
        for line in block.lines
        for w in line.words
        if w.box[2] > limit_x or w.box[3] > limit_y
    )
    if bad:
        log.warning(
            "%d VL word boxes fall outside the %dx%d page; "
            "coordinates may be mis-scaled by the pipeline.",
            bad,
            page.width,
            page.height,
        )
```

Call it at the end of `build_page`, before `return page`:

```python
    _warn_out_of_bounds(page)
```

- [ ] **Step 4: Run and commit**

Run: `pytest tests/ -q` - expected: all PASS.

```bash
git add paperless_paddleocr/paddle_engine/vl.py tests/test_vl_ocr_page.py
git commit -m "Warn when VL word boxes fall outside the page"
```
