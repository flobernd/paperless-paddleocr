# Docs and Infra Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the retired release workflow, label unverified performance figures as guidance, document the CJK
word-granularity limitation, and strengthen the CI mypy job.

**Architecture:** Documentation and workflow edits only; no package code changes.

**Tech Stack:** GitHub Actions YAML, Markdown.

## Global Constraints

- `.yamllint` and `.markdownlint.yaml` rules must pass (CI lints both).
- No em-dashes in prose; the README currently uses ` - ` separators, keep that style.

---

### Task 1: Delete the retired release workflow

**Files:**

- Delete: `.github/workflows/release.yml`

The file is a no-op stub whose own header says: "To finish the cleanup,
`git rm .github/workflows/release.yml`." The removal context is preserved in git history
by the stub commit itself.

- [ ] **Step 1: Remove and commit**

```bash
git rm .github/workflows/release.yml
git commit -m "Remove retired release workflow stub"
```

---

### Task 2: Label performance figures as guidance

**Files:**

- Modify: `README.md` ("Choosing an engine" table, "Performance notes" section)

The numbers ("~1-3 pages/sec", "~20 s/page on an A100") read as measured facts but are
estimates. Until the benchmark from `2026-07-02-mkldnn-reverification.md` replaces them
with measured values, mark them as rough guidance.

- [ ] **Step 1: Edit the engine table cells**

In the "Choosing an engine" table, change the two throughput cells:

- classic-cpu: `Roughly 1-3 pages/sec on a modern CPU (rough guidance, not a benchmark). No GPU needed. Right default for most deployments.`
- vl-remote: `Much higher accuracy on small fonts, low-DPI scans, handwriting. Per-page latency depends on the remote
  GPU - expect tens of seconds per page.`

- [ ] **Step 2: Edit the matching "Performance notes" bullets**

Apply the same wording to the `classic-cpu` and `vl-remote` bullets in "Performance
notes" (replace "does roughly 1-3 pages/sec for typical scans" with "handles roughly 1-3
pages/sec on typical scans as rough guidance; measure on your own hardware" and "on an
A100 expect ~20 s/page" with "expect tens of seconds per page depending on the GPU").

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Mark throughput figures as rough guidance"
```

---

### Task 3: Document the CJK word-granularity limitation

**Files:**

- Modify: `README.md` ("Language handling" section)

Background: `classic._merge_subtokens` joins PaddleOCR's sub-tokens into words at
whitespace separators (`classic.py:112-136`). CJK text has no inter-word whitespace, so a
whole recognised line becomes a single hOCR word. Text content and search are unaffected;
only word-level boxes (search-hit highlighting in PDF viewers) degrade to line-level.

- [ ] **Step 1: Add a note at the end of "Language handling"**

```markdown
> [!NOTE]
> For languages written without inter-word spaces (Chinese, Japanese, Thai, ...) the
> invisible text layer carries line-level rather than word-level boxes, so search-hit
> highlighting in PDF viewers selects whole lines. Extracted text and search results are
> unaffected.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document line-level boxes for unspaced scripts"
```

---

### Task 4: Run mypy against real dependencies in CI

**Files:**

- Modify: `.github/workflows/ci.yml:83-108` (`static` job)

With no dependencies installed, mypy sees every import as `Any` and checks almost
nothing. Installing the light runtime deps (ocrmypdf, lxml, pillow - the same set the
test job uses) lets it check the ocrmypdf plugin surface for real;
`--ignore-missing-imports` keeps paddle and django soft.

- [ ] **Step 1: Edit the static job**

Replace:

```yaml
      - name: Install ruff and mypy
        run: python -m pip install --upgrade pip ruff mypy
```

with:

```yaml
      - name: Install ruff, mypy and typed runtime deps
        run: |
          python -m pip install --upgrade pip ruff mypy
          python -m pip install "ocrmypdf>=17.4" "lxml>=4.9" pillow numpy
```

(Keep `numpy` in sync with the test job if the rotate-and-deskew plan has landed;
otherwise omit it here.)

- [ ] **Step 2: Verify locally if possible, then commit**

Run (in any venv): `pip install "ocrmypdf>=17.4" lxml pillow numpy mypy && mypy --ignore-missing-imports paperless_paddleocr`
Expected: no errors. Fix any newly surfaced type errors before committing; they are real
findings this change exists to catch.

```bash
git add .github/workflows/ci.yml
git commit -m "Type-check against installed runtime deps in CI"
```
