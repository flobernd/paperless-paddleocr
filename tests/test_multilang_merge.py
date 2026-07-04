"""Coverage for MultiLangPaddleEngine's multi-language branch.

The per-language pass (PaddleOCREngine.generate_hocr) is stubbed to write
deterministic hOCR per language, so winner selection, NMS merging and the
failure paths are exercised without paddle installed.
"""

from __future__ import annotations

import types
from pathlib import Path

from PIL import Image

from paperless_paddleocr.ocrmypdf_plugin import (
    MultiLangPaddleEngine,
    _parse_hocr_words,
    _Word,
    _write_merged_hocr,
)
from paperless_paddleocr.paddle_engine import PaddleOCREngine


def _install_fake_passes(monkeypatch, words_by_lang, fail_langs=()):
    """Stub the base per-language pass with fixed words per language."""

    def fake(input_file: Path, output_hocr: Path, output_text: Path, options) -> None:
        lang = options.languages[0]
        if lang in fail_langs:
            raise RuntimeError(f"simulated engine failure for {lang}")
        words = words_by_lang[lang]
        _write_merged_hocr(output_hocr, words, 200, 100, hocr_lang=lang)
        output_text.write_text(" ".join(w.text for w in words), encoding="utf-8")

    monkeypatch.setattr(PaddleOCREngine, "generate_hocr", staticmethod(fake))


def _run(tmp_path, monkeypatch, strategy, words_by_lang, fail_langs=()):
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_MULTI_LANG_STRATEGY", strategy)
    _install_fake_passes(monkeypatch, words_by_lang, fail_langs)
    img = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(img)
    out_hocr = tmp_path / "out.hocr"
    out_text = tmp_path / "out.txt"
    options = types.SimpleNamespace(
        languages=list(words_by_lang),
        paddle_engine="classic-cpu",
    )
    MultiLangPaddleEngine.generate_hocr(img, out_hocr, out_text, options)
    return out_hocr, out_text


def test_winner_keeps_the_higher_aggregate_confidence_language(tmp_path, monkeypatch):
    words_by_lang = {
        "en": [_Word("cat", 0, 0, 50, 20, 60)],
        "german": [_Word("Katze", 0, 0, 50, 20, 95)],
    }
    out_hocr, out_text = _run(tmp_path, monkeypatch, "winner", words_by_lang)
    assert [w.text for w in _parse_hocr_words(out_hocr)] == ["Katze"]
    assert out_text.read_text(encoding="utf-8") == "Katze"


def test_merge_keeps_highest_confidence_box_per_location(tmp_path, monkeypatch):
    words_by_lang = {
        # Same location: german wins on confidence. Second en word is
        # elsewhere on the page and must survive the NMS.
        "en": [_Word("cat", 0, 0, 50, 20, 60), _Word("extra", 100, 50, 150, 70, 80)],
        "german": [_Word("Katze", 0, 0, 50, 20, 95)],
    }
    out_hocr, _ = _run(tmp_path, monkeypatch, "merge", words_by_lang)
    assert sorted(w.text for w in _parse_hocr_words(out_hocr)) == ["Katze", "extra"]


def test_partial_language_failure_uses_surviving_passes(tmp_path, monkeypatch):
    words_by_lang = {
        "en": [_Word("hello", 0, 0, 50, 20, 90)],
        "german": [],  # never produced: this pass raises
    }
    out_hocr, out_text = _run(
        tmp_path, monkeypatch, "winner", words_by_lang, fail_langs=("german",)
    )
    assert [w.text for w in _parse_hocr_words(out_hocr)] == ["hello"]
    assert out_text.read_text(encoding="utf-8") == "hello"


def test_all_languages_failing_writes_empty_outputs(tmp_path, monkeypatch):
    words_by_lang = {"en": [], "german": []}
    out_hocr, out_text = _run(
        tmp_path, monkeypatch, "winner", words_by_lang, fail_langs=("en", "german")
    )
    assert _parse_hocr_words(out_hocr) == []
    assert out_text.read_text(encoding="utf-8") == ""


def test_unknown_strategy_falls_back_to_winner(tmp_path, monkeypatch):
    words_by_lang = {
        "en": [_Word("cat", 0, 0, 50, 20, 60)],
        "german": [_Word("Katze", 0, 0, 50, 20, 95)],
    }
    out_hocr, _ = _run(tmp_path, monkeypatch, "bogus-strategy", words_by_lang)
    assert [w.text for w in _parse_hocr_words(out_hocr)] == ["Katze"]


def _count_passes(tmp_path, monkeypatch, engine: str) -> int:
    calls: list[str] = []

    def fake(input_file, output_hocr, output_text, options) -> None:
        calls.append(options.languages[0])
        _write_merged_hocr(
            output_hocr,
            [_Word("x", 0, 0, 10, 10, 90)],
            200,
            100,
            hocr_lang=options.languages[0],
        )
        output_text.write_text("x", encoding="utf-8")

    monkeypatch.setattr(PaddleOCREngine, "generate_hocr", staticmethod(fake))
    img = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(img)
    options = types.SimpleNamespace(languages=["en", "german"], paddle_engine=engine)
    MultiLangPaddleEngine.generate_hocr(img, tmp_path / "o.hocr", tmp_path / "o.txt", options)
    return len(calls)


def test_vl_remote_runs_a_single_pass_for_multiple_languages(tmp_path, monkeypatch):
    assert _count_passes(tmp_path, monkeypatch, "vl-remote") == 1


def test_classic_still_runs_one_pass_per_language(tmp_path, monkeypatch):
    assert _count_passes(tmp_path, monkeypatch, "classic-cpu") == 2


def test_single_language_keeps_the_engine_sidecar(tmp_path, monkeypatch):
    # The engine now writes a reading-ordered sidecar itself; the plugin
    # must not rebuild (and re-order) it from the hOCR.
    def fake(input_file, output_hocr, output_text, options) -> None:
        _write_merged_hocr(output_hocr, [_Word("x", 0, 0, 10, 10, 90)], 200, 100, hocr_lang="en")
        output_text.write_text("ENGINE ORDER", encoding="utf-8")

    monkeypatch.setattr(PaddleOCREngine, "generate_hocr", staticmethod(fake))
    img = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(img)
    out_text = tmp_path / "o.txt"
    options = types.SimpleNamespace(languages=["en"], paddle_engine="classic-cpu")
    MultiLangPaddleEngine.generate_hocr(img, tmp_path / "o.hocr", out_text, options)
    assert out_text.read_text(encoding="utf-8") == "ENGINE ORDER"


def test_merged_hocr_groups_words_into_lines(tmp_path):
    from lxml import html

    out = tmp_path / "m.hocr"
    words = [
        _Word("hello", 0, 0, 50, 20, 90),
        _Word("world", 60, 0, 110, 20, 90),
        _Word("below", 0, 40, 50, 60, 90),
    ]
    _write_merged_hocr(out, words, 200, 100, hocr_lang="german")
    tree = html.parse(str(out))
    lines = tree.findall('.//{*}span[@class="ocr_line"]')
    assert len(lines) == 2  # not one line per word
    first_line_words = ["".join(s.itertext()) for s in lines[0].findall("{*}span")]
    assert first_line_words == ["hello", "world"]
