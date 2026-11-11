"""Coverage for the language passes of PaddleOCREngine.generate_hocr.

The classic adapter (classic.build_page) is stubbed to return a fixed page
per language, so the failure path, NMS merging, the preparation of each
page's words and winner selection in paddle_engine.merge are exercised
without paddle installed.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from paperless_paddleocr import ocrmypdf_plugin
from paperless_paddleocr.paddle_engine import PaddleOCREngine, classic, vl, worker
from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, Word, write_document
from paperless_paddleocr.paddle_engine.layout import reading_blocks
from paperless_paddleocr.paddle_engine.merge import _merged_page


@pytest.fixture(autouse=True)
def _no_model_answer(monkeypatch):
    _models(monkeypatch, {})


def _models(monkeypatch, models: dict[str, list[str]]) -> None:
    """Answer the op ``models`` from ``models``, with null for a language it lacks."""

    def call(request: dict[str, Any]) -> dict[str, Any]:
        # Another op fails as it does in the worker, so a rename on one side
        # of the wire fails these tests.
        if request["op"] != "models":
            raise worker.PaddleWorkerError(f"unknown op {request['op']!r}")
        return {"models": [models.get(lang) for lang in request["languages"]]}

    monkeypatch.setattr(worker, "call", call)


def _merged_hocr(words: list[Word], hocr_lang: str) -> Page:
    """The page a base pass would build for ``words`` on a 200x100 image."""
    return _merged_page(
        reading_blocks(words, bbox_of=lambda w: w.box), 200, 100, hocr_lang=hocr_lang
    )


def _hocr_words(hocr: Path) -> list[str]:
    from lxml import html

    tree = html.parse(str(hocr))
    return ["".join(s.itertext()) for s in tree.findall('.//{*}span[@class="ocrx_word"]')]


def _install_fake_passes(monkeypatch, words_by_lang, fail_langs=()):
    """Stub the classic adapter with fixed words per language."""

    def fake(input_file: Path, options) -> Page:
        lang = options.languages[0]
        if lang in fail_langs:
            raise RuntimeError(f"simulated engine failure for {lang}")
        return _merged_hocr(words_by_lang[lang], lang)

    monkeypatch.setattr(classic, "build_page", fake)


def _opts(**extra) -> types.SimpleNamespace:
    return types.SimpleNamespace(paddle_engine="classic-cpu", **extra)


def _run(tmp_path, monkeypatch, strategy, words_by_lang, fail_langs=()):
    _install_fake_passes(monkeypatch, words_by_lang, fail_langs)
    return _generate(tmp_path, strategy, list(words_by_lang))


def _generate(tmp_path, strategy, langs):
    img = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(img)
    out_hocr = tmp_path / "out.hocr"
    out_text = tmp_path / "out.txt"
    options = _opts(languages=langs, paddle_multi_lang_strategy=strategy)
    PaddleOCREngine.generate_hocr(img, out_hocr, out_text, options)
    return out_hocr, out_text


def _region(text: str, box) -> Block:
    """One classic detection region: a block holding one line of one word."""
    line = Line(box=box, confidence=90, text=text, words=[Word(text, box, 90)])
    return Block(box=box, lines=[line])


def test_winner_keeps_the_higher_aggregate_confidence_language(tmp_path, monkeypatch):
    words_by_lang = {
        "en": [Word("cat", (0, 0, 50, 20), 60)],
        "german": [Word("Katze", (0, 0, 50, 20), 95)],
    }
    out_hocr, out_text = _run(tmp_path, monkeypatch, "winner", words_by_lang)
    assert _hocr_words(out_hocr) == ["Katze"]
    assert out_text.read_text(encoding="utf-8") == "Katze"


def test_merge_keeps_highest_confidence_box_per_location(tmp_path, monkeypatch):
    words_by_lang = {
        # Same location: german wins on confidence. Second en word is
        # elsewhere on the page and must survive the NMS.
        "en": [Word("cat", (0, 0, 50, 20), 60), Word("extra", (100, 50, 150, 70), 80)],
        "german": [Word("Katze", (0, 0, 50, 20), 95)],
    }
    out_hocr, _ = _run(tmp_path, monkeypatch, "merge", words_by_lang)
    assert sorted(_hocr_words(out_hocr)) == ["Katze", "extra"]


def test_failed_pass_raises_with_the_language_in_the_message(tmp_path, monkeypatch):
    words_by_lang = {"en": [Word("Cat", (0, 0, 40, 20), 90)], "german": []}
    with pytest.raises(RuntimeError, match="language 'german'"):
        _run(tmp_path, monkeypatch, "winner", words_by_lang, fail_langs=("german",))


def test_merge_leaves_out_a_word_that_is_empty_after_cleaning(tmp_path, monkeypatch):
    words_by_lang = {
        "en": [Word("Cat", (0, 0, 40, 20), 90), Word("\x00", (50, 0, 90, 20), 90)],
        "german": [],
    }
    out_hocr, _ = _run(tmp_path, monkeypatch, "merge", words_by_lang)
    assert _hocr_words(out_hocr) == ["Cat"]


def test_merge_cleans_word_text_before_stripping_it(tmp_path, monkeypatch):
    # The NUL stops strip() from reaching the space until the text is cleaned.
    words_by_lang = {"en": [Word("\x00 Cat\n", (0, 0, 40, 20), 90)], "german": []}
    out_hocr, _ = _run(tmp_path, monkeypatch, "merge", words_by_lang)
    assert _hocr_words(out_hocr) == ["Cat"]


def test_merge_compares_boxes_clamped_to_the_page(tmp_path, monkeypatch):
    # An estimated last word can run past the right edge of the 200 px page.
    # Clamped, the german box covers the en word and wins the NMS.
    words_by_lang = {
        "en": [Word("Cat", (150, 0, 200, 20), 90)],
        "german": [Word("Katze", (150, 0, 420, 20), 95)],
    }
    out_hocr, _ = _run(tmp_path, monkeypatch, "merge", words_by_lang)
    assert _hocr_words(out_hocr) == ["Katze"]


def test_merge_breaks_a_confidence_tie_in_reading_order(tmp_path, monkeypatch):
    # PaddleOCR lists regions in detection order. These two overlap past the
    # NMS threshold with equal confidence, so the word that reads first wins.
    pages = {
        "en": Page(
            width=200,
            height=100,
            lang="en",
            ocr_system="test",
            blocks=[_region("second", (2, 1, 52, 21)), _region("first", (0, 0, 50, 20))],
        ),
        "german": Page(width=200, height=100, lang="german", ocr_system="test"),
    }
    monkeypatch.setattr(
        classic, "build_page", lambda input_file, options: pages[options.languages[0]]
    )
    out_hocr, _ = _generate(tmp_path, "merge", ["en", "german"])
    assert _hocr_words(out_hocr) == ["first"]


def test_check_options_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="paddle_multi_lang_strategy"):
        ocrmypdf_plugin.check_options(_opts(paddle_multi_lang_strategy="vote"))


def _passes(tmp_path, monkeypatch, engine: str, **extra: Any) -> list[str]:
    calls: list[str] = []

    def fake(input_file, options) -> Page:
        calls.append(options.languages[0])
        return _merged_hocr([Word("x", (0, 0, 10, 10), 90)], options.languages[0])

    monkeypatch.setattr(classic, "build_page", fake)
    monkeypatch.setattr(vl, "build_page", fake)
    img = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(img)
    options = types.SimpleNamespace(
        **({"languages": ["en", "german"], "paddle_engine": engine} | extra)
    )
    PaddleOCREngine.generate_hocr(img, tmp_path / "o.hocr", tmp_path / "o.txt", options)
    return calls


def test_vl_remote_runs_a_single_pass_for_multiple_languages(tmp_path, monkeypatch):
    assert _passes(tmp_path, monkeypatch, "vl-remote") == ["en"]


def test_model_option_runs_a_single_pass_for_multiple_languages(tmp_path, monkeypatch):
    passes = _passes(tmp_path, monkeypatch, "classic-cpu", paddle_rec_model_dir="/models/rec")
    assert passes == ["en"]


def test_empty_model_option_keeps_one_pass_per_language(tmp_path, monkeypatch):
    passes = _passes(tmp_path, monkeypatch, "classic-cpu", paddle_rec_model_name="")
    assert passes == ["en", "german"]


def test_each_group_gets_one_pass_by_its_first_language_in_configured_order(tmp_path, monkeypatch):
    # The groups share the detection model, as the PP-OCRv5 languages do, and the first group's
    # pair sorts after the second's, so passes ordered by their models start with ru.
    models = {"en": ["z-det", "z-rec"], "german": ["z-det", "z-rec"], "ru": ["z-det", "a-rec"]}
    _models(monkeypatch, models)
    passes = _passes(tmp_path, monkeypatch, "classic-cpu", languages=["en", "german", "ru"])
    assert passes == ["en", "ru"]


def test_language_without_a_models_answer_keeps_its_own_pass(tmp_path, monkeypatch):
    models = {"en": ["z-det", "z-rec"], "german": ["z-det", "z-rec"], "ru": ["z-det", "a-rec"]}
    _models(monkeypatch, models)
    passes = _passes(tmp_path, monkeypatch, "classic-cpu", languages=["en", "fr", "german", "ru"])
    assert passes == ["en", "fr", "ru"]


def test_tesseract_codes_are_grouped_by_the_code_that_the_pass_sends(tmp_path, monkeypatch):
    _models(monkeypatch, {"en": ["z-det", "z-rec"], "german": ["z-det", "z-rec"]})
    assert _passes(tmp_path, monkeypatch, "classic-cpu", languages=["eng", "deu"]) == ["eng"]


def _spaced_page(lang: str) -> Page:
    """A page whose one line has two spaces, where a join of its words has one."""
    words = [Word("a", (0, 0, 20, 10), 90), Word("b", (40, 0, 60, 10), 90)]
    line = Line(box=(0, 0, 60, 10), confidence=90, text="a  b", words=words)
    block = Block(box=(0, 0, 60, 10), lines=[line])
    return Page(width=200, height=100, lang=lang, ocr_system="test", blocks=[block])


def test_single_language_keeps_the_engine_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(classic, "build_page", lambda input_file, options: _spaced_page("en"))
    _, out_text = _generate(tmp_path, "winner", ["en"])
    assert out_text.read_text(encoding="utf-8") == "a  b"


def test_winner_writes_the_line_text_of_the_winning_page(tmp_path, monkeypatch):
    pages = {
        "en": Page(width=200, height=100, lang="en", ocr_system="test"),
        "german": _spaced_page("german"),
    }
    monkeypatch.setattr(
        classic, "build_page", lambda input_file, options: pages[options.languages[0]]
    )
    _, out_text = _generate(tmp_path, "winner", ["en", "german"])
    assert out_text.read_text(encoding="utf-8") == "a  b"


def test_winner_orders_the_winning_page_as_a_single_pass_does(tmp_path, monkeypatch):
    # One row, read B C A. A and B share a bottom edge, so the rows depend on the order
    # the regions arrive in, and a page ordered twice reads B A C.
    pages = {
        "en": Page(
            width=200,
            height=100,
            lang="en",
            ocr_system="test",
            blocks=[
                _region("A", (100, 0, 150, 20)),
                _region("B", (0, 14, 40, 20)),
                _region("C", (50, 17, 90, 23)),
            ],
        ),
        "german": Page(width=200, height=100, lang="german", ocr_system="test"),
    }
    monkeypatch.setattr(
        classic, "build_page", lambda input_file, options: pages[options.languages[0]]
    )
    _, out_text = _generate(tmp_path, "winner", ["en", "german"])
    assert out_text.read_text(encoding="utf-8") == "B\nC\nA"


def test_merged_hocr_groups_words_into_lines(tmp_path):
    from lxml import html

    out = tmp_path / "m.hocr"
    words = [
        Word("hello", (0, 0, 50, 20), 90),
        Word("world", (60, 0, 110, 20), 90),
        Word("below", (0, 40, 50, 60), 90),
    ]
    write_document(_merged_hocr(words, "german"), out, tmp_path / "m.txt", sidecar="")
    tree = html.parse(str(out))
    lines = tree.findall('.//{*}span[@class="ocr_line"]')
    assert len(lines) == 2  # not one line per word
    first_line_words = ["".join(s.itertext()) for s in lines[0].findall("{*}span")]
    assert first_line_words == ["hello", "world"]
