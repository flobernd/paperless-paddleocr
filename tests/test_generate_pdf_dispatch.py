"""Regression tests for ``generate_pdf`` (Finding 13 + the ocrmypdf>=17 rewrite).

Two contracts are pinned here:

* **Polymorphic dispatch (F13):** ocrmypdf calls ``generate_pdf`` on the
  *instance* returned by ``get_ocr_engine()`` - a ``MultiLangPaddleEngine``.
  ``generate_pdf`` must be a classmethod so ``cls.generate_hocr`` reaches
  the subclass and the multi-language merge is not silently bypassed.

* **ocrmypdf>=17 rendering:** ``generate_pdf`` parses the hOCR with
  ``HocrParser`` and renders a real text-only PDF via ``Fpdf2PdfRenderer``
  (ocrmypdf 17 removed the old ``ocrmypdf.hocrtransform.HocrTransform``).
  The renderer is *not* stubbed, so this test fails loudly if the ocrmypdf
  hOCR→PDF surface drifts again.
"""

from __future__ import annotations

import types

from PIL import Image

from paperless_paddleocr.ocrmypdf_plugin import (
    MultiLangPaddleEngine,
    _Word,
    _write_merged_hocr,
)
from paperless_paddleocr.paddle_engine import PaddleOCREngine


def _tiny_png(path):
    Image.new("RGB", (120, 40), "white").save(path, dpi=(300, 300))


def test_generate_pdf_dispatches_and_renders_textonly_pdf(tmp_path, monkeypatch):
    img = tmp_path / "page.png"
    _tiny_png(img)
    out_pdf = tmp_path / "out.pdf"
    out_text = tmp_path / "out.txt"

    calls: list[str] = []

    def fake_multilang_generate_hocr(input_file, output_hocr, output_text, options):
        # Proves dispatch reached the subclass, and produces a *real* hOCR
        # in the exact format we ship so HocrParser/Fpdf2PdfRenderer run for
        # real (not stubbed).
        calls.append("multilang")
        words = [
            _Word("hallo", 4, 4, 54, 30, 95),
            _Word("welt", 60, 4, 110, 30, 93),
        ]
        _write_merged_hocr(output_hocr, words, 120, 40, hocr_lang="german")
        output_text.write_text("hallo welt", encoding="utf-8")

    def fake_base_classic(*a, **k):
        calls.append("base_classic")

    def fake_base_vl(*a, **k):
        calls.append("base_vl")

    monkeypatch.setattr(
        MultiLangPaddleEngine,
        "generate_hocr",
        staticmethod(fake_multilang_generate_hocr),
    )
    monkeypatch.setattr(PaddleOCREngine, "_generate_hocr_classic", staticmethod(fake_base_classic))
    monkeypatch.setattr(PaddleOCREngine, "_generate_hocr_vl", staticmethod(fake_base_vl))

    options = types.SimpleNamespace(languages=["en", "german"], paddle_engine="classic-cpu")

    # Call exactly as ocrmypdf's textonly-pdf pipeline does: on the instance.
    engine = MultiLangPaddleEngine()
    engine.generate_pdf(img, out_pdf, out_text, options)

    assert calls == ["multilang"], (
        f"generate_pdf must dispatch only to MultiLangPaddleEngine.generate_hocr, got {calls!r}"
    )

    data = out_pdf.read_bytes()
    assert data.startswith(b"%PDF"), "generate_pdf must emit a real PDF"
    assert data.rstrip().endswith(b"%%EOF"), "PDF must be well-formed"
    assert out_text.read_text(encoding="utf-8") == "hallo welt"


def test_generate_pdf_is_classmethod_for_subclass_dispatch():
    # A @staticmethod here would hardcode the base class and bypass the
    # multi-language merge.
    assert isinstance(PaddleOCREngine.__dict__["generate_pdf"], classmethod), (
        "generate_pdf must be a classmethod"
    )
