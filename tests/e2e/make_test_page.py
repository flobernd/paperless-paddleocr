"""A test page for the runs that send a document through the real engine.

The page is an A4 scan at 200 dpi with ten lines of Latin text. One line
holds the word ``Paperless``, which a run looks for in the recognized text.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DPI = 200
_SIZE = (1654, 2339)
_MARGIN = 150
_LINE_HEIGHT = 80
_LINES = (
    "Lorem ipsum dolor sit amet, consectetur",
    "adipiscing elit, sed do eiusmod tempor",
    "incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis nostrud",
    "exercitation ullamco laboris nisi ut aliquip",
    "Paperless servat omnia documenta.",
    "Duis aute irure dolor in reprehenderit",
    "in voluptate velit esse cillum dolore",
    "eu fugiat nulla pariatur. Excepteur sint",
    "occaecat cupidatat non proident.",
)


def write_page(path: Path) -> None:
    image = Image.new("RGB", _SIZE, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    for index, line in enumerate(_LINES):
        draw.text((_MARGIN, _MARGIN + index * _LINE_HEIGHT), line, fill="black", font=font)
    image.save(path, dpi=(DPI, DPI))
