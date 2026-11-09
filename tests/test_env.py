"""Environment readers: a blank value counts as unset, and a value the reader
cannot use falls back to the default with a warning.
"""

from __future__ import annotations

import logging

from paperless_paddleocr.env import env_choice, env_int


def test_env_choice_returns_default_and_warns_on_unknown_value(monkeypatch, caplog):
    monkeypatch.setenv("X_CHOICE", "bogus")
    with caplog.at_level(logging.WARNING):
        assert env_choice("X_CHOICE", ("a", "b"), "a") == "a"
    assert "bogus" in caplog.text


def test_env_choice_treats_blank_as_unset(monkeypatch, caplog):
    monkeypatch.setenv("X_CHOICE", "  ")
    with caplog.at_level(logging.WARNING):
        assert env_choice("X_CHOICE", ("a", "b"), "b") == "b"
    assert caplog.text == ""


def test_env_choice_ignores_case_and_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("X_CHOICE", " MERGE ")
    assert env_choice("X_CHOICE", ("merge", "winner"), "winner") == "merge"


def test_env_int_returns_default_and_warns_on_non_integer(monkeypatch, caplog):
    monkeypatch.setenv("X_INT", "ten")
    with caplog.at_level(logging.WARNING):
        assert env_int("X_INT", 15) == 15
    assert "ten" in caplog.text


def test_env_int_treats_blank_as_unset(monkeypatch, caplog):
    monkeypatch.setenv("X_INT", "  ")
    with caplog.at_level(logging.WARNING):
        assert env_int("X_INT", 15) == 15
    assert caplog.text == ""


def test_env_int_reads_an_integer(monkeypatch):
    monkeypatch.setenv("X_INT", "20")
    assert env_int("X_INT", 15) == 20
