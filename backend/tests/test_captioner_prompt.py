"""The captioner prompt comes from an .md file, re-read each run."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import captioner  # noqa: E402
import prompt_template as pt  # noqa: E402


def test_missing_file_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("SF_CAPTIONER_PROMPT_FILE", str(tmp_path / "nope.md"))
    assert captioner.load_prompt_template() == captioner.PROMPT


def test_empty_file_falls_back(tmp_path, monkeypatch):
    f = tmp_path / "p.md"; f.write_text("  \n")
    monkeypatch.setenv("SF_CAPTIONER_PROMPT_FILE", str(f))
    assert captioner.load_prompt_template() == captioner.PROMPT


def test_file_is_reread_every_call(tmp_path, monkeypatch):
    f = tmp_path / "p.md"; f.write_text("first")
    monkeypatch.setenv("SF_CAPTIONER_PROMPT_FILE", str(f))
    assert captioner.load_prompt_template() == "first"
    f.write_text("second")
    assert captioner.load_prompt_template() == "second"


def test_text_tag_is_filled_per_segment(tmp_path, monkeypatch):
    f = tmp_path / "p.md"; f.write_text("Name: {{text_tag}}. Describe it.")
    monkeypatch.setenv("SF_CAPTIONER_PROMPT_FILE", str(f))
    t = captioner.load_prompt_template()
    assert pt.expand_prompt_template(t, text_tag="Mickey") == "Name: Mickey. Describe it."
    assert pt.expand_prompt_template(t, text_tag=None) == "Name: . Describe it."


def test_plain_default_prompt_is_unchanged_by_expansion():
    assert pt.expand_prompt_template(captioner.PROMPT, text_tag="x") == captioner.PROMPT
