import json

import pytest

from surgscope import prompts
from surgscope.prompts import PREAMBLE, load_fo_definitions, system_prompt, user_content


def test_user_content_exact():
    s = user_content("How many clips?", 0, 3725, "Sigmoid Resection")
    assert s == ("<video>Procedure type: Sigmoid Resection.\n"
                 "Clip window: 00:00:00 - 01:02:05 (source-video timeline).\nHow many clips?")
    assert user_content("Q?", 60, 120, None) == \
        "<video>Clip window: 00:01:00 - 00:02:00 (source-video timeline).\nQ?"


def test_system_prompt_runtime(tmp_path):
    f = tmp_path / "FO_definitions.json"
    f.write_text(json.dumps("Sponge: a surgical sponge.\n"))
    assert system_prompt(load_fo_definitions(f)) == PREAMBLE + "Sponge: a surgical sponge.\n"
    assert PREAMBLE.endswith("\n\n")


@pytest.mark.parametrize("content", ["{not json", json.dumps(["a list"]), json.dumps("")])
def test_malformed_definitions_fall_back(tmp_path, monkeypatch, content):
    f = tmp_path / "FO_definitions.json"
    f.write_text(content)
    monkeypatch.setattr(prompts, "default_fo_definitions", lambda: "FALLBACK")
    assert load_fo_definitions(f) == "FALLBACK"
    assert load_fo_definitions(tmp_path / "missing.json") == "FALLBACK"
