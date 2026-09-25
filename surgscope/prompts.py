"""Prompt construction (identical in training and inference).

system  a fixed preamble followed by the object-class definitions, read at run time from
        the ``FO_definitions.json`` that ships with the input data, so that new
        (out-of-distribution) object classes enter the prompt without retraining
user    "<video>" + optional "Procedure type: ..." line + the window that is actually shown,
        on the source-video timeline, + the question text unchanged
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("surgscope")

PREAMBLE = (
    "You are a surgical assistant. You are given endoscopic video from a "
    "minimally invasive procedure. Analyze the footage and answer the surgical "
    "question based on the visual evidence. Be precise and concise.\n\n"
)


def hms(t) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def window_hint(start_s, end_s) -> str:
    return f"Clip window: {hms(start_s)} - {hms(end_s)} (source-video timeline).\n"


def user_content(question: str, start_s=None, end_s=None, procedure_type=None,
                 with_hint: bool = True) -> str:
    """User turn: ``<video>`` expands to the frames of the clip for [start_s, end_s]."""
    pt = f"Procedure type: {procedure_type}.\n" if procedure_type else ""
    hint = window_hint(start_s, end_s) if (with_hint and start_s is not None) else ""
    return "<video>" + pt + hint + question


def default_fo_definitions() -> str:
    """Definitions bundled with the official ``orena-focus`` package (fallback only)."""
    try:
        from importlib.resources import files
        return files("focus").joinpath("assets/FO_definitions.txt").read_text()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "No FO_definitions.json next to the input and the `orena-focus` package is not "
            "installed; pass --fo-definitions or `pip install orena-focus`.") from e


def load_fo_definitions(path=None) -> str:
    """Definitions text from ``FO_definitions.json`` (a JSON-encoded string).

    Falls back to the ``orena-focus`` definitions only if the file is missing or malformed.
    """
    if path is not None:
        p = Path(path)
        try:
            text = json.loads(p.read_text()) if p.suffix == ".json" else p.read_text()
            if isinstance(text, str) and text.strip():
                return text
            log.warning("%s does not hold a non-empty string; using the orena-focus definitions", p)
        except FileNotFoundError:
            log.warning("%s not found; using the orena-focus definitions", p)
        except (OSError, ValueError) as e:
            log.warning("%s unreadable (%s); using the orena-focus definitions", p, e)
    return default_fo_definitions()


def system_prompt(fo_definitions: str) -> str:
    return PREAMBLE + fo_definitions
