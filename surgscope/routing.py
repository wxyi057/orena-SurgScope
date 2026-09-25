"""Question-conditioned temporal windowing.

Every question is answered on the prefix [0, end] of a procedure video. Most questions,
however, only need a small part of it. The challenge questions are generated from a
fixed set of templates, and each template carries a short phrase signature that is
invariant across object classes and datasets. We match those signatures (in order, the
first match wins) and route each question to one of three tiers:

  tier 1  the question names an absolute time T and matches one of 11 anchored rules
          -> a rule-specific window around T
  tier 2  no time in the question, but the event of interest sits near the end of the
          video (8 "tail" families) -> the last 60 minutes, [end - 3600, end]
  tier 3  everything else (counting, class sets, durations, unseen wordings)
          -> the whole prefix [0, end]

Windows shorter than 4 minutes are grown around their centre. During training, a
time answer that falls outside a shortened window is replaced by the literal
``outside clip window``; at inference, that answer triggers a re-run on the whole prefix.

This module is the single source of truth for training-data generation, local
evaluation and the challenge container.
"""
from __future__ import annotations

import re

TS = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")
MIN_WIN = 240.0          # shortest window: 4 min (>= 1200 frames at 5 fps)
TAIL = 3600.0            # tier-2 tail-window length: 60 min
ABSTAIN = "outside clip window"   # training label / inference abstention literal

# (name, signature: all substrings must occur (case-insensitive), window(ts, end) -> (w0, w1))
RULES = [
    ("insert_first", ("first inserted", "visible in the frame at"), lambda ts, e: (ts[0] - 1200, ts[0] + 60)),
    ("create_first", ("first created", "visible in the frame at"), lambda ts, e: (ts[0] - 1200, ts[0] + 60)),
    ("retrieve_when", ("when is it retrieved",), lambda ts, e: (ts[0] - 60, ts[0] + 1800)),
    ("between", ("seen between",), lambda ts, e: (min(ts) - 30, max(ts) + 30)),
    ("inserted_at", ("inserted or created", " at "), lambda ts, e: (ts[0] - 180, ts[0] + 180)),
    ("positions_at", ("relative central positions",), lambda ts, e: (ts[0] - 120, ts[0] + 120)),
    ("quadrant_first", ("which quadrant", "first occurs"), lambda ts, e: (0.0, ts[0] + 60)),
    ("also_appear", ("also appear at",), lambda ts, e: (min(ts) - 120, max(ts) + 120)),
    ("reappear", ("last visible just before", "re-appear"), lambda ts, e: (ts[0] - 180, e)),
    ("retrieval_exists", ("does a retrieval of this object exist",), lambda ts, e: (ts[0] - 60, e)),
    ("count_at", ("how many", "at time point"), lambda ts, e: (0.0, ts[0] + 60)),
]

# Tier 2: no time in the question, event close to the end of the window.
# kind "safe":  the last occurrence inside the tail window is the global last one.
# kind "first": the first occurrence inside the tail window need not be the global first;
#               the abstention label teaches the model to answer "outside clip window".
TAIL_FAMILIES = [
    ("last_visible", ("last visible in the video",), "safe"),
    ("last_cooccur", ("last visible at the same time",), "safe"),
    ("final_retrieval", ("final and last retrieval",), "safe"),
    ("clips_list", ("at which time points were clips applied",), "safe"),
    ("first_visible", ("first visible in the video",), "first"),
    ("kth_insert", ("inserted in the abdomen for the first time",), "first"),
    ("first_cooccur", ("first visible at the same time",), "first"),
    ("leave_first", ("leave the surgical view for at least",), "first"),
]


def timestamps(question: str) -> list[int]:
    """All H:MM:SS / HH:MM:SS times in the text, in seconds."""
    return [int(h) * 3600 + int(m) * 60 + int(s) for h, m, s in TS.findall(question)]


def classify(question: str):
    """Tier 1: (rule name, timestamps), or (None, timestamps) if no rule matches."""
    ts = timestamps(question)
    if not ts:
        return None, ts
    q = question.lower()
    for name, keys, _ in RULES:
        if all(k in q for k in keys):
            return name, ts
    return None, ts


def tail_family(question: str):
    """Tier 2: family name, or None (questions that contain a time never qualify)."""
    if TS.search(question):
        return None
    q = question.lower()
    for name, keys, _ in TAIL_FAMILIES:
        if all(k in q for k in keys):
            return name
    return None


def _clamp(w0, w1, end_s):
    w0, w1 = max(0.0, w0), min(float(end_s), w1)
    if w1 - w0 < MIN_WIN:                       # too short: grow around the centre, then snap to the edges
        c = (w0 + w1) / 2
        w0, w1 = c - MIN_WIN / 2, c + MIN_WIN / 2
        if w0 < 0:
            w0, w1 = 0.0, min(float(end_s), MIN_WIN)
        if w1 > end_s:
            w1, w0 = float(end_s), max(0.0, end_s - MIN_WIN)
    return float(int(w0)), float(int(w1))


def window(question: str, end_s: float):
    """Tier-1 window only: (w0, w1, rule name) or None."""
    name, ts = classify(question)
    if name is None:
        return None
    fn = next(r[2] for r in RULES if r[0] == name)
    w0, w1 = _clamp(*fn(ts, float(end_s)), float(end_s))
    return w0, w1, name


def route(question: str, end_s: float) -> dict:
    """Three-tier routing -> dict(tier, rule, w0, w1, shrunk).

    ``shrunk`` is True when the window differs from the whole prefix [0, end].
    """
    end_s = float(end_s)
    w = window(question, end_s)
    if w is not None:
        w0, w1, name = w
        return dict(tier=1, rule=name, w0=w0, w1=w1, shrunk=(w0 > 0 or w1 < end_s))
    fam = tail_family(question)
    if fam is not None:
        w0, w1 = _clamp(end_s - TAIL, end_s, end_s)
        return dict(tier=2, rule=fam, w0=w0, w1=w1, shrunk=(w0 > 0))
    return dict(tier=3, rule=None, w0=0.0, w1=end_s, shrunk=False)


def gt_in_window(answer, answer_format, w0, w1) -> bool:
    """True unless a time answer has a timestamp outside [w0, w1] (inclusive)."""
    if answer_format != "time":
        return True
    ts = timestamps(str(answer))
    if not ts:
        return True
    return all(w0 <= t <= w1 for t in ts)


def train_label(answer, answer_format, r: dict):
    """Training target for a routed question: (label, in_window).

    Tier 3 or in-window -> the original answer; otherwise the abstention literal.
    """
    if r["tier"] == 3:
        return answer, True
    ok = gt_in_window(answer, answer_format, r["w0"], r["w1"])
    return (answer if ok else ABSTAIN), ok


def is_abstain(text) -> bool:
    return ABSTAIN in str(text).lower()
