"""Answer post-processing and the geometry of the coarse-to-fine temporal refinement."""
from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TS = re.compile(r"\b(\d{2}:\d{2}:\d{2})\b")

ZOOM_HALF_1 = 600.0      # pass 2: +-10 min around the pass-1 answer
ZOOM_HALF_2 = 120.0      # pass 3: +-2 min around the pass-2 answer
ANCHOR_SLACK = 60.0      # accept anchors up to 1 min outside [0, end]


def clean(text) -> str:
    """Strip any residual <think>...</think> span and surrounding whitespace."""
    return _THINK.sub("", text or "").strip()


def ts_seconds(s: str):
    h, m, sec = s.split(":")
    m, sec = int(m), int(sec)
    if not (0 <= m < 60 and 0 <= sec < 60):
        return None
    return int(h) * 3600 + m * 60 + sec


def first_timestamp(text):
    """Seconds of the first valid HH:MM:SS in the answer, or None."""
    m = _TS.search(clean(text))
    return ts_seconds(m.group(1)) if m else None


def normalize_time_answer(ans) -> str:
    """An answer made only of timestamps and separators collapses to its first HH:MM:SS.

    Free-text answers that merely mention a time are left unchanged.
    """
    a = clean(ans)
    if re.fullmatch(r"\d{2}:\d{2}:\d{2}", a):
        return a
    rest = re.sub(r"(?i)\band\b", "", _TS.sub("", a)).strip(" \t\n,;.&-")
    if rest:
        return a
    m = _TS.search(a)
    return m.group(1) if m else a


def format_guard(ans) -> str:
    """Final output cleaning applied to every answer."""
    a = clean(ans)
    if _TS.search(a):
        a = normalize_time_answer(a)
    return a


def zoom_anchor(question: str, answer, end_s: float):
    """Anchor time for refinement, or None if the answer is not an in-range timestamp.

    Duration questions ("how long ...") are answered in HH:MM:SS too but are not times.
    """
    if "how long" in question.lower():
        return None
    t = first_timestamp(answer)
    if t is None:
        return None
    if t < -ANCHOR_SLACK or t > float(end_s) + ANCHOR_SLACK:
        return None
    return min(max(float(t), 0.0), float(end_s))


def window_around(end_s: float, t: float, half: float):
    """[t - half, t + half] clipped to [0, end], shifted to keep its full length if possible."""
    z0, z1 = max(0.0, t - half), min(end_s, t + half)
    want = min(2 * half, end_s)
    if z1 - z0 < want:
        if z0 <= 0:
            z1 = want
        else:
            z0 = end_s - want
    return z0, z1
