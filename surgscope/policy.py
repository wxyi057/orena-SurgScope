"""Question-conditioned frame budget ("sampling regimes").

Each question gets roughly the same visual-token budget (~46k tokens), but it is spent
differently depending on what the question asks for, read from the question text only:

  dense  questions that ask for a time ("hh:mm:ss"), a count, or one of two aggregation
         templates -> many frames at low resolution (temporal density matters)
  mid    everything else -> fewer frames at higher resolution

The spatial resolution is baked into the pixels of each mini-clip (``pick_grid``), and the
frame count travels with every request as ``chat_template_kwargs = {min_frames, max_frames}``.
Three regimes were used to train the three fine-tunes that make up the released weights:

  r768         768 frames for every question, native mezzanine resolution (<=128 tokens/pair)
  dense-mix    dense 1536 frames @ 60 tokens/pair, mid 1152 frames @ 84 tokens/pair
  dense-mix-2  dense 2880 frames @ 32 tokens/pair, mid 1512 frames @ 64 tokens/pair

``dense-mix`` and ``dense-mix-2`` spend the same budget per branch: frames x tokens-per-pair
is 92,160 for dense and 96,768 for mid in both (i.e. ~46k / ~48k visual tokens, since two
consecutive frames share one token grid). Inference always uses ``dense-mix``.
"""
from __future__ import annotations

import math
import re

_HH = re.compile(r"hh:mm:ss", re.I)
_CNT = re.compile(r"how many|maximum number of|total count|how often", re.I)
# Aggregation class-list template: "After the [first] X was inserted/created ...,
# which other/different foreign object classes ..."
_AGGCLS = re.compile(r"^after the .{0,80}?(?:inserted|created)"
                     r".{0,200}?which (?:other|different) foreign object classes",
                     re.I | re.S)
# Longest-duration template: "... visible for the longest (not necessarily consecutive)
# total duration ..."
_LONGEST = re.compile(r"longest[^.?]{0,60}duration", re.I)

# nframes: frames per clip; budget: tokens per frame pair baked into the pixels
# (None = keep the mezzanine resolution and let the processor resize to <=128 tokens).
REGIMES = {
    "r768": {
        "dense": {"nframes": 768, "budget": None},
        "mid":   {"nframes": 768, "budget": None},
    },
    "dense-mix": {
        "dense": {"nframes": 1536, "budget": 60},
        "mid":   {"nframes": 1152, "budget": 84},
    },
    "dense-mix-2": {
        "dense": {"nframes": 2880, "budget": 32},
        "mid":   {"nframes": 1512, "budget": 64},
    },
}

# Processor settings that go with each regime (training and inference must agree).
#   max_length   longest sequence (visual tokens + ~7.5 timestamp tokens per pair + prompt)
#   vid_min_tok  must stay below the smallest baked grid, or the processor would upsample it
#   per_row_kwargs  frame counts are written into every training row (else set by env only)
REGIME_META = {
    "r768":        {"max_length": 57344, "fps_min_frames": 768, "fps_max_frames": 768,
                    "vid_max_tok": 128, "vid_min_tok": 64, "per_row_kwargs": False},
    "dense-mix":   {"max_length": 57344, "fps_min_frames": 1152, "fps_max_frames": 1536,
                    "vid_max_tok": 128, "vid_min_tok": 32, "per_row_kwargs": True},
    "dense-mix-2": {"max_length": 65536, "fps_min_frames": 1512, "fps_max_frames": 2880,
                    "vid_max_tok": 128, "vid_min_tok": 16, "per_row_kwargs": True},
}

INFERENCE_REGIME = "dense-mix"


def branch(question: str) -> str:
    """'dense' for time / counting / aggregation questions, 'mid' otherwise."""
    q = question or ""
    if _HH.search(q) or _CNT.search(q) or _AGGCLS.search(q) or _LONGEST.search(q):
        return "dense"
    return "mid"


def profile(question: str, regime: str = INFERENCE_REGIME) -> dict:
    """{'nframes', 'budget'} for one question under a regime."""
    return REGIMES[regime][branch(question)]


def nframes(question: str, regime: str = INFERENCE_REGIME) -> int:
    return profile(question, regime)["nframes"]


def frame_kwargs(question: str, regime: str = INFERENCE_REGIME) -> dict:
    """``chat_template_kwargs`` frame counts for a training row / inference request."""
    n = nframes(question, regime)
    return {"min_frames": n, "max_frames": n}


def clip_tree(question: str, regime: str) -> str:
    """Directory name of the mini-clip tree a question's clip lives in."""
    return f"frames{profile(question, regime)['nframes']}"   # frame counts are unique across regimes


def pick_grid(width: int, height: int, budget: int) -> tuple[int, int]:
    """32-px grid with the smallest aspect-ratio distortion within a token budget.

    Picks (w, h), multiples of 32, with (w/32)*(h/32) <= budget and >= 85 % of it, e.g.
    960x540 @60 -> 320x192, 720x576 @60 -> 256x224, 960x540 @84 -> 384x224,
    720x576 @84 -> 320x256. Such grids pass the processor's smart_resize unchanged.
    """
    ar = width / height
    best = None
    for nh in range(3, 33):
        nw = budget // nh
        if nw < 3:
            break
        tok = nw * nh
        if tok < 0.85 * budget:
            continue
        cand = (abs(math.log((nw / nh) / ar)), -tok, nw, nh)
        if best is None or cand < best:
            best = cand
    _, _, nw, nh = best
    return nw * 32, nh * 32
