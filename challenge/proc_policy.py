"""Question-conditioned frame budget used by the challenge container.

Identical rules to surgscope/policy.py (only the regimes needed here):
  dense (time / counting / aggregation questions, matched on the question text)
        -> 1536 frames at 60 tokens per frame pair
  mid   (everything else) -> 1152 frames at 84 tokens per pair
The resolution is baked into the mini-clip pixels (pick_grid); VIDEO_MIN_TOKEN_NUM must
stay below the smallest baked grid.
"""
import math
import re

_HH = re.compile(r'hh:mm:ss', re.I)
_CNT = re.compile(r'how many|maximum number of|total count|how often', re.I)
# aggregation class-list template: "After the [first] X was inserted/created ...,
# which other/different foreign object classes ..."
_AGGCLS = re.compile(r'^after the .{0,80}?(?:inserted|created)'
                     r'.{0,200}?which (?:other|different) foreign object classes',
                     re.I | re.S)
# longest-duration template: "... visible for the longest (not necessarily
# consecutive) total duration ..."
_LONGEST = re.compile(r'longest[^.?]{0,60}duration', re.I)

POLICIES = {
    'dense-mix': {
        'dense': {'nframes': 1536, 'budget': 60, 'tree': 'frames1536'},
        'mid':   {'nframes': 1152, 'budget': 84, 'tree': 'frames1152'},
    },
    'dense-mix-2': {
        'dense': {'nframes': 2880, 'budget': 32, 'tree': 'frames2880'},
        'mid':   {'nframes': 1512, 'budget': 64, 'tree': 'frames1512'},
    },
}


def branch(question):
    q = question or ''
    if _HH.search(q) or _CNT.search(q) or _AGGCLS.search(q) or _LONGEST.search(q):
        return 'dense'
    return 'mid'


def pick_grid(width, height, budget):
    """32-px grid with the smallest aspect-ratio distortion within a token budget
    (>= 85 % of it): 960x540@60 -> 320x192, 720x576@60 -> 256x224,
    960x540@84 -> 384x224, 720x576@84 -> 320x256."""
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
