"""The challenge container script and the library implement the same rules."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

import surgscope.policy as P
import surgscope.postprocess as PP
import surgscope.prompts as PR
import surgscope.routing as R
import surgscope.video as V

CH = Path(__file__).resolve().parents[1] / "challenge"
QUESTIONS = [
    ("When was the Sponge visible in the frame at 00:30:00 first inserted in the abdomen? hh:mm:ss", 3600),
    ("There is one Specimen in the frame at 00:23:22. When is it retrieved from the surgical site?", 1590),
    ("At what time point was the final and last retrieval of a Sponge in the video? hh:mm:ss", 7200),
    ("At what time point was the final and last retrieval of a Sponge in the video? hh:mm:ss", 1860),
    ("How many distinct foreign object instances are visible in this video?", 5000),
    ("Does the Needle, last visible just before 00:08:31, re-appear later in the video?", 790),
    ("In which quadrant of the frame is the center of the Clip that first occurs at 00:40:00?", 9000),
    ("What types of foreign objects are seen between 00:13:00 and 00:27:01?", 1640),
]


@pytest.fixture(scope="module")
def ch():
    env = dict(os.environ)
    sys.path.insert(0, str(CH))
    try:
        spec = importlib.util.spec_from_file_location("challenge_inference", CH / "inference.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(CH))
        os.environ.clear()
        os.environ.update(env)
    return mod


def test_rules_and_routing(ch):
    assert [(n, k) for n, k, _ in ch.RULES] == [(n, k) for n, k, _ in R.RULES]
    assert [(n, k) for n, k, _ in ch.TAIL_FAMILIES] == [(n, k) for n, k, _ in R.TAIL_FAMILIES]
    for q, end in QUESTIONS:
        assert ch.route(q, end) == R.route(q, end), q


def test_policy_and_prompt(ch):
    for q, end in QUESTIONS:
        assert ch.profile_for(q)["nframes"] == P.nframes(q)
        assert ch.profile_for(q)["budget"] == P.profile(q)["budget"]
        assert ch.user_content(q, 0, end, "Sigmoid Resection") == PR.user_content(q, 0, end, "Sigmoid Resection")
    assert ch.PREAMBLE == PR.PREAMBLE


def test_postprocess_and_sampling(ch):
    for a in ["00:31:33, 00:32:17", "Sponge at 00:12:00", "<think>x</think>00:01:02", "yes", "00:12:00 and 00:13:00."]:
        assert ch.normalize_time_answer(a) == PP.normalize_time_answer(a)
        assert ch.first_timestamp(a) == PP.first_timestamp(a)
    for total, n in [(10, 4), (4, 3), (18000, 1536), (1400, 1152)]:
        assert ch.sample_indices(total, n) == V.sample_indices(total, n)
    assert ch.ENC == V.ENC and ch.ZOOM_HALF_1 == PP.ZOOM_HALF_1 and ch.ZOOM_HALF_2 == PP.ZOOM_HALF_2
