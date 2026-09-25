import pytest

from surgscope.policy import REGIME_META, REGIMES, branch, frame_kwargs, pick_grid


@pytest.mark.parametrize("w,h,b,expect", [
    (960, 540, 60, (320, 192)), (720, 576, 60, (256, 224)), (1024, 576, 60, (320, 192)),
    (960, 540, 84, (384, 224)), (720, 576, 84, (320, 256)),
    (960, 540, 32, (256, 128)), (720, 576, 32, (192, 160)), (1024, 576, 32, (256, 128)),
    (960, 540, 64, (320, 192)), (720, 576, 64, (288, 224)), (1024, 576, 64, (320, 192)),
])
def test_pick_grid_anchors(w, h, b, expect):
    assert pick_grid(w, h, b) == expect


def test_branch():
    assert branch("At what time was a Clip first visible? Please provide hh:mm:ss.") == "dense"
    assert branch("How many Needles are visible?") == "dense"
    assert branch("After the first Sponge was inserted in the abdomen in this video, which other "
                  "foreign object classes were inserted?") == "dense"
    assert branch("Which class is visible for the longest (not necessarily consecutive) total duration?") == "dense"
    assert branch("Which foreign object classes appear in this video?") == "mid"


def test_equal_nominal_budget_and_min_tokens():
    for br in ("dense", "mid"):
        a, b = REGIMES["dense-mix"][br], REGIMES["dense-mix-2"][br]
        assert a["nframes"] * a["budget"] == b["nframes"] * b["budget"]
    for reg, prof in REGIMES.items():
        grids = [pick_grid(w, h, p["budget"]) for p in prof.values() if p["budget"]
                 for w, h in ((960, 540), (720, 576), (1024, 576))]
        if grids:
            assert REGIME_META[reg]["vid_min_tok"] < min((gw // 32) * (gh // 32) for gw, gh in grids)


def test_frame_kwargs():
    assert frame_kwargs("How many clips?") == {"min_frames": 1536, "max_frames": 1536}
    assert frame_kwargs("Which classes appear?", "dense-mix-2") == {"min_frames": 1512, "max_frames": 1512}
