import numpy as np

from surgscope.video import Video, sample_indices


def test_sample_indices_round_half_even():
    assert sample_indices(10, 4) == [0, 3, 6, 9]
    assert sample_indices(4, 3) == [0, 2, 3]          # 1.5 -> 2
    assert sample_indices(6, 3) == [0, 2, 5]          # 2.5 -> 2
    assert sample_indices(7, 1) == [0]
    idx = sample_indices(18000, 1536)
    assert len(idx) == 1536 and len(set(idx)) == 1536 and idx[-1] == 17999


def _video(n_frames, key_every, dead_ranges=()):
    v = Video.__new__(Video)
    v.n_frames = n_frames
    dead = np.zeros(n_frames, bool)
    for a, b in dead_ranges:
        dead[a:b] = True
    v.key_idx = np.arange(0, n_frames, key_every)
    v.key_dead = dead[v.key_idx]
    cover = np.clip(np.searchsorted(v.key_idx, np.arange(n_frames), side="right") - 1, 0, len(v.key_idx) - 1)
    v.dead = v.key_dead[cover]
    return v


def test_select_keyframes_when_dense_enough():
    v = _video(20000, 25)
    mode, sel, n = v.select(0, 20000, 768)
    assert mode == "key" and len(sel) == 768 and n == 800


def test_select_skips_dead_frames():
    v = _video(3000, 25, dead_ranges=[(1000, 2000)])
    mode, sel, n = v.select(0, 3000, 1536)
    assert mode == "seq" and n == 2000 and len(sel) == 1536
    assert not any(1000 <= s < 2000 for s in sel)
