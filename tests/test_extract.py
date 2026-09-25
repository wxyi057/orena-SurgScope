from surgscope.data.extract import group_windows


def _it(qid, q, end, stem="v1", split="train"):
    return dict(ds="heico", split=split, qid=qid, stem=stem, end_s=end, question=q)


def test_group_windows_dedup_and_branch():
    q_time = "At what time point was the final and last retrieval of a Sponge in the video? hh:mm:ss"
    q_cls = "Which foreign object classes appear in this video?"
    items = [_it(2, q_time, 7200), _it(1, q_time, 7200), _it(3, q_cls, 7200), _it(4, q_time, 1800)]
    (ds, stem, wl), = group_windows(items, "dense-mix", branch="dense")
    assert (ds, stem) == ("heico", "v1")
    assert [(w["w0"], w["w1"], w["qids"]) for w in wl] == [(0.0, 1800.0, [4]), (3600.0, 7200.0, [1, 2])]
    (_, _, wl_mid), = group_windows(items, "dense-mix", branch="mid")
    assert [(w["w0"], w["w1"], w["qids"]) for w in wl_mid] == [(0.0, 7200.0, [3])]
