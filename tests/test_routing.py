from surgscope.routing import ABSTAIN, MIN_WIN, RULES, TAIL_FAMILIES, gt_in_window, route, train_label


def test_rule_tables():
    assert len(RULES) == 11 and len(TAIL_FAMILIES) == 8


def test_tier1_anchored_window():
    q = ("When was the Sponge visible in the frame at 00:30:00 first inserted in the abdomen? "
         "Please provide an answer in the format hh:mm:ss.")
    r = route(q, 3600)
    assert r == dict(tier=1, rule="insert_first", w0=600.0, w1=1860.0, shrunk=True)


def test_tier1_window_clipped_to_prefix():
    q = "There is one Specimen in the frame at 00:23:22. When is it retrieved from the surgical site?"
    r = route(q, 1590)
    assert (r["tier"], r["rule"], r["w0"], r["w1"], r["shrunk"]) == (1, "retrieve_when", 1342.0, 1590.0, True)


def test_short_window_grows_to_minimum():
    q = "At timepoint 00:01:00 please provide all relative central positions of foreign objects present."
    r = route(q, 3000)
    assert r["rule"] == "positions_at" and (r["w0"], r["w1"]) == (0.0, MIN_WIN)


def test_tier2_tail_window():
    q = "At what time point was the final and last retrieval of a Sponge in the video?"
    assert route(q, 7200) == dict(tier=2, rule="final_retrieval", w0=3600.0, w1=7200.0, shrunk=True)
    # a prefix shorter than the tail window is not shortened
    assert route(q, 1860) == dict(tier=2, rule="final_retrieval", w0=0.0, w1=1860.0, shrunk=False)


def test_tier3_whole_prefix():
    q = "How many distinct foreign object instances are visible in this video? Please provide a number."
    assert route(q, 5000) == dict(tier=3, rule=None, w0=0.0, w1=5000.0, shrunk=False)
    # a time in the question without a matching signature also falls back to the prefix
    assert route("Is anything visible at 00:10:00?", 5000)["tier"] == 3


def test_abstention_label():
    r = dict(tier=2, rule="first_visible", w0=3600.0, w1=7200.0, shrunk=True)
    assert train_label("00:20:00", "time", r) == (ABSTAIN, False)
    assert train_label("01:30:00", "time", r) == ("01:30:00", True)
    assert train_label("3", "number", r) == ("3", True)
    assert gt_in_window("none", "time", 0, 10)
