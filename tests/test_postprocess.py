from surgscope.postprocess import (clean, first_timestamp, format_guard, normalize_time_answer,
                                   window_around, zoom_anchor)


def test_clean_and_guard():
    assert clean("<think>\nhmm\n</think>\n\n00:12:03") == "00:12:03"
    assert format_guard("00:31:33, 00:32:17") == "00:31:33"
    assert format_guard("00:12:00 and 00:13:00.") == "00:12:00"
    assert format_guard("Sponge at 00:12:00") == "Sponge at 00:12:00"
    assert normalize_time_answer("yes") == "yes"


def test_first_timestamp():
    assert first_timestamp("00:24:41") == 1481
    assert first_timestamp("00:61:00") is None
    assert first_timestamp("none") is None


def test_zoom_anchor():
    q = "At what time was a Clip first visible in the video? Please answer in hh:mm:ss."
    assert zoom_anchor(q, "00:10:00", 1800) == 600.0
    assert zoom_anchor(q, "00:31:30", 1860) == 1860.0          # within the 60 s slack, clipped
    assert zoom_anchor(q, "01:00:00", 1800) is None
    assert zoom_anchor("For how long is a Sponge visible?", "00:10:00", 1800) is None


def test_window_around():
    assert window_around(7200, 3000, 600) == (2400, 3600)
    assert window_around(7200, 100, 600) == (0.0, 1200)          # shifted right at the start
    assert window_around(7200, 7150, 600) == (6000, 7200)        # shifted left at the end
    assert window_around(900, 450, 600) == (0.0, 900)            # prefix shorter than the window
