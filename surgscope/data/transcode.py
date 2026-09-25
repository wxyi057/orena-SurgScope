"""Source video -> challenge-format mezzanines (5 fps, height <= 576, H.264, GOP 5 s).

Two variants per video, as in the challenge input:
  plain      the video itself
  overlayed  the same frames with the source-timeline clock burned in (top left,
             DejaVu Sans 40 px, white). The clock is drawn after the 5 fps
             normalisation, so frame n shows floor(n / 5) seconds.

Two build paths, matching how the training data was produced:
  two-pass  (HeiCo)     source -> plain mezzanine -> clock burned onto the plain mezzanine
  one-pass  (LapChole)  source decoded once, split into plain and overlayed encodes
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

MEZZ_VF = "fps=5,scale=-2:'min(ih,576)'"
MEZZ_ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-g", "25", "-keyint_min", "25", "-sc_threshold", "0",
            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart"]

DATASET_MODE = {"heico": "two-pass", "lapchole": "one-pass"}


def overlay_text(font: str, start_s: float = 0) -> str:
    """drawtext filter that prints floor(start_s + n / 5) as HH:MM:SS on frame n."""
    if re.search(r"[:'\\,;\[\]= ]", font):
        raise ValueError(f"font path {font!r} contains filtergraph special characters; "
                         "set SURGSCOPE_FONT to a plain path")
    t = f"(n/5+{start_s})"
    return (f"drawtext=fontfile={font}:fontcolor=white:fontsize=40:x=20:y=20:"
            f"text='%{{eif\\:trunc({t}/3600)\\:d\\:2}}\\:"
            f"%{{eif\\:trunc(mod({t}/60,60))\\:d\\:2}}\\:"
            f"%{{eif\\:trunc(mod({t},60))\\:d\\:2}}'")


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={r.returncode}): {r.stderr[-800:]}")


def make_mezzanines(ffmpeg: str, font: str, src, plain_dst, overlay_dst,
                    mode: str = "two-pass", threads: int = 4) -> None:
    """Write the plain and overlayed mezzanines of one source video (atomic, idempotent)."""
    plain_dst, overlay_dst = Path(plain_dst), Path(overlay_dst)
    done = lambda p: p.exists() and p.stat().st_size > 0  # noqa: E731
    if done(plain_dst) and done(overlay_dst):
        return
    plain_dst.parent.mkdir(parents=True, exist_ok=True)
    overlay_dst.parent.mkdir(parents=True, exist_ok=True)
    base = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", str(threads)]
    tmp_p, tmp_o = f"{plain_dst}.part.mp4", f"{overlay_dst}.part.mp4"
    try:
        if mode == "two-pass":
            if not done(plain_dst):
                _run([*base, "-i", str(src), "-vf", MEZZ_VF, *MEZZ_ENC, tmp_p])
                os.replace(tmp_p, plain_dst)
            _run([*base, "-i", str(plain_dst), "-vf", overlay_text(font), *MEZZ_ENC, tmp_o])
            os.replace(tmp_o, overlay_dst)
        elif mode == "one-pass":
            fc = f"[0:v]{MEZZ_VF},split=2[p][o];[o]{overlay_text(font)}[ov]"
            _run([*base, "-i", str(src), "-filter_complex", fc,
                  "-map", "[p]", *MEZZ_ENC, tmp_p, "-map", "[ov]", *MEZZ_ENC, tmp_o])
            os.replace(tmp_p, plain_dst)
            os.replace(tmp_o, overlay_dst)
        else:
            raise ValueError(f"unknown mode {mode!r}")
    finally:
        for t in (tmp_p, tmp_o):
            if os.path.exists(t):
                os.unlink(t)


def cut_prefix(ffmpeg: str, mezz, dst, end_s: float) -> None:
    """Challenge-layout clip: the prefix [0, end_s] of a mezzanine, stream-copied."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    _run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(mezz), "-t", str(end_s),
          "-c", "copy", "-an", "-movflags", "+faststart", str(dst)])
