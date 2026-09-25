"""Locate and check the ffmpeg binary and the clock-overlay font."""
from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from pathlib import Path

FONT = Path(__file__).parent / "resources" / "fonts" / "DejaVuSans.ttf"


def find_ffmpeg(explicit=None, drawtext: bool = False) -> str:
    """ffmpeg to use: argument > $SURGSCOPE_FFMPEG > imageio-ffmpeg (pinned) > PATH.

    The pinned imageio-ffmpeg build has no ``drawtext`` filter; when the clock has to be
    burned in (``drawtext=True``), an ffmpeg on PATH that has it is preferred.
    """
    if explicit:
        return str(explicit)
    env = os.environ.get("SURGSCOPE_FFMPEG")
    if env:
        return env
    cands = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        cands.append(get_ffmpeg_exe())
    except Exception:  # noqa: BLE001
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        cands.append(exe)
    if not cands:
        raise RuntimeError("ffmpeg not found: `pip install imageio-ffmpeg` or set SURGSCOPE_FFMPEG")
    if drawtext:
        for c in cands:
            if capabilities(c)["drawtext"]:
                return c
    return cands[0]


def find_font(explicit=None) -> str:
    """TrueType font for the burned-in clock (DejaVu Sans, bundled)."""
    return str(explicit or os.environ.get("SURGSCOPE_FONT") or FONT)


def _run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@functools.lru_cache(maxsize=None)
def capabilities(ffmpeg: str) -> dict:
    """Version string and the features SurgScope relies on."""
    ver = _run([ffmpeg, "-hide_banner", "-version"]).stdout.splitlines()
    fps_mode = _run([ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                     "color=size=32x32:rate=5", "-frames:v", "2", "-fps_mode", "passthrough",
                     "-f", "null", "-"]).returncode == 0
    encoders = _run([ffmpeg, "-hide_banner", "-encoders"]).stdout
    filters = _run([ffmpeg, "-hide_banner", "-filters"]).stdout
    return {
        "version": ver[0] if ver else "unknown",
        "fps_mode": fps_mode,                              # ffmpeg >= 5.1
        "libx264": re.search(r"\blibx264\b", encoders) is not None,
        "drawtext": re.search(r"\bdrawtext\b", filters) is not None,
    }


def check(ffmpeg: str, drawtext: bool = False) -> dict:
    """Raise with a clear message if ffmpeg lacks a required feature."""
    cap = capabilities(ffmpeg)
    missing = [k for k in ("fps_mode", "libx264") if not cap[k]]
    if drawtext and not cap["drawtext"]:
        missing.append("drawtext (libfreetype)")
    if missing:
        raise RuntimeError(f"{ffmpeg} ({cap['version']}) lacks: {', '.join(missing)}. "
                           "SurgScope needs ffmpeg >= 5.1 with libx264"
                           + (" and libfreetype" if drawtext else "") + ".")
    return cap


def header_info(ffmpeg: str, path) -> dict:
    """Duration (s), width and height from the container header, without decoding."""
    p = subprocess.run([ffmpeg, "-nostdin", "-i", str(path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    err = p.stderr or ""
    out = {"duration": None, "width": None, "height": None}
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", err)
    if m:
        out["duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    m = re.search(r"Stream #0:\d+.*?Video:.*?, (\d{2,5})x(\d{2,5})", err)
    if m:
        out["width"], out["height"] = int(m[1]), int(m[2])
    return out
