"""Inference-side video handling: dead-frame masking, frame selection, mini-clip encoding.

Input clips follow the challenge layout: H.264, exactly 5 fps, height <= 576 px, a keyframe
every 5 s, and an HH:MM:SS clock burned into every frame (the ``overlayed`` variant).

Privacy-masked stretches (pure-blue frames) and black frames carry no information; about
15 % of all source frames are such "dead" frames. A keyframe-only decode at 32x18 px marks
them, and every frame inherits the status of its keyframe. The frames of a window are then
sampled uniformly from the remaining frames and re-encoded as a mini-clip whose frame rate
is N / window length, so the processor's per-frame-pair time stamps stay a linear map of
the source timeline; exact times are read from the burned-in clock.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from surgscope.policy import pick_grid

log = logging.getLogger("surgscope")

CLIP_FPS = 5.0           # challenge clips are exactly 5 fps
DEAD_BLUE_T = 0.90       # fraction of blue thumbnail pixels (B>150, R<80, G<80)
DEAD_BLACK_T = 0.90      # fraction of black thumbnail pixels (max channel < 20)

# x264 settings of the mini-clips (identical in training and inference).
# -fps_mode passthrough keeps every frame under a fractional -framerate.
ENC = ["-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast",
       "-crf", "23", "-g", "768", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
       "-an", "-movflags", "+faststart"]


def sample_indices(total: int, n: int) -> list[int]:
    """Replica of ``torch.linspace(0, total-1, n).round()`` (round half to even)."""
    if n == 1:
        return [0]
    step = (total - 1) / (n - 1)
    out = []
    for i in range(n):
        v = i * step
        r = int(v)
        frac = v - r
        if frac > 0.5 or (frac == 0.5 and r % 2 == 1):
            r += 1
        out.append(min(r, total - 1))
    return out


class Video:
    """One input clip: header info + keyframe dead-frame mask (computed once)."""

    def __init__(self, ffmpeg, path, end_hint):
        self.ffmpeg = ffmpeg
        self.path = str(path)
        self.duration = self._header_duration()
        if self.duration is None or self.duration <= 0:
            self.duration = float(end_hint)
        self.n_frames = max(1, int(round(self.duration * CLIP_FPS)))
        self.width = self.height = None
        self.key_idx, self.key_dead = self._scan_keyframes()
        # frame-level dead mask: frame f inherits its covering keyframe's status
        dead = np.zeros(self.n_frames, dtype=bool)
        if len(self.key_idx):
            cover = np.searchsorted(self.key_idx, np.arange(self.n_frames), side="right") - 1
            cover = np.clip(cover, 0, len(self.key_idx) - 1)
            dead = self.key_dead[cover]
        self.dead = dead

    def _header_duration(self):
        p = subprocess.run([self.ffmpeg, "-nostdin", "-i", self.path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", p.stderr or "")
        if not m:
            return None
        return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])

    def _scan_keyframes(self):
        """Decode keyframes only: 32x18 thumbnails + their pts (via showinfo)."""
        # -fps_mode passthrough is essential: without it ffmpeg re-times the output to the
        # source frame rate and duplicates every keyframe, breaking the ordinal mapping.
        cmd = [self.ffmpeg, "-nostdin", "-skip_frame", "nokey", "-i", self.path,
               "-vf", "scale=32:18,showinfo", "-fps_mode", "passthrough",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw = np.frombuffer(p.stdout, dtype=np.uint8)
        k = len(raw) // (32 * 18 * 3)
        pts = [float(x) for x in re.findall(r"pts_time:\s*([0-9.]+)", p.stderr.decode("utf-8", "ignore"))]
        wh = re.search(r"Stream #0:0.*?, (\d{2,5})x(\d{2,5})", p.stderr.decode("utf-8", "ignore"))
        if wh:
            self.width, self.height = int(wh[1]), int(wh[2])
        if k == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
        if len(pts) != k:
            log.warning("keyframe scan: %d raw frames vs %d showinfo pts (%s)", k, len(pts), Path(self.path).name)
        arr = raw[: k * 32 * 18 * 3].reshape(k, 18, 32, 3).astype(np.float32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        blue = ((b > 150) & (r < 80) & (g < 80)).reshape(k, -1).mean(1)
        black = (arr.max(-1) < 20).reshape(k, -1).mean(1)
        dead = (blue > DEAD_BLUE_T) | (black > DEAD_BLACK_T)
        if len(pts) >= k:
            idx = np.round(np.array(pts[:k]) * CLIP_FPS).astype(np.int64)
        else:                                  # showinfo missing: assume a 5 s GOP
            idx = np.arange(k, dtype=np.int64) * int(round(5 * CLIP_FPS))
        return idx, dead

    # ---- frame selection ------------------------------------------------
    def select(self, i0, i1, nframes):
        """(mode, indices, n_candidates) for frames [i0, i1).

        mode 'key': enough live keyframes -> sample among keyframes (keyframe ordinals);
        mode 'seq': sample among all live frames (frame indices).
        """
        i0, i1 = max(0, i0), min(self.n_frames, i1)
        kmask = (self.key_idx >= i0) & (self.key_idx < i1) & (~self.key_dead)
        kords = np.flatnonzero(kmask)
        if kords.size >= nframes:
            sel = kords[sample_indices(kords.size, nframes)]
            return "key", sel.tolist(), int(kords.size)
        cand = np.flatnonzero(~self.dead[i0:i1]) + i0
        if cand.size == 0:
            cand = np.arange(i0, i1)
        sel = cand[sample_indices(cand.size, nframes)] if cand.size > nframes else cand
        return "seq", sel.tolist(), int(cand.size)

    # ---- mini-clip encoding ----------------------------------------------
    def make_clip(self, dst, i0, i1, span_s, nframes, budget):
        """Encode ``nframes`` live frames of [i0, i1) as a mini-clip at fps = n / span_s.

        The resolution is baked in with ``pick_grid(width, height, budget)``
        (``budget=None`` keeps the source resolution).
        """
        dst = Path(dst)
        mode, sel, n_cand = self.select(i0, i1, nframes)
        if not sel:
            raise RuntimeError("no frames selected")
        if mode == "key":
            last_ord = sel[-1]
            dec_cmd = [self.ffmpeg, "-nostdin", "-v", "error", "-skip_frame", "nokey",
                       "-i", self.path, "-fps_mode", "passthrough", "-frames:v", str(last_ord + 1),
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            first_read = 0
        else:
            first_read = sel[0]
            dec_cmd = [self.ffmpeg, "-nostdin", "-v", "error",
                       "-ss", f"{first_read / CLIP_FPS:.3f}", "-i", self.path,
                       "-fps_mode", "passthrough", "-frames:v", str(sel[-1] - first_read + 1),
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        keep = set(sel)
        if not self.width:
            raise RuntimeError("unknown frame size")
        scale = []
        if budget is not None:
            tw, th = pick_grid(self.width, self.height, budget)
            scale = ["-vf", f"scale={tw}:{th}:flags=bicubic"]
        fb = self.width * self.height * 3
        fps_out = f"{len(sel)}/{max(span_s, 1e-3):.3f}"
        tmp = str(dst) + ".tmp.mp4"
        dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        enc = subprocess.Popen(
            [self.ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.width}x{self.height}", "-framerate", fps_out, "-i", "-",
             *scale, *ENC, tmp],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        got = 0
        k = first_read
        want = len(sel)
        while got < want:
            buf = dec.stdout.read(fb)
            if len(buf) < fb:
                break
            if k in keep:
                enc.stdin.write(buf)
                got += 1
            k += 1
        dec.stdout.close()
        dec.terminate()
        dec.wait()
        enc.stdin.close()
        enc.wait()
        if enc.returncode != 0 or got == 0:
            raise RuntimeError(f"encode failed rc={enc.returncode} got={got}/{want}")
        if got < want:
            log.warning("clip %s: got %d/%d frames (decoder ended early)", dst.name, got, want)
        os.replace(tmp, dst)
        return dict(mode=mode, n=got, n_cand=n_cand)
