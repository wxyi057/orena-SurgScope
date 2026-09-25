"""Per-frame appearance statistics used to mark dead frames (privacy mask / black).

Every frame is decoded at 32x18 px and summarised by 8 numbers:
mean R,G,B | spatial std R,G,B | blue-pixel fraction | black-pixel fraction.
A frame is dead if its blue fraction (B>150, R<80, G<80) or black fraction (max<20)
exceeds 0.9. The raw statistics are stored so thresholds can be revisited.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

W, H = 32, 18
FRAME_BYTES = W * H * 3
DEAD_BLUE_T = 0.90
DEAD_BLACK_T = 0.90


def probe(path) -> dict:
    """Frame rate (exact rational), size, duration and frame count via PyAV (no decode)."""
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        rate = s.base_rate or s.average_rate
        dur = float(c.duration / av.time_base) if c.duration else float(s.duration * s.time_base)
        return {"fps": float(rate), "width": s.codec_context.width,
                "height": s.codec_context.height, "duration_s": dur, "frames": int(s.frames or 0)}


def stats(arr: np.ndarray) -> np.ndarray:
    """arr: (n, H, W, 3) uint8 -> (n, 8) float32."""
    f = arr.astype(np.float32)
    flat = f.reshape(f.shape[0], -1, 3)
    mean = flat.mean(1)
    std = flat.std(1)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    blue = ((b > 150) & (r < 80) & (g < 80)).reshape(f.shape[0], -1).mean(1)
    black = (f.max(-1) < 20).reshape(f.shape[0], -1).mean(1)
    return np.concatenate([mean, std, blue[:, None], black[:, None]], axis=1)


def scan(ffmpeg: str, path, threads: int = 1):
    meta = probe(path)
    cmd = [ffmpeg, "-v", "error", "-threads", str(threads), "-i", str(path),
           "-vf", f"scale={W}:{H}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=FRAME_BYTES * 256)
    chunks, buf = [], b""
    while True:
        data = proc.stdout.read(FRAME_BYTES * 512)
        if not data:
            break
        buf += data
        n = len(buf) // FRAME_BYTES
        if n:
            arr = np.frombuffer(buf[:n * FRAME_BYTES], dtype=np.uint8).reshape(n, H, W, 3)
            chunks.append(stats(arr))
            buf = buf[n * FRAME_BYTES:]
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {err[:2000]}")
    if not chunks:
        raise RuntimeError("no frames decoded")
    return meta, np.concatenate(chunks, axis=0)


def scan_to_npz(ffmpeg: str, video, out, threads: int = 1) -> Path:
    """Scan one video and store the statistics (idempotent)."""
    out = Path(out)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    meta, s = scan(ffmpeg, video, threads)
    tmp = str(out) + ".tmp.npz"
    np.savez_compressed(tmp, stats=s.astype(np.float16), fps=meta["fps"],
                        duration_s=meta["duration_s"], width=meta["width"],
                        height=meta["height"], video=Path(video).name)
    os.replace(tmp, out)
    return out


def dead_mask(npz, n_frames: int) -> np.ndarray:
    """Per-frame dead mask on the 5 fps timeline of a mezzanine with ``n_frames`` frames.

    Statistics scanned at another frame rate (e.g. a 25/30 fps source) are mapped by
    nearest index; a few frames of metadata mismatch are padded as live.
    """
    z = np.load(npz)
    s = z["stats"].astype(np.float32)
    fps_nat = float(z["fps"])
    dead_nat = (s[:, 6] > DEAD_BLUE_T) | (s[:, 7] > DEAD_BLACK_T)
    if abs(fps_nat - 5.0) < 1e-3:
        dead = dead_nat
    else:
        idx = np.minimum(np.round(np.arange(n_frames) * fps_nat / 5.0).astype(np.int64),
                         len(dead_nat) - 1)
        dead = dead_nat[idx]
    if len(dead) < n_frames:
        dead = np.concatenate([dead, np.zeros(n_frames - len(dead), bool)])
    return dead[:n_frames]
