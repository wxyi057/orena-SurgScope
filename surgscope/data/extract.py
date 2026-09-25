"""Training mini-clips: one per unique (video, split, window) and regime.

For every question the routed window (``routing.route``) is cut from the overlayed
mezzanine: dead frames are removed, N frames are sampled uniformly (round half to even,
the same rule the processor uses) and re-encoded at fps = N / window length, with the
spatial resolution of the regime baked in. Each video is decoded once per batch of
windows that share a start time, feeding several encoders in parallel; questions with the
same window share one clip (symlinks). A JSON sidecar records the selected frame indices.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from surgscope import policy
from surgscope.data.scan import dead_mask, probe
from surgscope.routing import route
from surgscope.video import ENC, sample_indices


def hms_to_s(s) -> int:
    h, m, sec = str(s).split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def packet_count(path) -> int:
    """Number of video packets (= frames for our encodes), without decoding."""
    import av
    with av.open(str(path)) as c:
        return sum(1 for p in c.demux(video=0) if p.size)


def group_windows(items, regime: str, branch=None):
    """items: dicts with ds, split, qid, stem, end_s, question.

    Returns [(ds, stem, [window, ...])], a window being
    dict(ds, split, end_s, w0, w1, tier, rule, qids). ``branch`` keeps only questions of
    one regime branch ('dense' or 'mid'), since the two branches use different clip trees.
    """
    vids = {}
    for it in items:
        if branch and policy.branch(it["question"]) != branch:
            continue
        r = route(it["question"], it["end_s"])
        w0, w1 = (r["w0"], r["w1"]) if r["shrunk"] else (0.0, float(it["end_s"]))
        key = (it["split"], w0, w1, it["end_s"])
        wins = vids.setdefault((it["ds"], it["stem"]), {})
        w = wins.setdefault(key, dict(ds=it["ds"], split=it["split"], end_s=it["end_s"], w0=w0, w1=w1,
                                      tier=r["tier"] if r["shrunk"] else None,
                                      rule=r["rule"] if r["shrunk"] else None, qids=[]))
        w["qids"].append(it["qid"])
    out = []
    for (ds, stem), wins in vids.items():
        wl = [wins[k] for k in sorted(wins)]
        for w in wl:
            w["qids"] = sorted(w["qids"], key=lambda q: (len(str(q)), str(q)))
        out.append((ds, stem, wl))
    return out


class ClipExtractor:
    """Writes ``<clips_root>/<tree>/<ds>/<split>/<qid>.mp4`` (+ ``.json`` sidecar)."""

    def __init__(self, ffmpeg: str, clips_root, regime: str, branch: str, enc_batch: int = 40,
                 max_frames: int | None = None):
        prof = policy.REGIMES[regime][branch]
        self.ffmpeg, self.regime, self.branch = ffmpeg, regime, branch
        self.nframes, self.budget = prof["nframes"], prof["budget"]
        if max_frames:                       # smoke tests on small GPUs only
            self.nframes = min(self.nframes, max_frames)
        self.tree = Path(clips_root) / f"frames{self.nframes}"
        self.enc_batch = enc_batch

    def out_dir(self, w) -> Path:
        return self.tree / w["ds"] / w["split"]

    def clip_path(self, ds, split, qid) -> Path:
        return self.tree / ds / split / f"{qid}.mp4"

    def _done(self, w) -> bool:
        d = self.out_dir(w)
        p = d / f"{w['qids'][0]}.mp4"
        return p.exists() and p.stat().st_size > 0 and all(
            os.path.lexists(d / f"{q}.mp4") for q in w["qids"][1:])

    def _link_aliases(self, w):
        d = self.out_dir(w)
        for q in w["qids"][1:]:
            for ext in (".mp4", ".json"):
                dst = d / f"{q}{ext}"
                if not os.path.lexists(dst):
                    os.symlink(f"{w['qids'][0]}{ext}", dst)

    def extract_video(self, ds, stem, windows, mezz, stats_npz) -> list[dict]:
        """All pending windows of one video. Returns a list of failures."""
        fails = []
        try:
            meta = probe(mezz)
            wpx, hpx = meta["width"], meta["height"]
            n_mezz = meta["frames"] or int(round(meta["duration_s"] * 5))
        except Exception as e:  # noqa: BLE001
            return [dict(ds=ds, video=stem, err=f"probe: {e!r}")]
        dead = dead_mask(stats_npz, n_mezz)
        frame_bytes = wpx * hpx * 3
        enc_scale, out_res = [], None
        if self.budget is not None:
            tw, th = policy.pick_grid(wpx, hpx, self.budget)
            enc_scale, out_res = ["-vf", f"scale={tw}:{th}:flags=bicubic"], [tw, th]

        todo = []
        for w in windows:
            if self._done(w):
                self._link_aliases(w)
                continue
            i0 = int(round(w["w0"] * 5))
            i1 = min(n_mezz, int(round(w["w1"] * 5)))
            cand = i0 + np.flatnonzero(~dead[i0:i1])
            if cand.size == 0:                  # fallback: a window made only of dead frames
                cand = np.arange(i0, i1)
            sel = cand[sample_indices(cand.size, self.nframes)].tolist() if cand.size > self.nframes \
                else cand.tolist()
            todo.append((w, sel, int(cand.size)))
        if not todo:
            return fails

        todo.sort(key=lambda t: (t[0]["w0"], t[0]["w1"]))      # one decode stream per start time
        batches, cur = [], []
        for t in todo:
            if cur and (t[0]["w0"] != cur[0][0]["w0"] or len(cur) >= self.enc_batch):
                batches.append(cur)
                cur = []
            cur.append(t)
        if cur:
            batches.append(cur)

        for batch in batches:
            need = {}
            for bi, (_, sel, _) in enumerate(batch):
                for j in sel:
                    need.setdefault(j, []).append(bi)
            max_idx = max(need)
            encs, tmps = [], []
            for w, sel, _ in batch:
                d = self.out_dir(w)
                d.mkdir(parents=True, exist_ok=True)
                tmp = str(d / f"{w['qids'][0]}.mp4.tmp.mp4")
                fps_out = f"{len(sel)}/{int(w['w1'] - w['w0'])}"
                encs.append(subprocess.Popen(
                    [self.ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", f"{wpx}x{hpx}", "-framerate", fps_out, "-i", "-",
                     *enc_scale, *ENC, "-video_track_timescale", "12800", tmp],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                tmps.append(tmp)
            w0_batch = batch[0][0]["w0"]
            i0 = int(round(w0_batch * 5))
            seek = ["-ss", str(int(w0_batch))] if i0 > 0 else []     # mezzanine is CFR 5 fps
            dec = subprocess.Popen(
                [self.ffmpeg, "-nostdin", "-v", "error", "-threads", "2", *seek, "-i", str(mezz),
                 "-frames:v", str(max_idx - i0 + 1), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                stdout=subprocess.PIPE)
            left = {bi: len(sel) for bi, (_, sel, _) in enumerate(batch)}
            j = i0
            while left:
                buf = dec.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                for bi in need.get(j, ()):
                    if bi not in left:
                        continue
                    try:
                        encs[bi].stdin.write(buf)
                    except BrokenPipeError:
                        left.pop(bi, None)
                        continue
                    left[bi] -= 1
                    if left[bi] == 0:
                        encs[bi].stdin.close()
                        left.pop(bi)
                j += 1
            dec.stdout.close()
            dec.wait()
            for bi, (w, sel, n_cand) in enumerate(batch):
                d = self.out_dir(w)
                if bi in left and encs[bi].stdin and not encs[bi].stdin.closed:
                    encs[bi].stdin.close()
                rc = encs[bi].wait()
                ok = rc == 0 and bi not in left
                got = None
                if ok:
                    try:
                        got = packet_count(tmps[bi])
                        ok = got == len(sel)
                    except Exception as e:  # noqa: BLE001
                        ok, got = False, repr(e)
                if not ok:
                    if os.path.exists(tmps[bi]):
                        os.unlink(tmps[bi])
                    fails.append(dict(ds=ds, video=stem, qid=w["qids"][0],
                                      err=f"enc rc={rc} short={bi in left} got={got} want={len(sel)}"))
                    continue
                side = dict(qid=w["qids"][0], ds=ds, split=w["split"], video=stem,
                            end_s=w["end_s"], w0=w["w0"], w1=w["w1"], tier=w["tier"], rule=w["rule"],
                            n_mezz=n_mezz, n_candidates=n_cand, n_frames=len(sel),
                            fps_out=f"{len(sel)}/{int(w['w1'] - w['w0'])}", mezz_indices=sel,
                            regime=self.regime, branch=self.branch, out_res=out_res)
                side_path = d / f"{w['qids'][0]}.json"
                Path(str(side_path) + ".tmp").write_text(json.dumps(side))
                os.replace(str(side_path) + ".tmp", side_path)
                os.replace(tmps[bi], d / f"{w['qids'][0]}.mp4")    # the clip lands last
                self._link_aliases(w)
        return fails


def run_extraction(job):
    """Process-pool entry point: job = (ffmpeg, clips_root, regime, branch, ds, stem, windows,
    mezz, stats_npz, enc_batch, max_frames)."""
    ffmpeg, root, regime, branch, ds, stem, windows, mezz, npz, enc_batch, max_frames = job
    t0 = time.time()
    fails = ClipExtractor(ffmpeg, root, regime, branch, enc_batch, max_frames).extract_video(
        ds, stem, windows, mezz, npz)
    print(f"  {ds}/{stem} [{regime}/{branch}]: {len(windows)} windows in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return fails
