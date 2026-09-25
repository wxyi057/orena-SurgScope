"""Data preparation, end to end: source videos -> mezzanines -> mini-clips -> JSONL.

A dataset root follows the Hugging Face layout of the challenge data::

    <root>/data/procedure/{train,test}.parquet
    <root>/videos/<video file>

and the work directory collects everything derived from it::

    <work>/mezz/<ds>/{plain,overlayed}/<stem>.mp4   5 fps, height <= 576 (+ burned-in clock)
    <work>/stats/<ds>/<stem>.npz                    per-frame dead-frame statistics
    <work>/clips/frames<N>/<ds>/<split>/<id>.mp4    training mini-clips (+ .json sidecars)
    <work>/sft/<regime>.jsonl                       ms-swift training rows

A challenge-layout input directory (such as ``examples/train``) can stand in for a
dataset root: its per-question clips already are 5 fps prefixes of the source video.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from surgscope import policy
from surgscope.data.extract import group_windows, hms_to_s, run_extraction
from surgscope.data.scan import scan_to_npz
from surgscope.data.transcode import DATASET_MODE, make_mezzanines


def _stem(video: str) -> str:
    return os.path.splitext(os.path.basename(video))[0]


def _shard(seq, shard: str):
    """'K/N' snake sharding over a list sorted by decreasing cost."""
    k, n = map(int, shard.split("/"))
    return [x for i, x in enumerate(seq) if i % (2 * n) in (k, 2 * n - 1 - k)]


# --------------------------------------------------------------------------- sources
class DatasetSource:
    """A challenge dataset root (parquet + source videos)."""

    def __init__(self, name: str, root, work):
        self.name, self.root, self.work = name, Path(root), Path(work)
        self.mode = DATASET_MODE.get(name, "one-pass")

    def rows(self, split: str) -> list[dict]:
        import pyarrow.parquet as pq
        t = pq.read_table(self.root / "data" / "procedure" / f"{split}.parquet")
        return [dict(r, ds=self.name, split=split) for r in t.to_pylist()]

    def items(self, split: str) -> list[dict]:
        return [dict(ds=self.name, split=split, qid=int(r["id"]), stem=_stem(r["video"]),
                     end_s=hms_to_s(r["timestamp_end"]), question=r["question"])
                for r in self.rows(split)]

    def source_video(self, stem: str) -> Path:
        hits = sorted((self.root / "videos").glob(f"{stem}.*"))
        if not hits:
            raise FileNotFoundError(self.root / "videos" / stem)
        return hits[0]

    def plain(self, stem) -> Path:
        return self.work / "mezz" / self.name / "plain" / f"{stem}.mp4"

    def overlayed(self, stem) -> Path:
        return self.work / "mezz" / self.name / "overlayed" / f"{stem}.mp4"

    def stats(self, stem) -> Path:
        return self.work / "stats" / self.name / f"{stem}.npz"

    def scan_input(self, stem) -> Path:
        # HeiCo statistics were computed on the 5 fps plain mezzanine, LapChole on the source
        return self.plain(stem) if self.mode == "two-pass" else self.source_video(stem)


class InputDirSource(DatasetSource):
    """A challenge-layout input directory whose questions have references in a parquet."""

    def __init__(self, input_dir, reference, work, name: str = "input"):
        super().__init__(name, input_dir, work)
        self.reference = Path(reference)
        self.requests = {str(r["qID"]): r for r in
                         json.loads((self.root / "request.json").read_text())}

    def rows(self, split: str) -> list[dict]:
        import pyarrow.parquet as pq
        out = []
        for r in pq.read_table(self.reference).to_pylist():
            if str(r["id"]) in self.requests:
                out.append(dict(r, ds=self.name, split=split))
        return out

    def items(self, split: str) -> list[dict]:
        return [dict(ds=self.name, split=split, qid=int(r["id"]), stem=str(r["id"]),
                     end_s=hms_to_s(r["timestamp_end"]), question=r["question"])
                for r in self.rows(split)]

    def plain(self, stem) -> Path:
        return self.root / "plain" / f"{stem}.mp4"

    def overlayed(self, stem) -> Path:
        return self.root / "overlayed" / f"{stem}.mp4"

    def scan_input(self, stem) -> Path:
        return self.plain(stem)


# --------------------------------------------------------------------------- stages
def _mezz_job(job):
    ffmpeg, font, src, plain, over, mode, stats_in, stats_out = job
    make_mezzanines(ffmpeg, font, src, plain, over, mode=mode)
    scan_to_npz(ffmpeg, stats_in, stats_out)
    return str(over)


def transcode(source: DatasetSource, splits, ffmpeg: str, font: str, procs: int = 8,
              shard: str = "0/1", only=None) -> int:
    """Mezzanines + dead-frame statistics for every video referenced by ``splits``
    (or only the videos whose file stem is in ``only``)."""
    stems = sorted({it["stem"] for s in splits for it in source.items(s)})
    if only:
        stems = [s for s in stems if s in set(only)]
    jobs = []
    for stem in stems:
        src = source.source_video(stem)
        stats_in = source.plain(stem) if source.mode == "two-pass" else src
        jobs.append((os.path.getsize(src), (ffmpeg, font, str(src), str(source.plain(stem)),
                                            str(source.overlayed(stem)), source.mode,
                                            str(stats_in), str(source.stats(stem)))))
    jobs = [j for _, j in sorted(jobs, key=lambda x: -x[0])]
    jobs = _shard(jobs, shard)
    with ProcessPoolExecutor(max_workers=procs) as ex:
        for out in ex.map(_mezz_job, jobs):
            print("  mezzanine ready:", out, flush=True)
    return len(jobs)


def scan_inputs(source: InputDirSource, split: str, ffmpeg: str) -> None:
    for it in source.items(split):
        scan_to_npz(ffmpeg, source.scan_input(it["stem"]), source.stats(it["stem"]))


def extract_clips(sources, split: str, regime: str, work, ffmpeg: str, procs: int = 8,
                  shard: str = "0/1", enc_batch: int = 40, max_frames: int | None = None) -> list[dict]:
    """Training mini-clips of ``regime`` for all questions of ``split``. Returns failures."""
    by_name = {s.name: s for s in sources}
    items = [it for s in sources for it in s.items(split)]
    prof = policy.REGIMES[regime]
    branches = ["dense", "mid"] if prof["dense"] != prof["mid"] else [None]
    jobs = []
    for br in branches:
        for ds, stem, wl in group_windows(items, regime, branch=br):
            src = by_name[ds]
            cost = sum(w["w1"] - w["w0"] for w in wl)
            jobs.append((cost, (ffmpeg, str(Path(work) / "clips"), regime, br or "dense", ds, stem, wl,
                                str(src.overlayed(stem)), str(src.stats(stem)), enc_batch, max_frames)))
    jobs = [j for _, j in sorted(jobs, key=lambda x: -x[0])]
    jobs = _shard(jobs, shard)
    fails = []
    with ProcessPoolExecutor(max_workers=procs) as ex:
        for fl in ex.map(run_extraction, jobs):
            fails.extend(fl)
    return fails


def clip_path(work, regime: str, row: dict, max_frames: int | None = None) -> Path:
    n = policy.nframes(row["question"], regime)
    if max_frames:
        n = min(n, max_frames)
    return Path(work) / "clips" / f"frames{n}" / row["ds"] / row["split"] / f"{int(row['id'])}.mp4"
