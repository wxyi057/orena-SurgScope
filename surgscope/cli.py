"""``surgscope`` command line."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("surgscope")

DEFAULT_REPO = "wxyi088/orena-SurgScope"


def _setup_logging(verbose: bool = False):
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.propagate = False


def _data_args(values):
    out = []
    for v in values or []:
        name, _, root = v.partition("=")
        if not root:
            raise SystemExit(f"--data expects NAME=ROOT (e.g. heico=/data/heico-focus-vqa), got {v!r}")
        out.append((name, root))
    return out


# ----------------------------------------------------------------------------- commands
def cmd_doctor(a):
    import importlib.metadata as md
    from surgscope.ffmpeg import capabilities, find_ffmpeg
    print("SurgScope environment check")
    for pkg in ("torch", "transformers", "ms-swift", "peft", "qwen-vl-utils", "av",
                "imageio-ffmpeg", "orena-focus", "flash-attn"):
        try:
            print(f"  {pkg:<15} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:<15} not installed")
    try:
        ff = find_ffmpeg(a.ffmpeg)
        cap = capabilities(ff)
        print(f"  ffmpeg          {ff}\n                  {cap['version']}")
        print(f"                  -fps_mode: {cap['fps_mode']}  libx264: {cap['libx264']}  "
              f"drawtext: {cap['drawtext']}")
        if not cap["drawtext"]:
            alt = find_ffmpeg(None, drawtext=True)
            ok = alt != ff and capabilities(alt)["drawtext"]
            print(f"  ffmpeg (clock)  {alt if ok else 'none with drawtext found: needed only by `surgscope transcode`'}")
    except Exception as e:  # noqa: BLE001
        print(f"  ffmpeg          PROBLEM: {e}")
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            print(f"  GPU             {torch.cuda.get_device_name(0)}, {total / 2**30:.0f} GiB "
                  f"({free / 2**30:.0f} GiB free); inference needs ~40 GiB")
        else:
            print("  GPU             none visible (inference needs a CUDA GPU)")
    except Exception as e:  # noqa: BLE001
        print(f"  torch           PROBLEM: {e}")


def cmd_download(a):
    from huggingface_hub import snapshot_download
    print(snapshot_download(a.repo, local_dir=a.local_dir, allow_patterns=["adapter_*"]))


def cmd_infer(a):
    from surgscope.io import write_answers, write_json
    from surgscope.pipeline import SurgScope
    model = SurgScope.from_pretrained(a.model, base_model=a.base_model, ffmpeg=a.ffmpeg,
                                      attn_impl=a.attn_impl, work_dir=a.work_dir)
    budget = a.time_budget
    if a.challenge_budget:
        n = len(json.loads((Path(a.input) / "request.json").read_text()))
        budget = 120.0 + 30.0 * n
    questions, answers, traces = model.answer_dir(a.input, time_budget=budget)
    latency = traces[0]["seconds_per_question"] if traces else 0.0
    write_answers(a.output, [q.qID for q in questions], answers, latency)
    if a.trace:
        write_json(a.trace, traces)
    for q, ans in zip(questions, answers):
        print(f"{q.qID}\t{ans}")
    try:
        import torch
        if torch.cuda.is_available():
            log.info("peak GPU memory: %.1f GiB allocated, %.1f GiB reserved",
                     torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30)
    except Exception:  # noqa: BLE001
        pass
    model.close()


def cmd_infer_dataset(a):
    """Answer a whole dataset split directly from the overlayed mezzanines.

    A challenge clip is the prefix [0, end] of the mezzanine, so windows inside [0, end] see
    exactly the same frames; no per-question copies are needed.
    """
    from surgscope.data.prepare import DatasetSource
    from surgscope.data.extract import hms_to_s
    from surgscope.io import Question, write_answers, write_json
    from surgscope.pipeline import SurgScope
    from surgscope.prompts import load_fo_definitions
    qs = []
    for name, root in _data_args(a.data):
        src = DatasetSource(name, root, a.work)
        for r in src.rows(a.split):
            stem = os.path.splitext(os.path.basename(r["video"]))[0]
            qs.append(Question(qID=str(r["id"]), question=r["question"], end_time=float(hms_to_s(r["timestamp_end"])),
                               procedure_type=r.get("procedure_type") or None, video=str(src.overlayed(stem)),
                               videoID=r["video"]))
    k, n = map(int, a.shard.split("/"))
    qs = [q for i, q in enumerate(qs) if i % n == k]
    model = SurgScope.from_pretrained(a.model, base_model=a.base_model, ffmpeg=a.ffmpeg,
                                      attn_impl=a.attn_impl, work_dir=a.work_dir)
    answers, traces = model.answer_batch(qs, load_fo_definitions(a.fo_definitions))
    write_answers(a.output, [q.qID for q in qs], answers, traces[0]["seconds_per_question"] if traces else 0.0)
    if a.trace:
        write_json(a.trace, traces)
    print(f"{len(qs)} answers -> {a.output}")
    model.close()


def cmd_ask(a):
    from surgscope.pipeline import SurgScope
    from surgscope.prompts import load_fo_definitions
    model = SurgScope.from_pretrained(a.model, base_model=a.base_model, ffmpeg=a.ffmpeg,
                                      attn_impl=a.attn_impl)
    ans, trace = model.answer(a.question, a.video, end_time=a.end_time,
                              procedure_type=a.procedure_type,
                              fo_definitions=load_fo_definitions(a.fo_definitions))
    print(json.dumps(trace, indent=2))
    print(ans)
    model.close()


def cmd_evaluate(a):
    from surgscope.evaluate import evaluate, format_report
    res = evaluate(a.pred, a.reference, judge=a.judge, output_dir=a.output_dir)
    print(format_report(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


def cmd_transcode(a):
    from surgscope.data.prepare import DatasetSource, transcode
    from surgscope.ffmpeg import check, find_ffmpeg, find_font
    ffmpeg = find_ffmpeg(a.ffmpeg, drawtext=True)
    check(ffmpeg, drawtext=True)
    for name, root in _data_args(a.data):
        n = transcode(DatasetSource(name, root, a.work), a.splits, ffmpeg, find_font(a.font),
                      procs=a.procs, shard=a.shard, only=a.videos)
        print(f"{name}: {n} video(s) processed (shard {a.shard})")


def cmd_make_input(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from surgscope.data.prepare import DatasetSource
    from surgscope.data.transcode import cut_prefix
    from surgscope.ffmpeg import find_ffmpeg
    from surgscope.prompts import load_fo_definitions
    ffmpeg = find_ffmpeg(a.ffmpeg)
    (name, root), = _data_args([a.data])
    src = DatasetSource(name, root, a.work)
    rows = [r for r in src.rows(a.split) if not a.ids or int(r["id"]) in set(a.ids)]
    out = Path(a.out)
    reqs = []
    for r in rows:
        stem = os.path.splitext(os.path.basename(r["video"]))[0]
        end_s = sum(int(x) * m for x, m in zip(r["timestamp_end"].split(":"), (3600, 60, 1)))
        for variant, mezz in (("plain", src.plain(stem)), ("overlayed", src.overlayed(stem))):
            cut_prefix(ffmpeg, mezz, out / variant / f"{r['id']}.mp4", end_s)
        reqs.append(dict(qID=str(r["id"]), videoID=r["video"], start_time=0.0, end_time=float(end_s),
                         procedure_type=str(r.get("procedure_type") or ""), question=r["question"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "request.json").write_text(json.dumps(reqs, indent=1, ensure_ascii=False))
    (out / "batch.json").write_text(json.dumps({"qIDs": [q["qID"] for q in reqs]}))
    (out / "FO_definitions.json").write_text(json.dumps(load_fo_definitions(a.fo_definitions)))
    table = pa.Table.from_pylist([{k: v for k, v in r.items() if k not in ("ds", "split")} for r in rows])
    pq.write_table(table, out / "reference.parquet")
    print(f"{len(reqs)} question(s) -> {out}")


def cmd_prepare(a):
    from surgscope.data import sft
    from surgscope.data.prepare import (DatasetSource, InputDirSource, clip_path, extract_clips,
                                        scan_inputs)
    from surgscope.ffmpeg import check, find_ffmpeg
    from surgscope.prompts import load_fo_definitions
    ffmpeg = find_ffmpeg(a.ffmpeg)
    check(ffmpeg)
    if a.from_input:
        ref = a.reference or str(Path(a.from_input) / "reference.parquet")
        sources = [InputDirSource(a.from_input, ref, a.work)]
        scan_inputs(sources[0], a.split, ffmpeg)
        defs_path = a.fo_definitions or str(Path(a.from_input) / "FO_definitions.json")
    else:
        sources = [DatasetSource(n, r, a.work) for n, r in _data_args(a.data)]
        defs_path = a.fo_definitions
    if not a.jsonl_only:
        fails = extract_clips(sources, a.split, a.regime, a.work, ffmpeg, procs=a.procs,
                              shard=a.shard, max_frames=a.max_frames)
        if fails:
            raise SystemExit(f"{len(fails)} window(s) failed, e.g. {fails[:3]}")
    if a.shard != "0/1" and not a.jsonl_only:
        print("clips done for this shard; run again with --jsonl-only once all shards finished")
        return
    items = [r for s in sources for r in s.rows(a.split)]
    rows, missing = sft.build_rows(items, a.regime, lambda r: clip_path(a.work, a.regime, r, a.max_frames),
                                   load_fo_definitions(defs_path), max_frames=a.max_frames)
    if missing:
        raise SystemExit(f"{len(missing)} clip(s) missing, e.g. {missing[:3]}")
    out = a.out or str(Path(a.work) / "sft" / f"{a.regime}.jsonl")
    summary = sft.write_jsonl(rows, out)
    print(f"{out}: {json.dumps(summary)}")


def cmd_train(a):
    from surgscope.train.recipes import build_command, load_recipe
    cfg = load_recipe(a.config)
    if a.model:
        cfg["model"] = a.model
    argv, env = build_command(cfg, a.dataset, a.output_dir, nproc_per_node=a.nproc_per_node,
                              nnodes=a.nnodes, attn_impl=a.attn_impl, max_steps=a.max_steps,
                              max_frames=a.max_frames, extra=a.extra)
    print(" ".join(f"{k}={v}" for k, v in env.items()) + " \\\n  " + shlex.join(argv))
    if a.dry_run:
        return
    raise SystemExit(subprocess.call(argv, env={**os.environ, **env}))


def cmd_soup(a):
    from surgscope.train.soup import make_soup
    out = make_soup(a.adapters, a.out, names=a.names)
    print(json.loads((out / "SOUP_MEMBERS.json").read_text())["sha256"], out)


def cmd_merge(a):
    from surgscope.train.soup import merge
    print(shlex.join(merge(a.adapter, a.out, base_model=a.base, dry_run=a.dry_run)))


# ----------------------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(prog="surgscope", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def model_opts(sp):
        sp.add_argument("--model", default=DEFAULT_REPO,
                        help="adapter (HF repo id or local directory) or a merged model directory")
        sp.add_argument("--base-model", help="base model for the adapter (default: Qwen/Qwen3.5-9B)")
        sp.add_argument("--attn-impl", default="sdpa")
        sp.add_argument("--ffmpeg", help="ffmpeg binary (default: imageio-ffmpeg, then PATH)")

    sp = sub.add_parser("doctor", help="check packages, ffmpeg and GPU")
    sp.add_argument("--ffmpeg")
    sp.set_defaults(fn=cmd_doctor)

    sp = sub.add_parser("download", help="download the SurgScope adapter")
    sp.add_argument("--repo", default=DEFAULT_REPO)
    sp.add_argument("--local-dir")
    sp.set_defaults(fn=cmd_download)

    sp = sub.add_parser("infer", help="answer a challenge-layout input directory")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", default="answer.json")
    sp.add_argument("--trace", help="write per-question routing / refinement trace (JSON)")
    sp.add_argument("--work-dir")
    sp.add_argument("--time-budget", type=float, help="seconds for the whole batch (latency guard)")
    sp.add_argument("--challenge-budget", action="store_true", help="budget = 120 s + 30 s per question")
    model_opts(sp)
    sp.set_defaults(fn=cmd_infer)

    sp = sub.add_parser("infer-dataset", help="answer a dataset split from the overlayed mezzanines")
    sp.add_argument("--data", action="append", required=True, metavar="NAME=ROOT")
    sp.add_argument("--work", required=True, help="work directory of `surgscope transcode`")
    sp.add_argument("--split", default="test")
    sp.add_argument("--fo-definitions", help="FO_definitions.json (default: orena-focus definitions)")
    sp.add_argument("--output", default="answer.json")
    sp.add_argument("--trace")
    sp.add_argument("--work-dir")
    sp.add_argument("--shard", default="0/1", help="K/N: every N-th question (one process per GPU)")
    model_opts(sp)
    sp.set_defaults(fn=cmd_infer_dataset)

    sp = sub.add_parser("ask", help="answer one question about an overlayed clip")
    sp.add_argument("--video", required=True, help="clip covering [0, end] of the procedure, with clock")
    sp.add_argument("--question", required=True)
    sp.add_argument("--end-time", type=float, help="seconds (default: clip duration)")
    sp.add_argument("--procedure-type")
    sp.add_argument("--fo-definitions", help="FO_definitions.json (default: orena-focus definitions)")
    model_opts(sp)
    sp.set_defaults(fn=cmd_ask)

    sp = sub.add_parser("evaluate", help="score answer.json with the official metric")
    sp.add_argument("--pred", required=True, nargs="+", help="answer.json file(s), e.g. one per shard")
    sp.add_argument("--reference", required=True, nargs="+", help="parquet file(s) with references")
    sp.add_argument("--judge", default="auto", help="auto | none | judge model id")
    sp.add_argument("--output-dir")
    sp.add_argument("--json", help="write the full result as JSON")
    sp.set_defaults(fn=cmd_evaluate)

    def data_opts(sp):
        sp.add_argument("--data", action="append", metavar="NAME=ROOT",
                        help="dataset root in the HF layout (repeatable), e.g. heico=/data/heico-focus-vqa")
        sp.add_argument("--work", required=True, help="work directory for all derived data")
        sp.add_argument("--ffmpeg")
        sp.add_argument("--procs", type=int, default=8)
        sp.add_argument("--shard", default="0/1", help="K/N: process every N-th video (multi-node)")

    sp = sub.add_parser("transcode", help="source videos -> 5 fps mezzanines (+ clock) + dead-frame stats")
    data_opts(sp)
    sp.add_argument("--splits", nargs="+", default=["train", "test"])
    sp.add_argument("--videos", nargs="+", help="only these video file stems")
    sp.add_argument("--font")
    sp.set_defaults(fn=cmd_transcode)

    sp = sub.add_parser("make-input", help="cut a challenge-layout input directory from mezzanines")
    sp.add_argument("--data", required=True, metavar="NAME=ROOT")
    sp.add_argument("--work", required=True)
    sp.add_argument("--split", default="test")
    sp.add_argument("--ids", type=int, nargs="*", help="question ids (default: all)")
    sp.add_argument("--out", required=True)
    sp.add_argument("--fo-definitions")
    sp.add_argument("--ffmpeg")
    sp.set_defaults(fn=cmd_make_input)

    sp = sub.add_parser("prepare", help="training mini-clips + ms-swift JSONL for one regime")
    data_opts(sp)
    sp.add_argument("--regime", required=True, choices=["r768", "dense-mix", "dense-mix-2"])
    sp.add_argument("--split", default="train")
    sp.add_argument("--from-input", help="use a challenge-layout directory (e.g. the examples)")
    sp.add_argument("--reference", help="parquet with references for --from-input")
    sp.add_argument("--fo-definitions", help="FO_definitions.json for the system prompt")
    sp.add_argument("--out", help="JSONL path (default: <work>/sft/<regime>.jsonl)")
    sp.add_argument("--jsonl-only", action="store_true", help="skip clip extraction")
    sp.add_argument("--max-frames", type=int, help="cap frames per clip (smoke tests only)")
    sp.set_defaults(fn=cmd_prepare)

    sp = sub.add_parser("train", help="LoRA fine-tuning with ms-swift")
    sp.add_argument("--config", required=True, help="configs/train/<regime>.yaml")
    sp.add_argument("--dataset", required=True, help="JSONL from `surgscope prepare`")
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--model", help="override the base model (HF id or local directory)")
    sp.add_argument("--nproc-per-node", type=int, default=1)
    sp.add_argument("--nnodes", type=int, default=1)
    sp.add_argument("--attn-impl", default="flash_attn")
    sp.add_argument("--max-steps", type=int)
    sp.add_argument("--max-frames", type=int, help="must match `prepare --max-frames`")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("extra", nargs=argparse.REMAINDER, help="extra `swift sft` arguments after --")
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("soup", help="uniform average of LoRA adapters")
    sp.add_argument("--adapters", nargs="+", required=True)
    sp.add_argument("--names", nargs="+")
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_soup)

    sp = sub.add_parser("merge", help="merge a LoRA adapter into Qwen3.5-9B")
    sp.add_argument("--adapter", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--base", default="Qwen/Qwen3.5-9B")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(fn=cmd_merge)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if getattr(a, "extra", None) and a.extra[:1] == ["--"]:
        a.extra = a.extra[1:]
    _setup_logging(a.verbose)
    a.fn(a)


if __name__ == "__main__":
    main()
