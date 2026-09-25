"""SurgScope inference: route -> pass 1 -> abstention re-ask -> pass 2 -> pass 3 -> guard.

  route    the question text selects a window (19 rules) and a frame budget (dense / mid)
  pass 1   one forward pass on a mini-clip of the routed window
  abstain  a shortened window answered "outside clip window" -> re-ask on the whole prefix
  pass 2   an answer holding an in-range time T -> re-ask on [T - 10 min, T + 10 min]
  pass 3   re-ask on [T' - 2 min, T' + 2 min] around the pass-2 time T'; kept only if it
           contains a time
  guard    strip reasoning tags; timestamp-only answers collapse to one HH:MM:SS

Questions are processed as a batch in three sweeps (all pass-1 answers, then all pass-2
refinements, then all pass-3 refinements); decoding of the next clip is prefetched on a
worker thread while the GPU runs. An optional time budget reproduces the challenge's
latency guard (refinement passes are dropped, pass 3 first, rather than overrunning).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from surgscope import policy
from surgscope.ffmpeg import check as check_ffmpeg, find_ffmpeg
from surgscope.io import Question, read_input_dir
from surgscope.postprocess import (ZOOM_HALF_1, ZOOM_HALF_2, clean, first_timestamp,
                                   format_guard, window_around, zoom_anchor)
from surgscope.prompts import hms, load_fo_definitions, system_prompt, user_content
from surgscope.routing import is_abstain, route
from surgscope.video import CLIP_FPS, ENC, Video

log = logging.getLogger("surgscope")

DEFAULT_MODEL = "wxyi088/orena-SurgScope"
SELF_MARGIN = 45.0          # latency guard: keep this much of the budget in reserve


class SurgScope:
    """Question-conditioned windowing around a fine-tuned Qwen3.5-9B."""

    def __init__(self, model_dir, adapter=None, ffmpeg=None, attn_impl: str = "sdpa",
                 work_dir=None, warmup: bool = True):
        from surgscope.engine import SwiftEngine

        self.ffmpeg = find_ffmpeg(ffmpeg)
        check_ffmpeg(self.ffmpeg)
        self._own_work = work_dir is None
        self.work_dir = Path(work_dir or tempfile.mkdtemp(prefix="surgscope_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.regime = policy.INFERENCE_REGIME
        t0 = time.monotonic()
        self.engine = SwiftEngine(model_dir, adapter=adapter, attn_impl=attn_impl)
        log.info("model loaded from %s%s in %.1f s", model_dir,
                 f" + {adapter}" if adapter else "", time.monotonic() - t0)
        self._needs_warmup = warmup

    @classmethod
    def from_pretrained(cls, repo_or_dir: str = DEFAULT_MODEL, revision=None, base_model=None,
                        **kw):
        """Load the SurgScope adapter (Hub id or local directory) and merge it into its base
        model, or load a merged model directory."""
        path = Path(repo_or_dir)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(repo_or_dir, revision=revision,
                                          allow_patterns=["adapter_*"]))
        cfg = path / "adapter_config.json"
        if not cfg.exists():
            return cls(path, **kw)
        base = base_model or json.loads(cfg.read_text())["base_model_name_or_path"]
        return cls(base, adapter=path, **kw)

    def close(self):
        if self._own_work:
            shutil.rmtree(self.work_dir, ignore_errors=True)

    # ------------------------------------------------------------------ helpers
    def _warmup(self, system):
        """One generation at the largest mid shape (1152 frames @ 384x224) so that kernel
        selection happens before the first real question."""
        import subprocess
        self._needs_warmup = False
        try:
            wpath = self.work_dir / "_warmup.mp4"
            subprocess.run([self.ffmpeg, "-nostdin", "-y", "-v", "error", "-f", "lavfi",
                            "-i", "color=c=gray:size=384x224:rate=5", "-t", f"{1152 / 5:.1f}",
                            "-frames:v", "1152", *ENC, str(wpath)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            t = time.monotonic()
            self.engine.generate(system, user_content("Is a foreign object visible?", 0, 1800, None),
                                 wpath, 1152, max_tokens=4)
            log.info("warmup %.1f s", time.monotonic() - t)
        except Exception:  # noqa: BLE001
            log.exception("warmup failed (continuing)")

    def _infer(self, system, q: Question, clip, w0, w1, tag):
        try:
            return self.engine.generate(system, user_content(q.question, w0, w1, q.procedure_type),
                                        clip, policy.nframes(q.question, self.regime))
        except Exception:  # noqa: BLE001
            log.exception("[%s] %s: inference failed; empty answer", tag, q.qID)
            return ""

    # ------------------------------------------------------------------ main API
    def answer_batch(self, questions: list[Question], fo_definitions: str,
                     time_budget: float | None = None):
        """Answer a batch of questions. Returns (answers, traces)."""
        t_start = time.monotonic()
        B = len(questions)
        deadline = (time_budget - SELF_MARGIN) if time_budget else float("inf")
        system = system_prompt(fo_definitions)
        if self._needs_warmup:
            self._warmup(system)

        def elapsed():
            return time.monotonic() - t_start

        videos, routes = {}, {}
        traces = [dict(qID=q.qID, branch=policy.branch(q.question),
                       nframes=policy.nframes(q.question, self.regime)) for q in questions]

        def prep_window(i, w0, w1, tag):
            """Dead-frame scan (cached per question) + mini-clip of [w0, w1] seconds."""
            q = questions[i]
            try:
                v = videos.get(q.qID)
                if v is None:
                    v = Video(self.ffmpeg, q.video, q.end_time)
                    videos[q.qID] = v
                prof = policy.profile(q.question, self.regime)
                i0 = max(0, int(round(w0 * CLIP_FPS)))
                i1 = min(v.n_frames, int(round(w1 * CLIP_FPS)))
                dst = self.work_dir / f"{q.qID}_{tag}.mp4"
                meta = v.make_clip(dst, i0, i1, max(w1 - w0, 1e-3), prof["nframes"], prof["budget"])
                return dst, meta, None
            except Exception as e:  # noqa: BLE001
                log.exception("[%s] clip preparation failed for %s", tag, q.qID)
                return None, None, repr(e)

        def prep_pass1(i):
            rt = route(questions[i].question, float(questions[i].end_time))
            routes[i] = rt
            return prep_window(i, rt["w0"], rt["w1"], "p1")

        def drop(path):
            try:
                os.remove(path)
            except (OSError, TypeError):
                pass

        pool = ThreadPoolExecutor(max_workers=1)
        answers = [""] * B
        t_p1 = elapsed()
        fut = pool.submit(prep_pass1, 0) if B else None
        for i, q in enumerate(questions):
            t_q = time.monotonic()
            clip, meta, err = fut.result()
            if i + 1 < B:
                fut = pool.submit(prep_pass1, i + 1)        # decode the next clip meanwhile
            rt = routes.get(i) or dict(tier=3, rule=None, w0=0.0, w1=float(q.end_time), shrunk=False)
            tr = traces[i]
            tr.update(tier=rt["tier"], rule=rt["rule"], window=[hms(rt["w0"]), hms(rt["w1"])])
            if clip is None:
                log.error("[pass1] %s: no clip (%s) -> empty answer", q.qID, err)
                tr["error"] = err
                continue
            tr.update(frames=meta["n"], live_frames=meta["n_cand"], sampling=meta["mode"])
            answers[i] = clean(self._infer(system, q, clip, int(rt["w0"]), int(rt["w1"]), "pass1"))
            tr["pass1"] = answers[i]
            drop(clip)
            # abstention: a shortened window "outside clip window" -> whole prefix
            if rt["shrunk"] and is_abstain(answers[i]) and elapsed() + 30.0 < deadline:
                fclip, _, _ = prep_window(i, 0.0, float(q.end_time), "p1full")
                if fclip is not None:
                    a2 = clean(self._infer(system, q, fclip, 0, int(q.end_time), "abstain"))
                    tr["abstain_reask"] = {"window": [hms(0), hms(q.end_time)], "answer": a2}
                    if a2:
                        answers[i] = a2
                    drop(fclip)
            log.info("[pass1] %d/%d %s tier=%s rule=%s window=%s-%s %s frames=%d/%d %.1fs -> %r",
                     i + 1, B, q.qID, rt["tier"], rt["rule"], hms(rt["w0"]), hms(rt["w1"]),
                     tr["branch"], meta["n"], meta["n_cand"], time.monotonic() - t_q, answers[i][:40])
        per_q = max(1.0, (elapsed() - t_p1) / max(B, 1))

        # ---- coarse-to-fine temporal refinement (self-routed by the answer format)
        zoom = [(i, zoom_anchor(q.question, answers[i], q.end_time)) for i, q in enumerate(questions)]
        zoom = [(i, t) for i, t in zoom if t is not None]
        log.info("refinement candidates: %d/%d", len(zoom), B)

        def zoom_pass(cands, half, tag, est_per):
            out = {}
            if not cands:
                return out

            def prep(item):
                i, t = item
                z0, z1 = window_around(float(questions[i].end_time), t, half)
                clip, _, _ = prep_window(i, z0, z1, tag)
                return i, clip, z0, z1

            f = pool.submit(prep, cands[0])
            for j, _ in enumerate(cands):
                i, clip, z0, z1 = f.result()
                if j + 1 < len(cands):
                    f = pool.submit(prep, cands[j + 1])
                if elapsed() + est_per > deadline:
                    log.warning("[%s] time budget: skipping %d question(s)", tag, len(cands) - j)
                    drop(clip)
                    break
                if clip is None:
                    continue
                out[i] = (clean(self._infer(system, questions[i], clip, z0, z1, tag)), z0, z1)
                log.info("[%s] %d/%d %s window=%s-%s -> %r", tag, j + 1, len(cands), questions[i].qID,
                         hms(z0), hms(z1), out[i][0][:40])
                drop(clip)
            return out

        est2 = per_q * 0.9
        if zoom and elapsed() + est2 < deadline:
            for i, (a, z0, z1) in zoom_pass(zoom, ZOOM_HALF_1, "pass2", est2).items():
                traces[i]["pass2"] = {"window": [hms(z0), hms(z1)], "answer": a}
                if a:
                    answers[i] = a
        elif zoom:
            log.warning("time budget: skipping pass 2")
            zoom = []

        cands3 = []
        for i, _ in zoom:
            t = first_timestamp(answers[i])
            if t is None or not (0 <= t <= float(questions[i].end_time)):
                continue
            cands3.append((i, float(t)))
        est3 = per_q * 0.7
        if cands3 and elapsed() + est3 < deadline:
            for i, (a, z0, z1) in zoom_pass(cands3, ZOOM_HALF_2, "pass3", est3).items():
                keep = first_timestamp(a) is not None      # pass 3 must yield a time
                traces[i]["pass3"] = {"window": [hms(z0), hms(z1)], "answer": a, "kept": keep}
                if keep:
                    answers[i] = a
        elif cands3:
            log.warning("time budget: skipping pass 3")
        pool.shutdown(wait=False)

        final = [format_guard(a) for a in answers]
        for tr, a in zip(traces, final):
            tr["answer"] = a
        total = elapsed()
        for tr in traces:
            tr["seconds_per_question"] = round(total / max(B, 1), 2)
        return final, traces

    def answer(self, question: str, video, end_time=None, procedure_type=None,
               fo_definitions=None, qid: str = "q0"):
        """Answer one question about an overlayed clip covering [0, end_time] of a procedure.

        ``end_time`` defaults to the clip duration. Returns (answer, trace).
        """
        if end_time is None:
            from surgscope.ffmpeg import header_info
            end_time = header_info(self.ffmpeg, video)["duration"]
        defs = fo_definitions if fo_definitions is not None else load_fo_definitions(None)
        q = Question(qID=qid, question=question, end_time=float(end_time),
                     procedure_type=procedure_type, video=str(video))
        answers, traces = self.answer_batch([q], defs)
        return answers[0], traces[0]

    def answer_dir(self, input_dir, fo_definitions=None, time_budget=None):
        """Answer every question of a challenge-layout input directory."""
        d = Path(input_dir)
        questions = read_input_dir(d)
        defs = fo_definitions if fo_definitions is not None else load_fo_definitions(d / "FO_definitions.json")
        answers, traces = self.answer_batch(questions, defs, time_budget=time_budget)
        return questions, answers, traces
