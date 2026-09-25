"""SurgScope — challenge container entry point (ORena SAVE FOCUS 2026, PROCEDURE track).

This is the inference script of the final test-set submission (official score 0.6592),
kept byte-for-byte in its logic; only comments and names were edited for release.

Reads /input (request.json, FO_definitions.json, overlayed/<qID>.mp4) and writes
/output/answer.json. Model: Qwen3.5-9B with the SurgScope weights merged in
(resources/SurgScope-9B), run with ms-swift's TransformersEngine (bf16, SDPA, batch 1).

  route    the question text selects a window (19 rules, vendored below from
           surgscope/routing.py) and a frame budget (proc_policy.py):
           time / counting / aggregation questions -> 1536 frames at 60 tokens per
           frame pair, all others -> 1152 frames at 84 tokens per pair; the resolution
           is baked into the mini-clip pixels and the frame count travels with every
           request as chat_template_kwargs {min_frames, max_frames}
  scan     keyframe-only decode marks dead frames (privacy mask / black, > 0.9)
  pass 1   frames sampled uniformly from the live frames of the routed window
  abstain  a shortened window answered "outside clip window" -> re-run on the prefix
  zoom     answers holding an in-range HH:MM:SS (not "how long"): +-10 min around it,
           then +-2 min around the pass-2 answer
  guard    strip <think> residue; timestamp-only answers collapse to one HH:MM:SS

Latency guard: allowed = 120 s + B x 30 s; refinement passes are dropped (pass 3 first,
then pass 2) rather than overrunning. The next clip is decoded while the GPU runs.

The system prompt is the trained preamble followed by the current
/input/FO_definitions.json, read at run time, so new classes enter the prompt as given.
"""

import os

# Sampling / visual-token config, identical to training. Must precede any
# swift/transformers import so qwen-vl-utils picks them up (ms-swift copies
# these env vars into qwen_vl_utils.vision_process at model load). FPS_MIN/MAX
# are only the fallback — every InferRequest carries explicit min/max_frames.
os.environ["FPS_MIN_FRAMES"] = "1536"
os.environ["FPS_MAX_FRAMES"] = "1536"
os.environ["VIDEO_MAX_TOKEN_NUM"] = "128"
os.environ["VIDEO_MIN_TOKEN_NUM"] = "32"
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import logging
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from proc_policy import POLICIES, branch as policy_branch, pick_grid

# Own handler, no propagation: ms-swift reconfigures the root logger on import.
log = logging.getLogger("procalgo")
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(_h)
log.setLevel(logging.INFO)
log.propagate = False

RESOURCES_PATH = Path(__file__).parent / "resources"
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = RESOURCES_PATH / "SurgScope-9B"
VIDEO_DIR = INPUT_PATH / "overlayed"       # trained on the overlay variant
WORK_DIR = Path("/tmp/proc_work")

PROFILES = POLICIES["dense-mix"]           # dense: 1536@60, mid: 1152@84
CLIP_FPS = 5.0                             # platform clips arrive at exactly 5 fps
DEAD_BLUE_T = 0.90                         # thumbnail blue-pixel fraction (identical to training scan)
DEAD_BLACK_T = 0.90
MAX_NEW_TOKENS = 64

# Zoom geometry: +-10 min, then +-2 min
ZOOM_HALF_1 = 600.0
ZOOM_HALF_2 = 120.0
ANCHOR_SLACK = 60.0

# Budget fuse
SETUP_ALLOWANCE = 120.0
PER_QUESTION_BUDGET = 30.0
SELF_MARGIN = 45.0

PREAMBLE = (
    "You are a surgical assistant. You are given endoscopic video from a "
    "minimally invasive procedure. Analyze the footage and answer the surgical "
    "question based on the visual evidence. Be precise and concise.\n\n"
)

# x264 settings byte-identical to the training-side mini-clip encoder
# (-fps_mode passthrough keeps all frames under fractional -framerate).
ENC = ["-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast",
       "-crf", "23", "-g", "768", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
       "-an", "-movflags", "+faststart"]

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TS = re.compile(r"\b(\d{2}:\d{2}:\d{2})\b")
_TS_ANY = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")


def profile_for(question):
    """dense-mix regime for one question: {'nframes', 'budget', 'tree'}."""
    return PROFILES[policy_branch(question)]


# ──────────────── window rules (vendored verbatim from surgscope/routing.py) ────────────────
# Same rules as training-data generation and the local evaluation. Signatures are
# per question template, class- and dataset-agnostic; first match wins.
MIN_WIN = 240.0          # shortest window 4 min
TAIL = 3600.0            # tier-2 tail-window length
ABSTAIN = "outside clip window"

RULES = [
    ("insert_first", ("first inserted", "visible in the frame at"), lambda ts, e: (ts[0] - 1200, ts[0] + 60)),
    ("create_first", ("first created", "visible in the frame at"), lambda ts, e: (ts[0] - 1200, ts[0] + 60)),
    ("retrieve_when", ("when is it retrieved",), lambda ts, e: (ts[0] - 60, ts[0] + 1800)),
    ("between", ("seen between",), lambda ts, e: (min(ts) - 30, max(ts) + 30)),
    ("inserted_at", ("inserted or created", " at "), lambda ts, e: (ts[0] - 180, ts[0] + 180)),
    ("positions_at", ("relative central positions",), lambda ts, e: (ts[0] - 120, ts[0] + 120)),
    ("quadrant_first", ("which quadrant", "first occurs"), lambda ts, e: (0.0, ts[0] + 60)),
    ("also_appear", ("also appear at",), lambda ts, e: (min(ts) - 120, max(ts) + 120)),
    ("reappear", ("last visible just before", "re-appear"), lambda ts, e: (ts[0] - 180, e)),
    ("retrieval_exists", ("does a retrieval of this object exist",), lambda ts, e: (ts[0] - 60, e)),
    ("count_at", ("how many", "at time point"), lambda ts, e: (0.0, ts[0] + 60)),
]

TAIL_FAMILIES = [
    ("last_visible", ("last visible in the video",), "safe"),
    ("last_cooccur", ("last visible at the same time",), "safe"),
    ("final_retrieval", ("final and last retrieval",), "safe"),
    ("clips_list", ("at which time points were clips applied",), "safe"),
    ("first_visible", ("first visible in the video",), "first"),
    ("kth_insert", ("inserted in the abdomen for the first time",), "first"),
    ("first_cooccur", ("first visible at the same time",), "first"),
    ("leave_first", ("leave the surgical view for at least",), "first"),
]


def rule_timestamps(question):
    return [int(h) * 3600 + int(m) * 60 + int(s) for h, m, s in _TS_ANY.findall(question)]


def _clamp(w0, w1, end_s):
    w0, w1 = max(0.0, w0), min(float(end_s), w1)
    if w1 - w0 < MIN_WIN:                       # too short: grow around the centre, then snap to edges
        c = (w0 + w1) / 2
        w0, w1 = c - MIN_WIN / 2, c + MIN_WIN / 2
        if w0 < 0:
            w0, w1 = 0.0, min(float(end_s), MIN_WIN)
        if w1 > end_s:
            w1, w0 = float(end_s), max(0.0, end_s - MIN_WIN)
    return float(int(w0)), float(int(w1))


def route(question, end_s):
    """Three-tier routing -> dict(tier, rule, w0, w1, shrunk)."""
    end_s = float(end_s)
    ts = rule_timestamps(question)
    if ts:
        q = question.lower()
        for name, keys, fn in RULES:
            if all(k in q for k in keys):
                w0, w1 = _clamp(*fn(ts, end_s), end_s)
                return dict(tier=1, rule=name, w0=w0, w1=w1, shrunk=(w0 > 0 or w1 < end_s))
    else:
        q = question.lower()
        for name, keys, _ in TAIL_FAMILIES:
            if all(k in q for k in keys):
                w0, w1 = _clamp(end_s - TAIL, end_s, end_s)
                return dict(tier=2, rule=name, w0=w0, w1=w1, shrunk=(w0 > 0))
    return dict(tier=3, rule=None, w0=0.0, w1=end_s, shrunk=False)


def is_abstain(text):
    return ABSTAIN in str(text).lower()


# ─────────────────────────── text helpers ───────────────────────────
def clean(text):
    return _THINK.sub("", text or "").strip()


def hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def ts_seconds(s):
    h, m, sec = s.split(":")
    m, sec = int(m), int(sec)
    if not (0 <= m < 60 and 0 <= sec < 60):
        return None
    return int(h) * 3600 + m * 60 + sec


def first_timestamp(text):
    m = _TS.search(clean(text))
    return ts_seconds(m.group(1)) if m else None


def user_content(question, start_s, end_s, procedure_type):
    """Byte-identical to the training-side prompt builder (surgscope/prompts.py)."""
    pt = f"Procedure type: {procedure_type}.\n" if procedure_type else ""
    hint = f"Clip window: {hms(start_s)} - {hms(end_s)} (source-video timeline).\n"
    return "<video>" + pt + hint + question


def normalize_time_answer(ans):
    """Timestamp-only answers must be exactly one HH:MM:SS (format guard)."""
    a = clean(ans)
    if re.fullmatch(r"\d{2}:\d{2}:\d{2}", a):
        return a
    rest = re.sub(r"(?i)\band\b", "", _TS.sub("", a)).strip(" \t\n,;.&-")
    if rest:
        return a
    m = _TS.search(a)
    return m.group(1) if m else a


def sample_indices(total, n):
    """torch.linspace(0,total-1,n).round() replica (half-to-even) — as in training."""
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


# ─────────────────────────── video helpers ───────────────────────────
class Video:
    """One /input clip: header info + keyframe dead-mask (computed once per qID)."""

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
        """Decode keyframes only, 32x18 thumbnails + pts via showinfo. ~3 s per 4 h."""
        # -fps_mode passthrough is essential: without it ffmpeg re-times the output to
        # the source frame rate and DUPLICATES every keyframe ~25x (log shows dup=24),
        # which silently breaks the ordinal<->keyframe mapping used below.
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
        else:                                  # showinfo missing: assume 5 s GOP
            idx = np.arange(k, dtype=np.int64) * int(round(5 * CLIP_FPS))
        return idx, dead

    # ---- frame selection ------------------------------------------------
    def select(self, i0, i1, nframes):
        """(mode, indices): mode 'key' = keyframe ordinals; 'seq' = frame indices."""
        i0, i1 = max(0, i0), min(self.n_frames, i1)
        kmask = (self.key_idx >= i0) & (self.key_idx < i1) & (~self.key_dead)
        kords = np.flatnonzero(kmask)
        if kords.size >= nframes:            # tier A: dense enough in keyframes alone
            sel = kords[sample_indices(kords.size, nframes)]
            return "key", sel.tolist(), int(kords.size)
        cand = np.flatnonzero(~self.dead[i0:i1]) + i0
        if cand.size == 0:
            cand = np.arange(i0, i1)
        sel = cand[sample_indices(cand.size, nframes)] if cand.size > nframes else cand
        return "seq", sel.tolist(), int(cand.size)

    # ---- mini-clip encoding ----------------------------------------------
    def make_clip(self, dst, i0, i1, span_s, nframes, budget):
        """Sample nframes non-dead frames in [i0,i1) -> mini-clip at fps=n/span_s,
        resolution burned to pick_grid(src, budget) — as in training extraction."""
        mode, sel, n_cand = self.select(i0, i1, nframes)
        if not sel:
            raise RuntimeError("no frames selected")
        if mode == "key":
            first_ord, last_ord = sel[0], sel[-1]
            dec_cmd = [self.ffmpeg, "-nostdin", "-v", "error", "-skip_frame", "nokey",
                       "-i", self.path, "-fps_mode", "passthrough", "-frames:v", str(last_ord + 1),
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            keep = set(sel)
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
        tw, th = pick_grid(self.width, self.height, budget)
        fb = self.width * self.height * 3
        fps_out = f"{len(sel)}/{max(span_s, 1e-3):.3f}"
        tmp = str(dst) + ".tmp.mp4"
        dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        enc = subprocess.Popen(
            [self.ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.width}x{self.height}", "-framerate", fps_out, "-i", "-",
             "-vf", f"scale={tw}:{th}:flags=bicubic", *ENC, tmp],
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


def log_environment(torch):
    """Report the CUDA stack once at startup (from the official template)."""
    log.info("--- Environment ---")
    log.info("  torch          : %s (built against CUDA %s)", torch.__version__, torch.version.cuda)
    arch = torch.cuda.get_arch_list() or (torch._C._cuda_getArchFlags() or "").split()
    log.info("  torch kernels  : %s", " ".join(arch) or "unknown")
    drv = Path("/proc/driver/nvidia/version")
    log.info("  host driver    : %s", drv.read_text().strip().splitlines()[0] if drv.exists() else "none visible")
    if not torch.cuda.is_available():
        log.warning("  No GPU visible — CPU only")
        return
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    log.info("  GPU            : %s | sm_%d%d | %.1f GiB free of %.1f GiB",
             torch.cuda.get_device_name(0), *cap, free / 1024**3, total / 1024**3)
    ncpu = os.cpu_count()
    log.info("  CPUs visible   : %s", ncpu)
    # /input layout — the one line that pinpoints data-side failures in try-out logs
    for d in ("plain", "overlayed"):
        p = INPUT_PATH / d
        log.info("  /input/%-10s: %s", d, f"{len(list(p.glob('*.mp4')))} clip(s)" if p.is_dir() else "MISSING")


# ─────────────────────────────── main ───────────────────────────────
def run() -> int:
    t_start = time.monotonic()
    log.info("=== SurgScope — ORena SAVE FOCUS PROCEDURE track ===")

    from focus import Response, load_requests, save_items
    import torch
    from swift import InferRequest, RequestConfig, TransformersEngine
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:            # local dry-run outside the image
        get_ffmpeg_exe = lambda: os.environ.get("FFMPEG_BIN", "ffmpeg")

    log_environment(torch)
    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        log.error("request.json contains no requests")
        return 1
    B = len(requests)
    allowed = SETUP_ALLOWANCE + B * PER_QUESTION_BUDGET
    deadline = allowed - SELF_MARGIN
    log.info("Batch of %d question(s); allowed %.0f s, self-deadline %.0f s", B, allowed, deadline)

    system = PREAMBLE + json.loads((INPUT_PATH / "FO_definitions.json").read_text())
    log.info("System prompt: %d chars (FO definitions taken from /input)", len(system))

    ffmpeg = get_ffmpeg_exe()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    def elapsed():
        return time.monotonic() - t_start

    # ---- Model: loaded once per run --------------------------------------
    log.info("Loading engine (merged model=%s)", MODEL_PATH)
    engine = TransformersEngine(str(MODEL_PATH), model_type="qwen3_5",
                                torch_dtype=torch.bfloat16, attn_impl="sdpa",
                                max_batch_size=1)
    req_cfg = RequestConfig(max_tokens=MAX_NEW_TOKENS, temperature=0.0)
    log.info("Engine ready (setup %.1f s)", elapsed())

    def build_ir(req, clip, w0, w1):
        n = profile_for(req.question)["nframes"]
        return InferRequest(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user_content(req.question, w0, w1, req.procedure_type)}],
            videos=[str(clip)],
            chat_template_kwargs={"enable_thinking": False,
                                  "min_frames": n, "max_frames": n})

    def infer_one(ir, tag):
        try:
            r = engine.infer([ir], req_cfg, use_tqdm=False)
            return r[0].choices[0].message.content
        except Exception:
            log.exception("[%s] inference failed; empty answer", tag)
            return ""

    # Warmup at the largest trained shape (mid: 1152 frames @84 tok/pair ~48k
    # tokens): absorbs CUDA context, Triton autotune and cuDNN heuristics inside
    # the setup allowance instead of on the first real question.
    try:
        wpath = WORK_DIR / "_warmup.mp4"
        subprocess.run([ffmpeg, "-nostdin", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "color=c=gray:size=384x224:rate=5", "-t", f"{1152 / 5:.1f}",
                        "-frames:v", "1152", *ENC, str(wpath)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t_w = time.monotonic()
        engine.infer([InferRequest(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user_content("Is a foreign object visible?", 0, 1800, None)}],
            videos=[str(wpath)],
            chat_template_kwargs={"enable_thinking": False,
                                  "min_frames": 1152, "max_frames": 1152})],
            RequestConfig(max_tokens=4, temperature=0.0), use_tqdm=False)
        log.info("Warmup generation done (%.1f s); setup total %.1f s", time.monotonic() - t_w, elapsed())
    except Exception:
        log.exception("Warmup failed (continuing)")

    # ---- Pass 1: routed window per question, decode prefetched ---------------
    videos = {}
    routes = {}

    def prep_window(req, w0, w1, tag):
        """Video scan (cached per qID) + regime mini-clip for [w0, w1] seconds."""
        try:
            v = videos.get(req.qID)
            if v is None:
                v = Video(ffmpeg, VIDEO_DIR / f"{req.qID}.mp4", req.end_time)
                videos[req.qID] = v
            prof = profile_for(req.question)
            i0 = max(0, int(round(w0 * CLIP_FPS)))
            i1 = min(v.n_frames, int(round(w1 * CLIP_FPS)))
            dst = WORK_DIR / f"{req.qID}_{tag}.mp4"
            meta = v.make_clip(dst, i0, i1, max(w1 - w0, 1e-3), prof["nframes"], prof["budget"])
            return dst, meta, None
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] clip prep failed for %s", tag, req.qID)
            return None, None, repr(e)

    def prep_pass1(req):
        rt = route(req.question, float(req.end_time))
        routes[req.qID] = rt
        return prep_window(req, rt["w0"], rt["w1"], "p1")

    pool = ThreadPoolExecutor(max_workers=1)
    answers = [""] * B
    t_p1 = elapsed()
    fut = pool.submit(prep_pass1, requests[0])
    n_abstain = 0
    for i, req in enumerate(requests):
        t_q = time.monotonic()
        clip, meta, err = fut.result()
        if i + 1 < B:
            fut = pool.submit(prep_pass1, requests[i + 1])   # decode next while GPU runs
        t_prep = time.monotonic() - t_q
        rt = routes.get(req.qID) or dict(tier=3, rule=None, w0=0.0, w1=float(req.end_time), shrunk=False)
        if clip is None:
            log.error("[pass1] %s: no clip (%s) -> empty answer", req.qID, err)
            continue
        raw = infer_one(build_ir(req, clip, int(rt["w0"]), int(rt["w1"])), "pass1")
        answers[i] = clean(raw)
        try:
            os.remove(clip)
        except OSError:
            pass
        # Abstain fallback: shrunk-window answer says "outside clip window" -> full prefix
        if rt["shrunk"] and is_abstain(answers[i]) and elapsed() + 30.0 < deadline:
            n_abstain += 1
            fclip, _, ferr = prep_window(req, 0.0, float(req.end_time), "p1full")
            if fclip is not None:
                raw2 = infer_one(build_ir(req, fclip, 0, int(req.end_time)), "abstain")
                a2 = clean(raw2)
                if a2:
                    answers[i] = a2
                try:
                    os.remove(fclip)
                except OSError:
                    pass
        log.info("[pass1] %d/%d %s tier=%s rule=%s win=%s-%s regime=%s mode=%s frames=%d/%d prep-wait=%.1fs total=%.1fs -> %r  (elapsed %.0f/%.0f s)",
                 i + 1, B, req.qID, rt["tier"], rt["rule"], hms(rt["w0"]), hms(rt["w1"]),
                 policy_branch(req.question), meta["mode"], meta["n"], meta["n_cand"], t_prep,
                 time.monotonic() - t_q, answers[i][:40], elapsed(), deadline)
    per_q = max(1.0, (elapsed() - t_p1) / max(B, 1))     # measured pass-1 cost per question
    log.info("Pass 1 done: %d abstain fallback(s); %.1f s/question", n_abstain, per_q)

    # ---- Zoom routing (answer-format self-routing) ---------------------------
    def zoom_anchor(req, ans):
        if "how long" in req.question.lower():        # duration question: not a timestamp
            return None
        t = first_timestamp(ans)
        if t is None:
            return None
        if t < -ANCHOR_SLACK or t > float(req.end_time) + ANCHOR_SLACK:
            return None
        return min(max(float(t), 0.0), float(req.end_time))

    def window_around(end_s, t, half):
        z0, z1 = max(0.0, t - half), min(end_s, t + half)
        want = min(2 * half, end_s)
        if z1 - z0 < want:
            if z0 <= 0:
                z1 = want
            else:
                z0 = end_s - want
        return z0, z1

    zoom = [(i, zoom_anchor(r, answers[i])) for i, r in enumerate(requests)]
    zoom = [(i, t) for i, t in zoom if t is not None]
    log.info("Zoom candidates: %d/%d (elapsed %.0f s, deadline %.0f s)", len(zoom), B, elapsed(), deadline)

    def zoom_pass(cands, half, tag, est_per):
        """Cut sub-window clips (prefetched) + re-infer; budget-gated per question."""
        out = {}
        if not cands:
            return out
        def prep(item):
            i, t = item
            req = requests[i]
            end_s = float(req.end_time)
            z0, z1 = window_around(end_s, t, half)
            clip, _, err = prep_window(req, z0, z1, tag)
            return i, clip, z0, z1
        f = pool.submit(prep, cands[0])
        for j, item in enumerate(cands):
            i, clip, z0, z1 = f.result()
            if j + 1 < len(cands):
                f = pool.submit(prep, cands[j + 1])
            if elapsed() + est_per > deadline:
                log.warning("[%s] budget: skipping remaining %d question(s)", tag, len(cands) - j)
                if clip is not None:
                    try: os.remove(clip)
                    except OSError: pass
                break
            if clip is None:
                continue
            raw = infer_one(build_ir(requests[i], clip, z0, z1), tag)
            out[i] = clean(raw)
            try:
                os.remove(clip)
            except OSError:
                pass
        return out

    # ---- Pass 2 (+-10 min) ---------------------------------------------------
    est2 = per_q * 0.9
    if zoom and elapsed() + est2 < deadline:
        for i, a in zoom_pass(zoom, ZOOM_HALF_1, "pass2", est2).items():
            if a:
                answers[i] = a
    elif zoom:
        log.warning("Skipping pass 2 entirely (elapsed %.0f s)", elapsed())
        zoom = []
    log.info("Pass 2 done (elapsed %.0f s)", elapsed())

    # ---- Pass 3 (+-2 min around the pass-2 answer) ---------------------------
    cands3 = []
    for i, _ in zoom:
        req = requests[i]
        t = first_timestamp(answers[i])
        if t is None or not (0 <= t <= float(req.end_time)):
            continue                      # nothing sensible to refine
        cands3.append((i, float(t)))
    est3 = per_q * 0.7
    if cands3 and elapsed() + est3 < deadline:
        for i, a in zoom_pass(cands3, ZOOM_HALF_2, "pass3", est3).items():
            if first_timestamp(a) is not None:   # pass 3 must yield a timestamp
                answers[i] = a
    elif cands3:
        log.warning("Skipping pass 3 entirely (elapsed %.0f s)", elapsed())
    log.info("Pass 3 done (elapsed %.0f s)", elapsed())
    pool.shutdown(wait=False)

    # ---- Format guard + output ------------------------------------------------
    contents = []
    for req, ans in zip(requests, answers):
        a = clean(ans)
        if _TS.search(a):
            a = normalize_time_answer(a)
        contents.append(a)
    total = elapsed()
    responses = [Response(qID=r.qID, content=c, latency=total / max(B, 1))
                 for r, c in zip(requests, contents)]
    n_empty = sum(1 for r in responses if not r.content.strip())
    if n_empty:
        log.error("ALERT: %d/%d answers are EMPTY — check exception logs above", n_empty, len(responses))
    for r in responses[:5]:
        log.info("  %s -> %r", r.qID, r.content)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    save_items(responses, OUTPUT_PATH / "answer.json")
    log.info("Wrote %d response(s); total %.1f s (allowed %.0f s)", len(responses), total, allowed)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
