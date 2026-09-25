"""ms-swift training rows (JSONL).

One row per training question::

    {"messages": [{"role": "system", "content": PREAMBLE + definitions},
                  {"role": "user", "content": "<video>Procedure type: ...\\nClip window: ...\\n<question>"},
                  {"role": "assistant", "content": "<answer or 'outside clip window'>"}],
     "videos": ["<mini-clip>.mp4"],
     "chat_template_kwargs": {"min_frames": N, "max_frames": N}}     # dense-mix regimes only

The window line states the routed window when it is shortened, else [start, end]; a time
answer outside a shortened window is replaced by the abstention literal. Rows are
shuffled with seed 42. A sidecar JSONL carries per-row metadata for analysis.
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

from surgscope import policy
from surgscope.data.extract import hms_to_s
from surgscope.prompts import system_prompt, user_content
from surgscope.routing import route, train_label

SEED = 42


def build_rows(items, regime: str, clip_for, fo_definitions: str, max_frames: int | None = None):
    """items: parquet rows (dicts) with an extra 'ds' key, in a fixed order.

    ``clip_for(item) -> Path`` gives the mini-clip of a question. Returns (rows, missing).
    """
    system = system_prompt(fo_definitions)
    per_row = policy.REGIME_META[regime]["per_row_kwargs"]
    rows, missing = [], []
    for r in items:
        start, end = hms_to_s(r["timestamp_start"]), hms_to_s(r["timestamp_end"])
        rt = route(r["question"], end)
        clip = Path(clip_for(r))
        if not (clip.exists() and clip.stat().st_size > 0):
            missing.append(str(clip))
            continue
        label, in_win = train_label(r["answer"], r["answer_format"], rt)
        hint_s, hint_e = (int(rt["w0"]), int(rt["w1"])) if rt["shrunk"] else (start, end)
        sample = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content(r["question"], hint_s, hint_e,
                                                         procedure_type=r.get("procedure_type"))},
                {"role": "assistant", "content": label},
            ],
            "videos": [str(clip.resolve())],
        }
        n = policy.nframes(r["question"], regime)
        if max_frames:
            n = min(n, max_frames)
        if per_row:
            sample["chat_template_kwargs"] = {"min_frames": n, "max_frames": n}
        side = {"dataset": r["ds"], "qid": int(r["id"]), "video": r["video"],
                "answer_format": r["answer_format"], "primary_capability": r["primary_capability"],
                "ood": bool(r.get("ood", False)), "timestamp_start": r["timestamp_start"],
                "timestamp_end": r["timestamp_end"], "tier": rt["tier"], "rule": rt["rule"],
                "w0": rt["w0"], "w1": rt["w1"], "shrunk": rt["shrunk"], "gt_in_window": in_win,
                "abstain": label != r["answer"], "branch": policy.branch(r["question"]),
                "nframes": n}
        rows.append((sample, side))
    return rows, missing


def write_jsonl(rows, out) -> dict:
    """Shuffle (seed 42) and write ``out`` plus ``out.sidecar.jsonl``; returns a summary."""
    rows = list(rows)
    random.Random(SEED).shuffle(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    side = out.with_name(out.stem + ".sidecar.jsonl")
    with open(out, "w") as f, open(side, "w") as g:
        for s, sc in rows:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            g.write(json.dumps(sc, ensure_ascii=False) + "\n")
    return {"rows": len(rows),
            "formats": dict(collections.Counter(sc["answer_format"] for _, sc in rows)),
            "abstain_labels": sum(sc["abstain"] for _, sc in rows),
            "shrunk": sum(sc["shrunk"] for _, sc in rows)}
