"""Scoring with the official challenge metric (``orena-focus``).

Each answer is scored by its answer format (binary, number, percentage, class set, time
with a 1-5 s tolerance, ...); open-ended, matching and multiple-choice answers are judged
by an LLM (Qwen3.5-4B by default). The challenge score is the unweighted mean over the
populated buckets of (5 capability groups) x (in-distribution, out-of-distribution), each
bucket being the flat per-question accuracy.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

log = logging.getLogger("surgscope")

CONVENTION = ("flat per-question accuracy inside each (capability group x ID/OOD) bucket, "
              "unweighted mean over the populated buckets (official challenge score)")


def _load_rows(references):
    import pandas as pd
    frames = [pd.read_parquet(p) for p in references]
    return pd.concat(frames, ignore_index=True).to_dict("records")


def evaluate(predictions, references, judge: str = "auto", output_dir=None,
             only_answered: bool = True) -> dict:
    """Score one or more ``answer.json`` files against parquet references.

    judge: 'auto' (default LLM judge, Qwen/Qwen3.5-4B on GPU if available), 'none'
    (skip judge-scored formats) or a model id / path for the judge.
    """
    from focus import Evaluator, Response
    from focus.data.base_dataset import FocusDataset
    from focus.data.formats import JUDGE_FORMATS

    if isinstance(predictions, (str, Path)):
        predictions = [predictions]
    preds = {str(r["qID"]): r for p in predictions for r in json.loads(Path(p).read_text())}
    rows = _load_rows(references)
    requests, refs, skipped = [], [], []
    for row in rows:
        req, ref = FocusDataset._parse_row(row)
        ref = dataclasses.replace(ref, ood=bool(row.get("ood", False)))
        if only_answered and req.qID not in preds:
            continue
        if judge == "none" and ref._format in JUDGE_FORMATS:
            skipped.append(req.qID)
            continue
        requests.append(req)
        refs.append(ref)
    if not requests:
        raise ValueError("no prediction matches a reference id (answer.json qIDs must be the "
                         "dataset ids)")
    responses = [Response(qID=q, content=str(preds[q]["content"]),
                          latency=float(preds[q].get("latency", 0.0)))
                 for q in (r.qID for r in requests) if q in preds]
    judges = None
    if judge not in ("auto", "none"):
        import torch
        from focus.evaluation.judges import TransformersJudge
        judges = [TransformersJudge(model_name=judge, device="cuda" if torch.cuda.is_available() else "cpu")]
    elif judge == "auto" and any(r._format in JUDGE_FORMATS for r in refs):
        import torch
        from focus.evaluation.judges import TransformersJudge
        judges = [TransformersJudge(device="cuda" if torch.cuda.is_available() else "cpu")]
    ev = Evaluator(judges=judges)
    results, _ = ev.run(requests, refs, responses, output_dir=output_dir)
    score, buckets = ev.pre_evaluation_score(results)
    ref_by_id = {r.qID: r for r in refs}
    per_q = [{"qID": r["qID"], "format": r["answer_format"], "capability": r["primary"],
              "ood": bool(r["ood"]), "answer": str(preds[r["qID"]]["content"]) if r["qID"] in preds else None,
              "reference": ref_by_id[r["qID"]].answer, "correct": bool(r["correctness"])}
             for r in results.to_dict("records")]
    return {"score": float(score), "convention": CONVENTION, "n": len(per_q),
            "skipped_judge_formats": skipped,
            "buckets": buckets.to_dict("records"), "questions": per_q}


def format_report(res: dict) -> str:
    lines = []
    for q in res["questions"]:
        mark = "correct" if q["correct"] else "wrong"
        lines.append(f"  {q['qID']:>10}  [{q['format']}]  answer={q['answer']!r}  "
                     f"reference={q['reference']!r}  -> {mark}")
    lines.append("  buckets: " + ", ".join(
        f"{b['group']}/{'OOD' if b['ood'] else 'ID'}={b['accuracy']:.3f} (n={b['count']})"
        for b in res["buckets"]))
    lines.append(f"  score = {res['score']:.4f} over {res['n']} question(s)  [{res['convention']}]")
    if res["skipped_judge_formats"]:
        lines.append(f"  skipped (judge formats, --judge none): {len(res['skipped_judge_formats'])}")
    return "\n".join(lines)
