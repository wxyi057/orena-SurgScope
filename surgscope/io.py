"""Challenge input/output layout.

Input directory (one batch)::

    request.json          [{qID, videoID, start_time, end_time, procedure_type, question}, ...]
    FO_definitions.json   object-class definitions (a JSON-encoded string)
    plain/<qID>.mp4       prefix [0, end_time] of the source video, 5 fps, height <= 576
    overlayed/<qID>.mp4   the same with a burned-in HH:MM:SS clock (what SurgScope reads)

Output: ``answer.json`` = [{qID, content, latency}, ...].
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Question:
    qID: str
    question: str
    end_time: float
    procedure_type: str | None = None
    video: str | None = None          # path to the overlayed clip
    videoID: str = ""
    start_time: float = 0.0


def read_input_dir(input_dir, variant: str = "overlayed") -> list[Question]:
    d = Path(input_dir)
    rows = json.loads((d / "request.json").read_text())
    out = []
    for r in rows:
        out.append(Question(
            qID=str(r["qID"]), question=r["question"], end_time=float(r["end_time"]),
            procedure_type=r.get("procedure_type") or None,
            video=str(d / variant / f"{r['qID']}.mp4"),
            videoID=r.get("videoID", ""), start_time=float(r.get("start_time", 0.0))))
    return out


def write_answers(path, qids, contents, latency: float = 0.0) -> None:
    rows = [{"qID": q, "content": c, "latency": latency} for q, c in zip(qids, contents)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def question_to_request(q: Question) -> dict:
    d = asdict(q)
    d.pop("video")
    d["procedure_type"] = d["procedure_type"] or ""
    return d
