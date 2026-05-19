"""Load gold-standard QA data and VLM responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def _scene_ids(root: Path) -> list[str]:
    return sorted(p.stem.removesuffix("_qa") for p in root.glob("*_qa.json"))


def load_questions(root: Path, scene_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """Return per-question gold metadata: scene_id, question_id, type, relation, answer, regime, n_objects, scene-level features."""
    ids = list(scene_ids) if scene_ids is not None else _scene_ids(root)
    rows: list[dict] = []
    for sid in ids:
        qa_path = root / f"{sid}_qa.json"
        scene_path = root / f"{sid}_scene.json"
        if not qa_path.exists() or not scene_path.exists():
            continue
        scene = json.loads(scene_path.read_text())
        objs = scene["objects"]
        n_objects = len(objs)
        regime = scene.get("regime", "unknown")
        # Pre-compute scene object summary used by scoring (color, shape, id) and label index.
        obj_summary = [
            {"id": o["id"], "color": o["color"], "shape": o["shape"], "size": o.get("size")}
            for o in objs
        ]
        for q in json.loads(qa_path.read_text()):
            rows.append(
                {
                    "scene_id": sid,
                    "question_id": int(q["question_id"]),
                    "question": q["question"],
                    "question_type": q["question_type"],
                    "relation_type": q.get("relation_type"),
                    "gold": str(q["answer"]).strip().lower(),
                    "regime": regime,
                    "n_objects": n_objects,
                    "objects": obj_summary,
                }
            )
    return pd.DataFrame(rows)


def load_responses(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["question_id"] = df["question_id"].astype(int)
    df["response"] = df["response"].fillna("")
    return df


def write_jsonl(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True)
