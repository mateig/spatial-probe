from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.config import ModelSpec

Key = tuple[str, str, int, str]


def make_row(
    spec: ModelSpec,
    scene_id: str,
    question_id: int,
    question_text: str,
    modalities: tuple[str, ...],
    response: str | None,
    error: str | None,
    elapsed_s: float,
) -> dict:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "model_key": spec.key,
        "model_id": spec.model_id,
        "scene_id": scene_id,
        "question_id": question_id,
        "modalities": list(modalities),
        "modality_key": "+".join(modalities),
        "question": question_text,
        "response": response,
        "error": error,
        "elapsed_s": round(elapsed_s, 4),
    }


def completed(path: Path) -> set[Key]:
    if not path.exists():
        return set()
    seen: set[Key] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("error") is None:
                seen.add(
                    (
                        row["model_key"],
                        row["scene_id"],
                        int(row["question_id"]),
                        row["modality_key"],
                    )
                )
    return seen


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
