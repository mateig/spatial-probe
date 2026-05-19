from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Question:
    id: int
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Scene:
    id: str
    rgb_path: Path
    depth_path: Path
    description: str
    questions: tuple[Question, ...]


def discover(root: Path) -> list[str]:
    return sorted(p.stem.removesuffix("_qa") for p in root.glob("*_qa.json"))


def select(
    all_ids: list[str],
    explicit: tuple[str, ...] | None,
    start: int,
    limit: int | None,
) -> list[str]:
    if explicit:
        return list(explicit)
    chosen = all_ids[start:]
    if limit is not None:
        chosen = chosen[:limit]
    return chosen


def load(root: Path, scene_id: str) -> Scene:
    qa = root / f"{scene_id}_qa.json"
    rgb = root / f"{scene_id}_rgb.png"
    depth = root / f"{scene_id}_depth.png"
    desc = root / f"{scene_id}_description.txt"

    missing = [str(p) for p in (qa, rgb, depth, desc) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for {scene_id}: {missing}")

    items = json.loads(qa.read_text())
    questions = tuple(
        Question(
            id=int(item["question_id"]),
            text=item["question"],
            metadata={
                k: v
                for k, v in item.items()
                if k not in {"question_id", "question", "answer"}
            },
        )
        for item in items
    )
    return Scene(
        id=scene_id,
        rgb_path=rgb,
        depth_path=depth,
        description=desc.read_text().strip(),
        questions=questions,
    )
