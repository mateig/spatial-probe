"""Flat-output loader for generated scene samples and their assets."""

import json
from pathlib import Path
from typing import Iterator

from PIL import Image


def _prefixes(root: Path):
    return sorted(p.stem.removesuffix("_scene") for p in root.glob("*_scene.json"))


def load_scene(root: Path, prefix: str) -> dict:
    scene_json = json.loads((root / f"{prefix}_scene.json").read_text())
    return {
        "scene_id": prefix,
        "rgb": Image.open(root / f"{prefix}_rgb.png").convert("RGB"),
        "depth": Image.open(root / f"{prefix}_depth.png"),
        "description": (root / f"{prefix}_description.txt").read_text(),
        "regime": scene_json["regime"],
        "qa": json.loads((root / f"{prefix}_qa.json").read_text()),
        "scene_json": scene_json,
    }


def iter_qa_samples(root: Path) -> Iterator[tuple[dict, str, str]]:
    """Yield each QA pair with the full scene asset bundle it belongs to."""
    for prefix in _prefixes(root):
        scene = load_scene(root, prefix)
        for qa in scene["qa"]:
            yield (scene, qa["question"], qa["answer"])
