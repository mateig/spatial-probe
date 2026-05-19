"""Summarize flat stage-0 style outputs."""

import argparse
import json
from pathlib import Path


def _prefixes(root: Path):
    return sorted(p.stem.removesuffix("_scene") for p in root.glob("*_scene.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output"))
    args = parser.parse_args()

    scene_count = 0
    qa_count = 0
    per_regime = {}
    per_type = {}
    for prefix in _prefixes(args.root):
        scene = json.loads((args.root / f"{prefix}_scene.json").read_text())
        qas = json.loads((args.root / f"{prefix}_qa.json").read_text())
        scene_count += 1
        qa_count += len(qas)
        per_regime[scene["regime"]] = per_regime.get(scene["regime"], 0) + 1
        for qa in qas:
            qt = qa["question_type"]
            per_type[qt] = per_type.get(qt, 0) + 1

    print(f"scenes: {scene_count}")
    print(f"qa pairs: {qa_count}")
    print(f"per regime: {per_regime}")
    print(f"per question type: {per_type}")


if __name__ == "__main__":
    main()
