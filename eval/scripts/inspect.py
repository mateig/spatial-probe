"""Print headline metrics from a previous run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    cfg = config.load(args.config)
    if not cfg.metrics_dir.exists():
        raise SystemExit(f"no metrics at {cfg.metrics_dir}; run `make run` first")

    overall = pd.read_csv(cfg.metrics_dir / "overall.csv").sort_values("accuracy", ascending=False)
    by_qtype = pd.read_csv(cfg.metrics_dir / "by_qtype.csv")
    by_modality = pd.read_csv(cfg.metrics_dir / "by_modality.csv")
    latency = pd.read_csv(cfg.metrics_dir / "latency.csv")

    print("== overall accuracy ==")
    for _, r in overall.iterrows():
        print(f"  {r['model_key']:<22} {r['accuracy']:.4f}  (n={int(r['n']):,})")

    print()
    print("== accuracy by question type (RGB rows averaged across qtypes shown) ==")
    pivot = by_qtype.pivot_table(index="question_type", columns="model_key", values="accuracy")
    print(pivot.round(3).to_string())

    print()
    print("== accuracy by modality ==")
    pivot = by_modality.pivot_table(index="modality_key", columns="model_key", values="accuracy")
    print(pivot.round(3).to_string())

    print()
    print("== median latency (s/question) ==")
    print(latency[["model_key", "median"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
