"""Aggregate scored rows into metric tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _acc_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by, dropna=False).agg(
        n=("is_correct", "size"),
        n_correct=("is_correct", "sum"),
    )
    g["accuracy"] = g["n_correct"] / g["n"]
    return g.reset_index()


def write_all(df: pd.DataFrame, metrics_dir: Path) -> dict[str, pd.DataFrame]:
    _ensure_dir(metrics_dir)
    out: dict[str, pd.DataFrame] = {}

    out["overall"] = _acc_table(df, ["model_key"])
    out["by_modality"] = _acc_table(df, ["model_key", "modality_key"])
    out["by_qtype"] = _acc_table(df, ["model_key", "question_type"])
    out["by_relation"] = _acc_table(df, ["model_key", "question_type", "relation_type"])
    out["by_regime"] = _acc_table(df, ["model_key", "regime"])
    out["by_n_objects"] = _acc_table(df, ["model_key", "n_objects"])
    out["modality_qtype"] = _acc_table(df, ["model_key", "modality_key", "question_type"])
    out["modality_regime"] = _acc_table(df, ["model_key", "modality_key", "regime"])

    # Latency summary
    lat = df.groupby("model_key")["elapsed_s"].agg(["mean", "median", "std"]).reset_index()
    out["latency"] = lat

    # Yes-bias for binary types (over rgb-only modality, the universally supported one).
    bin_types = {"binary_relation", "existence", "distance_comparison", "same_side"}
    bdf = df[df["question_type"].isin(bin_types) & (df["modality_key"] == "rgb")].copy()
    if not bdf.empty:
        bdf["said_yes"] = bdf["parsed"] == "yes"
        bdf["gold_yes"] = bdf["gold_norm"] == "yes"
        yb = bdf.groupby(["model_key", "question_type"]).agg(
            n=("said_yes", "size"),
            said_yes_rate=("said_yes", "mean"),
            gold_yes_rate=("gold_yes", "mean"),
        ).reset_index()
        out["yes_bias"] = yb

    for name, table in out.items():
        table.to_csv(metrics_dir / f"{name}.csv", index=False)
    return out
