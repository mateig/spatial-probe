"""Generate eval visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _setup(palette: str) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette(palette)


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def overall_accuracy(df: pd.DataFrame, out: Path, dpi: int) -> None:
    """RGB-only headline so all 5 models are compared on identical inputs.
    PaliGemma cannot run rgb+depth modalities; mixing modalities into a single
    'overall' would compare models on different sample sets."""
    rgb = df[df["modality_key"] == "rgb"]
    g = rgb.groupby("model_key")["is_correct"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(g.index, g.values, color=sns.color_palette("viridis", len(g)))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title("Headline accuracy by model (RGB only)")
    for b, v in zip(bars, g.values):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    _save(fig, out, dpi)


def accuracy_by_qtype(df: pd.DataFrame, out: Path, dpi: int) -> None:
    rgb = df[df["modality_key"] == "rgb"]
    g = rgb.groupby(["question_type", "model_key"])["is_correct"].mean().unstack("model_key")
    g = g.sort_values(by=g.columns[0])
    fig, ax = plt.subplots(figsize=(13, 7))
    g.plot(kind="bar", ax=ax, width=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by question type (RGB only)")
    ax.legend(title="model", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    _save(fig, out, dpi)


def heatmap_model_qtype(df: pd.DataFrame, out: Path, dpi: int) -> None:
    rgb = df[df["modality_key"] == "rgb"]
    pivot = rgb.pivot_table(index="model_key", columns="question_type", values="is_correct", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "accuracy"})
    ax.set_title("Model × question type accuracy (RGB only)")
    _save(fig, out, dpi)


def heatmap_model_modality(df: pd.DataFrame, out: Path, dpi: int) -> None:
    pivot = df.pivot_table(index="model_key", columns="modality_key", values="is_correct", aggfunc="mean")
    cols = [c for c in ("rgb", "rgb+depth", "rgb+description", "rgb+depth+description") if c in pivot.columns]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "accuracy"}, annot_kws={"size": 13})
    ax.set_title("Model × modality accuracy (all question types). White = modality unsupported.")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.tick_params(axis="y", rotation=0)
    _save(fig, out, dpi)


def modality_lift(df: pd.DataFrame, out: Path, dpi: int) -> None:
    base = df[df["modality_key"] == "rgb"].groupby("model_key")["is_correct"].mean()
    rows: list[dict] = []
    for mod in ["rgb+depth", "rgb+description", "rgb+depth+description"]:
        accs = df[df["modality_key"] == mod].groupby("model_key")["is_correct"].mean()
        for m, v in accs.items():
            if m in base.index:
                rows.append({"model_key": m, "modality": mod, "lift": v - base[m]})
    if not rows:
        return
    lift = pd.DataFrame(rows)
    pivot = lift.pivot(index="model_key", columns="modality", values="lift")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(pivot, annot=True, fmt="+.3f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "Δ accuracy vs RGB"})
    ax.set_title("Modality lift over RGB-only baseline")
    _save(fig, out, dpi)


def accuracy_by_regime(df: pd.DataFrame, out: Path, dpi: int) -> None:
    rgb = df[df["modality_key"] == "rgb"]
    pivot = rgb.pivot_table(index="regime", columns="model_key", values="is_correct", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(13, 6))
    pivot.plot(kind="bar", ax=ax, width=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by scene regime (RGB only)")
    ax.legend(title="model", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.xticks(rotation=20, ha="right")
    _save(fig, out, dpi)


def accuracy_by_n_objects(df: pd.DataFrame, out: Path, dpi: int) -> None:
    rgb = df[df["modality_key"] == "rgb"]
    g = rgb.groupby(["n_objects", "model_key"])["is_correct"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.lineplot(data=g, x="n_objects", y="is_correct", hue="model_key", marker="o", ax=ax, linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of objects in scene")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs scene complexity (RGB only)")
    ax.legend(title="model", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    _save(fig, out, dpi)


def accuracy_by_relation(df: pd.DataFrame, out: Path, dpi: int) -> None:
    rgb = df[df["modality_key"] == "rgb"]
    relations = rgb[rgb["question_type"] == "binary_relation"]
    if relations.empty:
        return
    pivot = relations.pivot_table(index="relation_type", columns="model_key", values="is_correct", aggfunc="mean")
    order = ["left_of", "right_of", "in_front_of", "behind", "closer_than", "farther_than"]
    pivot = pivot.reindex([r for r in order if r in pivot.index])
    fig, ax = plt.subplots(figsize=(13, 5.5))
    pivot.plot(kind="bar", ax=ax, width=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Binary-relation accuracy by relation type (RGB only)")
    ax.legend(title="model", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    plt.xticks(rotation=15, ha="right")
    _save(fig, out, dpi)


def yes_bias(df: pd.DataFrame, out: Path, dpi: int) -> None:
    bin_types = ["binary_relation", "existence", "distance_comparison", "same_side"]
    bdf = df[df["question_type"].isin(bin_types) & (df["modality_key"] == "rgb")].copy()
    if bdf.empty:
        return
    bdf["said_yes"] = (bdf["parsed"] == "yes").astype(int)
    bdf["gold_yes"] = (bdf["gold_norm"] == "yes").astype(int)
    said = bdf.groupby(["model_key", "question_type"])["said_yes"].mean().reset_index()
    gold = bdf.groupby("question_type")["gold_yes"].mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=said, x="question_type", y="said_yes", hue="model_key", ax=ax, order=bin_types, errorbar=None)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(model says yes)")
    ax.set_title("Yes-bias on binary questions (RGB only). Black ticks = gold P(yes).")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)

    # Overlay gold_yes_rate as a horizontal black tick per question type
    xticks = ax.get_xticks()
    for x, qt in zip(xticks, bin_types):
        if qt in gold.index:
            ax.hlines(gold[qt], x - 0.4, x + 0.4, colors="black", linewidth=2.5)
    ax.legend(title="model", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    _save(fig, out, dpi)


def latency(df: pd.DataFrame, out: Path, dpi: int) -> None:
    g = df.groupby("model_key")["elapsed_s"].median().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(g.index, g.values, color=sns.color_palette("magma", len(g)))
    for b, v in zip(bars, g.values):
        ax.text(v + g.max() * 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}s", va="center")
    ax.set_xlabel("Median seconds per question")
    ax.set_title("Per-question inference latency (median)")
    _save(fig, out, dpi)


def shape_confusion(df: pd.DataFrame, out: Path, dpi: int) -> None:
    """For object-label questions, what shape did the model say (vs gold shape)?"""
    obj_types = {"directional_extremum", "camera_extremum", "between"}
    sub = df[df["question_type"].isin(obj_types) & (df["modality_key"] == "rgb")].copy()
    if sub.empty:
        return
    import re
    from src.score import SHAPE_ALIASES, _gold_object

    def detect_shape(s: str) -> str:
        for t in re.findall(r"[a-z]+", s.lower()):
            if t in SHAPE_ALIASES:
                return SHAPE_ALIASES[t]
        return "none"

    sub["pred_shape"] = sub["response"].fillna("").map(detect_shape)
    sub["gold_shape"] = sub["gold_norm"].map(lambda g: (_gold_object(g)[1] or "none"))
    n = sub["model_key"].nunique()
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (m, g) in zip(axes, sub.groupby("model_key")):
        cm = pd.crosstab(g["gold_shape"], g["pred_shape"], normalize="index")
        order = ["sphere", "cube", "cylinder", "none"]
        cm = cm.reindex(index=[r for r in order if r in cm.index], columns=[c for c in order if c in cm.columns]).fillna(0)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax, cbar=False, annot_kws={"size": 11})
        ax.set_title(m, fontsize=12)
        ax.set_xlabel("predicted")
        if ax is axes[0]:
            ax.set_ylabel("gold shape")
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Shape confusion on object-label questions (RGB only) — row-normalized", y=1.02)
    _save(fig, out, dpi)


def heatmap_qtype_modality(df: pd.DataFrame, out: Path, dpi: int) -> None:
    """Per-model heatmap of qtype × modality accuracy."""
    models = sorted(df["model_key"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 8), sharey=True)
    if n == 1:
        axes = [axes]
    mod_order = ["rgb", "rgb+depth", "rgb+description", "rgb+depth+description"]
    qt_order = ["binary_relation", "existence", "distance_comparison", "same_side",
                "counting", "directional_extremum", "camera_extremum", "between", "id_extremum"]
    for ax, m in zip(axes, models):
        sub = df[df["model_key"] == m]
        pivot = sub.pivot_table(index="question_type", columns="modality_key", values="is_correct", aggfunc="mean")
        pivot = pivot.reindex(index=[r for r in qt_order if r in pivot.index],
                               columns=[c for c in mod_order if c in pivot.columns])
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax, cbar=False, annot_kws={"size": 11})
        ax.set_title(m, fontsize=12)
        ax.set_xlabel("")
        if ax is not axes[0]:
            ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Question type × modality accuracy, per model", y=1.005)
    _save(fig, out, dpi)


def counting_distribution(df: pd.DataFrame, out: Path, dpi: int) -> None:
    sub = df[(df["question_type"] == "counting") & (df["modality_key"] == "rgb")].copy()
    if sub.empty:
        return
    sub["pred_int"] = pd.to_numeric(sub["parsed"], errors="coerce")
    sub["gold_int"] = pd.to_numeric(sub["gold"], errors="coerce")
    valid = sub.dropna(subset=["pred_int", "gold_int"]).copy()
    valid["pred_int"] = valid["pred_int"].clip(upper=5).astype(int)
    valid["gold_int"] = valid["gold_int"].clip(upper=5).astype(int)
    models = sorted(valid["model_key"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        g = valid[valid["model_key"] == m]
        cm = pd.crosstab(g["gold_int"], g["pred_int"], normalize="index")
        idx = sorted(cm.index)
        cols = sorted(cm.columns)
        cm = cm.reindex(index=idx, columns=cols).fillna(0)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax, cbar=False, annot_kws={"size": 11})
        ax.set_title(m, fontsize=12)
        ax.set_xlabel("predicted count (≥5 grouped)")
        if ax is axes[0]:
            ax.set_ylabel("gold count")
        else:
            ax.set_ylabel("")
    fig.suptitle("Counting: predicted vs gold (RGB only) — row-normalized", y=1.02)
    _save(fig, out, dpi)


def axis_competence(df: pd.DataFrame, out: Path, dpi: int) -> None:
    """Aggregate binary-relation accuracy by spatial-axis family."""
    rgb = df[(df["modality_key"] == "rgb") & (df["question_type"] == "binary_relation")].copy()
    if rgb.empty:
        return
    axis_map = {
        "left_of": "lateral (L/R)", "right_of": "lateral (L/R)",
        "in_front_of": "egocentric in/out", "behind": "egocentric in/out",
        "closer_than": "depth ordering", "farther_than": "depth ordering",
    }
    rgb["axis"] = rgb["relation_type"].map(axis_map)
    rgb = rgb.dropna(subset=["axis"])
    pivot = rgb.pivot_table(index="axis", columns="model_key", values="is_correct", aggfunc="mean")
    pivot = pivot.reindex(["lateral (L/R)", "egocentric in/out", "depth ordering"])
    fig, ax = plt.subplots(figsize=(13, 5.5))
    pivot.plot(kind="bar", ax=ax, width=0.85)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("spatial axis family")
    ax.set_title("Spatial axis competence (binary_relation, RGB only)")
    ax.legend(title="model", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.xticks(rotation=0)
    _save(fig, out, dpi)


def all_plots(df: pd.DataFrame, plots_dir: Path, dpi: int, palette: str) -> None:
    _setup(palette)
    overall_accuracy(df, plots_dir / "01_overall_accuracy.png", dpi)
    accuracy_by_qtype(df, plots_dir / "02_accuracy_by_qtype.png", dpi)
    heatmap_model_qtype(df, plots_dir / "03_heatmap_model_qtype.png", dpi)
    heatmap_model_modality(df, plots_dir / "04_heatmap_model_modality.png", dpi)
    modality_lift(df, plots_dir / "05_modality_lift.png", dpi)
    accuracy_by_regime(df, plots_dir / "06_accuracy_by_regime.png", dpi)
    accuracy_by_n_objects(df, plots_dir / "07_accuracy_by_n_objects.png", dpi)
    accuracy_by_relation(df, plots_dir / "08_accuracy_by_relation.png", dpi)
    yes_bias(df, plots_dir / "09_yes_bias.png", dpi)
    latency(df, plots_dir / "10_latency.png", dpi)
    shape_confusion(df, plots_dir / "11_shape_confusion.png", dpi)
    heatmap_qtype_modality(df, plots_dir / "12_qtype_x_modality.png", dpi)
    counting_distribution(df, plots_dir / "13_counting_confusion.png", dpi)
    axis_competence(df, plots_dir / "14_axis_competence.png", dpi)
