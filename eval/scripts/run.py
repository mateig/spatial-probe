"""Score VLM responses, write metrics tables, and render plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd

from src import aggregate, config, data, plot, score


def _score(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for r in merged.itertuples(index=False):
        s = score.score_row(r.question_type, r.gold, r.response, r.objects)
        rows.append(s)
    scored = pd.DataFrame(rows, index=merged.index)
    out = pd.concat([merged.drop(columns=["objects"]), scored], axis=1)
    out["is_correct"] = out["is_correct"].astype(bool)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--limit-scenes", type=int, default=None)
    args = parser.parse_args()

    cfg = config.load(args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"data_root={cfg.data_root}")
    print(f"responses={cfg.responses_path}")

    t0 = perf_counter()
    print("loading responses...")
    responses = data.load_responses(cfg.responses_path)
    print(f"  responses: {len(responses):,} rows in {perf_counter() - t0:.1f}s")

    t0 = perf_counter()
    print("loading gold QA + scenes...")
    scene_ids = sorted(responses["scene_id"].unique())
    limit = args.limit_scenes if args.limit_scenes is not None else cfg.scene_limit
    if limit is not None:
        scene_ids = scene_ids[:limit]
        responses = responses[responses["scene_id"].isin(scene_ids)]
    questions = data.load_questions(cfg.data_root, scene_ids)
    print(f"  gold: {len(questions):,} questions, {questions['scene_id'].nunique()} scenes in {perf_counter() - t0:.1f}s")

    t0 = perf_counter()
    print("merging...")
    merged = responses.merge(
        questions[["scene_id", "question_id", "question_type", "relation_type", "gold", "regime", "n_objects", "objects"]],
        on=["scene_id", "question_id"],
        how="left",
        validate="many_to_one",
    )
    missing = merged["gold"].isna().sum()
    if missing:
        print(f"  WARN: {missing} response rows without matching gold; dropping")
        merged = merged.dropna(subset=["gold"])

    # Drop question types whose gold or template doesn't fit a clean text-match scorer:
    #   same_side: "(left or right)?" trailing parenthetical induces side answers, not yes/no.
    #   id_extremum: only 12 source questions; gold needs integer-id matching against same-
    #                color/shape objects which the canonical color+shape scorer can't resolve.
    drop_types = ["same_side", "id_extremum"]
    n_drop = merged["question_type"].isin(drop_types).sum()
    if n_drop:
        print(f"  dropping {n_drop:,} rows in {drop_types} (out-of-scope for current scorer)")
        merged = merged[~merged["question_type"].isin(drop_types)]

    print(f"  merged in {perf_counter() - t0:.1f}s")

    t0 = perf_counter()
    print("scoring...")
    scored = _score(merged)
    print(f"  scored {len(scored):,} rows in {perf_counter() - t0:.1f}s; overall accuracy={scored['is_correct'].mean():.4f}")

    print(f"writing scored rows -> {cfg.scored_path}")
    keep_cols = [
        "model_key", "model_id", "scene_id", "question_id", "modality_key",
        "question_type", "relation_type", "regime", "n_objects",
        "gold", "response", "parsed", "is_correct", "elapsed_s",
    ]
    keep_cols = [c for c in keep_cols if c in scored.columns]
    data.write_jsonl(cfg.scored_path, scored[keep_cols])

    print(f"writing metrics -> {cfg.metrics_dir}")
    aggregate.write_all(scored, cfg.metrics_dir)

    print(f"writing plots -> {cfg.plots_dir}")
    plot.all_plots(scored, cfg.plots_dir, cfg.plot.dpi, cfg.plot.palette)

    print("done.")


if __name__ == "__main__":
    main()
