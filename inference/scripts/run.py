from __future__ import annotations

import argparse
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from time import perf_counter

from src import config, dataset, inference, records

HEARTBEAT_EVERY = 100


def _modality_combos(optional: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [
        ("rgb", *combo)
        for size in range(len(optional) + 1)
        for combo in combinations(optional, size)
    ]


def _select_models(models, keys):
    enabled = [m for m in models if m.enabled]
    if keys is None:
        return enabled
    by_key = {m.key: m for m in enabled}
    missing = sorted(set(keys) - set(by_key))
    if missing:
        raise SystemExit(f"Unknown or disabled model keys: {missing}")
    return [by_key[k] for k in keys]


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = config.load(args.config)
    scene_ids = dataset.select(
        dataset.discover(cfg.data_root),
        tuple(args.scene_ids) if args.scene_ids else cfg.scene_ids,
        cfg.scene_start,
        args.limit_scenes if args.limit_scenes is not None else cfg.scene_limit,
    )
    models = _select_models(cfg.models, args.models)
    combos = _modality_combos(cfg.optional_modalities)
    seen = set() if args.no_resume else records.completed(cfg.output_path)

    print(f"started at {datetime.now(UTC).isoformat()}")
    print(f"data_root={cfg.data_root}")
    print(f"output_path={cfg.output_path}")
    print(f"scenes={len(scene_ids)} models={len(models)} modalities={len(combos)}")

    sweep_t0 = perf_counter()
    total_rows = 0
    total_errors = 0

    for spec in models:
        print(f"[{spec.key}] loading...", flush=True)
        load_t0 = perf_counter()
        loaded = inference.load_model(spec, cfg.inference)
        print(
            f"[{spec.key}] loaded in {_fmt_duration(perf_counter() - load_t0)}",
            flush=True,
        )

        model_t0 = perf_counter()
        rows = 0
        errors = 0

        try:
            for i, scene_id in enumerate(scene_ids, 1):
                scene = dataset.load(cfg.data_root, scene_id)
                for modalities in combos:
                    if not inference.supports(spec, modalities):
                        continue
                    mod_key = "+".join(modalities)
                    pending = [
                        q
                        for q in scene.questions
                        if not (
                            cfg.inference.resume
                            and (spec.key, scene.id, q.id, mod_key) in seen
                        )
                    ]
                    if not pending:
                        continue

                    started = perf_counter()
                    error: str | None = None
                    try:
                        responses = inference.generate(
                            loaded,
                            scene,
                            [q.text for q in pending],
                            modalities,
                            cfg.prompt,
                            cfg.inference,
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        responses = [None] * len(pending)
                    elapsed = (perf_counter() - started) / len(pending)

                    for q, response in zip(pending, responses):
                        records.append(
                            cfg.output_path,
                            records.make_row(
                                spec=spec,
                                scene_id=scene.id,
                                question_id=q.id,
                                question_text=q.text,
                                modalities=modalities,
                                response=response,
                                error=error,
                                elapsed_s=elapsed,
                            ),
                        )
                        rows += 1
                        if error is not None:
                            errors += 1

                if i % HEARTBEAT_EVERY == 0 or i == len(scene_ids):
                    el = perf_counter() - model_t0
                    eta = el / i * (len(scene_ids) - i)
                    pct = 100 * i / len(scene_ids)
                    print(
                        f"[{spec.key}] {i}/{len(scene_ids)} ({pct:.1f}%) "
                        f"elapsed={_fmt_duration(el)} eta={_fmt_duration(eta)} "
                        f"rows={rows} errors={errors}",
                        flush=True,
                    )
        finally:
            inference.free_model(loaded)

        print(
            f"[{spec.key}] done in {_fmt_duration(perf_counter() - model_t0)}: "
            f"rows={rows} errors={errors}",
            flush=True,
        )
        total_rows += rows
        total_errors += errors

    print(
        f"sweep complete: wall={_fmt_duration(perf_counter() - sweep_t0)} "
        f"rows={total_rows} errors={total_errors}",
        flush=True,
    )


if __name__ == "__main__":
    main()
