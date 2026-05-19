"""Entry point: render a synthetic spatial QA dataset per config."""

import argparse
import subprocess
import sys

from scripts.config import cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-count", type=int, default=None, help=argparse.SUPPRESS
    )
    return parser.parse_args()


def _run_worker(worker_id: int, worker_count: int) -> None:
    from src import render

    render.run(
        cfg["render"]["run"],
        count=cfg["count"],
        seed=cfg["seed"],
        start=worker_id,
        step=worker_count,
        worker_id=worker_id,
    )


def _run_parallel(worker_count: int) -> None:
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for worker_id in range(worker_count):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.run",
                        "--worker-id",
                        str(worker_id),
                        "--worker-count",
                        str(worker_count),
                    ]
                )
            )

        exit_code = 0
        for process in processes:
            code = process.wait()
            if code:
                exit_code = code

        if exit_code:
            raise SystemExit(exit_code)
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        raise


def main() -> None:
    args = _parse_args()

    if args.worker_id is not None:
        _run_worker(args.worker_id, args.worker_count)
        return

    worker_count = cfg.get("workers", 1)
    if worker_count == 1:
        _run_worker(0, 1)
        return

    _run_parallel(worker_count)


if __name__ == "__main__":
    main()
