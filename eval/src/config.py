from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PlotCfg:
    dpi: int = 130
    palette: str = "viridis"


@dataclass(frozen=True)
class Config:
    data_root: Path
    responses_path: Path
    output_dir: Path
    scored_path: Path
    metrics_dir: Path
    plots_dir: Path
    plot: PlotCfg
    scene_limit: int | None


def load(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text())
    return Config(
        data_root=Path(raw["data_root"]),
        responses_path=Path(raw["responses_path"]),
        output_dir=Path(raw["output_dir"]),
        scored_path=Path(raw["scored_path"]),
        metrics_dir=Path(raw["metrics_dir"]),
        plots_dir=Path(raw["plots_dir"]),
        plot=PlotCfg(**raw.get("plot", {})),
        scene_limit=raw.get("scene_limit"),
    )
