from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    model_id: str
    enabled: bool = True


@dataclass(frozen=True)
class InferenceCfg:
    device_map: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    resume: bool = True


@dataclass(frozen=True)
class Config:
    data_root: Path
    output_path: Path
    scene_ids: tuple[str, ...] | None
    scene_start: int
    scene_limit: int | None
    optional_modalities: tuple[str, ...]
    inference: InferenceCfg
    prompt: str
    models: tuple[ModelSpec, ...]


def load(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text())
    sel = raw.get("scene_selection", {})
    mod = raw.get("modalities", {})
    explicit = sel.get("scene_ids")
    return Config(
        data_root=Path(raw["data_root"]),
        output_path=Path(raw["output_path"]),
        scene_ids=tuple(explicit) if explicit else None,
        scene_start=int(sel.get("start", 0)),
        scene_limit=sel.get("limit"),
        optional_modalities=tuple(mod.get("optional", ())),
        inference=InferenceCfg(**raw.get("inference", {})),
        prompt=raw["prompt"],
        models=tuple(ModelSpec(**m) for m in raw["models"]),
    )
