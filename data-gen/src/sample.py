"""Pure uniform sampler for stage-0 style spatial QA scenes."""

import math
import random
from dataclasses import dataclass

import numpy as np

from src.constants import COLOR_NAMES, COLORS, SHAPES, SIZES, SIZE_NAMES


@dataclass(frozen=True)
class Object:
    id: int
    shape: str
    color: str
    size: str
    x: float
    y: float

    @property
    def half_extent(self) -> float:
        return SIZES[self.size]

    @property
    def z(self) -> float:
        return self.half_extent

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def color_rgb(self) -> tuple[float, float, float]:
        return COLORS[self.color]


@dataclass(frozen=True)
class Camera:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_degrees: float
    resolution: tuple[int, int]


@dataclass(frozen=True)
class Scene:
    scene_id: int
    regime: str
    camera: Camera
    objects: tuple[Object, ...]


def _u(rng: random.Random, r: list[float]) -> float:
    return rng.uniform(r[0], r[1])


def _camera(rng: random.Random, c: dict) -> Camera:
    radius = _u(rng, c["camera_radius"])
    az = rng.uniform(0.0, 2.0 * math.pi)
    el = c["camera_elevation"]
    pos = (
        radius * math.cos(az) * math.cos(el),
        radius * math.sin(az) * math.cos(el),
        radius * math.sin(el),
    )
    return Camera(
        position=pos,
        target=(0.0, 0.0, 0.0),
        fov_degrees=c["camera_fov_degrees"],
        resolution=tuple(c["resolution"]),
    )


def _random_attrs(rng: random.Random, n: int) -> list[dict]:
    return [
        {
            "shape": rng.choice(SHAPES),
            "color": rng.choice(COLOR_NAMES),
            "size": rng.choice(SIZE_NAMES),
        }
        for _ in range(n)
    ]


def _place(attrs: list[dict], rng: random.Random, c: dict) -> tuple[Object, ...] | None:
    disk_r = c["placement_disk_radius"]
    pad = c["placement_padding"]
    attempts = c["placement_attempts"]
    out: list[Object] = []
    for i, a in enumerate(attrs):
        size = SIZES[a["size"]]
        for _ in range(attempts):
            r = math.sqrt(rng.uniform(0.0, 1.0)) * disk_r
            theta = rng.uniform(0.0, 2.0 * math.pi)
            x, y = r * math.cos(theta), r * math.sin(theta)
            if any(
                (x - p.x) ** 2 + (y - p.y) ** 2
                < (size + p.half_extent + pad) ** 2
                for p in out
            ):
                continue
            out.append(Object(i, a["shape"], a["color"], a["size"], x, y))
            break
        else:
            return None
    return tuple(out)


def _sparse(rng: random.Random, c: dict):
    return _place(_random_attrs(rng, 3), rng, c)


def _dense(rng: random.Random, c: dict):
    return _place(_random_attrs(rng, rng.choice((5, 6))), rng, c)


def _size_mixed(rng: random.Random, c: dict):
    attrs = _random_attrs(rng, rng.randint(4, 6))
    attrs[0]["size"] = "small"
    attrs[1]["size"] = "large"
    rng.shuffle(attrs)
    return _place(attrs, rng, c)


def _same_color_cluster(rng: random.Random, c: dict):
    attrs = _random_attrs(rng, rng.randint(4, 6))
    shared = rng.choice(COLOR_NAMES)
    attrs[0]["color"] = shared
    attrs[1]["color"] = shared
    rng.shuffle(attrs)
    return _place(attrs, rng, c)


def _vertical(rng: random.Random, c: dict):
    attrs = _random_attrs(rng, rng.randint(3, 5))
    attrs[0] = {"shape": "cylinder", "color": rng.choice(COLOR_NAMES), "size": "large"}
    for a in attrs[1:]:
        a["size"] = rng.choice(("small", "medium"))
    rng.shuffle(attrs)
    return _place(attrs, rng, c)


def _camera_basis(camera: Camera):
    pos = np.array(camera.position, dtype=np.float64)
    target = np.array(camera.target, dtype=np.float64)
    forward = target - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return pos, forward, right, up


def _nearest_surface_depth(obj: Object, basis) -> float:
    pos, forward, _right, _up = basis
    center = np.array(obj.position)
    if obj.shape == "sphere":
        return float(np.dot(center - pos, forward) - obj.half_extent)
    s = obj.half_extent
    corners = np.array(
        [
            center + np.array((sx, sy, sz)) * s
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    return float(min(np.dot(c - pos, forward) for c in corners))


def _has_depth_ambiguous_pair(objects: tuple[Object, ...], camera: Camera) -> bool:
    pos, forward, right, _up = _camera_basis(camera)
    entries = []
    for o in objects:
        center = np.array(o.position)
        entries.append(
            (
                float(np.dot(center - pos, right)),
                _nearest_surface_depth(o, (pos, forward, right, _up)),
            )
        )
    for i, (lxi, di) in enumerate(entries):
        for lxj, dj in entries[i + 1 :]:
            if abs(lxi - lxj) < 0.35 and abs(di - dj) > 1.0:
                return True
    return False


def _depth_ambiguous(rng: random.Random, c: dict):
    n = rng.randint(4, 6)
    for _ in range(20):
        placed = _place(_random_attrs(rng, n), rng, c)
        if placed is None:
            continue
        camera = _camera(rng, c)
        if _has_depth_ambiguous_pair(placed, camera):
            return placed, camera
    return None, None


def scene(rng: random.Random, c: dict, scene_id: int) -> Scene:
    regimes = c["regimes"]
    regime = regimes[scene_id % len(regimes)]
    for _ in range(c["regime_retries"]):
        camera = _camera(rng, c)
        if regime == "sparse":
            objects = _sparse(rng, c)
        elif regime == "dense":
            objects = _dense(rng, c)
        elif regime == "size_mixed":
            objects = _size_mixed(rng, c)
        elif regime == "same_color_cluster":
            objects = _same_color_cluster(rng, c)
        elif regime == "vertical":
            objects = _vertical(rng, c)
        elif regime == "depth_ambiguous":
            objects, camera = _depth_ambiguous(rng, c)
        else:
            raise ValueError(f"unknown regime {regime!r}")
        if objects is not None and camera is not None:
            return Scene(scene_id=scene_id, regime=regime, camera=camera, objects=objects)

    objects = _sparse(rng, c)
    if objects is None:
        raise RuntimeError(f"sample {scene_id}: could not place fallback scene")
    return Scene(scene_id=scene_id, regime="sparse", camera=_camera(rng, c), objects=objects)
