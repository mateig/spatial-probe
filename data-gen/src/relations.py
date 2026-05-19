"""Camera-frame geometry and pairwise spatial relations."""

from dataclasses import dataclass

import numpy as np

from src import sample


@dataclass(frozen=True)
class CameraBasis:
    position: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray


@dataclass(frozen=True)
class PairRelation:
    a: int
    b: int
    left_of: bool
    right_of: bool
    in_front_of: bool
    behind: bool
    closer_than: bool
    farther_than: bool


def camera_basis(camera: sample.Camera) -> CameraBasis:
    pos = np.array(camera.position, dtype=np.float64)
    target = np.array(camera.target, dtype=np.float64)
    forward = target - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return CameraBasis(pos, forward, right, up)


def to_camera_frame(p_world: np.ndarray, basis: CameraBasis) -> tuple[float, float, float]:
    rel = p_world - basis.position
    return (
        float(np.dot(rel, basis.right)),
        float(np.dot(rel, basis.up)),
        float(np.dot(rel, basis.forward)),
    )


def aabb_corners(obj: sample.Object) -> np.ndarray:
    s = obj.half_extent
    c = np.array(obj.position)
    return np.array(
        [
            c + np.array((sx, sy, sz)) * s
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )


def nearest_surface_depth(obj: sample.Object, basis: CameraBasis) -> float:
    if obj.shape == "sphere":
        return to_camera_frame(np.array(obj.position), basis)[2] - obj.half_extent
    return float(min(to_camera_frame(c, basis)[2] for c in aabb_corners(obj)))


def lateral_x(obj: sample.Object, basis: CameraBasis) -> float:
    return to_camera_frame(np.array(obj.position), basis)[0]


def derive(scene: sample.Scene, lateral_threshold=0.15, depth_threshold=0.20):
    basis = camera_basis(scene.camera)
    ns_depth = {o.id: nearest_surface_depth(o, basis) for o in scene.objects}
    lat_x = {o.id: lateral_x(o, basis) for o in scene.objects}
    out: list[PairRelation] = []
    for a in scene.objects:
        for b in scene.objects:
            if a.id == b.id:
                continue
            dx = lat_x[a.id] - lat_x[b.id]
            dd = ns_depth[a.id] - ns_depth[b.id]
            left_of = dx < -lateral_threshold
            right_of = dx > lateral_threshold
            in_front_of = dd < -depth_threshold
            behind = dd > depth_threshold
            out.append(
                PairRelation(
                    a=a.id,
                    b=b.id,
                    left_of=left_of,
                    right_of=right_of,
                    in_front_of=in_front_of,
                    behind=behind,
                    closer_than=in_front_of,
                    farther_than=behind,
                )
            )
    return out


def relation_holds(relations: list[PairRelation], a_id: int, b_id: int, name: str) -> bool:
    for r in relations:
        if r.a == a_id and r.b == b_id:
            return bool(getattr(r, name))
    raise KeyError(f"no relation for pair ({a_id},{b_id})")


def object_depths(scene: sample.Scene) -> dict:
    basis = camera_basis(scene.camera)
    out = {}
    for o in scene.objects:
        lx, vy, cd = to_camera_frame(np.array(o.position), basis)
        out[o.id] = {
            "nearest_surface_depth": nearest_surface_depth(o, basis),
            "centroid_depth": cd,
            "lateral_x": lx,
            "vertical_y": vy,
        }
    return out


def relations_to_dict(relations: list[PairRelation]) -> list[dict]:
    keys = ("left_of", "right_of", "in_front_of", "behind", "closer_than", "farther_than")
    return [{"a": r.a, "b": r.b, **{k: getattr(r, k) for k in keys}} for r in relations]
