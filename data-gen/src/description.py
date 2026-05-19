"""Deterministic text serialization for each generated scene sample."""

import numpy as np

from src import relations, sample

def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _label(obj: sample.Object) -> str:
    return f"{obj.color} {obj.shape}"


def serialize(scene: sample.Scene, rels: list[relations.PairRelation]) -> str:
    basis = relations.camera_basis(scene.camera)
    lines: list[str] = []
    lines.append(f"Scene with {len(scene.objects)} objects viewed from a camera.")
    cx, cy, cz = scene.camera.position
    lines.append(f"Camera position (world): ({_fmt(cx)}, {_fmt(cy)}, {_fmt(cz)}).")
    lines.append("Camera looks at the world origin (0, 0, 0).")
    lines.append("")
    lines.append("Objects:")
    for o in scene.objects:
        lx, vy, cd = relations.to_camera_frame(np.array(o.position), basis)
        ns = relations.nearest_surface_depth(o, basis)
        lines.append(
            f"- id={o.id}: {_label(o)} ({o.size}); "
            f"world_position=({_fmt(o.x)}, {_fmt(o.y)}, {_fmt(o.z)}); "
            f"camera_frame=(lateral={_fmt(lx)}, vertical={_fmt(vy)}, depth={_fmt(cd)}); "
            f"nearest_surface_depth={_fmt(ns)}."
        )
    lines.append("")
    lines.append("Spatial relations (camera viewpoint). Each relation listed once, from A to B:")
    seen = set()
    for r in rels:
        pair_key = tuple(sorted((r.a, r.b)))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        a = scene.objects[r.a]
        b = scene.objects[r.b]
        al, bl = _label(a), _label(b)
        facts = []
        if r.left_of:
            facts.append(f"{al} is to the left of {bl}")
        elif r.right_of:
            facts.append(f"{al} is to the right of {bl}")
        if r.in_front_of:
            facts.append(f"{al} is in front of {bl}")
        elif r.behind:
            facts.append(f"{al} is behind {bl}")
        if r.closer_than:
            facts.append(f"{al} is closer to the camera than {bl}")
        elif r.farther_than:
            facts.append(f"{al} is farther from the camera than {bl}")
        for fact in facts:
            lines.append(f"- {fact}.")
    return "\n".join(lines) + "\n"
