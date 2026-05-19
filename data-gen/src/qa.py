"""Template-based spatial QA generation."""

import random
from collections import Counter

import numpy as np

from src import relations, sample

RELATIONS_BINARY = (
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "closer_than",
    "farther_than",
)

RELATION_PHRASE = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "in_front_of": "in front of",
    "behind": "behind",
    "closer_than": "closer to the camera than",
    "farther_than": "farther from the camera than",
}

BINARY_ANSWER_TYPES = {
    "binary_relation",
    "existence",
    "distance_comparison",
    "same_side",
}


def _label(o: sample.Object) -> str:
    return f"{o.color} {o.shape}"


def _unique_by_color_shape(scene: sample.Scene) -> list[sample.Object]:
    counts = Counter((o.color, o.shape) for o in scene.objects)
    return [o for o in scene.objects if counts[(o.color, o.shape)] == 1]


def _world_distance(a: sample.Object, b: sample.Object) -> float:
    return float(np.linalg.norm(np.array(a.position) - np.array(b.position)))


def gen_binary_relation(scene, rels, rng: random.Random):
    uniques = _unique_by_color_shape(scene)
    if len(uniques) < 2:
        return None
    a, b = rng.sample(uniques, 2)
    rel = rng.choice(RELATIONS_BINARY)
    truth = relations.relation_holds(rels, a.id, b.id, rel)
    return {
        "question_type": "binary_relation",
        "relation_type": rel,
        "question": f"Is the {_label(a)} {RELATION_PHRASE[rel]} the {_label(b)}?",
        "answer": "yes" if truth else "no",
    }


def _extremum(scene, axis: str, kind: str):
    basis = relations.camera_basis(scene.camera)
    vals = {}
    for o in scene.objects:
        if axis == "lateral_x":
            vals[o.id] = relations.lateral_x(o, basis)
        elif axis == "vertical_y":
            vals[o.id] = relations.to_camera_frame(np.array(o.position), basis)[1]
        elif axis == "ns_depth":
            vals[o.id] = relations.nearest_surface_depth(o, basis)
        else:
            raise ValueError(axis)
    winner = min(vals, key=vals.get) if kind == "min" else max(vals, key=vals.get)
    sorted_vals = sorted(vals.values())
    gap = sorted_vals[1] - sorted_vals[0] if kind == "min" else sorted_vals[-1] - sorted_vals[-2]
    if gap < 0.3:
        return None
    uniques = {o.id for o in _unique_by_color_shape(scene)}
    if winner not in uniques:
        return None
    return next(o for o in scene.objects if o.id == winner)


def gen_directional_extremum(scene, rels, rng: random.Random):
    direction = rng.choice(("leftmost", "rightmost", "frontmost", "backmost"))
    if direction == "leftmost":
        obj = _extremum(scene, "lateral_x", "min")
    elif direction == "rightmost":
        obj = _extremum(scene, "lateral_x", "max")
    elif direction == "frontmost":
        obj = _extremum(scene, "ns_depth", "min")
    else:
        obj = _extremum(scene, "ns_depth", "max")
    if obj is None:
        return None
    question = {
        "leftmost": "Which object is the leftmost (from the camera's viewpoint)?",
        "rightmost": "Which object is the rightmost (from the camera's viewpoint)?",
        "frontmost": "Which object is the frontmost (closest to the camera)?",
        "backmost": "Which object is the backmost (farthest from the camera)?",
    }[direction]
    return {
        "question_type": "directional_extremum",
        "relation_type": direction,
        "question": question,
        "answer": _label(obj),
    }


def gen_camera_extremum(scene, rels, rng: random.Random):
    kind = rng.choice(("closest", "farthest"))
    obj = _extremum(scene, "ns_depth", "min" if kind == "closest" else "max")
    if obj is None:
        return None
    preposition = "to" if kind == "closest" else "from"
    return {
        "question_type": "camera_extremum",
        "relation_type": kind,
        "question": f"Which object is {kind} {preposition} the camera?",
        "answer": _label(obj),
    }


def gen_counting(scene, rels, rng: random.Random):
    uniques = _unique_by_color_shape(scene)
    if not uniques:
        return None
    ref = rng.choice(uniques)
    by_color = bool(rng.randint(0, 1))
    if by_color:
        candidates = list({o.color for o in scene.objects if o.id != ref.id})
        if not candidates:
            return None
        attr = rng.choice(candidates)

        def filter_fn(o):
            return o.color == attr

        attr_phrase = f"{attr} objects"
    else:
        candidates = list({o.shape for o in scene.objects if o.id != ref.id})
        if not candidates:
            return None
        attr = rng.choice(candidates)

        def filter_fn(o):
            return o.shape == attr

        attr_phrase = f"{attr}s"
    rel = rng.choice(RELATIONS_BINARY)
    count = sum(
        1
        for o in scene.objects
        if o.id != ref.id and filter_fn(o) and relations.relation_holds(rels, o.id, ref.id, rel)
    )
    return {
        "question_type": "counting",
        "relation_type": rel,
        "question": f"How many {attr_phrase} are {RELATION_PHRASE[rel]} the {_label(ref)}?",
        "answer": str(count),
    }


def gen_existence(scene, rels, rng: random.Random):
    uniques = _unique_by_color_shape(scene)
    if not uniques:
        return None
    ref = rng.choice(uniques)
    colors = list({o.color for o in scene.objects if o.id != ref.id})
    if not colors:
        return None
    attr = rng.choice(colors)
    rel = rng.choice(RELATIONS_BINARY)
    found = any(
        o.color == attr
        and o.id != ref.id
        and relations.relation_holds(rels, o.id, ref.id, rel)
        for o in scene.objects
    )
    return {
        "question_type": "existence",
        "relation_type": rel,
        "question": f"Is there a {attr} object {RELATION_PHRASE[rel]} the {_label(ref)}?",
        "answer": "yes" if found else "no",
    }


def gen_distance_comparison(scene, rels, rng: random.Random):
    uniques = _unique_by_color_shape(scene)
    if len(uniques) < 3:
        return None
    a, b, c = rng.sample(uniques, 3)
    d_ab = _world_distance(a, b)
    d_cb = _world_distance(c, b)
    if abs(d_ab - d_cb) < 0.25:
        return None
    return {
        "question_type": "distance_comparison",
        "relation_type": "closer_to",
        "question": f"Is the {_label(a)} closer to the {_label(b)} than the {_label(c)} is?",
        "answer": "yes" if d_ab < d_cb else "no",
    }


def gen_between(scene, rels, rng: random.Random):
    uniques = _unique_by_color_shape(scene)
    if len(uniques) < 2:
        return None
    a, b = rng.sample(uniques, 2)
    a_pos = np.array(a.position)
    b_pos = np.array(b.position)
    ab = b_pos - a_pos
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-6:
        return None
    matches = []
    for o in scene.objects:
        if o.id in (a.id, b.id):
            continue
        op = np.array(o.position) - a_pos
        t = float(np.dot(op, ab) / ab_len_sq)
        if not 0.15 < t < 0.85:
            continue
        proj = a_pos + t * ab
        if float(np.linalg.norm(np.array(o.position) - proj)) < 0.5:
            matches.append(o)
    if len(matches) > 1:
        return None
    if len(matches) == 1:
        uniques_ids = {o.id for o in uniques}
        if matches[0].id not in uniques_ids:
            return None
        answer = _label(matches[0])
    else:
        answer = "none"
    return {
        "question_type": "between",
        "relation_type": "between",
        "question": f"Which object is between the {_label(a)} and the {_label(b)}?",
        "answer": answer,
    }


def gen_same_side(scene, rels, rng: random.Random):
    uniques = _unique_by_color_shape(scene)
    if not uniques:
        return None
    ref = rng.choice(uniques)
    color_counts = Counter(o.color for o in scene.objects if o.id != ref.id)
    candidates = [c for c, n in color_counts.items() if n >= 2 and c != ref.color]
    if not candidates:
        return None
    target = rng.choice(candidates)
    basis = relations.camera_basis(scene.camera)
    ref_lx = relations.lateral_x(ref, basis)
    signs = set()
    for o in scene.objects:
        if o.id == ref.id or o.color != target:
            continue
        side = relations.lateral_x(o, basis) - ref_lx
        if abs(side) < 0.1:
            return None
        signs.add(side > 0)
    return {
        "question_type": "same_side",
        "relation_type": "same_side",
        "question": f"Are all {target} objects on the same side of the {_label(ref)} (left or right)?",
        "answer": "yes" if len(signs) == 1 else "no",
    }


TEMPLATES = {
    "binary_relation": gen_binary_relation,
    "directional_extremum": gen_directional_extremum,
    "camera_extremum": gen_camera_extremum,
    "counting": gen_counting,
    "existence": gen_existence,
    "distance_comparison": gen_distance_comparison,
    "between": gen_between,
    "same_side": gen_same_side,
}


def generate(scene, rels, rng: random.Random, n_target: int, max_attempts=1000) -> list[dict]:
    out: list[dict] = []
    seen_questions: set[str] = set()
    type_names = list(TEMPLATES.keys())
    per_type_counts = {t: Counter() for t in type_names}
    attempts = 0
    i = 0
    while len(out) < n_target and attempts < max_attempts:
        attempts += 1
        qtype = type_names[i % len(type_names)]
        i += 1
        candidate = None
        for _ in range(5):
            c = TEMPLATES[qtype](scene, rels, rng)
            if c is None:
                break
            if qtype in BINARY_ANSWER_TYPES:
                pc = per_type_counts[qtype]
                if pc["yes"] != pc["no"]:
                    minority = "yes" if pc["yes"] < pc["no"] else "no"
                    candidate = c
                    if c["answer"] == minority:
                        break
                else:
                    candidate = c
                    break
            else:
                candidate = c
                break
        if candidate is None:
            continue
        if candidate["question"] in seen_questions:
            continue
        record = dict(candidate)
        record["question_id"] = len(out)
        out.append(record)
        seen_questions.add(record["question"])
        per_type_counts[qtype][record["answer"]] += 1
    return out
