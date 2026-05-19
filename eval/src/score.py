"""Normalize VLM responses and score against gold answers per question type."""

from __future__ import annotations

import re
from typing import Any

YES_TOKENS = {"yes", "y", "true", "correct", "yeah", "yep"}
NO_TOKENS = {"no", "n", "false", "incorrect", "nope", "none", "nothing"}

NUMBER_WORDS = {
    "zero": 0, "no": 0, "none": 0,
    "one": 1, "a": 1, "single": 1,
    "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
}

# Shape vocabulary used by data-gen + permissive aliases models tend to use.
SHAPES = {"sphere", "cube", "cylinder"}
SHAPE_ALIASES = {
    "sphere": "sphere", "spheres": "sphere", "ball": "sphere", "balls": "sphere",
    "orb": "sphere", "circle": "sphere",
    "cube": "cube", "cubes": "cube", "box": "cube", "boxes": "cube",
    "block": "cube", "square": "cube",
    "cylinder": "cylinder", "cylinders": "cylinder",
    "can": "cylinder", "tube": "cylinder", "rod": "cylinder", "pillar": "cylinder",
}

COLORS = {
    "red", "green", "blue", "yellow", "orange", "purple", "magenta",
    "cyan", "brown", "gray", "grey", "white", "black", "pink",
}
COLOR_ALIASES = {"grey": "gray"}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _first_yes_no(tokens: list[str]) -> str | None:
    for t in tokens:
        if t in YES_TOKENS:
            return "yes"
        if t in NO_TOKENS:
            return "no"
    return None


def _first_int(tokens: list[str]) -> int | None:
    for t in tokens:
        if t.isdigit():
            return int(t)
        if t in NUMBER_WORDS:
            return NUMBER_WORDS[t]
    return None


def _detect_colors_shapes(tokens: list[str]) -> tuple[set[str], set[str]]:
    colors: set[str] = set()
    shapes: set[str] = set()
    for t in tokens:
        c = COLOR_ALIASES.get(t, t)
        if c in COLORS:
            colors.add(c)
        if t in SHAPE_ALIASES:
            shapes.add(SHAPE_ALIASES[t])
    return colors, shapes


def _gold_object(gold: str) -> tuple[str | None, str | None]:
    """Parse '<color> <shape>' or for id_extremum 'object N (size color shape)'."""
    s = gold.lower()
    # id_extremum: pull the parenthetical
    m = re.search(r"\(([^)]+)\)", s)
    if m:
        s = m.group(1)
    toks = _tokenize(s)
    color = next((COLOR_ALIASES.get(t, t) for t in toks if COLOR_ALIASES.get(t, t) in COLORS), None)
    shape = next((SHAPE_ALIASES[t] for t in toks if t in SHAPE_ALIASES), None)
    return color, shape


def _matches_object(
    response_tokens: list[str],
    gold_color: str | None,
    gold_shape: str | None,
    scene_objects: list[dict],
) -> bool:
    """Lenient: response identifies the gold object if it disambiguates within the scene.

    Strict color+shape match always works. Color- or shape-only matches are accepted
    when that property uniquely picks out the gold object in the scene.
    """
    colors, shapes = _detect_colors_shapes(response_tokens)
    if gold_color in colors and gold_shape in shapes:
        return True
    # Filter scene objects by what response mentions; correct iff that filter resolves to the gold.
    if not colors and not shapes:
        return False
    matching = [
        o for o in scene_objects
        if (not colors or o["color"] in colors) and (not shapes or o["shape"] in shapes)
    ]
    if len(matching) != 1:
        return False
    return matching[0]["color"] == gold_color and matching[0]["shape"] == gold_shape


def score_row(question_type: str, gold: str, response: str, scene_objects: list[dict]) -> dict[str, Any]:
    """Return {is_correct, parsed, gold_norm, mode} where mode describes the scorer used."""
    text = (response or "").strip()
    tokens = _tokenize(text)

    if question_type in {"binary_relation", "existence", "distance_comparison", "same_side"}:
        parsed = _first_yes_no(tokens)
        return {
            "is_correct": parsed is not None and parsed == gold,
            "parsed": parsed if parsed is not None else "",
            "gold_norm": gold,
            "mode": "yes_no",
        }

    if question_type == "counting":
        n = _first_int(tokens)
        try:
            gold_n = int(gold)
        except ValueError:
            gold_n = None
        return {
            "is_correct": n is not None and gold_n is not None and n == gold_n,
            "parsed": "" if n is None else str(n),
            "gold_norm": gold,
            "mode": "count",
        }

    if question_type == "between":
        # gold = "none" or a "<color> <shape>" label
        if gold in {"none", "no", "nothing"}:
            # Accept any no/none/nothing token. The "there is no object between the
            # {a} and the {b}" pattern echoes question referents, so we cannot also
            # require the response to be free of color/shape words.
            parsed = _first_yes_no(tokens)
            is_correct = parsed == "no"
            return {
                "is_correct": bool(is_correct),
                "parsed": "none" if is_correct else (text[:40] or ""),
                "gold_norm": gold,
                "mode": "between_none",
            }
        gc, gs = _gold_object(gold)
        return {
            "is_correct": _matches_object(tokens, gc, gs, scene_objects),
            "parsed": text[:40],
            "gold_norm": gold,
            "mode": "object_label",
        }

    if question_type in {"directional_extremum", "camera_extremum", "id_extremum"}:
        gc, gs = _gold_object(gold)
        return {
            "is_correct": _matches_object(tokens, gc, gs, scene_objects),
            "parsed": text[:40],
            "gold_norm": gold,
            "mode": "object_label",
        }

    return {"is_correct": False, "parsed": text[:40], "gold_norm": gold, "mode": "unknown"}
