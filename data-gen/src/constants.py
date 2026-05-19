"""Palette, shapes, and sizes shared by sampling, rendering, and labels."""

COLORS = {
    "red": (1.00, 0.00, 0.00),
    "green": (0.00, 0.70, 0.00),
    "blue": (0.00, 0.00, 1.00),
    "yellow": (1.00, 1.00, 0.00),
    "cyan": (0.00, 0.80, 0.80),
    "magenta": (1.00, 0.00, 1.00),
    "orange": (1.00, 0.50, 0.00),
    "purple": (0.50, 0.00, 0.50),
    "brown": (0.50, 0.25, 0.10),
    "gray": (0.50, 0.50, 0.50),
}
COLOR_NAMES = tuple(COLORS.keys())

SHAPES = ("cube", "sphere", "cylinder")

SIZES = {
    "small": 0.30,
    "medium": 0.45,
    "large": 0.60,
}
SIZE_NAMES = tuple(SIZES.keys())
