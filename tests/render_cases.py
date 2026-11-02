"""Synthetic volumes and planes behind the committed renderer goldens.

Each case is small enough to hand-check against ``docs/algorithm.md`` and
covers one required fixture kind: a 3-point plane, a 4-point trapezoid, an
out-of-volume clipping plane, and an exact half-to-even rounding tie. The
volumes are built in code (no binary input fixtures); only the rendered
golden frames are committed, under ``tests/golden/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ramp_volume(shape: tuple[int, int, int], axis: int = 0) -> np.ndarray:
    """uint8 volume whose value at each voxel is its index along ``axis``.

    With identity uint8 display scaling, a rendered pixel's value *is* the
    sampled index on that axis - hand-checkable without re-deriving geometry.
    """
    index = np.arange(shape[axis], dtype=np.uint8)
    view = [1, 1, 1]
    view[axis] = shape[axis]
    return np.broadcast_to(index.reshape(view), shape).copy()


def short_ramp_volume(shape: tuple[int, int, int]) -> np.ndarray:
    """int16 volume with one hand-chosen value per x: min, 0, max, +100, -100."""
    values = np.array([-32768, 0, 32767, 100, -100], dtype=np.int16)
    return np.broadcast_to(values[:, None, None], shape).copy()


@dataclass(frozen=True)
class RenderCase:
    name: str
    volume: np.ndarray
    points: tuple[tuple[float, float, float], ...]
    note: str


CASES: tuple[RenderCase, ...] = (
    RenderCase(
        name="triangle",
        volume=ramp_volume((6, 5, 5)),
        points=((2, 2, 0), (0, 2, 0), (4, 2, 0)),
        note="3-point plane: leg 2 -> 2 rows, base 4 -> 4 columns; rounding ties",
    ),
    RenderCase(
        name="trapezoid",
        volume=ramp_volume((8, 8, 5)),
        points=((2, 2, 0), (4, 2, 0), (0, 6, 0), (6, 6, 0)),
        note="4-point trapezoid: leg sqrt(20) -> 4 rows, base 6 -> 6 columns",
    ),
    RenderCase(
        name="out_of_volume",
        volume=np.full((4, 4, 4), 255, dtype=np.uint8),
        points=((2, 2, 2), (-2, 2, 2), (6, 2, 2)),
        note="plane wider than M: out-of-volume samples are black, not clipped",
    ),
    RenderCase(
        name="rounding_tie",
        volume=ramp_volume((3, 6, 7)),
        points=((2.5, 2, 0), (2.5, 0, 6), (2.5, 4, 6)),
        note="all samples land on x=2.5: np.rint -> even neighbour 2, not 3",
    ),
    RenderCase(
        name="short_scaling",
        volume=short_ramp_volume((5, 5, 1)),
        points=((2, 0, 0), (-0.5, 4, 0), (4.5, 4, 0)),
        note="int16 min/0/max/+100/-100 -> 0/128/255/128/127, full signed range",
    ),
)
