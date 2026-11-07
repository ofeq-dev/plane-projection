"""Scalar geometry helpers for plane validation.

Used by plane_validation predicates. The renderer uses vectorized numpy
geometry instead (different abstraction level).
"""

from __future__ import annotations

import math
from typing import Tuple

Point = Tuple[float, float, float]


def _dist(p: Point, q: Point) -> float:
    return math.dist(p, q)


def _cross(u: Point, v: Point) -> Point:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _sub(p: Point, q: Point) -> Point:
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def _dot(u: Point, v: Point) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]