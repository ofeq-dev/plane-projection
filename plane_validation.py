"""Plane-list parsing + per-row validation (FR-7..FR-11).

Public interface:
- parse_planes(path) -> list[Plane | PlaneError]
- count_planes(path) -> int

Dataclasses:
- Plane: one validated CSV plane row
- PlaneError: one CSV row that failed validation
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from csv_reader import iter_csv_rows
from errors import UsageError
from geometry import _cross, _dist, _dot, _sub

#: Plane validity tolerance (brief Q&A; SRS FR-10).
EPS = 0.001

#: Relative tolerance for the coplanarity predicate (rule 3, 4-point):
#: the scalar triple product of the leg directions and the apex separation
#: is compared against the product of the involved lengths, so the test is
#: scale-free. Genuinely non-coplanar point sets sit at O(0.01..1); the
#: corpus's rectangular planes sit at float-dust level (~1e-18).
COPLANAR_RTOL = 1e-9

Point = tuple[float, float, float]


@dataclass(frozen=True)
class Plane:
    """One validated CSV plane row (FR-8).

    ``points`` keeps the brief's column order: 3-point rows are
    ``(a, b, c)`` (legs measured from ``a``); 4-point rows are
    ``(a1, a2, b, c)`` (legs ``a1→b``/``a2→c``, bases ``a1→a2``/``b→c``).
    Canonical voxel-index space per ADR 0002 - no world-space conversion.
    """

    row: int
    points: tuple[Point, ...]

    @property
    def is_trapezoid(self) -> bool:
        return len(self.points) == 4


@dataclass(frozen=True)
class PlaneError:
    """One CSV row that failed validation (FR-10/FR-11).

    ``reason`` names the exact failing predicate with its numbers - the same
    text the future error-frame banner (FR-15) and CLI log line (FR-16)
    carry. ``banner()`` composes the full line, addressable by row index.
    """

    row: int
    reason: str

    def banner(self) -> str:
        return f"row {self.row}: {self.reason}"


def _fmt(value: float) -> str:
    """Compact %g formatting for banner numbers (100 not 100.00000)."""
    return f"{value:g}"


def _validate_triangle(a: Point, b: Point, c: Point) -> str | None:
    """3-point validity + constructibility: the failing rule, or None.

    Rule 1 (FR-10): ``|dist(a,b) - dist(a,c)| < EPS`` - the isosceles legs
    are measured from the first point. Constructibility (FR-11): no
    zero-length leg; ``b == c`` makes every row's width 0 (the two legs
    collapse onto one direction), a degenerate frame.
    """
    dab, dac = _dist(a, b), _dist(a, c)
    if dab == 0:
        return "zero-length leg a→b (a == b)"
    if dac == 0:
        return "zero-length leg a→c (a == c)"
    if abs(dab - dac) >= EPS:
        return (
            f"|dist(a,b) − dist(a,c)| = {_fmt(abs(dab - dac))} ≥ EPS = {EPS}"
            f" (dist(a,b) = {_fmt(dab)}, dist(a,c) = {_fmt(dac)})"
        )
    if b == c:
        return "degenerate: b == c (parallel legs give a zero-width frame)"
    return None


def _validate_trapezoid(a1: Point, a2: Point, b: Point, c: Point) -> str | None:
    """4-point validity + constructibility: the failing rule, or None.

    Rules in the brief's order (FR-10): legs-equal ``|dist(a1,b) -
    dist(a2,c)| < EPS``; base ``dist(a1,a2) <= dist(b,c)`` (with the EPS
    tolerance - the corpus's rectangles exceed it by ~4e-7 float dust);
    coplanarity of the two legs. Constructibility (FR-11): no zero-length
    leg; ``a1 == a2`` with same-direction legs maximizes the zero-width
    collapse, while ``a1 == a2`` with diverging legs (including opposite-way
    legs) is a triangle and renders fine.
    Parallel legs with a nonzero top base are legal rectangles (the whole
    head.csv corpus).
    """
    d1, d2 = _dist(a1, b), _dist(a2, c)
    if d1 == 0:
        return "zero-length leg a1→b (a1 == b)"
    if d2 == 0:
        return "zero-length leg a2→c (a2 == c)"
    if abs(d1 - d2) >= EPS:
        return (
            f"|dist(a1,b) − dist(a2,c)| = {_fmt(abs(d1 - d2))} ≥ EPS = {EPS}"
            f" (dist(a1,b) = {_fmt(d1)}, dist(a2,c) = {_fmt(d2)})"
        )
    d12, dbc = _dist(a1, a2), _dist(b, c)
    if d12 > dbc + EPS:
        return (
            f"dist(a1,a2) = {_fmt(d12)} > dist(b,c) = {_fmt(dbc)}"
            f" (rule: dist(a1,a2) ≤ dist(b,c))"
        )
    leg1, leg2, sep = _sub(b, a1), _sub(c, a2), _sub(a2, a1)
    triple = abs(_dot(sep, _cross(leg1, leg2)))
    if triple > COPLANAR_RTOL * d1 * d2 * (d12 or 1.0):
        return "legs a1→b and a2→c do not lie in the same plane (rule 3)"
    if d12 == 0:
        # a1 == a2: the frame is zero-width on every row exactly when the
        # legs step the same way (a2->c == a1->b). Opposite-way legs
        # (a2->c == -(a1->b)) are the 3-point triangle geometry — apex a1,
        # base through b and c — and render.
        if leg1 == leg2:
            return "degenerate: a1 == a2 with same-direction legs (zero-width frame)"
    return None


def _parse_row(row_idx: int, raw: list[str]) -> Plane | PlaneError:
    """Parse and validate one CSV data row into its final record (FR-8..FR-11)."""

    def fail(reason: str) -> PlaneError:
        return PlaneError(row=row_idx, reason=reason)

    count_cell = raw[0].strip() if raw else ""
    try:
        n = float(count_cell)
    except ValueError:
        return fail(f"point count is not a number: '{count_cell}'")
    if n not in (3.0, 4.0):
        return fail(f"point count must be 3 or 4, got '{count_cell}'")
    n = int(n)

    if len(raw) < 1 + 3 * n:
        return fail(f"expected {1 + 3 * n} columns for a {n}-point row, got {len(raw)}")
    points: list[Point] = []
    for i in range(n):
        cells = raw[1 + 3 * i : 4 + 3 * i]
        point: list[float] = []
        for j, cell in enumerate(cells):
            try:
                point.append(float(cell))
            except ValueError:
                return fail(f"coordinate is not a number: '{cell.strip()}' (point {i + 1})")
        points.append((point[0], point[1], point[2]))

    if n == 3:
        reason = _validate_triangle(*points)
    else:
        reason = _validate_trapezoid(*points)
    if reason is not None:
        return fail(reason)
    return Plane(row=row_idx, points=tuple(points))


def parse_planes(path: str | Path) -> list[Plane | PlaneError]:
    """Parse the CSV plane list into one record per data row, in row order.

    Each non-empty row becomes either a validated :class:`Plane` or a
    :class:`PlaneError` naming the exact failing rule (FR-10/FR-11) - no
    row ever stops the run (NFR-4 groundwork; the error-frame ticket turns
    these into banners). File-level problems (missing, directory, empty)
    stay whole-run :class:`UsageError`s (FR-2).

    Tolerant of what the corpus provides (FR-9): a missing trailing
    newline, integer- or float-coordinate cells, and blank separator rows.
    """
    return [_parse_row(i, row) for i, row in enumerate(iter_csv_rows(Path(path)), 1)]


def count_planes(path: str | Path) -> int:
    """Count the CSV plane-list's data rows (the header row is excluded).

    Full parsing/validation of each row lives in :func:`parse_planes`; the
    count is simply one record per non-empty row, so the video keeps one
    frame per row, in row order (FR-17). Tolerant already of
    what the corpus provides: integer- or float-coordinate cells and a
    missing trailing newline (FR-9); rows that are entirely empty are not
    planes.
    """
    return len(parse_planes(path))