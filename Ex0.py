"""Ex0.py — plane-projection CLI.

Brief's demo contract (FR-1): ``python Ex0.py file.nrrd file.csv file.mp4``.

Ticket 1 pieces (requirements are normative in ``docs/srs.md``):

* :func:`load_volume` — NRRD volume loader (FR-3..FR-6).
* :func:`parse_planes` — CSV plane-list parsing + per-row validation into
  :class:`Plane` / :class:`PlaneError` records (FR-7..FR-11); :func:`count_planes`
  counts the records (FR-17 groundwork).
* :func:`write_video` — the video container (FR-19); this ticket writes one
  black placeholder frame per CSV row at a placeholder canvas (FR-18's real
  canvas arrives with the renderer).
* :func:`main` — argument handling and the whole-run pipeline (FR-1, FR-2).

Canonical coordinate space is voxel-index space, per
``docs/adr/0002-canonical-voxel-index-space.md``; video encoding via OpenCV's
``cv2.VideoWriter`` (``mp4v`` codec) per
``docs/adr/0004-video-encoding-via-opencv-mp4v.md`` (superseding ADR 0003's
ffmpeg-CLI choice — everything stays pip-installable, no system binary).
"""

from __future__ import annotations

import csv
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast
from pathlib import Path

import cv2
import numpy as np
import nrrd


class UsageError(Exception):
    """A user-fixable input/usage problem reported as a one-line CLI error (FR-2)."""


def _require_file(label: str, path: Path) -> None:
    """One-line usage error when a named input path is a directory (FR-2).

    A missing file is NOT pre-checked: the open/read call raises FileNotFoundError
    (no time-of-check/time-of-use lie); :func:`_os_errors_as_usage` converts it.
    """
    if path.is_dir():
        raise UsageError(f"cannot read {label}: '{path}' is a directory, not a file")


@contextmanager
def _os_errors_as_usage(label: str, path: Path) -> Iterator[None]:
    """Convert file-access OS errors into one-line usage errors (FR-2)."""
    try:
        yield
    except FileNotFoundError as e:
        raise UsageError(f"cannot read {label} '{path}': no such file") from e
    except PermissionError as e:
        raise UsageError(f"cannot read {label} '{path}': permission denied") from e
    except OSError as e:
        raise UsageError(f"cannot read {label} '{path}': {e.strerror or e}") from e


# ---------------------------------------------------------------------------
# Volume loader (FR-3..FR-6)
# ---------------------------------------------------------------------------

def load_volume(path: str | Path) -> np.ndarray:
    """Load a 3D NRRD volume as canonical voxel-index space ``M[x][y][z]``.

    The returned array is indexed directly as the brief requires:
    ``M[x][y][z] == volume[x, y, z]`` with ``sx, sy, sz == volume.shape``.
    ``index_order='F'`` pins pynrrd's array axis 0 == the NRRD's axis 0 (x,
    the fast axis on disk, per the NRRD spec §1.4 axis ordering) instead of
    leaving the mapping to an implicit library default.

    World-space header fields (``space``, ``space origin``,
    ``space directions``) are read by pynrrd but deliberately **ignored**:
    the canonical coordinate space is voxel-index space per ADR 0002
    (``docs/adr/0002-canonical-voxel-index-space.md``, FR-6). Anisotropic
    spacing is ignored with it.

    Dtype contract (FR-4): ``uint8``/``uint16``/``uint32`` are the brief's
    8/16/32-bit unsigned contract; signed integer types (e.g. ``short``, as
    in the lecturer's ``MRHead_1.nrrd``) are tolerated, not rejected.
    Everything else (floats, ``uint64``, non-integer types) is outside the
    contract and rejected.

    Dimensions must satisfy ``1 <= s < 2**10`` per axis (FR-5).
    """
    path = Path(path)
    _require_file("volume", path)
    try:
        with _os_errors_as_usage("volume", path):
            data, _header = nrrd.read(path, index_order="F")  # FR-6: header deliberately unused
    except UsageError:
        raise
    except Exception as e:  # pynrrd parse errors
        raise UsageError(f"cannot load NRRD '{path}': {e}") from e

    if data.ndim != 3:
        raise UsageError(
            f"volume '{path}' has {data.ndim} dimensions; expected exactly 3 (sx, sy, sz)"
        )

    axis_names = ("sx", "sy", "sz")
    for name, size in zip(axis_names, data.shape):
        if not 1 <= size < 2**10:
            raise UsageError(
                f"volume '{path}' has {name}={size}; every dimension must satisfy"
                f" 1 <= s < 2^10 (FR-5)"
            )

    kind, itemsize = data.dtype.kind, data.dtype.itemsize
    if not (kind == "i" or (kind == "u" and itemsize in (1, 2, 4))):
        raise UsageError(
            f"volume '{path}' has unsupported element type '{data.dtype}'; expected"
            f" uint8/uint16/uint32 (the brief's 8/16/32-bit unsigned contract) or a"
            f" signed integer type (tolerated), per FR-4"
        )
    # nrrd.read is untyped (mypy: Any); it always returns an ndarray here.
    return cast(np.ndarray, data)


# ---------------------------------------------------------------------------
# Plane-list parsing + per-row validation (FR-7..FR-11)
# ---------------------------------------------------------------------------

#: Plane validity tolerance (brief Q&A; SRS FR-10).
EPS = 0.001

#: Relative tolerance for the coplanarity predicate (rule 3, 4-point): the
#: scalar triple product of the leg directions and the apex separation is
#: compared against the product of the involved lengths, so the test is
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
    leg; ``a1 == a2`` with parallel legs gives a zero-width frame, while
    ``a1 == a2`` with diverging legs is a triangle and renders fine.
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
        # a1 == a2: only parallel legs make d12 stay 0 on every row.
        if _cross(leg1, leg2) == (0.0, 0.0, 0.0):
            return "degenerate: a1 == a2 with parallel legs (zero-width frame)"
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


def _iter_plane_rows(path: Path) -> Iterator[list[str]]:
    """Yield the CSV's non-empty data rows (header excluded), FR-9-tolerant."""
    _require_file("plane list", path)
    with _os_errors_as_usage("plane list", path):
        with open(path, newline="") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header is None:
                raise UsageError(f"cannot read plane list '{path}': empty file (no header)")
            for row in reader:
                if any(cell.strip() for cell in row):
                    yield row


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
    return [_parse_row(i, row) for i, row in enumerate(_iter_plane_rows(Path(path)), 1)]


# ---------------------------------------------------------------------------
# Plane-list counting (FR-17 groundwork; superseded body: one record per row)
# ---------------------------------------------------------------------------

def count_planes(path: str | Path) -> int:
    """Count the CSV plane-list's data rows (the header row is excluded).

    Full parsing/validation of each row lives in :func:`parse_planes`; the
    count is simply one record per non-empty row, so the placeholder video
    keeps one frame per row, in row order (FR-17). Tolerant already of
    what the corpus provides: integer- or float-coordinate cells and a
    missing trailing newline (FR-9); rows that are entirely empty are not
    planes.
    """
    return len(parse_planes(path))


# ---------------------------------------------------------------------------
# Placeholder video writer (FR-19 container contract; FR-18's real canvas
# arrives with the renderer — encoder choice: docs/adr/0003)
# ---------------------------------------------------------------------------

#: Placeholder canvas (width, height) in pixels, until the renderer computes
#: the real one (max width/height across the CSV's renderable frames, FR-18).
#: Placeholder only — derived from no plane; the CSV drives row count here,
#: nothing else. Both are even so the yuv420p pixel format is satisfied
#: (FR-19).
PLACEHOLDER_CANVAS = (256, 256)

#: Container rate, in frames/second (FR-19). A container choice only —
#: deliberately decoupled from rendering throughput (NFR-2).
VIDEO_FPS = 30


def write_video(
    out_path: str | Path,
    n_frames: int,
    canvas: tuple[int, int],
    fps: int = VIDEO_FPS,
) -> None:
    """Write ``n_frames`` black placeholder frames at the given canvas (W, H).

    One black frame per CSV row at a uniform canvas stands in for the real
    projections (the renderer ticket replaces their content; FR-18's real
    canvas likewise). The container contract (FR-19) is final from this
    ticket on: uniform frame size, 30 FPS container rate, mp4, MPEG-4 Part 2
    (``mp4v``), even dimensions (yuv420p-style chroma subsampling compatibility).

    The encoder is OpenCV's ``cv2.VideoWriter`` with the ``mp4v`` fourcc —
    see docs/adr/0004-video-encoding-via-opencv-mp4v.md: pip-installable
    everywhere (supersedes the ffmpeg-CLI choice of ADR 0003, whose system
    binary requirement is gone).
    """
    if n_frames < 0:
        raise UsageError(f"cannot write video '{out_path}': negative frame count {n_frames}")
    width, height = canvas
    if width <= 0 or height <= 0:
        raise UsageError(f"cannot write video '{out_path}': canvas {canvas!r} must be positive")
    if width % 2 or height % 2:
        # Chroma-subsampling encoders require even dimensions (FR-19); rejecting
        # up front avoids a silently wrong container later.
        raise UsageError(
            f"cannot write video '{out_path}': canvas {canvas!r} must have even dimensions"
        )

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]  # stubs omit it
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise UsageError(
            f"cannot write video '{out_path}': cv2.VideoWriter could not be opened"
            " (check the output path is writable and the mp4v codec is available)"
        )
    try:
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(n_frames):
            writer.write(black_frame)
    finally:
        writer.release()  # flush and close the container even on write failure


# ---------------------------------------------------------------------------
# CLI (FR-1, FR-2)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the brief's demo contract: ``python Ex0.py file.nrrd file.csv file.mp4``.

    Exactly three positional arguments (FR-1). Every user-fixable problem
    exits non-zero with a one-line error naming the problem file (FR-2);
    per-row plane errors arrive with the parsing ticket (FR-14..FR-16).
    """
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print(
            "Ex0.py: expected exactly 3 arguments: volume.nrrd planes.csv output.mp4;"
            f" got {len(argv)}",
            file=sys.stderr,
        )
        return 2

    volume_path, plane_list_path, video_path = argv
    try:
        load_volume(volume_path)
        records = parse_planes(plane_list_path)
        write_video(video_path, len(records), canvas=PLACEHOLDER_CANVAS, fps=VIDEO_FPS)
    except UsageError as e:
        print(f"Ex0.py: {e}", file=sys.stderr)
        return 1

    # Per-row verdicts (FR-16 groundwork): the same text the future error
    # banners will carry, greppable without re-rendering. Valid rows say so;
    # error rows go to stderr, one line each, and never stop the run.
    for record in records:
        if isinstance(record, PlaneError):
            print(f"Ex0.py: {record.banner()}", file=sys.stderr)
        else:
            print(f"row {record.row}: ok ({'trapezoid' if record.is_trapezoid else 'triangle'})")

    errors = sum(isinstance(r, PlaneError) for r in records)
    print(
        f"wrote {len(records)} placeholder frames ({PLACEHOLDER_CANVAS[0]}x"
        f"{PLACEHOLDER_CANVAS[1]}, mp4v @{VIDEO_FPS} fps) to '{video_path}'"
        f" ({errors} error row(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
