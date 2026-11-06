"""Ex0.py — plane-projection CLI.

Brief's demo contract (FR-1): ``python Ex0.py file.nrrd file.csv file.mp4``.

Requirements are normative in ``docs/srs.md``:

* :func:`load_volume` — NRRD volume loader (FR-3..FR-6).
* :func:`parse_planes` — CSV plane-list parsing + per-row validation into
  :class:`Plane` / :class:`PlaneError` records (FR-7..FR-11); :func:`count_planes`
  counts the records (FR-17 groundwork).
* :func:`render_frames` — the render seam: renders each validated plane to a
  density-sized grayscale frame (FR-12/FR-13; geometry pinned in
  ``docs/algorithm.md``) and isolates per-row render failures.
* :func:`canvas_size` / :func:`pad_frame` — the video's fixed even canvas and
  top-left black padding (FR-18/FR-19/FR-21).
* :func:`render_error_frame` — a black canvas with a red, row-specific banner
  for validation and rendering failures (FR-14..FR-16).
* :func:`write_video` — the video container (FR-17/FR-19): one frame per CSV
  row, in row order.
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
from collections.abc import Iterable, Iterator, Sequence
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
# Frame rendering (FR-12, FR-13; geometry pinned in docs/algorithm.md)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    """One rendered plane projection, display-scaled to grayscale uint8.

    ``image`` is the frame at its own density-derived size (H = round(leg),
    W = round(base)); canvas padding is a separate assembler step (FR-18).
    """

    row: int
    image: np.ndarray


def _display_scale(values: np.ndarray) -> np.ndarray:
    """Map integer voxel values over their dtype's full range to uint8 (FR-12e).

    No content-based stretching: unsigned values use ``0..dtype_max``; signed
    values (tolerated per FR-4) use ``dtype_min..dtype_max``. Missing samples
    are not scaled by this helper - the caller leaves them black (FR-12d).
    """
    info = np.iinfo(values.dtype)
    lo, hi = info.min, info.max
    if hi == lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.rint((values.astype(np.float64) - lo) * (255.0 / (hi - lo)))
    return scaled.astype(np.uint8)


def _plane_corners(plane: Plane) -> tuple[Point, Point, Point, Point]:
    """The plane's ``(top_left, top_right, bottom_left, bottom_right)`` corners.

    CSV point order maps ``a1 → top_left``, ``a2 → top_right``,
    ``b → bottom_left``, ``c → bottom_right`` (FR-8). A 3-point plane is the
    same geometry with the top edge collapsed to the single apex
    ``a``: ``top_left == top_right == a``, so the top edge has zero width and
    rows widen linearly down to the base.
    """
    if plane.is_trapezoid:
        return (plane.points[0], plane.points[1], plane.points[2], plane.points[3])
    a, b, c = plane.points
    return (a, a, b, c)


def _render_plane(volume: np.ndarray, plane: Plane) -> Frame:
    """Render one validated plane as a display-scaled 2D uint8 frame (FR-12).

    Analytic geometry, no accumulation (docs/algorithm.md — same names there):
    row ``row_index`` samples ``row_fraction = (row_index + 1) / frame_height``
    (one leg step taken before each output row is written, FR-12b); column
    ``column_index`` samples the row at pixel-centre distance
    ``offset_from_centre = column_index + 0.5 - frame_width / 2`` from the row
    centre. Samples outside the plane's row span are black; sample points
    outside the volume are black (FR-12d). Nearest-cube lookup rounds each
    sample point with ``np.rint`` (half-to-even, FR-12a) and is resolved with
    a boolean in-bounds mask *before* fancy indexing, so every index handed
    to numpy is guaranteed in range.
    """
    top_left, top_right, bottom_left, bottom_right = _plane_corners(plane)
    leg_length = (_dist(top_left, bottom_left) + _dist(top_right, bottom_right)) / 2.0
    base_length = _dist(bottom_left, bottom_right)
    frame_height = max(1, int(np.rint(leg_length)))
    frame_width = max(1, int(np.rint(base_length)))

    left_apex, right_apex = np.asarray(top_left), np.asarray(top_right)
    left_base, right_base = np.asarray(bottom_left), np.asarray(bottom_right)
    row_fraction = (np.arange(frame_height) + 1.0) / frame_height
    left_edge = left_apex + row_fraction[:, None] * (left_base - left_apex)
    right_edge = right_apex + row_fraction[:, None] * (right_base - right_apex)

    row_centre = (left_edge + right_edge) / 2.0
    row_span = right_edge - left_edge
    row_width = np.linalg.norm(row_span, axis=1)
    row_direction = np.zeros_like(row_span)
    np.divide(row_span, row_width[:, None], out=row_direction,
              where=row_width[:, None] != 0)

    offset_from_centre = np.arange(frame_width) + 0.5 - frame_width / 2.0
    inside_row = np.abs(offset_from_centre)[None, :] <= row_width[:, None] / 2.0 + 1e-9

    # The sample point's axes are built by the same elementwise float ops as
    # the whole-array expression above — row-centre + offset ⋅ row-direction —
    # only axis-at-a-time, so the FP arithmetic (and thus every np.rint
    # rounding decision, FR-12a) is bit-identical while the temporaries drop
    # from (H, W, 3) float64 blocks to one (H, W) per axis.
    row_dx, row_dy, row_dz = (
        row_direction[:, None, axis] for axis in range(3)
    )
    sample_x = row_centre[:, None, 0] + offset_from_centre[None, :] * row_dx
    sample_y = row_centre[:, None, 1] + offset_from_centre[None, :] * row_dy
    sample_z = row_centre[:, None, 2] + offset_from_centre[None, :] * row_dz

    voxel_x = np.rint(sample_x).astype(np.intp)
    voxel_y = np.rint(sample_y).astype(np.intp)
    voxel_z = np.rint(sample_z).astype(np.intp)
    volume_sx, volume_sy, volume_sz = volume.shape
    # Unsigned-reinterpret bound test (one comparison per element): in the
    # C two's-complement representation, ``0 <= v < hi`` holds exactly when
    # the value reinterpreted as unsigned is ``< hi`` — negatives wrap to the
    # maximum unsigned order, so they fail the same comparison. Integer
    # arithmetic is exact, so the mask is identical to the signed compare.
    inside_volume = (
        (voxel_x.view(np.uintp) < volume_sx)
        & (voxel_y.view(np.uintp) < volume_sy)
        & (voxel_z.view(np.uintp) < volume_sz)
    )
    inside = inside_row & inside_volume

    image = np.zeros((frame_height, frame_width), dtype=np.uint8)
    if np.any(inside):
        x_idx, y_idx, z_idx = voxel_x[inside], voxel_y[inside], voxel_z[inside]
        gathered = volume[x_idx, y_idx, z_idx]
        info = np.iinfo(volume.dtype)
        if info.min == 0 and info.max == 255:
            # Full-range unsigned display scaling is the identity (FR-12e):
            # rint(v * 255/255) == v for every 0..255 value, bit-exactly —
            # skip the float round trip.
            image[inside] = gathered
        else:
            image[inside] = _display_scale(gathered)
    return Frame(row=plane.row, image=image)


def render_frames(
    volume: np.ndarray, records: Sequence[Plane | PlaneError]
) -> list[Frame | PlaneError]:
    """Render validated rows and isolate row failures, in list order (FR-14, FR-17).

    This is the single render seam: the CLI and every renderer test call it.
    Validation errors stay :class:`PlaneError` records. Exceptions raised
    while rendering an individual valid row are converted to the same
    per-row error record so later rows still render.
    """
    rendered: list[Frame | PlaneError] = []
    for record in records:
        if isinstance(record, PlaneError):
            rendered.append(record)
            continue
        try:
            rendered.append(_render_plane(volume, record))
        except Exception as error:
            reason = str(error).replace("\r", " ").replace("\n", " ").strip()
            if not reason:
                reason = type(error).__name__
            rendered.append(
                PlaneError(record.row, f"render failed ({type(error).__name__}): {reason}")
            )
    return rendered


def canvas_size(records: Sequence[Frame | PlaneError]) -> tuple[int, int]:
    """The video's fixed canvas ``(width, height)`` (FR-18, FR-19).

    The max width/height across renderable frames, rounded up to even so the
    yuv420p-style encoder accepts it. When no row renders (all errors, FR-20)
    there is no frame to measure, so the placeholder canvas carries the video.
    """
    frames = [record.image for record in records if isinstance(record, Frame)]
    if not frames:
        return PLACEHOLDER_CANVAS
    width = max(image.shape[1] for image in frames)
    height = max(image.shape[0] for image in frames)
    return (width + width % 2, height + height % 2)


def pad_frame(image: np.ndarray, canvas: tuple[int, int]) -> np.ndarray:
    """Place ``image`` top-left on a black canvas ``(width, height)`` (FR-18).

    Padding is background black with no aspect compensation (FR-21); a frame
    larger than the canvas is an internal error, never silently cropped.
    """
    width, height = canvas
    if image.shape[0] > height or image.shape[1] > width:
        raise ValueError(
            f"frame {image.shape[1]}x{image.shape[0]} is larger than canvas {width}x{height}"
        )
    padded = np.zeros((height, width), dtype=np.uint8)
    padded[: image.shape[0], : image.shape[1]] = image
    return padded


def render_error_frame(error: PlaneError, canvas: tuple[int, int]) -> np.ndarray:
    """Draw a black BGR frame with the row's error text in a red top banner (FR-14/FR-15)."""
    width, height = canvas
    image = np.zeros((height, width, 3), dtype=np.uint8)
    text = error.banner()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    horizontal_padding = 8
    vertical_padding = 5

    def text_width(value: str) -> int:
        return cv2.getTextSize(value, font, scale, thickness)[0][0]

    max_width = max(1, width - 2 * horizontal_padding)
    # Shrink until the widest single word fits its line: overlong words
    # cannot wrap, so the scale keeps dropping (no clipping — every drawn
    # glyph stays inside the canvas) until it does.
    while scale > 0.02 and max(text_width(word) for word in text.split()) > max_width:
        scale *= 0.8

    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and text_width(candidate) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)

    (_, line_height), baseline = cv2.getTextSize("Ag", font, scale, thickness)
    line_step = line_height + baseline + 2
    banner_height = min(height, len(lines) * line_step + 2 * vertical_padding)
    cv2.rectangle(image, (0, 0), (width - 1, banner_height - 1), (0, 0, 255), -1)
    for index, value in enumerate(lines):
        y = vertical_padding + line_height + index * line_step
        if y >= banner_height:
            break
        cv2.putText(
            image,
            value,
            (horizontal_padding, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return image


# ---------------------------------------------------------------------------
# Plane-list counting (FR-17 groundwork; superseded body: one record per row)
# ---------------------------------------------------------------------------

def iter_canvas_frames(
    rendered: Iterable[Frame | PlaneError], canvas: tuple[int, int]
) -> Iterator[np.ndarray]:
    """Yield canvas-sized padded / bannered frames in run order (FR-18/FR-14).

    The single canvas-assembly shape used by both the CLI and the benchmark
    harness: a valid row's image is padded top-left (FR-18/FR-21), an error
    row becomes the bannered frame (FR-14). A generator so canvas-sized
    copies stay transient inside :func:`write_video`.
    """
    for record in rendered:
        yield (
            pad_frame(record.image, canvas)
            if isinstance(record, Frame)
            else render_error_frame(record, canvas)
        )


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


# ---------------------------------------------------------------------------
# Video writer (FR-17..FR-21 container contract; encoder choice: docs/adr/0004)
# ---------------------------------------------------------------------------

#: Fallback canvas (width, height) when a CSV has no renderable row at all
#: (every row an error, FR-20). Real runs use :func:`canvas_size` (the max
#: width/height across renderable frames, FR-18). Both dimensions are even so
#: the yuv420p pixel format is satisfied (FR-19).
PLACEHOLDER_CANVAS = (256, 256)

#: Container rate, in frames/second (FR-19). A container choice only —
#: deliberately decoupled from rendering throughput (NFR-2).
VIDEO_FPS = 30


def write_video(
    out_path: str | Path,
    frames: Iterable[np.ndarray],
    canvas: tuple[int, int],
    fps: int = VIDEO_FPS,
) -> None:
    """Write grayscale or BGR uint8 frames of one fixed canvas to mp4 (FR-17).

    Frames are written in iteration order (one per CSV row); grayscale frames
    are converted to BGR and color frames are written directly. The container
    contract (FR-19) is: uniform frame size, 30 FPS container rate, mp4,
    MPEG-4 Part 2 (``mp4v``), even dimensions (yuv420p-style chroma
    subsampling compatibility). The encoder is OpenCV's ``cv2.VideoWriter``
    with the ``mp4v`` fourcc — see docs/adr/0004-video-encoding-via-opencv-mp4v.md.

    A generator is accepted so canvas-sized padded copies are transient;
    :func:`render_frames` itself returns one record per row up front.
    """
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
        for frame in frames:
            if frame.shape not in {(height, width), (height, width, 3)}:
                raise ValueError(
                    f"frame size {frame.shape[1]}x{frame.shape[0]} does not match canvas"
                    f" {width}x{height}"
                )
            if frame.dtype != np.uint8:
                raise ValueError(f"frame dtype {frame.dtype} is not uint8")
            bgr_frame = (
                cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if frame.ndim == 2
                else frame
            )
            writer.write(bgr_frame)
    finally:
        writer.release()  # flush and close the container even on write failure


# ---------------------------------------------------------------------------
# CLI (FR-1, FR-2)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the brief's demo contract: ``python Ex0.py file.nrrd file.csv file.mp4``.

    Exactly three positional arguments (FR-1). Every user-fixable problem
    exits non-zero with a one-line error naming the problem file (FR-2);
    per-row plane errors become bannered frames and do not stop the run
    (FR-14..FR-16).
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
        volume = load_volume(volume_path)
        records = parse_planes(plane_list_path)
        rendered = render_frames(volume, records)
        canvas = canvas_size(rendered)
        write_video(
            video_path,
            iter_canvas_frames(rendered, canvas),
            canvas=canvas,
            fps=VIDEO_FPS,
        )
    except UsageError as e:
        print(f"Ex0.py: {e}", file=sys.stderr)
        return 1

    # Error output uses the exact banner text so logs and video agree.
    for source, result in zip(records, rendered):
        if isinstance(result, PlaneError):
            print(result.banner(), file=sys.stderr)
        elif isinstance(source, Plane):
            print(f"row {source.row}: ok ({'trapezoid' if source.is_trapezoid else 'triangle'})")

    errors = sum(isinstance(r, PlaneError) for r in rendered)
    print(
        f"wrote {len(rendered)} frames ({canvas[0]}x{canvas[1]}, mp4v @{VIDEO_FPS} fps)"
        f" to '{video_path}' ({errors} error row(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
