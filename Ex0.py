"""Ex0.py — plane-projection CLI skeleton.

Brief's demo contract (FR-1): ``python Ex0.py file.nrrd file.csv file.mp4``.

Ticket 1 pieces (SRS ``docs/srs.md``):

* :func:`load_volume` — NRRD volume loader (FR-3..FR-6).
* :func:`count_planes` — CSV plane-list row count (FR-17 groundwork; full
  parsing/validation is the next ticket, FR-7..FR-11).
* :func:`write_placeholder_video` — one black placeholder frame per CSV row,
  at the SRS's video-container contract (FR-19; the real canvas arrives with
  the renderer, FR-18).
* :func:`main` — argument handling and the whole-run pipeline (FR-1, FR-2).

Requirements are normative in ``docs/srs.md``; the canonical coordinate space
(voxel-index space) is decided in ``docs/adr/0002-canonical-voxel-index-space.md``.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import nrrd


class UsageError(Exception):
    """A user-fixable input/usage problem reported as a one-line CLI error (FR-2)."""


# ---------------------------------------------------------------------------
# Volume loader (FR-3..FR-6)
# ---------------------------------------------------------------------------

def load_volume(path: str | Path) -> np.ndarray:
    """Load a 3D NRRD volume as canonical voxel-index space ``M[x][y][z]``.

    The returned array is indexed directly as brief semantics require:
    ``M[x][y][z] == volume[x, y, z]`` with ``sx, sy, sz == volume.shape``.
    ``index_order='F'`` keeps pynrrd's array axis 0 == the NRRD's axis 0 (x,
    the fastest axis on disk, per the NRRD spec §1.4 axis ordering) instead of
    letting an implicit convention decide it.

    World-space header fields (``space``, ``space origin``,
    ``space directions``) are read by pynrrd but deliberately **ignored**: the
    canonical coordinate space is voxel-index space per ADR 0002
    (``docs/adr/0002-canonical-voxel-index-space.md``, FR-6). Anisotropic
    spacing is ignored with it.

    Dtype contract (FR-4): ``uint8``/``uint16``/``uint32`` are the brief's
    8/16/32-bit unsigned contract; signed integer types (e.g. ``short``, as in
    the lecturer's ``MRHead_1.nrrd``) are tolerated, not rejected. Everything
    else (floats, ``uint64``, vectors, …) is out of contract and rejected.

    Dimensions must satisfy ``1 <= s < 2**10`` per axis (FR-5).
    """
    path = Path(path)
    if path.is_dir():
        raise UsageError(f"cannot read volume: '{path}' is a directory, not a file")
    try:
        data, _header = nrrd.read(path, index_order="F")  # FR-6: header intentionally ignored
    except FileNotFoundError as e:
        raise UsageError(f"cannot read volume '{path}': no such file") from e
    except PermissionError as e:
        raise UsageError(f"cannot read volume '{path}': permission denied") from e
    except OSError as e:
        raise UsageError(f"cannot read volume '{path}': {e.strerror or e}") from e
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
    return data


# ---------------------------------------------------------------------------
# Plane-list counting (FR-17 groundwork; full parsing: next ticket, FR-7..FR-11)
# ---------------------------------------------------------------------------

def count_planes(path: str | Path) -> int:
    """Count the CSV plane-list's data rows (the header row is excluded).

    Full parsing/validation of each row lands with the plane-list ticket
    (FR-7..FR-11); this ticket only needs the row count so the placeholder
    video has one frame per row, in row order (FR-17). Tolerant already of
    what the corpus provides: integer- or float-coordinate cells and a missing
    trailing newline (FR-9); rows that are entirely empty are not planes.
    """
    if Path(path).is_dir():
        raise UsageError(f"cannot read plane list: '{path}' is a directory, not a file")
    try:
        with open(path, newline="") as file:
            reader = csv.reader(file)
            try:
                next(reader)  # header row (FR-7) — not counted
            except StopIteration:
                raise UsageError(f"cannot read plane list '{path}': empty file (no header)") from None
            try:
                return sum(1 for row in reader if any(cell.strip() for cell in row))
            except csv.Error as e:
                raise UsageError(f"cannot parse plane list '{path}': {e}") from e
    except FileNotFoundError as e:
        raise UsageError(f"cannot read plane list '{path}': no such file") from e
    except PermissionError as e:
        raise UsageError(f"cannot read plane list '{path}': permission denied") from e
    except OSError as e:
        raise UsageError(f"cannot read plane list '{path}': {e.strerror or e}") from e
