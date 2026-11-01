"""Ex0.py — plane-projection CLI.

Brief's demo contract (FR-1): ``python Ex0.py file.nrrd file.csv file.mp4``.

Ticket 1 pieces (requirements are normative in ``docs/srs.md``):

* :func:`load_volume` — NRRD volume loader (FR-3..FR-6).
* :func:`count_planes` — CSV plane-list row count (FR-17 groundwork; full
  parsing/validation is the next ticket, FR-7..FR-11).
* :func:`write_video` — the video container (FR-19); this ticket writes one
  black placeholder frame per CSV row at a placeholder canvas (FR-18's real
  canvas arrives with the renderer).
* :func:`main` — argument handling and the whole-run pipeline (FR-1, FR-2).

Canonical coordinate space is voxel-index space, per
``docs/adr/0002-canonical-voxel-index-space.md``; H.264 encoding via the
system ffmpeg CLI per ``docs/adr/0003-video-encoding-via-ffmpeg-cli.md``.
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
    if Path(path).is_dir():
        raise UsageError(f"cannot read volume: '{path}' is a directory, not a file")
    try:
        data, _header = nrrd.read(path, index_order="F")  # FR-6: header deliberately unused
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
    what the corpus provides: integer- or float-coordinate cells and a
    missing trailing newline (FR-9); rows that are entirely empty are not
    planes.
    """
    if Path(path).is_dir():
        raise UsageError(f"cannot read plane list: '{path}' is a directory, not a file")
    try:
        with open(path, newline="") as file:
            reader = csv.reader(file)
            try:
                next(reader)  # header row (FR-7) — not counted
            except StopIteration:
                raise UsageError(
                    f"cannot read plane list '{path}': empty file (no header)"
                ) from None
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
    ticket on: uniform frame size, 30 FPS container rate, mp4, H.264/AVC,
    even dimensions (yuv420p codec compatibility).

    The encoder is the system ``ffmpeg`` CLI (rawvideo/rgb24 piped on stdin),
    not OpenCV's VideoWriter — see docs/adr/0003-video-encoding-via-ffmpeg-cli.md:
    opencv-python's bundled ffmpeg build cannot encode H.264, and FR-19
    demands a genuine H.264 fourcc.
    """
    if n_frames < 0:
        raise UsageError(f"cannot write video '{out_path}': negative frame count {n_frames}")
    width, height = canvas
    if width <= 0 or height <= 0:
        raise UsageError(f"cannot write video '{out_path}': canvas {canvas!r} must be positive")
    if width % 2 or height % 2:
        # yuv420p chroma subsampling requires even dimensions (FR-19); the
        # ffmpeg CLI would silently crop odd frames instead (FR-19's point).
        raise UsageError(
            f"cannot write video '{out_path}': canvas {canvas!r} must have even dimensions"
        )
    if shutil.which("ffmpeg") is None:
        raise UsageError(
            f"cannot write video '{out_path}': ffmpeg not found on PATH"
            " (H.264 encoding relies on it; see docs/adr/0003-video-encoding-via-ffmpeg-cli.md)"
        )

    command = [
        "ffmpeg",
        "-y",  # overwrite an existing output file
        "-hide_banner",
        "-loglevel", "error",
        # stdin rawvideo: one frame = canvas pixels of rgb24 bytes, row-major
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",  # no audio track
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",  # moov first: progressively playable mp4
        str(out_path),
    ]
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except OSError as e:
        raise UsageError(f"cannot write video '{out_path}': ffmpeg failed to start: {e}") from e

    try:
        for _ in range(n_frames):
            process.stdin.write(black_frame.tobytes())
        process.stdin.close()
    except BrokenPipeError as e:
        _raise_encoder_failure(out_path, process, e)
    except OSError as e:
        raise UsageError(f"cannot write video '{out_path}': {e}") from e

    stderr_tail = process.stderr.read().decode(errors="replace").strip()
    return_code = process.wait()
    if return_code != 0:
        reason = " ".join(stderr_tail.replace("\n", " | ").split())[:300]
        raise UsageError(
            f"cannot write video '{out_path}': ffmpeg exited with status {return_code}"
            + (f": {reason}" if reason else "")
        )


def _raise_encoder_failure(
    out_path: str | Path, process: subprocess.Popen, cause: Exception
) -> None:
    """Report an encoder that stopped reading frames before the run finished."""
    return_code = process.wait()
    raise UsageError(
        f"cannot write video '{out_path}': ffmpeg stopped accepting frames"
        f" (exit status {return_code})"
    ) from cause


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
        plane_count = count_planes(plane_list_path)
        write_video(video_path, plane_count, canvas=PLACEHOLDER_CANVAS, fps=VIDEO_FPS)
    except UsageError as e:
        print(f"Ex0.py: {e}", file=sys.stderr)
        return 1

    print(
        f"wrote {plane_count} placeholder frames ({PLACEHOLDER_CANVAS[0]}x"
        f"{PLACEHOLDER_CANVAS[1]}, H.264 @{VIDEO_FPS} fps) to '{video_path}'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
