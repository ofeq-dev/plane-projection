"""bench.py — benchmark harness (NFR-1, NFR-2).

Measures end-to-end throughput (frames ÷ wall time over **load + render**)
with the [``time.perf_counter()``](https://docs.python.org/3/library/time.html#time.perf_counter)
monotonic high-resolution clock, reported as min-of-repeats per the
[timeit methodology](https://docs.python.org/3/library/timeit.html). The
container encode is decoupled from the container's 30 FPS metadata (NFR-2):
its wall time is measured and reported **out of** the throughput figure.

Runs the exact production pipeline from :mod:`Ex0` (load → parse → render →
canvas → encode), so a harness number is a number for the tool itself.

Usage::

    python bench.py volume.nrrd planes.csv [--repeats N] [--out results.md]

Results land in a versioned markdown file (see ``benchmarks/``) whose header
carries the methodology sentence, hardware context, and — one per row — the
exact one-line command that reproduced it, so the experiment write-up (AGENTS
TODO #7) can cite it wholesale.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy

import Ex0


Clock = Callable[[], float]


@dataclass(frozen=True)
class Timing:
    """One timed run, phase-split per NFR-2."""

    frames: int
    load_seconds: float
    render_seconds: float
    encode_seconds: float


@dataclass(frozen=True)
class ResultRow:
    """One results-file row: a reproducible command and its best timing."""

    label: str
    command: str
    timing: Timing


@dataclass(frozen=True)
class ReportContext:
    """Hardware + software context so cross-machine numbers aren't apples-to-oranges."""

    date: str
    cpu: str
    python: str
    numpy: str
    codec_note: str


def timed_pipeline(
    volume_path: str | Path,
    plane_list_path: str | Path,
    video_path: str | Path | None,
    *,
    clock: Clock,
) -> Timing:
    """Time one full pipeline run; write the mp4 when ``video_path`` is given.

    Phase boundaries are three/four ``clock()`` checkpoints (start, after
    load, after render, after encode); every phase time is a difference of
    consecutive checkpoints, so the scriptable clock in tests makes every
    figure deterministic.
    """
    start = clock()
    volume = Ex0.load_volume(volume_path)
    after_load = clock()
    records = Ex0.parse_planes(plane_list_path)
    rendered = Ex0.render_frames(volume, records)
    canvas = Ex0.canvas_size(rendered)
    after_render = clock()

    frames = sum(isinstance(record, Ex0.Frame) for record in rendered)
    encode_seconds = 0.0
    if video_path is not None:
        Ex0.write_video(
            video_path,
            (
                Ex0.pad_frame(record.image, canvas)
                if isinstance(record, Ex0.Frame)
                else Ex0.render_error_frame(record, canvas)
                for record in rendered
            ),
            canvas=canvas,
            fps=Ex0.VIDEO_FPS,
        )
        encode_seconds = clock() - after_render

    return Timing(
        frames=frames,
        load_seconds=after_load - start,
        render_seconds=after_render - after_load,
        encode_seconds=encode_seconds,
    )


def throughput(timing: Timing) -> float:
    """Frames ÷ (load + render) wall time — the NFR-1 figure.

    The encoder's wall time is deliberately excluded (NFR-2: throughput is
    decoupled from the container's 30 FPS metadata); it is reported alongside.
    """
    if timing.frames == 0:
        raise ValueError(
            "no rendered frames: throughput is undefined for a run that rendered nothing"
        )
    seconds = timing.load_seconds + timing.render_seconds
    return timing.frames / seconds


def choose_min(timings: Sequence[Timing]) -> Timing:
    """min-of-repeats (NFR-2): the repeat with the shortest wall time.

    With a fixed frame count the shortest wall time is the highest FPS, so
    the chosen figures form one coherent (least favourable noise) repeat.
    """
    if not timings:
        raise IndexError("min-of-repeats needs at least one timing")
    return min(
        timings,
        key=lambda timing: timing.load_seconds + timing.render_seconds,
    )


def detect_cpu() -> str:
    """One human-readable line naming this machine's processor and cores."""
    model = platform.processor() or platform.machine()
    cores = len(__import__("os").sched_getaffinity(0)) if hasattr(__import__("os"), "sched_getaffinity") else __import__("os").cpu_count() or 1
    return f"{model} ({cores} cores)"


def report_context() -> ReportContext:
    """The hardware + software context recorded next to every results file."""
    return ReportContext(
        date=time.strftime("%Y-%m-%d %H:%M %z"),
        cpu=detect_cpu(),
        python=sys.version.split()[0],
        numpy=numpy.__version__,
        codec_note="encode: cv2.VideoWriter mp4v @30 fps (ADR 0004), excluded from throughput",
    )


def write_results(
    results_path: Path,
    context: ReportContext,
    rows: Sequence[ResultRow],
) -> None:
    """Write the versioned results file: methodology, hardware, quoted commands.

    Every figure's provenance lives in the file itself, so the experiment
    write-up (TODO #7) cites the file, not folklore.
    """
    lines = [
        "# Benchmark results",
        "",
        f"Recorded: {context.date}",
        f"Machine: {context.cpu}",
        f"Software: Python {context.python}, numpy {context.numpy}, {context.codec_note}",
        "",
        "Methodology (NFR-2): wall time via `time.perf_counter()` (monotonic,"
        " highest resolution); best = min-of-repeats (timeit methodology);"
        " scope = load + render; encode reported separately and excluded from"
        " every throughput figure; throughput = rendered frames ÷ wall time.",
        "",
        "| input | command | frames | load s | render s | encode s | FPS (load+render) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        frames = row.timing.load_seconds + row.timing.render_seconds
        best = throughput(row.timing)
        lines.append(
            f"| {row.label} | `{row.command}` | {row.timing.frames}"
            f" | {row.timing.load_seconds:.3f} | {row.timing.render_seconds:.3f}"
            f" | {row.timing.encode_seconds:.3f} | {best:.1f} |"
        )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the harness: ``python bench.py volume.nrrd planes.csv [--repeats N]``."""
    parser = argparse.ArgumentParser(
        prog="bench.py",
        description="Benchmark the plane-projection pipeline (NFR-1/NFR-2).",
    )
    parser.add_argument("volume", help="input .nrrd volume")
    parser.add_argument("planes", help="input .csv plane list")
    parser.add_argument("--repeats", type=int, default=3, metavar="N",
                        help="min-of-repeats count (default: 3)")
    parser.add_argument("--out", metavar="MARKDOWN", default=None,
                        help="also append the best figures to this results file")
    args = parser.parse_args(argv)

    timings = [
        timed_pipeline(args.volume, args.planes, None, clock=time.perf_counter)
        for _ in range(args.repeats)
    ]
    best = choose_min(timings)
    print(
        f"{args.volume} x {args.planes}: {best.frames} frames,"
        f" load {best.load_seconds:.3f}s, render {best.render_seconds:.3f}s,"
        f" encode {best.encode_seconds:.3f}s ->"
        f" {throughput(best):.1f} FPS (load+render, min-of-{args.repeats})"
    )

    if args.out is not None:
        row = ResultRow(
            label=f"{Path(args.volume).name} x {Path(args.planes).name}",
            command=f"python bench.py {args.volume} {args.planes} --repeats {args.repeats}",
            timing=best,
        )
        write_results(args.out, report_context(), [row])
        print(f"appended 1 row to '{args.out}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
