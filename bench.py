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
from plane_validation import parse_planes


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
    records = parse_planes(plane_list_path)
    rendered = Ex0.render_frames(volume, records)
    canvas = Ex0.canvas_size(rendered)
    after_render = clock()

    frames = sum(isinstance(record, Ex0.Frame) for record in rendered)
    encode_seconds = 0.0
    if video_path is not None:
        Ex0.write_video(
            video_path,
            Ex0.iter_canvas_frames(rendered, canvas),
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


def measure_best(
    volume_path: str | Path,
    plane_list_path: str | Path,
    repeats: int,
) -> Timing:
    """min-of-repeats over fresh full runs, timed with the real clock (NFR-2)."""
    timings = [
        timed_pipeline(volume_path, plane_list_path, None, clock=time.perf_counter)
        for _ in range(repeats)
    ]
    return choose_min(timings)


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
    import os

    model = platform.processor() or platform.machine()
    if sys.platform == "linux" and os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name\t"):
                    model = line.split(":", 1)[1].strip()
                    break
    cores = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else range(os.cpu_count() or 1)
    return f"{model} ({len(cores)} cores)"


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
    verdicts: Sequence[str] = (),
) -> None:
    """Write the versioned results file: methodology, hardware, quoted commands.

    Every figure's provenance lives in the file itself, so the experiment
    write-up (TODO #7) cites the file, not folklore. ``verdicts`` carries the
    evidence sentences (hard bar, target, measured drift) so a citation
    never has to reconstruct them from stdout.
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
        " the **load** scope is the NRRD read; the **render** scope is parse +"
        " render + canvas assembly; the mp4 write is *not* exercised:"
        " throughput runs do not write the mp4 — encode seconds appear only in"
        " the dedicated encode-probe row; throughput = rendered frames ÷"
        " (load + render) wall time.",
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
    if verdicts:
        lines += ["", *verdicts]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines) + "\n")


@dataclass(frozen=True)
class CorpusPair:
    """One (volume, plane list) benchmark pair from the corpus cross-product."""

    volume: Path
    csv: Path


def corpus_pairs(nrrd_dir: Path, csv_dir: Path) -> list[CorpusPair]:
    """Every CSV crossed with every volume, volumes sorted descending by size seed.

    Preferring the heaviest volume first keeps timing warm-up consistent:
    slower pairs always measure under similar cache conditions.
    """
    volumes = sorted(nrrd_dir.glob("*.nrrd"), reverse=True)
    csvs = sorted(csv_dir.glob("*.csv"))
    return [
        CorpusPair(volume=volume, csv=plane_list)
        for volume in volumes
        for plane_list in csvs
    ]


def hard_bar_verdict(fps: float, *, hard: float, target: float) -> str:
    """The NFR-1 evidence sentence: hard bar and 100 FPS target, explicitly."""
    if fps >= hard:
        target_text = (
            f"target of {target:g} FPS met" if fps >= target
            else f"target of {target:g} FPS NOT met"
        )
        return f"met the {hard:g} FPS hard bar ({target_text})"
    return f"NOT met the {hard:g} FPS hard bar"


def main(argv: list[str] | None = None) -> int:
    """Run the harness: ``python bench.py volume.nrrd planes.csv [--repeats N]``."""
    parser = argparse.ArgumentParser(
        prog="bench.py",
        description="Benchmark the plane-projection pipeline (NFR-1/NFR-2).",
    )
    parser.add_argument("volume", nargs="?", help="input .nrrd volume")
    parser.add_argument("planes", nargs="?", help="input .csv plane list")
    parser.add_argument("--corpus", action="store_true",
                        help="benchmark every assets/csv/*.csv against every assets/nrrd/*.nrrd")
    parser.add_argument("--repeats", type=int, default=3, metavar="N",
                        help="min-of-repeats count (default: 3)")
    parser.add_argument("--out", metavar="MARKDOWN", default=None,
                        help="write/overwrite this results file (corpus requires it)")
    parser.add_argument("--encode-probe", action="store_true",
                        help="also run one full run writing a throwaway mp4, to report the encoder's wall time")
    args = parser.parse_args(argv)

    if args.corpus:
        if not (args.volume is None and args.planes is None and args.out is not None):
            parser.error("--corpus takes no positional inputs and requires --out")
        repo_root = Path(__file__).parent
        pairs = corpus_pairs(repo_root / "assets" / "nrrd", repo_root / "assets" / "csv")
        rows: list[ResultRow] = []
        croissant_full: float | None = None
        drift: float | None = None
        for pair in pairs:
            best = measure_best(pair.volume, pair.csv, args.repeats)
            figure = throughput(best)
            label = f"{pair.volume.name} x {pair.csv.name}"
            volume_ref = pair.volume.relative_to(repo_root)
            csv_ref = pair.csv.relative_to(repo_root)
            command = f"python bench.py {volume_ref} {csv_ref} --repeats {args.repeats}"
            rows.append(ResultRow(label=label, command=command, timing=best))
            print(f"{label}: {figure:.1f} FPS (min-of-{args.repeats})")
            if pair.volume.name == "croissant.nrrd" and pair.csv.name == "croissant.csv":
                croissant_full = figure
                # drift criterion (AC): two independent min-of-repeats runs of the
                # canonical pair must agree within ~10% for the number to be citable
                second = measure_best(pair.volume, pair.csv, args.repeats)
                second_figure = throughput(second)
                drift = abs(second_figure - figure) / figure * 100.0
                print(f"{label}: repeat run {second_figure:.1f} FPS (drift {drift:.1f}%)")

        if croissant_full is None:
            raise ValueError(
                "corpus missing the canonical croissant.nrrd x croissant.csv pair:"
                " the NFR-1 hard bar cannot be evidenced"
            )

        # encode probe: the throughput runs never write the mp4 (NFR-2), so a
        # dedicated measured run records what the encoder itself costs.
        probe_out = Path(args.out).with_name("bench-encode-probe.mp4")
        probe = timed_pipeline(
            repo_root / "assets" / "nrrd" / "croissant.nrrd",
            repo_root / "assets" / "csv" / "croissant.csv",
            probe_out,
            clock=time.perf_counter,
        )
        probe_out.unlink(missing_ok=True)
        rows.append(ResultRow(
            label="croissant.nrrd x croissant.csv (encode probe: throwaway mp4 written)",
            command=(
                "python bench.py assets/nrrd/croissant.nrrd assets/csv/croissant.csv"
                " --repeats 1 --encode-probe"
            ),
            timing=probe,
        ))
        print(f"encode probe (croissant): {probe.encode_seconds:.3f}s")

        write_results(
            Path(args.out),
            report_context(),
            rows,
            verdicts=[
                "NFR-1 hard bar (croissant.nrrd x croissant.csv, 500³ uint8):"
                f" {croissant_full:.1f} FPS —"
                f" {hard_bar_verdict(croissant_full, hard=30.0, target=100.0)}.",
                f"Repeatability (drift criterion < ~10%): {drift:.1f}% between two"
                f" independent min-of-{args.repeats} runs of the canonical pair.",
            ],
        )
        print(f"wrote '{args.out}' ({len(rows)} rows)")
        return 0

    if args.volume is None or args.planes is None:
        parser.error("expected: volume.nrrd planes.csv (or --corpus)")

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

    if args.encode_probe:
        probe_out = Path(args.out).with_name("bench-encode-probe.mp4") if args.out else Path("bench-encode-probe.mp4")
        probe = timed_pipeline(args.volume, args.planes, probe_out, clock=time.perf_counter)
        probe_out.unlink(missing_ok=True)
        print(f"encode probe: {probe.encode_seconds:.3f}s (throwaway '{probe_out.name}')")

    if args.out is not None:
        row = ResultRow(
            label=f"{Path(args.volume).name} x {Path(args.planes).name}",
            command=f"python bench.py {args.volume} {args.planes} --repeats {args.repeats}",
            timing=best,
        )
        write_results(Path(args.out), report_context(), [row])
        print(f"wrote 1 row to '{args.out}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
