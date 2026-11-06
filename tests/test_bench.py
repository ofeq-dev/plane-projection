"""Benchmark harness seam tests (NFR-1, NFR-2).

Under test is the ``bench`` module: :func:`bench.timed_pipeline` (one timed
run, phases split per NFR-2), :func:`bench.throughput` (frames ÷ load+render),
:func:`bench.choose_min` (min-of-repeats), and :func:`bench.write_results`
(the versioned results file). Timing itself is deterministic in tests via an
injected clock; no timing test asserts a real runtime.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import nrrd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import bench


def read_frames(path: Path) -> list[np.ndarray]:
    """Read every frame back out of a written mp4 (same probe as test_cli)."""
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


@pytest.fixture(scope="module")
def tiny_case(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A 4x4x4 uint8 volume and a two-row CSV, small enough to time fast."""
    base = tmp_path_factory.mktemp("tiny_case")
    volume = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    volume_path = base / "tiny.nrrd"
    nrrd.write(str(volume_path), volume)

    csv_path = base / "tiny.csv"
    csv_path.write_text(
        "0,0,0,4,4,4\n"        # triangle through the diagonal
        "0,0,0,0,3,2,3\n"      # trapezoid
    )
    return volume_path, csv_path


class ScriptedClock:
    """Monkeypatch-friendly clock: yields scripted ``time()`` values in order.

    Values are consumed one per call, exactly like Grafana'd checkpoints:
    callers get whatever sequence the test scripts, so every measured wall
    time is deterministic.
    """

    def __init__(self, *values: float) -> None:
        self.values = list(values)
        self.i = 0

    def __call__(self) -> float:
        value = self.values[self.i]
        self.i += 1
        return value


def test_timed_pipeline_splits_phases_and_counts_frames(tiny_case, tmp_path):
    volume_path, csv_path = tiny_case
    video_path = tmp_path / "out.mp4"
    # 4 checkpoints per repeat: start, after load, after render, after encode.
    clock = ScriptedClock(0.0, 0.100, 0.350, 0.500)
    timing = bench.timed_pipeline(volume_path, csv_path, video_path, clock=clock)
    assert timing.frames == 2
    assert timing.load_seconds == pytest.approx(0.100)
    assert timing.render_seconds == pytest.approx(0.250)  # 0.350 - 0.100
    assert timing.encode_seconds == pytest.approx(0.150)  # 0.500 - 0.350


def test_timed_pipeline_without_video_reports_zero_encode(tiny_case):
    volume_path, csv_path = tiny_case
    # 3 checkpoints per repeat: start, after load, after render (no encode).
    clock = ScriptedClock(0.0, 0.100, 0.350)
    timing = bench.timed_pipeline(volume_path, csv_path, None, clock=clock)
    assert timing.encode_seconds == 0.0
    assert timing.render_seconds == pytest.approx(0.250)


def test_timed_pipeline_produces_decodable_video(tiny_case, tmp_path):
    volume_path, csv_path = tiny_case
    video_path = tmp_path / "out.mp4"
    bench.timed_pipeline(volume_path, csv_path, video_path, clock=time.perf_counter)
    assert len(read_frames(video_path)) == 2


def test_throughput_is_frames_over_load_plus_render():
    timing = bench.Timing(
        frames=100, load_seconds=1.0, render_seconds=2.0, encode_seconds=0.5
    )
    assert bench.throughput(timing) == pytest.approx(100 / 3.0)


def test_throughput_raises_when_nothing_renders():
    timing = bench.Timing(
        frames=0, load_seconds=1.0, render_seconds=0.0, encode_seconds=0.0
    )
    with pytest.raises(ValueError, match="no rendered frames"):
        bench.throughput(timing)


def test_choose_min_picks_fatest_repeat_by_throughput_clock():
    """min-of-repeats (NFR-2) keeps the repeat with the shortest wall time,
    which by a fixed frame count is the highest FPS."""
    slow = bench.Timing(frames=100, load_seconds=0.2, render_seconds=1.0, encode_seconds=0.0)
    fast = bench.Timing(frames=100, load_seconds=0.2, render_seconds=0.8, encode_seconds=0.0)
    other = bench.Timing(frames=99, load_seconds=0.0, render_seconds=0.5, encode_seconds=0.0)
    best = bench.choose_min([slow, fast, other])
    assert best is fast
    with pytest.raises(IndexError):
        bench.choose_min([])


def test_write_results_includes_command_methodology_and_hardware(tmp_path):
    timing = bench.Timing(frames=100, load_seconds=1.0, render_seconds=2.0, encode_seconds=0.5)
    context = bench.ReportContext(
        date="2026-11-06",
        cpu="Fake CPU 1GHz (2 cores)",
        python="3.12.0",
        numpy="1.0.0",
        ffmpeg_note="codec: cv2.VideoWriter mp4v",
    )
    results_path = tmp_path / "results-test.md"
    row = bench.ResultRow(
        label="croissant.nrrd x croissant.csv",
        command="python bench.py assets/nrrd/croissant.nrrd assets/csv/croissant.csv",
        timing=timing,
    )
    bench.write_results(results_path, context, [row])
    text = results_path.read_text()
    assert "min-of-repeats" in text
    assert "time.perf_counter" in text
    assert "load + render" in text
    assert "Fake CPU 1GHz (2 cores)" in text
    # every reported row cites the one-line command that reproduced it
    assert row.command in text


def test_cli_rejects_wrong_argument_count(capsys):
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["just", "one", "extra", "line"])
    assert excinfo.value.code == 2
