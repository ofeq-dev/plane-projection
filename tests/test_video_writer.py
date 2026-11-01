"""Video writer tests — decode the produced mp4, never re-render it (CLI seam).

Container contract (FR-19) is checked against the actual encoded bytes via
ffprobe: codec, container, frame count, fps, uniform size. ffmpeg/ffprobe are
the project's system video tools (ADR 0003); tests skip when they're absent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import Ex0

has_ffmpeg = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = pytest.mark.skipif(not has_ffmpeg, reason="ffmpeg/ffprobe not available")


def ffprobe(path, *args):
    run = subprocess.run(
        ["ffprobe", "-v", "error", *args, str(path)], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    return run.stdout.strip()


def decode_frames(path, canvas):
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout


def test_container_contract(tmp_path):
    # FR-17/FR-19: one file, N frames in row order (one per plane), 30 FPS
    # container rate, H.264 codec in an mp4 container, even canvas.
    out = tmp_path / "out.mp4"
    Ex0.write_video(out, 4, canvas=Ex0.PLACEHOLDER_CANVAS, fps=Ex0.VIDEO_FPS)

    assert "h264" in ffprobe(out, "-show_entries", "stream=codec_name")
    assert "mp4" in ffprobe(out, "-show_entries", "format=format_name")
    assert ffprobe(out, "-count_frames", "-select_streams", "v:0",
                   "-show_entries", "stream=nb_read_frames",
                   "-of", "default=nw=1:nk=1") == "4"
    assert ffprobe(out, "-show_entries", "stream=r_frame_rate",
                   "-of", "default=nw=1:nk=1") == "30/1"
    size = ffprobe(out, "-show_entries", "stream=width,height",
                   "-of", "csv=p=0").strip()
    assert size == f"{Ex0.PLACEHOLDER_CANVAS[0]},{Ex0.PLACEHOLDER_CANVAS[1]}"


def test_placeholders_are_uniform_black(tmp_path):
    # Placeholder policy of this ticket: every frame identical (black) at the
    # given canvas; encoding must not alter frame count or ordering.
    out = tmp_path / "uniform.mp4"
    canvas = (64, 48)
    Ex0.write_video(out, 3, canvas=canvas)
    raw = decode_frames(out, canvas)
    height, width = canvas[1], canvas[0]
    frames = [
        raw[i * height * width * 3:(i + 1) * height * width * 3]
        for i in range(len(raw) // (height * width * 3))
    ]
    assert len(frames) == 3
    assert len(set(frames)) == 1  # placeholders are uniform


def test_zero_rows_still_writes_the_container(tmp_path):
    # FR-20 groundwork: the run's output exists even when it has no frames to
    # carry (the empty CSV decision point stays with the plane-list ticket).
    out = tmp_path / "empty.mp4"
    Ex0.write_video(out, 0, canvas=Ex0.PLACEHOLDER_CANVAS)
    assert out.exists()
    assert "mp4" in ffprobe(out, "-show_entries", "format=format_name")


def test_odd_canvas_is_rejected_not_silently_cropped(tmp_path):
    # FR-19: yuv420p codec-compatibility is the writer's job — the ffmpeg CLI
    # would otherwise crop odd rows/columns silently.
    with pytest.raises(Ex0.UsageError, match="even dimensions"):
        Ex0.write_video(tmp_path / "odd.mp4", 1, canvas=(63, 48))
    with pytest.raises(Ex0.UsageError, match="even dimensions"):
        Ex0.write_video(tmp_path / "odd.mp4", 1, canvas=(64, 49))


def test_nonpositive_canvas_is_rejected(tmp_path):
    with pytest.raises(Ex0.UsageError, match="must be positive"):
        Ex0.write_video(tmp_path / "bad.mp4", 1, canvas=(0, 48))
    with pytest.raises(Ex0.UsageError, match="must be positive"):
        Ex0.write_video(tmp_path / "bad.mp4", 1, canvas=(64, -2))


def test_negative_frame_count_is_rejected(tmp_path):
    with pytest.raises(Ex0.UsageError, match="negative frame count"):
        Ex0.write_video(tmp_path / "neg.mp4", -1, canvas=Ex0.PLACEHOLDER_CANVAS)


def test_missing_ffmpeg_names_the_output_file(tmp_path, monkeypatch):
    # FR-2 shape: the one-line error names the problem file (the output).
    monkeypatch.setattr(Ex0.shutil, "which", lambda name: None)
    out = tmp_path / "out.mp4"
    with pytest.raises(Ex0.UsageError) as excinfo:
        Ex0.write_video(out, 1, canvas=Ex0.PLACEHOLDER_CANVAS)
    assert "ffmpeg not found" in str(excinfo.value)
    assert str(out) in str(excinfo.value)


def test_unwritable_output_naming_the_file(tmp_path):
    # ffmpeg must fail writing under a missing directory; surfaced as a
    # one-line error naming the output file (FR-2), not a traceback.
    out = tmp_path / "no_such_dir" / "out.mp4"
    with pytest.raises(Ex0.UsageError, match=str(out)) as excinfo:
        Ex0.write_video(out, 1, canvas=(64, 48))
    assert "exited with status" in str(excinfo.value)


def test_overwrites_previous_output_deterministically(tmp_path):
    # A second run must replace the file (the writer is not append-only), and
    # identical inputs encode to identical bytes — the regression-safety net
    # the renderer ticket's golden comparison leans on.
    out = tmp_path / "out.mp4"
    Ex0.write_video(out, 1, canvas=(64, 48))
    encoded_once = out.read_bytes()
    Ex0.write_video(out, 1, canvas=(64, 48))
    assert out.read_bytes() == encoded_once
