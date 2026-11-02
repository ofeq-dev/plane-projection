"""Video writer tests — decode the produced mp4 with OpenCV itself.

Container contract (FR-19) is checked by decoding the actual encoded bytes:
codec, container, frame count, fps, uniform size. The writer is OpenCV's
cv2.VideoWriter (ADR 0004), so tests only need opencv-python — no system
tools, no skips.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

import Ex0


def decode(path):
    """Decode a video file to a list of BGR frames plus its fps."""
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), f"could not decode {path}"
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return frames, fps, fourcc, (width, height)


def black_frames(n, canvas=Ex0.PLACEHOLDER_CANVAS):
    return [np.zeros((canvas[1], canvas[0]), dtype=np.uint8) for _ in range(n)]


def test_container_contract(tmp_path):
    # FR-17/FR-19: one file, N frames in row order (one per plane), 30 FPS
    # container rate, mp4v codec in an mp4 container, even canvas.
    out = tmp_path / "out.mp4"
    Ex0.write_video(out, black_frames(4), canvas=Ex0.PLACEHOLDER_CANVAS, fps=Ex0.VIDEO_FPS)

    frames, fps, fourcc, size = decode(out)
    assert frames, "no frames decoded"
    assert os.path.splitext(out)[1] == ".mp4"
    assert os.path.getsize(out) > 0
    # MPEG-4 Part 2 fourcc (ADR 0004). OpenCV stores the codec's canonical
    # ffmpeg tag ('FMP4'); 'mp4v' is the equivalent OpenCV fourcc spelling.
    assert bytes(int(fourcc).to_bytes(4, "little")).decode("ascii") in {"mp4v", "FMP4"}
    assert round(fps) == Ex0.VIDEO_FPS
    assert size == (Ex0.PLACEHOLDER_CANVAS[0], Ex0.PLACEHOLDER_CANVAS[1])


def test_black_frames_decode_uniform(tmp_path):
    # Every frame is identical (black) at the given canvas; encoding must not
    # alter frame count or ordering. mp4v is lossy, so decoded pixels may drift
    # slightly — allow a small tolerance and require mutual uniformity.
    out = tmp_path / "uniform.mp4"
    canvas = (64, 48)
    Ex0.write_video(out, black_frames(3, canvas), canvas=canvas)
    frames, _, _, size = decode(out)
    assert len(frames) == 3
    assert size == (canvas[0], canvas[1])
    stacked = np.stack(frames)
    assert stacked.max() < 16  # near-black after lossy encoding
    assert stacked.std(axis=0).max() < 4.0  # frames are mutually uniform


def test_zero_rows_still_writes_the_container(tmp_path):
    # FR-20 groundwork: the run's output exists even when it has no frames to
    # carry (the empty CSV decision point stays with the plane-list ticket).
    # A container with no video stream is valid mp4 but cv2.VideoCapture cannot
    # open it (there is no stream to decode), so only the file is checked.
    out = tmp_path / "empty.mp4"
    Ex0.write_video(out, [], canvas=Ex0.PLACEHOLDER_CANVAS)
    assert out.exists()
    assert os.path.getsize(out) > 0


def test_odd_canvas_is_rejected_not_silently_cropped(tmp_path):
    # FR-19: chroma-subsampling codec-compatibility is the writer's job — the
    # encoder must not be handed odd frames it would crop silently.
    with pytest.raises(Ex0.UsageError, match="even dimensions"):
        Ex0.write_video(tmp_path / "odd.mp4", black_frames(1, (63, 48)), canvas=(63, 48))
    with pytest.raises(Ex0.UsageError, match="even dimensions"):
        Ex0.write_video(tmp_path / "odd.mp4", black_frames(1, (64, 49)), canvas=(64, 49))


def test_nonpositive_canvas_is_rejected(tmp_path):
    with pytest.raises(Ex0.UsageError, match="must be positive"):
        Ex0.write_video(tmp_path / "bad.mp4", [], canvas=(0, 48))
    with pytest.raises(Ex0.UsageError, match="must be positive"):
        Ex0.write_video(tmp_path / "bad.mp4", [], canvas=(64, -2))


def test_wrongly_sized_frame_is_rejected(tmp_path):
    # The writer's frame-size contract is the canvas's; a mismatch is an
    # internal error, never a silently cropped/oddly encoded frame.
    out = tmp_path / "out.mp4"
    with pytest.raises(ValueError, match="frame size"):
        Ex0.write_video(out, [np.zeros((48, 64), dtype=np.uint8)], canvas=(32, 48))


def test_unwritable_output_naming_the_file(tmp_path):
    # cv2.VideoWriter fails to open under a missing directory; surfaced as a
    # one-line error naming the output file (FR-2), not a traceback.
    out = tmp_path / "no_such_dir" / "out.mp4"
    with pytest.raises(Ex0.UsageError) as excinfo:
        Ex0.write_video(out, black_frames(1, (64, 48)), canvas=(64, 48))
    assert str(out) in str(excinfo.value)


def test_overwrites_previous_output(tmp_path):
    # A second run must replace the file (the writer is not append-only), and
    # identical inputs decode to the same frame count and size — the
    # regression-safety net the renderer ticket's comparison leans on.
    out = tmp_path / "out.mp4"
    Ex0.write_video(out, black_frames(1, (64, 48)), canvas=(64, 48))
    first = decode(out)
    Ex0.write_video(out, black_frames(1, (64, 48)), canvas=(64, 48))
    second = decode(out)
    assert len(first[0]) == len(second[0]) == 1
    assert first[3] == second[3]
