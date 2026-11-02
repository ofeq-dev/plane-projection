"""CLI tests (FR-1, FR-2): argument handling, one-line errors, exit codes.

The end-to-end probe runs the real brief command on the provided corpus
assets and decodes the produced mp4 (never re-rendering it) — the same seam
the owner's acceptance run uses.
"""

from __future__ import annotations

import cv2
import subprocess
import sys
from pathlib import Path

import pytest

import Ex0

ASSETS = Path(__file__).parent.parent / "assets"


# --- usage errors (FR-2) ---------------------------------------------------

def test_wrong_argument_counts_exit_non_zero_with_one_line(capsys):
    for argv in ([], ["a.nrrd"], ["a.nrrd", "a.csv"], ["a.nrrd", "a.csv", "out.mp4", "extra"]):
        code = Ex0.main(argv)
        err = capsys.readouterr().err
        assert code != 0
        assert err.strip().count("\n") == 0, f"error must be one line, got: {err!r}"
        assert "3 arguments" in err


def test_usage_error_is_printed_without_traceback(capsys):
    # Framed as a one-line CLI error, not a Python traceback (FR-2).
    code = Ex0.main(["a.nrrd", "a.csv", "out.mp4"])
    err = capsys.readouterr().err
    assert code == 1
    assert "Traceback" not in err
    assert err.startswith("Ex0.py: ")
    assert err.strip().count("\n") == 0


def test_errors_go_to_stderr(capsys, tmp_path):
    # Usage errors are stderr-only; stdout stays silent so automated runs can
    # trust it (FR-2's 'one-line CLI error').
    Ex0.main([str(tmp_path / "no.nrrd"), "a.csv", "out.mp4"])
    captured = capsys.readouterr()
    assert "no.nrrd" in captured.err
    assert captured.out == ""


# --- file-level usage errors (FR-2: naming the problem argument/file) -------

def test_missing_volume_named(tmp_path, capsys):
    code = Ex0.main([str(tmp_path / "absent.nrrd"), "a.csv", "out.mp4"])
    err = capsys.readouterr().err
    assert code == 1
    assert "absent.nrrd" in err


def test_directory_as_volume_named(tmp_path, capsys):
    code = Ex0.main([str(tmp_path), "a.csv", "out.mp4"])
    err = capsys.readouterr().err
    assert code == 1
    assert "is a directory" in err
    assert str(tmp_path) in err


def test_missing_and_directory_plane_list_named(tmp_path, capsys, tmp_nrrd):
    code_missing = Ex0.main([str(tmp_nrrd), str(tmp_path / "absent.csv"), "out.mp4"])
    err_missing = capsys.readouterr().err
    assert code_missing == 1 and "absent.csv" in err_missing

    code_dir = Ex0.main([str(tmp_nrrd), str(tmp_path), "out.mp4"])
    err_dir = capsys.readouterr().err
    assert code_dir == 1 and "is a directory" in err_dir


@pytest.fixture
def tmp_nrrd(tmp_path):
    """Smallest valid NRRD (uint8, 1^3) for CLI tests not targeting volumes."""
    import numpy as np
    import nrrd

    file = tmp_path / "tiny.nrrd"
    nrrd.write(str(file), np.zeros((1, 1, 1), dtype=np.uint8), index_order="F")
    return file


def test_rejecting_volume_happens_before_counting(tmp_path, capsys, tmp_nrrd):
    # Order matters for the error's message: the volume named, even while the
    # plane list is also broken.
    code = Ex0.main([str(tmp_path / "absent.nrrd"), str(tmp_path / "absent.csv"), "out.mp4"])
    err = capsys.readouterr().err
    assert code == 1
    assert "absent.nrrd" in err


# --- end-to-end on the real corpus (acceptance probe) -----------------------

@pytest.mark.skipif(
    not (ASSETS / "csv" / "lecturer_example.csv").exists(),
    reason="corpus assets unavailable",
)
def test_brief_command_on_lecturer_example(tmp_path, capsys):
    # SRS §5.3.1 groundwork: the exact brief command renders valid rows and
    # keeps one frame per CSV row (FR-17), on the volume the lecturer
    # encourages. Rows 1-2 render projections at the max frame size
    # (500x447 -> even 500x448); rows 3-4 keep black placeholder frames
    # until the error-frame ticket adds their banners.
    out = tmp_path / "out.mp4"
    code = Ex0.main(
        [str(ASSETS / "nrrd" / "MRHead_1.nrrd"), str(ASSETS / "csv" / "lecturer_example.csv"), str(out)]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert out.exists()

    capture = cv2.VideoCapture(str(out))
    assert capture.isOpened()
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    assert len(frames) == 4
    assert capture.get(cv2.CAP_PROP_FPS) == 30
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 500
    assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 448
    capture.release()

    # MRHead is int16 0..279: full-range display scaling puts its in-volume
    # samples around mid gray, visibly distinct from the black placeholder of
    # error rows (the planes extend past the volume, so most of each frame is
    # black by FR-12d). Lossy mp4v keeps that separation comfortably.
    assert frames[0].max() > 100 and frames[1].max() > 100
    assert frames[2].max() < 16 and frames[3].max() < 16

    # FR-16 groundwork: per-row verdicts are logged to the CLI with the same
    # content the future error banners will carry. lecturer_example's rows
    # 3 and 4 fail validation (isosceles 20.53; legs 100) and are logged to
    # stderr while the run continues; rows 1-2 are ok on stdout.
    assert "wrote 4 frames (500x448" in captured.out
    assert captured.out.count("ok (") == 2
    err_lines = captured.err.strip().splitlines()
    assert len(err_lines) == 2
    assert err_lines[0].startswith("Ex0.py: row 3: |dist(a,b) − dist(a,c)| = 20.5")
    assert err_lines[1].startswith("Ex0.py: row 4: |dist(a1,b) − dist(a2,c)| = 100")


@pytest.mark.parametrize(
    ("csv_name", "volume_name", "frames", "canvas"),
    [
        ("croissant.csv", "croissant", 100, (500, 560)),
        ("onion.csv", "onion", 120, (256, 256)),
        ("head.csv", "MRHead_1", 280, (256, 256)),
    ],
)
def test_corpus_run_produces_a_video_with_real_content(
    tmp_path, capsys, csv_name, volume_name, frames, canvas
):
    # SRS §5.3.2/§5.3.4 groundwork: the corpus pairs run end-to-end, one
    # frame per row at the shared even canvas. Croissant is the large case
    # (500x559 -> 500x560); head/onion pad 255 -> 256.
    out = tmp_path / f"{csv_name}.mp4"
    code = Ex0.main(
        [str(ASSETS / "nrrd" / f"{volume_name}.nrrd"), str(ASSETS / "csv" / csv_name), str(out)]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert f"wrote {frames} frames ({canvas[0]}x{canvas[1]}" in captured.out

    capture = cv2.VideoCapture(str(out))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == frames
    assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == canvas
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert frame.max() > 0  # the decoded video carries rendered content


def test_brief_command_via_actual_python_invocation(tmp_path):
    # FR-1 is the literal contract: `python Ex0.py file.nrrd file.csv file.mp4`.
    run = subprocess.run(
        [
            sys.executable, "Ex0.py",
            str(ASSETS / "nrrd" / "MRHead_1.nrrd"),
            str(ASSETS / "csv" / "lecturer_example.csv"),
            str(tmp_path / "out.mp4"),
        ],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    assert "wrote 4" in run.stdout


def test_brief_command_exit_codes_through_process(tmp_path):
    # FR-2 as a process: wrong arg count → exit 2; broken volume → exit 1.
    run = subprocess.run(
        [sys.executable, "Ex0.py"], capture_output=True, text=True, cwd=Path.cwd()
    )
    assert run.returncode == 2
    assert run.stderr.count("\n") <= 1  # one line (possibly no trailing NL)
    assert "Traceback" not in (run.stderr + run.stdout)
