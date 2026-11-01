"""CLI tests (FR-1, FR-2): argument handling, one-line errors, exit codes.

The end-to-end probe runs the real brief command on the provided corpus
assets and decodes the produced mp4 (never re-rendering it) — the same seam
the owner's acceptance run uses.
"""

from __future__ import annotations

import shutil
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
    not ((ASSETS / "csv" / "lecturer_example.csv").exists() and shutil.which("ffmpeg")),
    reason="corpus assets or ffmpeg unavailable",
)
def test_brief_command_on_lecturer_example(tmp_path, capsys):
    # SRS §5.3.1 groundwork: the exact brief command runs without a traceback
    # and yields one placeholder frame per CSV row (rows → frames 1:1), on the
    # volume the lecturer encourages. Row-2's error frame arrives with the
    # error-frame ticket; here the video decodes to 4 uniform frames.
    out = tmp_path / "out.mp4"
    code = Ex0.main(
        [str(ASSETS / "nrrd" / "MRHead_1.nrrd"), str(ASSETS / "csv" / "lecturer_example.csv"), str(out)]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert out.exists()

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate,codec_name",
            "-of", "default=nw=1", str(out),
        ],
        capture_output=True, text=True,
    )
    assert probe.returncode == 0
    stream = probe.stdout
    assert "nb_read_frames=4" in stream
    assert "codec_name=h264" in stream
    assert "r_frame_rate=30/1" in stream
    assert "width=256" in stream and "height=256" in stream

    # stdout carries one summary line; nothing on stderr for the happy path.
    assert "wrote 4 placeholder frames" in captured.out
    assert captured.err == ""


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
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
