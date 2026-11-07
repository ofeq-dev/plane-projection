"""Plane-list counting tests (FR-16 groundwork; full parsing is the next ticket).

This ticket only consumes the CSV row count (one placeholder frame per row),
so the pinned corpus row counts here are the same numbers the video must
carry on every later ticket.
"""

from __future__ import annotations

import pytest

from plane_validation import count_planes
from errors import UsageError


@pytest.mark.parametrize(
    ("name", "rows"),
    [
        ("lecturer_example.csv", 4),
        ("croissant.csv", 100),
        ("head.csv", 280),
        ("onion.csv", 120),
        ("swiss_roll.csv", 120),
        ("biscuit_cake.csv", 120),
    ],
)
def test_corpus_plane_lists_count_their_rows(name, rows):
    # FR-7..FR-9 across the whole provided corpus, including
    # lecturer_example.csv (missing trailing newline, mixed 3-/4-point rows)
    # and croissant.csv (float-coordinate rows).
    assert count_planes(f"assets/csv/{name}") == rows


def test_missing_trailing_newline_counts_identically(tmp_path):
    # FR-9: a CSV missing its trailing newline parses the same.
    with_newline = tmp_path / "with_newline.csv"
    without = tmp_path / "without.csv"
    body = "Number of Points (3 or 4),x1,y1,z1,,\n3,1,2,3,,\n3,4,5,6,,"
    with_newline.write_text(body + "\n")
    without.write_text(body)
    assert count_planes(with_newline) == count_planes(without) == 2


def test_float_and_integer_cells_count_identically(tmp_path):
    # FR-9: integer- or float-point coordinates load identically (counting
    # sees both as one row per line; parsing of the cell content is the next
    # ticket's business).
    integer = tmp_path / "int.csv"
    floating = tmp_path / "float.csv"
    header = "Number of Points (3 or 4),x1,y1,z1,,"
    integer.write_text(f"{header}\n3,1,2,3,,\n3,4,5,6,,\n")
    floating.write_text(f"{header}\n3,1.5,2.5,3.0,,\n3,4.5,5.5,6.25,,\n")
    assert count_planes(integer) == count_planes(floating) == 2


def test_empty_rows_are_not_planes(tmp_path):
    # Blank rows (trailing or interior) are separators, not plane rows.
    file = tmp_path / "blanks.csv"
    header = "Number of Points (3 or 4),x1,y1,z1,,"
    file.write_text(f"{header}\n3,1,2,3,,\n\n3,4,5,6,,\n\n")
    assert count_planes(file) == 2


def test_crlf_line_endings_count_identically(tmp_path):
    header = "Number of Points (3 or 4),x1,y1,z1,,"
    file = tmp_path / "crlf.csv"
    file.write_bytes(f"{header}\r\n3,1,2,3,,\r\n3,4,5,6,,\r\n".encode())
    assert count_planes(file) == 2


def test_empty_file_is_rejected_naming_it(tmp_path):
    file = tmp_path / "empty.csv"
    file.write_text("")
    with pytest.raises(UsageError, match=str(file.name)):
        count_planes(file)


def test_directory_as_plane_list_is_rejected(tmp_path):
    with pytest.raises(UsageError, match="is a directory"):
        count_planes(tmp_path)


def test_missing_plane_list_is_rejected(tmp_path):
    with pytest.raises(UsageError, match="no such file"):
        count_planes(tmp_path / "absent.csv")