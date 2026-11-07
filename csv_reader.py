"""CSV row iteration with FR-9 tolerances.

Yields non-empty data rows (header excluded). Tolerant of:
- missing trailing newline
- CRLF line endings
- blank separator rows
- integer- or float-coordinate cells
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from errors import UsageError


def _require_file(label: str, path: Path) -> None:
    """One-line usage error when a named input path is a directory."""
    if path.is_dir():
        raise UsageError(f"cannot read {label}: '{path}' is a directory, not a file")


def iter_csv_rows(path: str | Path) -> Iterator[list[str]]:
    """Yield the CSV's non-empty data rows (header excluded), FR-9-tolerant.

    Args:
        path: Path to the CSV file.

    Yields:
        List of cell strings for each non-empty data row.

    Raises:
        UsageError: If the file is missing, a directory, or empty (no header).
    """
    path = Path(path)
    _require_file("plane list", path)
    try:
        with open(path, newline="") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header is None:
                raise UsageError(f"cannot read plane list '{path}': empty file (no header)")
            for row in reader:
                if any(cell.strip() for cell in row):
                    yield row
    except FileNotFoundError as e:
        raise UsageError(f"cannot read plane list '{path}': no such file") from e
    except PermissionError as e:
        raise UsageError(f"cannot read plane list '{path}': permission denied") from e
    except OSError as e:
        raise UsageError(f"cannot read plane list '{path}': {e.strerror or e}") from e