"""Regenerate the committed renderer goldens.

Run from the repo root::

    python tests/generate_goldens.py

The goldens are the *output of this PR's code*; the independent oracle is the
hand-derived table in ``docs/algorithm.md`` and the pixel assertions in
``tests/test_renderer.py``. Regenerating must leave ``git status`` clean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import Ex0  # noqa: E402
from render_cases import CASES  # noqa: E402

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def main() -> None:
    GOLDEN_DIR.mkdir(exist_ok=True)
    for case in CASES:
        records = Ex0.render_frames(
            case.volume, [Ex0.Plane(row=1, points=case.points)]
        )
        frame = records[0]
        assert isinstance(frame, Ex0.Frame), f"{case.name}: expected a rendered frame"
        path = GOLDEN_DIR / f"{case.name}.npy"
        np.save(path, frame.image)
        height, width = frame.image.shape
        print(f"wrote {path.relative_to(REPO_ROOT)} ({width}x{height})")


if __name__ == "__main__":
    main()
