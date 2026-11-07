# Plane Projection

[![CI](https://github.com/ofeq-dev/plane-projection/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ofeq-dev/plane-projection/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-githubpages-22d3ee.svg)](https://ofeq-dev.github.io/plane-projection/)

Render plane projections of a 3D voxel volume: one NRRD volume + one CSV
plane-list → one mp4, one frame per plane (requirements: `docs/srs.md`;
exercise brief: `I2SE_Exe1.docx`).

**Live demo & interactive walkthrough:** <https://ofeq-dev.github.io/plane-projection/>

## Build & run

Python ≥ 3.12 (numpy 2.5.x requires it). Video encoding uses OpenCV's `cv2.VideoWriter`
with the `mp4v` codec — everything is pip-installable, no system binary needed
([ADR 0004](docs/adr/0004-video-encoding-via-opencv-mp4v.md)):

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# development extras (tests, typechecking): pip install -r requirements-dev.txt
```

The brief's demo contract ([SRS FR-1](docs/srs.md)):

```sh
python Ex0.py file.nrrd file.csv file.mp4
# example: python Ex0.py assets/nrrd/MRHead_1.nrrd assets/csv/lecturer_example.csv out.mp4
```

Tests: `python -m pytest`.

Benchmark (NFR-1/NFR-2 — throughput over load + render, min-of-repeats,
encoder reported separately):

```sh
python bench.py assets/nrrd/croissant.nrrd assets/csv/croissant.csv --repeats 3
python bench.py --corpus --repeats 3 --out benchmarks/results-2026-11-06.md
```

Latest committed figures: see [`benchmarks/results-2026-11-06.md`](benchmarks/results-2026-11-06.md)
— croissant (500³ uint8) reached 42.8 FPS on the reference machine with the
brief's ≥ 30 FPS hard bar met (100 FPS target not met on that machine). Each row of
that file cites the exact one-line command that reproduced it.

Docs: [`docs/srs.md`](docs/srs.md) (requirements) ·
[`docs/algorithm.md`](docs/algorithm.md) (frame geometry; interactive
plain-words walkthrough in [`docs/algorithm.html`](docs/algorithm.html)) ·
[`docs/testing-strategy.md`](docs/testing-strategy.md) (seams, oracles,
golden protocol) ·
[`docs/example-run.md`](docs/example-run.md) (a real run, frame by frame,
with the exported videos in [`assets/videos/`](assets/videos/)).

Example outputs: [`assets/videos/lecturer_example.mp4`](assets/videos/lecturer_example.mp4)
(the brief's demo — 2 projections + 2 error frames), `head.mp4` (280 axial
slices) and `onion.mp4` (120 uint8 frames); see the example-run write-up for
the commands and per-frame explanation.

## License

[MIT](LICENSE) — © 2026 Josh Hall (ofeq-dev). Sample NRRD volumes retain their
original provenance; see `assets/nrrd/` for details.
