# Plane Projection

Render plane projections of a 3D voxel volume: one NRRD volume + one CSV
plane-list → one mp4, one frame per plane (requirements: `docs/srs.md`;
exercise brief: `I2SE_Exe1.docx`).

## Build & run

Python ≥ 3.10 (exercised on 3.12). Video encoding uses OpenCV's `cv2.VideoWriter`
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
