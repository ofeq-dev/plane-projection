# Software Requirements Specification — Plane Projection

Status: **draft** — pending owner review. Context: exercise brief `I2SE_Exe1.docx`
(To-Do #2). Language follows the [Wiegers SRS template](https://research.cs.queensu.ca/home/emads/teaching/slides/srs_template_sep14.pdf)
(outline only; the full [ISO/IEC/IEEE 29148](https://www.iso.org/standard/72089.html)
process machinery is explicitly out of scope for this project size). Domain terms are
used as defined in [`GLOSSARY.md`](../GLOSSARY.md); decision background lives in
[`docs/adr/`](adr/0001-store-volume-and-video-assets-in-git-lfs.md) and
[`docs/research/`](research/).

## 1. Introduction

### 1.1 Purpose

Specify the requirements of a command-line tool that, given a 3D volume `M` and a list
of planes `p`, renders each plane's projection as a 2D frame and writes the frames, in
list order, to a single video file.

### 1.2 Scope

In scope: NRRD input, CSV plane input, frame rendering, error handling, mp4 output,
runtime efficiency, CLI ergonomics.
Out of scope: interactive/guided use, world-space (physically scaled) rendering — see
[ADR 0002](adr/0002-canonical-voxel-index-space.md), out-of-plane interpolation or other
non-nearest sampling ([justification: scipy `order=0` vs. orders 1–5](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html)).

### 1.3 References

- Exercise brief: `I2SE_Exe1.docx` (repo root; problem definition, plane validity rules, Q\&A)
- [NRRD file format spec](https://teem.sourceforge.net/nrrd/format.html) — normative for the input format; [`type:` table](https://teem.sourceforge.net/nrrd/format.html#type), [`space:` fields](https://teem.sourceforge.net/nrrd/format.html#space)
- [pynrrd docs](https://pynrrd.readthedocs.io/en/stable/) — the volume-loading dependency
- 3D Slicer, source of the `MRHead` sample data: [coordinate systems](https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html), [MRHead provenance & license](https://github.com/Slicer/Slicer/blob/main/Modules/Scripted/SampleData/SampleData.py)
- Research notes with verified citations: [`docs/research/`](research/)

### 1.4 Definitions

See [`GLOSSARY.md`](../GLOSSARY.md) (matrix, voxel, plane, frame, canvas, error frame,
canonical space, sample rule).

## 2. Overall description

The tool is a batch renderer: one run consumes one volume + one plane list and produces
one video. Its users are the exercise staff and other students comparing results; its
primary non-functional thrusts are correctness/stability and rendering throughput.
Everything about *how* the projection is computed is the algorithm document's business;
this SRS pins what the tool must accept, produce, and how fast.

## 3. Functional requirements

### 3.1 Invocation

- **FR-1** The tool runs as `python Ex0.py file.nrrd file.csv file.mp4`, exactly three
  positional arguments, per the brief's demo contract.
- **FR-2** Invalid usage (missing files, wrong argument count) exits non-zero with a
  one-line CLI error naming the problem argument or file.

### 3.2 Volume input

- **FR-3** The tool reads [NRRD](https://teem.sourceforge.net/nrrd/format.html) files via
  [pynrrd](https://pynrrd.readthedocs.io/en/stable/) and treats the array as
  `M[x][y][z]` with dimensions `sx, sy, sz` per [NRRD §1.4 axis ordering](https://teem.sourceforge.net/nrrd/format.html#general.4).
- **FR-4** Contract: element types `uint8`, `uint16`, `uint32` (the brief's 8/16/32-bit
  unsigned values); [per the spec's `type:` table](https://teem.sourceforge.net/nrrd/format.html#type).
  Signed integer types (e.g. `short`) are **tolerated**, not rejected — the provided
  `MRHead_1.nrrd`, the lecturer's encouraged starter file, is `short`.
- **FR-5** Dimensions outside `1 <= s < 2^10` are rejected with a clear CLI error; this
  documents the brief's constraint, not a runtime guard the tool is expected to hit.
- **FR-6** The volume's world-space header fields are ignored per
  [ADR 0002](adr/0002-canonical-voxel-index-space.md).

### 3.3 Plane list input

- **FR-7** The CSV has a header row of 13 columns — `Number of Points (3 or 4),
  x1,y1,z1,...,x4,y4,z4` — with the trailing fields empty for 3-point rows.
- **FR-8** Each row defines one plane: 3 points `(a, b, c)` or 4 points in the column
  order `(a1, a2, b, c)`. Column order is the validity-rule order referenced below.
- **FR-9** A CSV file missing its trailing newline, and integer- or float-point
  coordinates, must parse identically.
- **FR-10** A plane is **valid** iff it satisfies the brief's rules with `EPS = 0.001`:
  3-point: `|dist(a,b) − dist(a,c)| < EPS`; 4-point: `|dist(a1,b) − dist(a2,c)| < EPS`,
  `dist(a1,a2) ≤ dist(b,c)` — checked with the same EPS slack, since the corpus's
  rectangular rows exceed exact equality by ~4e-7 float dust (so the rejection
  predicate is `dist(a1,a2) > dist(b,c) + EPS`) — and both legs `a1→b`, `a2→c`
  lie in one plane.
- **FR-11** A plane is **constructible** iff, additionally, no leg has length 0
  (division-by-zero guard) and its geometry cannot collapse to a zero-width frame:
  a triangle with `b == c`, or a 4-point row with `a1 == a2` whose legs step the
  same way (`a1→b == a2→c`). `a1 == a2` with opposite-way legs is the 3-point
  triangle geometry (apex `a1`, base `b→c`) and renders; parallel legs under a
  nonzero top edge are legal rectangles (the whole `head.csv` corpus).
  Constructibility is an extension of validity, not an alternative: both raise
  for the same downstream treatment.

### 3.4 Frame rendering

- **FR-12** For each row, in file order, the tool renders that plane's projection frame.
  Requirements on the rendered image:
  - **FR-12a** Sampling: nearest voxel by the file's documented tie rule —
    half-to-even, per [`numpy.rint`](https://numpy.org/doc/stable/reference/generated/numpy.rint.html).
  - **FR-12b** Density: one frame sample per voxel-length unit along each plane edge —
    frame height `= round(leg length)`, width `= round(base length)` in index units,
    clamped to at least 1 so a sub-voxel plane still renders (docs/algorithm.md).
  - **FR-12c** Orientation: apex `a` at top-center; the two equal legs run downward to
    `b` (left) and `c` (right); each row's samples spread symmetrically about the center
    column; 4-point planes render rows whose width interpolates between the top edge
    `dist(a1,a2)` and the base `dist(b,c)` — FR-10's rule keeps the top edge no wider
    than the base, so no rendered row is wider than the base, which the last row spans
    exactly. No mirroring or flipping. Anchored to the lecturer's reference pseudocode for
    cross-student comparability; verified against a classmate's output on common planes
    (see §5.3) before freezing.
  - **FR-12d** Out-of-volume samples (`p` not in `M`) render as 0 (black), per the brief.
  - **FR-12e** Display scaling: voxel values are shown over the file's full dtype range
    without content-based stretching ([scikit-image dtype-range table](https://scikit-image.org/docs/stable/user_guide/data_types.html)).
- **FR-13** Sample points are looked up in canonical voxel-index space ([ADR 0002](adr/0002-canonical-voxel-index-space.md)).

### 3.5 Error handling

- **FR-14** For any row failing FR-10/FR-11, the tool raises internally, catches, and
  emits one **error frame in place of that row's projection**; all subsequently valid
  rows still render. Per the Q\&A in the brief: error frames do not stop the video.
- **FR-15** The error frame is black with a red banner in a fixed corner containing the
  CSV row index and the *specific* failing rule, e.g. `row 7: |dist(a1,b)−dist(a2,c)|
  = 100 ≥ EPS=0.001`.
- **FR-16** Each per-row exception is logged to the CLI with the same content as the
  banner (row index + reason), so the log is greppable without re-rendering.

### 3.6 Video output

- **FR-17** One output video per run, frames in CSV row order — one frame per plane.
- **FR-18** Canvas: all frames of a video share one frame size — the max width and max
  height across the CSV's renderable frames — padded top-left with black. One fixed
  frame size is a hard requirement of the mp4 writer
  ([`cv::VideoWriter.write()`](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html)):
  each written frame "must have the same size as has been specified when opening".
- **FR-19** Encoding: 30 FPS, MPEG-4 Part 2 (`mp4v`) fourcc, mp4 container, canvas
  dimensions made codec-compatible (even, per the writer's truncation behavior). The 30
  figures is a *container* choice only — decoupled from rendering throughput (§4).
  H.264 was originally required (ADR 0003) but was dropped for the pip-only dependency
  policy (see [ADR 0004](adr/0004-video-encoding-via-opencv-mp4v.md)).
- **FR-20** The output file is written even if every row errored (a video of error
  frames communicates the failure better than an empty/absent file).
- **FR-21** Aspect compensation of padding is *not* applied: padding is background
  black exactly as FR-18 states.

## 4. Performance requirements

- **NFR-1** On a 500³-byte-voxel volume (`croissant.nrrd` subset of the corpus), the
  renderer achieves **≥ 30 FPS** measured end-to-end over a full CSV run
  (frames rendered ÷ wall time), with a **100 FPS target** — per the brief; the lecturer
  cites 300+ FPS as achieved elsewhere, so the bar is not aspirational.
- **NFR-2** Throughput is measured with [`time.perf_counter()`](https://docs.python.org/3/library/time.html#time.perf_counter)
  (monotonic, highest resolution) and reported as min-of-repeats
  ([`timeit` methodology](https://docs.python.org/3/library/timeit.html)), covering
  load + render but decoupled from the container's 30 FPS metadata (§3.6).

## 5. Non-functional requirements / verification

### 5.1 Implementation

- **NFR-3** Python + numpy + pynrrd; per-[the numpy arch's stated design](https://numpy.org/doc/stable/user/whatisnumpy.html)
  the render path is vectorized (no per-voxel Python stepping).

### 5.2 Robustness

- **NFR-4** The tool never crashes out on invalid plane rows: FR-14 is the only
  acceptable behavior, and the video is produced regardless (FR-20).

### 5.3 Acceptance criteria

1. `python Ex0.py assets/nrrd/MRHead_1.nrrd assets/csv/lecturer_example.csv out.mp4`
   runs without a traceback and produces 4 frames; row 3 fails the 3-point
   isosceles rule and row 4's set (`dist(a1,b)=300` vs `dist(a2,c)=400`, diff
   `≥ EPS`) renders an error frame whose banner names that rule; rows 1-2
   render projections.
2. `head.csv` against `MRHead_1` produces a video of axial slices, black above/below
   (rows whose z-coordinate leaves the volume).
3. Cross-student check: the same CSV, rendered by a classmate's implementation, matches
   FR-12c's orientation (or the flip is fixed and re-anchored before freeze).
4. Benchmark harness (§5.4) reports ≥30 FPS on croissant with the corpus's own CSVs.

### 5.4 Testing strategy

Frame-vs-golden-master comparisons using fixtures per
[pytest fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html) and the
[xUnit verification vocabulary](http://xunitpatterns.com/Result%20Verification%20Patterns.html).
Detailed test-plan decisions live in
[`docs/testing-strategy.md`](testing-strategy.md) (seams, oracles, fixtures,
golden protocol, traceability), not here. A worked run of §5.3.1/§5.3.2 is in
[`docs/example-run.md`](example-run.md).

## 6. History

- v0.1 (2026-10-31): first draft from the design session; every FR traces to a decision
  made in that session's ledger (glossary + ADR 0002).
- v0.2 (2026-11-05): alignment pass with `docs/algorithm.md` and the brief: FR-10
  documents rule 2's EPS-slack predicate; FR-11 states the constructibility rules as
  implemented (same-direction-leg collapse; rectangles stay legal); FR-12b pins the
  ≥1 clamp; FR-12c's row-width wording matches the interpolation guarantee
  (a valid 4-point quad's rows are capped at the base width, not guaranteed to widen
  monotonically).
