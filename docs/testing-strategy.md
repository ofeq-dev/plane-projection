# Testing strategy

Status: **active** — the plan behind the suite in `tests/`. Normative
requirements live in [`docs/srs.md`](srs.md) (FR-1…FR-21, NFR-1…NFR-4);
the discretization the tests encode is pinned in
[`docs/algorithm.md`](algorithm.md). This document exists because SRS §5.4
defers the detailed test plan to it.

The short version: the renderer is verified at **one primary seam** with
independent oracles, the command-line contract at **one thin seam** by
decoding the produced mp4, and nothing is asserted about internals,
vectorization, or speed.

## 1. What "correct" means

A frame is correct when it matches a source of truth that is independent of
the code that produced it. The suite recognises four such oracles:

| Oracle | Lives in | Independent because |
|---|---|---|
| Hand-derived pixel tables | `docs/algorithm.md`, `tests/test_renderer.py`, `tests/test_plane_validation.py` | Worked out from the algorithm by hand, not captured from the renderer |
| Corpus verdicts | `tests/test_plane_validation.py`, `tests/test_plane_list.py` | The provided CSVs are external inputs; the expected pass/fail/row counts are hand-checked |
| Decoded mp4 bytes | `tests/test_video_writer.py`, `tests/test_cli.py`, `tests/test_bench.py` | `cv2.VideoCapture` reads what a player would read, not our writer's own state |
| A scripted clock | `tests/test_bench.py` | Time is injected, so timing logic is checked without depending on machine speed |

Committed goldens (`tests/golden/*.npy`) are a **regression lock**, not an
oracle: they are the current code's output, and each one has a hand-derived
table behind it (see §6). They catch accidental output drift; they cannot
justify a new output.

## 2. Seams under test

### 2.1 Render seam (primary)

```python
records = Ex0.render_frames(volume, planes)   # list[Frame | PlaneError]
```

`volume` is a numpy array, `planes` is the list returned by
`Ex0.parse_planes(csv_path)`. The seam returns one record per CSV row, in
row order: a `Frame` (`row` + 2-D `uint8` image) or a `PlaneError` (`row` +
reason). Canvas helpers (`Ex0.canvas_size`, `Ex0.pad_frame`,
`Ex0.build_error_frame`) are exercised through the same seam.

This is where every functional rendering assertion lives: density,
orientation, rounding ties, out-of-volume black, display scaling, canvas
padding, error-frame content. Tiny synthetic volumes (3³ / 4³ / 8³) make the
expected pixels hand-computable; no real asset is needed to prove FR-12.

### 2.2 CLI seam (thin)

```python
code = Ex0.main([volume_path, csv_path, out_path])   # or: python Ex0.py ...
```

The CLI seam asserts the *contract*, not the rendering: exit codes, one-line
errors on stderr, and the decoded mp4 (frame count, size, fps, codec,
banner presence, black padding band). End-to-end tests use the real corpus
assets; they re-decode the mp4 rather than re-rendering it, so a renderer
change cannot silently agree with itself.

There are no other seams. `load_volume`, `parse_planes`, `write_video` and
`bench` are tested at their own module boundary because each is directly
observable (an array, records, an mp4, a results file); no test reaches
inside them.

## 3. Test layers

`python -m pytest` collects **123 tests** in about 11 s (reference machine).

| File | Tests | Requirements covered | What it proves |
|---|---:|---|---|
| `tests/test_volume_loader.py` | 15 | FR-3…FR-6 | uint8/16/32 accepted, signed tolerated, other dtypes rejected, `1 ≤ s < 2^10`, `M[x][y][z]` disk-order mapping, world-space headers ignored, corpus volume loads |
| `tests/test_plane_list.py` | 13 | FR-7…FR-9 | row counts for every corpus CSV, missing trailing newline, integer vs float cells, blank rows |
| `tests/test_plane_validation.py` | 26 | FR-7…FR-11, FR-14 groundwork | 3-point isosceles rule, 4-point legs/base/coplanarity rules, EPS slack, constructibility (zero-length legs, degenerate collapses), error records instead of crashes, one-based row index, corpus verdicts |
| `tests/test_renderer.py` | 32 | FR-12, FR-13, FR-14…FR-16, FR-18, FR-20 | golden frames, hand-derived pixels, density clamp, orientation, half-to-even ties, OOB black, full-range scaling, canvas/padding, error banner wrapping/truncation |
| `tests/test_video_writer.py` | 8 | FR-17, FR-19, FR-21 | decoded container contract (30 fps, mp4v, even dimensions), uniform black frames, zero rows, invalid canvas/frame sizes, unwritable output, overwrite |
| `tests/test_cli.py` | 15 | FR-1, FR-2, FR-14, FR-16, FR-20 | argument handling, one-line errors and exit codes, real `lecturer_example.csv` run with banner/log equality, render exception between valid rows, all-invalid video, corpus pairs, literal `python Ex0.py` invocation |
| `tests/test_bench.py` | 14 | NFR-1, NFR-2 | phase split, throughput arithmetic, min-of-repeats selection, corpus cross-product, verdict persistence, results-file methodology/hardware/command, error paths |

Support files (not collected as tests):

- `tests/render_cases.py` — the synthetic `RenderCase` table behind the
  goldens (hand-checkable volumes + planes).
- `tests/generate_goldens.py` — regenerates `tests/golden/*.npy`.
- `conftest.py` — puts the repo root on `sys.path` so tests `import Ex0`.

## 4. Fixtures

- **Synthetic volumes are built in code**, never committed as binaries.
  Shapes are 3³–8³ so a whole frame fits in a readable assertion (or is
  checked by an exact `.tolist()`).
- **NRRD files for loader tests are written into `tmp_path`** with pynrrd;
  the axis-mapping case uses a hand-written ASCII NRRD whose disk byte order
  is known.
- **CSV fixtures are written into `tmp_path`** by `write_csv`, with the real
  header line, so parsing and validation see exactly what the corpus sees.
- **Corpus assets are read from `assets/`** by the CLI and bench tests.
  Asset-dependent tests skip with a reason if the files are absent; they are
  never duplicated into the test tree.
- **Parametrization carries the case** (dtype, corpus pair, render case,
  verdict) so one failing case names itself in the pytest id.

## 5. The golden protocol

`tests/golden/` holds five `.npy` frames: `triangle`, `trapezoid`,
`out_of_volume`, `rounding_tie`, `short_scaling`. Each covers one
hand-checkable behaviour.

Protocol:

1. The golden is compared with `np.testing.assert_array_equal` — byte-level
   equality of the rendered array, no tolerance.
2. `python tests/generate_goldens.py` rewrites the goldens and **must leave
   `git status` clean**. Any diff means either the code or the case changed
   for a reason that has to be explained.
3. A golden diff is only approvable together with an update to the
   independent oracle: the hand-derived table in `docs/algorithm.md` or the
   explicit pixel assertions in `tests/test_renderer.py`.
4. Adding a case means adding a `RenderCase` in `tests/render_cases.py`,
   documenting its hand derivation, regenerating, and committing the new
   `.npy` in the same commit.
5. Goldens are never regenerated "to make the suite pass".

This is what makes an output-preserving optimization safe: any accidental
change to the sampled values shows up immediately, byte-for-byte.

## 6. What the suite deliberately does not assert

- **Real timing.** `tests/test_bench.py` injects a scripted clock. FPS,
  hardware, drift and the NFR-1 verdict belong to `bench.py` and
  [`benchmarks/results-2026-11-06.md`](../benchmarks/results-2026-11-06.md);
  asserting wall time in the suite would only make it flaky.
- **Encoder-exact pixels.** mp4v is lossy, so video tests assert decoded
  properties (frame count, size, fps, presence/absence of content, black
  padding bands), never exact decoded pixel values.
- **Internal structure.** No test asserts array shapes inside the renderer,
  the rounding helper directly, or that a particular loop was vectorized.
  Those are implementation choices; the frame is the contract.
- **Multiprocessing/GPU.** Out of scope per SRS §1.2.

## 7. Running the suite

```sh
. .venv/bin/activate
python -m pytest                 # all 123 tests
python -m pytest tests/test_renderer.py -q   # one file while iterating
mypy Ex0.py bench.py             # types for the two entry points
```

The suite must be green and mypy clean before a PR is opened. A red golden
test is a behavioural change until proven otherwise; do not regenerate.

## 8. Traceability (requirements → tests)

| Requirement | Primary test evidence |
|---|---|
| FR-1, FR-2 | `test_cli.py` (process invocation, exit codes, one-line errors) |
| FR-3…FR-6 | `test_volume_loader.py` |
| FR-7…FR-9 | `test_plane_list.py`, `test_plane_validation.py` |
| FR-10, FR-11 | `test_plane_validation.py` (predicates + corpus verdicts) |
| FR-12a…FR-12e | `test_renderer.py` (goldens, hand tables, ties, OOB, scaling) |
| FR-13 | `test_renderer.py`, ADR 0002 (no world-space lookups) |
| FR-14…FR-16 | `test_renderer.py` (error frames), `test_cli.py` (banner = log) |
| FR-17…FR-19, FR-21 | `test_video_writer.py`, `test_cli.py` (decoded mp4) |
| FR-20 | `test_cli.py::test_all_invalid_rows_still_produce_error_video` |
| NFR-1, NFR-2 | `test_bench.py` (logic) + `benchmarks/results-*.md` (numbers) |
| NFR-3 | `Ex0.py` (vectorized render path), reviewed rather than asserted |
| NFR-4 | `test_cli.py` (interleaved render failure, all-invalid run) |

## 9. Checklist for a new test

1. Name the requirement (FR/NFR) in the docstring.
2. Use the render seam with a hand-computable volume/plane when the claim is
   about pixels; use the CLI seam only for contract claims.
3. Prefer exact assertions (`==`, `assert_array_equal`) over tolerances;
   tolerate floats only where the SRS defines EPS.
4. If it needs a new golden, add the case and the hand derivation together.
5. If it depends on `assets/`, skip with a reason when absent.
6. Do not assert timings, internal shapes, or the rounding helper directly.
