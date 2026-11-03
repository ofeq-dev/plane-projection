# Algorithm — frame geometry

> Prefer pictures to equations? [`docs/algorithm.html`](algorithm.html) walks
> through this same algorithm animated, in plain words, step by step.

Normative requirements live in [`docs/srs.md`](srs.md) (FR-12/FR-13); this
document pins the discretization the renderer implements. Coordinates are
canonical voxel-index coordinates ([ADR 0002](adr/0002-canonical-voxel-index-space.md)):
no world-space header field, origin, or spacing is applied.

## The algorithm in plain words

Think of the input volume as a block of small cubes, where every cube holds
one brightness number. A plane from the CSV is a flat triangle- or
trapezoid-shaped sheet placed inside (or reaching beyond) that block. The
renderer turns that sheet into a small picture — a frame — by asking, for
every pixel of the sheet: **which cube does this pixel sit closest to?** The
cube's brightness number is copied into the pixel. Pixels that fall outside
the sheet's row span, or closest to a cube outside the block, stay black.

Step by step, per plane:

1. The plane's corners set the sheet's size: one pixel row per voxel-length
   step down its two slanted edges (the legs), and one pixel column per
   voxel-length step across its base.
2. Walking top to bottom, each row's two ends slide down those two edges,
   so the sheet's rows widen (triangle) or widen then stay wide (trapezoid).
3. Each pixel of a row samples a point half a step to the side of the row's
   centre, so the row is symmetric about the middle.
4. Every sample point is rounded to the nearest cube — on an exact tie to
   the even side — and that cube's value is scaled to a 0..255 gray level
   over the file's full value range.
5. Misses stay black. Nothing is clamped to the block's edge, and nothing is
   stretched to the picture's own brightest value.

All frames are then padded onto one shared black canvas (the video's frame
size, FR-18) and written out as an mp4, one frame per CSV row (FR-17).

## Names used here

The CSV column letters (`a, b, c` / `a1, a2, b, c`) come from the brief's
validity rules (SRS FR-8/FR-10) and stay in error messages; every computed
quantity below uses a plain name, the same names the code in `Ex0.py` uses:

| name | meaning | CSV letter(s) |
| --- | --- | --- |
| `top_left`, `top_right` | the plane's two top corners (for a triangle both are the single apex) | `a1`, `a2` (triangle: `a`) |
| `bottom_left`, `bottom_right` | the plane's two bottom corners; the plane's base runs between them | `b`, `c` |
| `leg_length` | average of the two leg lengths (top corner → opposite bottom corner); the legs agree within `EPS = 0.001` (FR-10) | — |
| `base_length` | distance between the two bottom corners | — |
| `frame_height`, `frame_width` | the frame's row/column counts: `round(leg_length)`, `round(base_length)`, each clamped to ≥ 1 (FR-12b) | — |
| `row_index` | which output row, `0 .. frame_height - 1`, top to bottom | — |
| `row_fraction` | how far down the legs that row sits: `(row_index + 1) / frame_height` | — |
| `left_edge`, `right_edge` | the row's two end points, on the left/right leg | — |
| `row_centre` | midpoint of the row's two ends | — |
| `row_width` | distance between the row's two ends | — |
| `row_direction` | a one-index-unit step pointing along the row, from `left_edge` to `right_edge` | — |
| `column_index` | which pixel within the row, `0 .. frame_width - 1`, left to right | — |
| `offset_from_centre` | the pixel's signed distance from the row centre, in index units | — |
| `sample_point` | the 3D point one pixel's worth of distance from the row centre | — |
| `voxel_index` | `sample_point` rounded to whole indices (nearest cube, FR-12a) | — |
| `volume` | the 3D input array, indexed `volume[x][y][z]` (the brief's `M[x][y][z]`) | — |

## Inputs

A validated row is either a triangle `(a, b, c)` or a trapezoid
`(a1, a2, b, c)` in the CSV's column order (FR-8). A triangle is the same
geometry with the top edge collapsed: `top_left = top_right = a`.

- `leg_length = (dist(top_left, bottom_left) + dist(top_right, bottom_right)) / 2`
  — the two legs agree within `EPS = 0.001` by FR-10.
- `base_length = dist(bottom_left, bottom_right)`.
- `frame_height = round(leg_length)`, `frame_width = round(base_length)`
  (FR-12b), clamped to at least 1 pixel so a sub-voxel plane still produces
  a renderable frame. `round` is NumPy/Python half-to-even.

## Sample grid

### Rows slide down the legs

Rows run top to bottom, `row_index = 0 .. frame_height - 1`, with

```
row_fraction = (row_index + 1) / frame_height
```

This is the lecturer's pseudocode anchor: it steps one leg sample *before*
writing each output row, so the bottom row lies on the bottom edge
(`bottom_left`/`bottom_right`) and the top row is one sample below the top
edge. The legs are divided into `frame_height` equal steps, so sample
density is one sample per voxel-length unit (FR-12b).

Each row's two ends, its centre, and its sideways direction:

```
left_edge     = top_left  + row_fraction * (bottom_left  - top_left)   # slides down the left leg
right_edge    = top_right + row_fraction * (bottom_right - top_right)  # slides down the right leg
row_centre    = (left_edge + right_edge) / 2
row_width     = dist(left_edge, right_edge)
row_direction = (right_edge - left_edge) / row_width                   # one unit across the row
```

### Columns sit at pixel centres

Column `column_index = 0 .. frame_width - 1` samples the row at

```
offset_from_centre = column_index + 0.5 - frame_width / 2
sample_point       = row_centre + offset_from_centre * row_direction
```

`offset_from_centre` is the signed distance from the row centre in index
units; columns are one unit apart, and each row is symmetric about the
canvas centre. A pixel is inside the plane's row when
`|offset_from_centre| <= row_width / 2` (with `1e-9` of float-dust
tolerance); otherwise it is background black. A triangle's top rows are
therefore narrow and centred, while a rectangle fills every column on every
row. There is no mirroring: `bottom_left` is the left edge and
`bottom_right` the right edge.

Every sample point is computed directly from `(row_index, column_index)` —
no `point += step` accumulation, so floating-point drift cannot build up
across a frame.

### Orientation freeze

Orientation is frozen to this convention for cross-student comparability
(FR-12c/SRS §5.3.3): no mirroring, `bottom_left` (`b`) left /
`bottom_right` (`c`) right. It has not yet been checked against a
classmate's implementation; if that check diverges, the fix is a documented
constant flip in a follow-up, not a silent match.

## Voxel lookup and display

- Nearest cube: every `sample_point` is rounded componentwise with
  `np.rint` (half-to-even — an exact `.5` goes to the even side, FR-12a),
  producing whole-number `voxel_index` coordinates.
- A boolean mask keeps only samples that are both inside the plane's row
  and inside the volume (`0 <= voxel_index < volume.shape`); the mask is
  applied **before** indexing into the volume. Samples outside the volume
  are black `0` (FR-12d) — never a clamped edge voxel.
- Display scaling (FR-12e, 0..255 gray levels): every voxel value is an
  integer stored at a fixed bit width — numpy calls this the array's
  **dtype** ("data type", the value type: *unsigned* or signed, and how
  many bits each value gets). Scaling maps the value **type's** full span
  onto 0..255, never the picture's own content:
  - unsigned types (`uint8` = 0..255, `uint16` = 0..65535,
    `uint32` = 0..4294967295) map `0 .. type_max` to `0..255`;
  - signed types (tolerated input per FR-4 — e.g. the corpus's `int16`
    MRHead, values −32768..32767) map `type_min .. type_max` to `0..255`.
  Rounding to a 0..255 level is again half-to-even. Values are never
  stretched to the frame's content range. Missing/outside samples stay at
  0 (black) after scaling.

## Canvas

`canvas_size` takes the maximum width and height over renderable frames and
rounds each up to even (FR-18/FR-19). `pad_frame` places each frame
top-left on a black canvas of that size — no centring, no aspect
compensation (FR-21). If no row renders, the placeholder canvas carries the
video (FR-20).

## Hand-checked fixtures

The four `uint8` fixtures use `volume[x][y][z] = x` as a ramp, so identity
display scaling makes every pixel value the rounded sampled x-index. The
fifth fixture is a signed 16-bit (`int16`) ramp with hand-chosen full-range
values, exercising FR-12e's signed-value scaling. The committed goldens in
`tests/golden/` are regenerated by `python tests/generate_goldens.py`;
these tables are the independent oracle.

**Triangle** apex `a=(2,2,0)`, `bottom_left=(0,2,0)`, `bottom_right=(4,2,0)`:
`leg_length = 2 → frame_height = 2`, `base_length = 4 → frame_width = 4`.

| `row_index` | `row_fraction` | sample x at columns 0..3 | `rint` | expected row |
| --- | --- | --- | --- | --- |
| 0 | 1/2 | –, 1.5, 2.5, – | –, 2, 2, – | `0 2 2 0` |
| 1 | 1 | 0.5, 1.5, 2.5, 3.5 | 0, 2, 2, 4 | `0 2 2 4` |

**Trapezoid** `top_left=(2,2,0)`, `top_right=(4,2,0)`,
`bottom_left=(0,6,0)`, `bottom_right=(6,6,0)`:
`leg_length = sqrt(20) = 4.472 → frame_height = 4`,
`base_length = 6 → frame_width = 6`.

| `row_index` | `row_width` | in-plane sample x | `rint` | expected row |
| --- | --- | --- | --- | --- |
| 0 | 3 | 1.5, 2.5, 3.5, 4.5 | 2, 2, 4, 4 | `0 2 2 4 4 0` |
| 1 | 4 | 1.5, 2.5, 3.5, 4.5 | 2, 2, 4, 4 | `0 2 2 4 4 0` |
| 2 | 5 | 0.5..5.5 | 0, 2, 2, 4, 4, 6 | `0 2 2 4 4 6` |
| 3 | 6 | 0.5..5.5 | 0, 2, 2, 4, 4, 6 | `0 2 2 4 4 6` |

**Out-of-volume** `top_left = top_right = (2,2,2)`, `bottom_left=(-2,2,2)`,
`bottom_right=(6,2,2)` against a `4^3` volume of 255:
`frame_height = 4`, `frame_width = 8`. In-volume samples are 255; every
sample rounding outside `0..3` is 0.

| `row_index` | in-volume sample x | expected row |
| --- | --- | --- |
| 0 | 1.5, 2.5 | `0 0 0 255 255 0 0 0` |
| 1 | 0.5, 1.5, 2.5 | `0 0 255 255 255 0 0 0` |
| 2 | -0.5, 0.5, 1.5, 2.5 | `0 255 255 255 255 0 0 0` |
| 3 | -0.5, 0.5, 1.5, 2.5 | `0 255 255 255 255 0 0 0` |

**Rounding tie** plane `a=(2.5,2,0), b=(2.5,0,6), c=(2.5,4,6)`: every
in-volume sample is exactly `x = 2.5` — a tie, and `np.rint(2.5) = 2` (the
even side), whereas half-up rounding would read value `3`. Expected frame
(6 rows x 4 columns):

```
0 0 0 0
0 2 2 0
0 2 2 0
0 2 2 0
2 2 2 2
2 2 2 2
```

Row 0's half-width is 1/3, so no pixel centre lies inside the plane there
(black row); the remaining rows widen until row 5 is full width and samples
`z = 6`, still inside the fixture volume's 7 values along z.

**Signed short scaling** `top_left = top_right = (2,0,0)`,
`bottom_left = (-0.5,4,0)`, `bottom_right = (4.5,4,0)` against an `int16`
volume whose x-slices are `-32768, 0, 32767, 100, -100`:
`leg_length = 4.717 → frame_height = 5`, `base_length = 5 → frame_width = 5`.
Samples are consecutive integer x on the bottom rows:

| x value | -32768 | 0 | 32767 | 100 | -100 |
| --- | --- | --- | --- | --- | --- |
| scaled gray level | 0 | 128 | 255 | 128 | 127 |

Expected frame (5 rows x 5 columns):

```
0   0 255   0   0
0 128 255 128   0
0 128 255 128   0
0 128 255 128 127
0 128 255 128 127
```
