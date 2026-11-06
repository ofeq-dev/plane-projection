"""Renderer seam tests (FR-12): volume + validated rows -> frame records.

The single seam under test is :func:`Ex0.render_frames`; canvas assembly is
tested through :func:`Ex0.canvas_size` / :func:`Ex0.pad_frame`. Golden frames
live in ``tests/golden/`` and are regenerated with
``python tests/generate_goldens.py``; the hand-derived tables here are the
independent oracle for those goldens (see ``docs/algorithm.md``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import Ex0
from render_cases import CASES, ramp_volume, short_ramp_volume

GOLDEN_DIR = Path(__file__).parent / "golden"


def frame_of(volume, points):
    records = Ex0.render_frames(volume, [Ex0.Plane(row=1, points=tuple(points))])
    assert len(records) == 1
    frame = records[0]
    assert isinstance(frame, Ex0.Frame)
    return frame


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_committed_golden_frame_matches(case):
    # The committed golden is this code's output; the independent oracle for
    # it is the hand-derived table in docs/algorithm.md and the explicit pixel
    # assertions below. Regenerate with `python tests/generate_goldens.py`;
    # the regeneration must leave `git status` clean.
    frame = frame_of(case.volume, case.points)
    golden = np.load(GOLDEN_DIR / f"{case.name}.npy")
    np.testing.assert_array_equal(frame.image, golden)


def test_trapezoid_matches_hand_derived_pixels():
    # a1=(2,2,0), a2=(4,2,0), b=(0,6,0), c=(6,6,0): legs sqrt(20) -> 4 rows;
    # top base 2, bottom base 6 -> 6 columns. Row t=(y+1)/4:
    #   y=0: width 3, x = 1.5, 2.5, 3.5, 4.5 -> 2, 2, 4, 4
    #   y=1: width 4, x = 1.5 .. 4.5 -> 2, 2, 4, 4
    #   y=2: width 5, x = 0.5 .. 5.5 -> 0, 2, 2, 4, 4, 6
    #   y=3: width 6, same column samples -> 0, 2, 2, 4, 4, 6
    frame = frame_of(
        ramp_volume((8, 8, 5)),
        [(2, 2, 0), (4, 2, 0), (0, 6, 0), (6, 6, 0)],
    )
    assert frame.image.tolist() == [
        [0, 2, 2, 4, 4, 0],
        [0, 2, 2, 4, 4, 0],
        [0, 2, 2, 4, 4, 6],
        [0, 2, 2, 4, 4, 6],
    ]


def test_frame_dimensions_follow_the_density_rule():
    # Rectangle a1=(0,0,0), a2=(0,2.4,0), b=(2.6,0,0), c=(2.6,2.4,0):
    # leg 2.6 -> round 3 rows; base 2.4 -> round 2 columns (FR-12b).
    frame = frame_of(
        np.zeros((3, 3, 1), dtype=np.uint8),
        [(0, 0, 0), (0, 2.4, 0), (2.6, 0, 0), (2.6, 2.4, 0)],
    )
    assert frame.image.shape == (3, 2)


def test_zero_top_base_opposite_legs_render_as_a_triangle():
    # a1 == a2 (collapsed top edge) with opposite-way legs: exactly the
    # 3-point triangle geometry, apex a1, base b->c — leg 4 -> 4 rows,
    # base 8 -> 8 columns (FR-11 rejects only the same-direction collapse).
    frame = frame_of(
        np.zeros((3, 8, 5), dtype=np.uint8),
        [(0, 0, 0), (0, 0, 0), (0, 4, 0), (0, -4, 0)],
    )
    assert frame.image.shape == (4, 8)


@pytest.mark.parametrize(
    ("dtype", "value", "expected"),
    [
        (np.uint8, 255, 255),
        (np.uint8, 100, 100),  # no content-based stretching to 255
        (np.uint16, 65535, 255),
        (np.uint16, 32768, 128),  # 127.50... rounds half-to-even
        (np.uint32, 4294967295, 255),
        (np.int16, 32767, 255),
        (np.int16, 0, 128),  # midpoint of the full signed range
        (np.int16, -32768, 0),
    ],
)
def test_display_scaling_uses_the_full_dtype_range(dtype, value, expected):
    # FR-12e: a constant volume renders every in-volume sample at the value's
    # fixed position in the dtype range - never stretched to the data's max.
    volume = np.full((4, 4, 1), value, dtype=dtype)
    frame = frame_of(volume, [(1, 1, 0), (1, 3, 0), (3, 1, 0)])
    assert frame.image[1, 1] == expected
    assert frame.image[0, 1] == expected


@pytest.mark.parametrize(
    ("csv_name", "volume_name", "frame_shape", "canvas"),
    [
        ("croissant.csv", "croissant", (559, 500), (500, 560)),
        ("onion.csv", "onion", (255, 255), (256, 256)),
        ("head.csv", "MRHead_1", (255, 255), (256, 256)),
    ],
)
def test_corpus_rows_render_at_density_dimensions_with_content(
    csv_name, volume_name, frame_shape, canvas
):
    # FR-12b on the real corpus: head/onion's 255-length edges give 255-pixel
    # frames; croissant's 559.02 leg gives 559 rows, 500 base gives 500.
    volume = Ex0.load_volume(f"assets/nrrd/{volume_name}.nrrd")
    records = Ex0.parse_planes(f"assets/csv/{csv_name}")
    rendered = Ex0.render_frames(volume, records)
    assert not any(isinstance(r, Ex0.PlaneError) for r in rendered)

    first = rendered[0]
    assert isinstance(first, Ex0.Frame)
    assert first.image.shape == frame_shape
    assert first.image.max() > 0  # real content, not an empty placeholder
    assert Ex0.canvas_size(rendered) == canvas


def test_head_frames_beyond_the_volume_are_black_regions_not_errors():
    # FR-12d: head.csv's oblique rows extend past z=0, so part of the frame
    # samples outside M and renders black while the rest carries the slice.
    volume = Ex0.load_volume("assets/nrrd/MRHead_1.nrrd")
    records = Ex0.parse_planes("assets/csv/head.csv")
    rendered = Ex0.render_frames(volume, records)
    frame = rendered[200]  # row 201: z runs 146.46 down to -17.46
    assert isinstance(frame, Ex0.Frame)
    assert frame.image.max() > 0
    assert (frame.image == 0).any()


def test_render_frames_passes_error_rows_through_in_order():
    volume = np.zeros((4, 4, 1), dtype=np.uint8)
    ok1 = Ex0.Plane(row=1, points=((1, 1, 0), (1, 3, 0), (3, 1, 0)))
    bad = Ex0.PlaneError(row=2, reason="zero-length leg a→b (a == b)")
    ok3 = Ex0.Plane(row=3, points=((0, 0, 0), (0, 2, 0), (2, 0, 0)))
    records = Ex0.render_frames(volume, [ok1, bad, ok3])
    assert [r.row for r in records] == [1, 2, 3]
    assert isinstance(records[0], Ex0.Frame)
    assert records[1] is bad
    assert isinstance(records[2], Ex0.Frame)


def test_canvas_is_the_max_frame_size_made_even():
    # FR-18/FR-19: one canvas per video = max width and max height over the
    # renderable frames, rounded up to even for the mp4v writer.
    records = [
        Ex0.Frame(row=1, image=np.zeros((3, 2), dtype=np.uint8)),
        Ex0.PlaneError(row=2, reason="bad plane"),
        Ex0.Frame(row=3, image=np.zeros((2, 3), dtype=np.uint8)),
    ]
    assert Ex0.canvas_size(records) == (4, 4)


def test_canvas_falls_back_to_placeholder_when_no_row_renders():
    # All-error CSV still yields a video (FR-20); there is no renderable
    # frame size to take a max over, so the placeholder canvas carries it.
    records = [Ex0.PlaneError(row=1, reason="bad plane")]
    assert Ex0.canvas_size(records) == Ex0.PLACEHOLDER_CANVAS
    assert Ex0.canvas_size([]) == Ex0.PLACEHOLDER_CANVAS


def test_error_frame_has_red_banner_white_text_and_black_remainder():
    canvas = Ex0.render_error_frame(
        Ex0.PlaneError(row=7, reason="|dist(a1,b) − dist(a2,c)| = 100 ≥ EPS=0.001"),
        (256, 256),
    )
    assert canvas.shape == (256, 256, 3)
    assert canvas[0, 0].tolist() == [0, 0, 255]
    assert ((canvas[:80] == [255, 255, 255]).all(axis=2)).any()
    assert canvas[-1, -1].tolist() == [0, 0, 0]


def test_error_frame_wrapping_and_determinism():
    # FR-15: the banner carries the row index followed by the specific rule;
    # long reasons wrap to fill the fixed top corner, and the same record
    # always produces the identical frame (banner determinism, FR-16).
    error = Ex0.PlaneError(
        row=9,
        reason=(
            "|dist(a1,b) − dist(a2,c)| = 100 ≥ EPS = 0.001"
            " (dist(a1,b) = 300, dist(a2,c) = 400)"
        ),
    )
    wrapped = Ex0.render_error_frame(error, (120, 90))
    # Text lines occupy at least two distinct bands of the top banner: rows
    # with white pixels separated by banner-only rows.
    text_rows = {int(y) for y, _ in np.argwhere(
        (wrapped[:45] == (255, 255, 255)).all(axis=2))}
    bands = [y for i, y in enumerate(sorted(text_rows)) if i == 0 or y - sorted(text_rows)[i - 1] > 5]
    assert len(bands) >= 2
    assert (wrapped == Ex0.render_error_frame(error, (120, 90))).all()


def test_error_frame_truncation_keeps_text_inside_canvas():
    # FR-18: the banner is drawn inside the frame. Multi-word reasons wrap
    # (white text, none below the banner band); the pathological single
    # overlong word cannot wrap, so it is drawn at the minimum scale and
    # blends into the banner rather than crashing or escaping the canvas.
    error = Ex0.PlaneError(row=3, reason=" ".join(["ox"] * 60))
    canvas = Ex0.render_error_frame(error, (64, 32))
    assert canvas.shape == (32, 64, 3)
    assert canvas[0, 0].tolist() == [0, 0, 255]
    assert ((canvas[:20] == (255, 255, 255)).all(axis=2)).any()
    assert ((canvas[20:] == (255, 255, 255)).all(axis=2)).any() == False  # noqa: E712

    giant = Ex0.render_error_frame(Ex0.PlaneError(row=3, reason="x" * 200), (64, 32))
    assert giant.shape == (32, 64, 3)
    assert ((giant[..., 2] == 255) & (giant[..., 0] == 0)).any()  # red banner intact


def test_pad_frame_places_content_top_left_and_black_fills():
    # FR-18: padding is top-left, background black - no centring, no aspect
    # compensation (FR-21).
    image = np.full((3, 2), 7, dtype=np.uint8)
    padded = Ex0.pad_frame(image, (5, 4))
    assert padded.shape == (4, 5)
    assert padded[:3, :2].tolist() == [[7, 7], [7, 7], [7, 7]]
    assert padded[3, :].tolist() == [0, 0, 0, 0, 0]
    assert padded[:, 2:].tolist() == [[0, 0, 0]] * 4


def test_pad_frame_refuses_a_frame_larger_than_the_canvas():
    with pytest.raises(ValueError, match="larger than canvas"):
        Ex0.pad_frame(np.zeros((3, 3), dtype=np.uint8), (2, 2))


def test_out_of_volume_samples_are_black_not_clipped():
    # 4^3 volume all 255; triangle a=(2,2,2), b=(-2,2,2), c=(6,2,2):
    # leg 4 -> 4 rows, base 8 -> 8 columns. Rows widen; samples rounding to
    # x<0 or x>3 are outside M and render 0 (FR-12d), never a clipped edge
    # voxel. Column 3's top-row samples x=1.5,2.5 are the first in-bounds pair.
    volume = np.full((4, 4, 4), 255, dtype=np.uint8)
    frame = frame_of(volume, [(2, 2, 2), (-2, 2, 2), (6, 2, 2)])
    assert frame.image.tolist() == [
        [0, 0, 0, 255, 255, 0, 0, 0],
        [0, 0, 255, 255, 255, 0, 0, 0],
        [0, 255, 255, 255, 255, 0, 0, 0],
        [0, 255, 255, 255, 255, 0, 0, 0],
    ]


def test_rounding_ties_use_half_to_even():
    # Plane x = 2.5 (vertical y-z plane), volume value = x. Every in-volume
    # sample is exactly 2.5 -> np.rint -> 2 (even), not half-up's 3 (FR-12a).
    # leg sqrt(40) = 6.32 -> 6 rows; base 4 -> 4 columns.
    volume = ramp_volume((3, 6, 7))
    frame = frame_of(volume, [(2.5, 2, 0), (2.5, 0, 6), (2.5, 4, 6)])
    assert frame.image.tolist() == [
        [0, 0, 0, 0],
        [0, 2, 2, 0],
        [0, 2, 2, 0],
        [0, 2, 2, 0],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
    ]


def test_short_dtype_scales_over_the_full_signed_range():
    # int16 volume with x-values min(-32768), 0, max(32767), +100, -100.
    # Triangle a=(2,0,0), b=(-0.5,4,0), c=(4.5,4,0): legs sqrt(22.25)=4.72
    # -> 5 rows; base 5 -> 5 columns. Rows widen 1,2,3,4,5 units; samples are
    # consecutive integer x at the bottom, so the full signed range is visibly
    # distinct: 0, 128, 255, 128, 127 (FR-12e).
    frame = frame_of(
        short_ramp_volume((5, 5, 1)),
        [(2, 0, 0), (-0.5, 4, 0), (4.5, 4, 0)],
    )
    assert frame.image.tolist() == [
        [0, 0, 255, 0, 0],
        [0, 128, 255, 128, 0],
        [0, 128, 255, 128, 0],
        [0, 128, 255, 128, 127],
        [0, 128, 255, 128, 127],
    ]


def test_triangle_matches_hand_derived_pixels():
    # a=(2,2,0), b=(0,2,0), c=(4,2,0): leg 2, base 4 -> 2x4 frame.
    # Row t=(y+1)/2; columns are pixel centres at offsets -1.5..1.5 from the
    # row centre; |offset| <= row half-width keeps the triangle, black outside.
    #   y=0: row width 2, samples x = 1.5, 2.5 -> rint -> 2, 2
    #   y=1: row width 4, samples x = 0.5, 1.5, 2.5, 3.5 -> 0, 2, 2, 4
    frame = frame_of(ramp_volume((6, 5, 5)), [(2, 2, 0), (0, 2, 0), (4, 2, 0)])
    assert frame.image.tolist() == [
        [0, 2, 2, 0],
        [0, 2, 2, 4],
    ]
