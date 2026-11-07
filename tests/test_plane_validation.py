"""Plane parsing + per-row validation tests (FR-7..FR-11, FR-16 groundwork).

Predicates are one-for-one against the brief's wording (SRS FR-10/FR-11):
3-point isosceles legs from the first point; 4-point legs ``a1→b``/``a2→c``
with bases ``a1→a2``/``b→c``; EPS = 0.001. Fixtures are hand-computable.

Corpus verdicts are pinned here because they encode two decisions the raw
numbers prove: parallel legs are legal (every head.csv/onion-family 4-point
row is a rectangle and must validate), and the base rule needs the EPS
tolerance (onion-family rows exceed ``dist(b,c)`` by ~4e-7 float dust).
"""

from __future__ import annotations

import pytest

from plane_validation import Plane, PlaneError, parse_planes, count_planes


def write_csv(tmp_path, rows: str, header: str = "Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4"):
    file = tmp_path / "planes.csv"
    file.write_text(f"{header}\n{rows}")
    return file


# --- corpus verdicts (FR-10/FR-11 against the real corpus) ------------------

def test_lecturer_example_row_verdicts():
    # Hand-checked: row 1 isosceles legs 447.21/447.21; row 2 a rectangle
    # (legs 400/400, bases 500/500, parallel legs legal); row 3 fails the
    # 3-point isosceles rule (|497.39 - 476.86| = 20.53); row 4 fails the
    # 4-point legs rule (|300 - 400| = 100 >= EPS) - the ticket's
    # hand-checkable "100" case.
    records = parse_planes("assets/csv/lecturer_example.csv")
    assert [type(r).__name__ for r in records] == [
        "Plane",
        "Plane",
        "PlaneError",
        "PlaneError",
    ]
    assert "row 3" in records[2].banner() and "dist(a,b)" in records[2].banner()
    assert "20.5" in records[2].banner()
    assert "row 4" in records[3].banner() and "dist(a1,b)" in records[3].banner()
    assert "100" in records[3].banner()


@pytest.mark.parametrize(
    "name",
    ["croissant.csv", "head.csv", "onion.csv", "swiss_roll.csv", "biscuit_cake.csv"],
)
def test_corpus_plane_lists_fully_validate(name):
    # head.csv's 220 rectangles prove parallel legs are legal; the
    # onion-family files' ~4e-7 base-length float dust proves rule 2 needs
    # the EPS tolerance. Every row of these files must come out valid.
    records = parse_planes(f"assets/csv/{name}")
    errors = [r for r in records if isinstance(r, PlaneError)]
    assert errors == [], f"{name} unexpectedly flagged: {[e.banner() for e in errors[:3]]}"
    assert all(isinstance(r, Plane) for r in records)


# --- 3-point predicates (FR-10) ---------------------------------------------

def test_triangle_within_eps_is_valid(tmp_path):
    # legs 5 and 5 + 0.0005 -> inside EPS = 0.001
    records = parse_planes(write_csv(tmp_path, "3,0,0,0,5,0,0,0,5.0005,0,,\n"))
    assert isinstance(records[0], Plane)


def test_triangle_outside_eps_flags_the_isosceles_rule(tmp_path):
    # legs 5 and 6 -> |5 - 6| = 1 >= EPS
    records = parse_planes(write_csv(tmp_path, "3,0,0,0,5,0,0,0,6,0,,\n"))
    record = records[0]
    assert isinstance(record, PlaneError)
    assert record.row == 1
    assert "dist(a,b)" in record.reason and "dist(a,c)" in record.reason
    assert "1" in record.reason and "0.001" in record.reason


def test_triangle_legs_are_measured_from_the_first_point(tmp_path):
    # dist(a,b)=5, dist(a,c)=5, but dist(b,c)=8: valid regardless of the
    # base length - the rule only compares the two legs from a.
    records = parse_planes(write_csv(tmp_path, "3,0,0,0,5,0,0,0,-5,0,,\n"))
    assert isinstance(records[0], Plane)


# --- 4-point predicates (FR-10) ----------------------------------------------

def test_trapezoid_valid(tmp_path):
    # a1=(0,0,0) a2=(2,0,0) b=(0,4,0) c=(5,2.6458,0): legs 4/4, bases
    # 2 <= 5.18, coplanar (z=0), legs not parallel.
    records = parse_planes(write_csv(tmp_path, "4,0,0,0,2,0,0,0,4,0,5,2.6457513,0\n"))
    assert isinstance(records[0], Plane)


def test_trapezoid_legs_rule_names_the_exact_predicate(tmp_path):
    # legs a1->b = 3, a2->c = 4 -> |3 - 4| = 1 >= EPS
    records = parse_planes(write_csv(tmp_path, "4,0,0,0,2,0,0,0,3,0,2,4,0\n"))
    record = records[0]
    assert isinstance(record, PlaneError)
    assert "dist(a1,b)" in record.reason and "dist(a2,c)" in record.reason
    assert "0.001" in record.reason


def test_trapezoid_base_rule_dist_a1a2_le_dist_bc(tmp_path):
    # legs 4/4 (b=(0,4,0), c=(1,0,0) from a2=(5,0,0)); top base a1a2 = 5
    # > bottom base bc = 4.123 -> rule 2 fails.
    records = parse_planes(write_csv(tmp_path, "4,0,0,0,5,0,0,0,4,0,1,0,0\n"))
    record = records[0]
    assert isinstance(record, PlaneError)
    assert "dist(a1,a2)" in record.reason and "dist(b,c)" in record.reason


def test_trapezoid_base_rule_tolerates_float_dust(tmp_path):
    # top base 5.0000004, bottom base 5 (onion-family dust) -> valid.
    records = parse_planes(write_csv(tmp_path, "4,0,0,0,0,0,5.0000004,0,4,0,0,4,5\n"))
    assert isinstance(records[0], Plane)


def test_trapezoid_coplanarity_rule(tmp_path):
    # legs both 4 (b=(0,4,0), c=(2,3,√7) from a2=(2,0,0)), bases 2 <= 3.46,
    # but the legs' directions span different planes (triple product ≠ 0).
    records = parse_planes(write_csv(tmp_path, "4,0,0,0,2,0,0,0,4,0,2,3,2.6457513\n"))
    record = records[0]
    assert isinstance(record, PlaneError)
    assert "same plane" in record.reason or "coplanar" in record.reason


def test_trapezoid_parallel_legs_are_legal(tmp_path):
    # Rectangle: legs a1->b and a2->c both (0,4,0). head.csv is full of
    # these; they must validate (FR-11's degeneracy is zero-width frames,
    # not parallel directions).
    records = parse_planes(write_csv(tmp_path, "4,0,0,0,0,0,5,0,4,0,0,4,5\n"))
    assert isinstance(records[0], Plane)


# --- constructibility gates (FR-11) ------------------------------------------

def test_zero_length_leg_flags_constructibility(tmp_path):
    records = parse_planes(write_csv(tmp_path, "3,0,0,0,0,0,0,0,5,0,,\n"))
    record = records[0]
    assert isinstance(record, PlaneError)
    assert "zero-length" in record.reason


def test_trapezoid_zero_length_leg_flags_constructibility(tmp_path):
    records = parse_planes(write_csv(tmp_path, "4,0,0,0,2,0,0,0,0,0,2,4,0\n"))
    record = records[0]
    assert isinstance(record, PlaneError)
    assert "zero-length" in record.reason


def test_triangle_with_coincident_b_and_c_is_degenerate(tmp_path):
    # legs 5/5 (isosceles holds) but b == c -> zero-width frame.
    records = parse_planes(write_csv(tmp_path, "3,0,0,0,5,0,0,5,0,0,,\n"))
    record = records[0]
    assert isinstance(record, PlaneError)
    assert "degenerate" in record.reason or "zero-width" in record.reason


def test_trapezoid_zero_top_base_with_same_direction_legs_is_degenerate(tmp_path):
    # legs 4/4 (isosceles holds), a1 == a2, and the legs step the same way
    # (a2->c == a1->b) -> every row's width is 0 -> zero-width frame.
    records = parse_planes(write_csv(tmp_path, "4,0,0,0,0,0,0,0,4,0,0,4,0\n"))
    record = records[0]
    assert isinstance(record, PlaneError)
    assert "degenerate" in record.reason or "zero-width" in record.reason


def test_trapezoid_zero_top_base_with_opposite_legs_renders_as_a_triangle(tmp_path):
    # a1 == a2 but leg a2->c = -leg a1->b: opposite-way legs are the 3-point
    # triangle geometry (apex a1, base through b and c) - valid and rendered,
    # so FR-11 only rejects the same-direction collapse.
    records = parse_planes(write_csv(tmp_path, "4,0,0,0,0,0,0,0,4,0,0,-4,0\n"))
    assert isinstance(records[0], Plane)


def test_trapezoid_zero_top_base_with_non_parallel_legs_is_a_triangle(tmp_path):
    # a1 == a2 but the legs diverge: the pseudocode renders this as a
    # triangle (constant apex, widening rows) - valid.
    records = parse_planes(write_csv(tmp_path, "4,0,0,0,0,0,0,0,4,0,0,0,4\n"))
    assert isinstance(records[0], Plane)


# --- row-level parsing (FR-7..FR-9) ------------------------------------------

def test_row_index_is_one_based_over_data_rows(tmp_path):
    rows = "3,0,0,0,5,0,0,0,5,,\n3,0,0,0,5,0,0,0,6,,\n3,0,0,0,1,0,0,0,5,,\n"
    records = parse_planes(write_csv(tmp_path, rows))
    assert [r.row for r in records] == [1, 2, 3]
    assert isinstance(records[2], PlaneError)


def test_wrong_point_count_is_an_error_record_not_a_crash(tmp_path):
    records = parse_planes(write_csv(tmp_path, "5,0,0,0,1,0,0,2,0,0,3,0,0,4,0,0\n"))
    record = records[0]
    assert isinstance(record, PlaneError)
    assert "3 or 4" in record.reason


def test_non_numeric_coordinate_is_an_error_record(tmp_path):
    records = parse_planes(write_csv(tmp_path, "3,0,0,0,5,0,0,oops,5,,\n"))
    record = records[0]
    assert isinstance(record, PlaneError)
    assert "number" in record.reason


def test_short_row_is_an_error_record(tmp_path):
    records = parse_planes(write_csv(tmp_path, "3,0,0,0,5,0\n"))
    record = records[0]
    assert isinstance(record, PlaneError)


def test_count_planes_comes_from_parse_planes(tmp_path):
    # One record per non-empty row, so the placeholder frame count (one per
    # row, error frames included) keeps matching the ticket-1 pinned counts.
    rows = "3,0,0,0,5,0,0,0,5,,\n\n3,0,0,0,5,0,0,0,6,,\n"
    assert count_planes(write_csv(tmp_path, rows)) == 2