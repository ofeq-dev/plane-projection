"""Volume loader tests (FR-3..FR-6): dtype contract, dimensions, index space.

Synthetic NRRDs are written with pynrrd into tmp_path (no binary fixtures in
git); the axis-mapping fixture is a hand-written ASCII NRRD whose disk byte
order is known, pinning the ``M[x][y][z]`` mapping (FR-3).
"""

from __future__ import annotations

import numpy as np
import nrrd
import pytest

import Ex0


def write_nrrd(path, array, header=None):
    nrrd.write(str(path), array, header=header or {}, index_order="F")


def test_accepts_the_8_16_32_bit_unsigned_contract(tmp_path):
    # FR-4: uint8 / uint16 / uint32 (M is 8-, 16-, or 32-bit unsigned values).
    for dtype in (np.uint8, np.uint16, np.uint32):
        array = np.array([[[0, 1], [2, 3]]], dtype=dtype)
        file = tmp_path / f"vol_{np.dtype(dtype).name}.nrrd"
        write_nrrd(file, array)
        loaded = Ex0.load_volume(file)
        assert loaded.dtype == dtype
        assert loaded.tolist() == array.tolist()


def test_tolerates_signed_short(tmp_path):
    # FR-4: signed integer types are tolerated (MRHead_1.nrrd is 'short').
    array = np.array([[[0, -1], [-2, -3]]], dtype=np.int16)
    file = tmp_path / "signed.nrrd"
    write_nrrd(file, array, header={"type": "short"})
    loaded = Ex0.load_volume(file)
    assert loaded.dtype == np.dtype("short")
    assert loaded.tolist() == array.tolist()


def test_other_signed_integers_are_tolerated(tmp_path):
    # FR-4 tolerates signed integer types generally (e.g. int64).
    array = np.array([[[np.iinfo(np.int64).min]]], dtype=np.int64)
    file = tmp_path / "int64.nrrd"
    write_nrrd(file, array)
    assert Ex0.load_volume(file).dtype == np.dtype("int64")


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(np.array([[[1.0]]], dtype=np.float32), id="float32"),
        pytest.param(np.array([[[1.0]]], dtype=np.float64), id="float64"),
        pytest.param(np.array([[[1]]], dtype=np.uint64), id="uint64"),
        pytest.param(np.array([[[[1, 1, 1]]]], dtype=np.uint8), id="vector-like-4d"),
    ],
)
def test_rejects_dtypes_outside_the_contract(tmp_path, array):
    # FR-4: only the unsigned 8/16/32-bit contract plus signed-integer
    # tolerance pass. Anything else is rejected with a one-line error naming
    # the offending type (the 4D case is the same check downstream of NRRD's
    # vector handling).
    if array.ndim == 4:
        with pytest.raises(Exception, match="dimensions"):
            Ex0.load_volume(_write(tmp_path, array))
        return
    file = _write(tmp_path, array)
    with pytest.raises(Ex0.UsageError) as excinfo:
        Ex0.load_volume(file)
    assert str(array.dtype) in str(excinfo.value)


def _write(tmp_path, array):
    file = tmp_path / f"reject_{array.dtype.name}_{array.ndim}d.nrrd"
    write_nrrd(file, array)
    return file


def test_rejects_non_3d_arrays(tmp_path):
    # FR-3: the tool treats the volume as M[x][y][z] — exactly three axes.
    array_2d = np.zeros((4, 4), dtype=np.uint8)
    file = tmp_path / "in2d.nrrd"
    write_nrrd(file, array_2d, header={"type": "uchar", "dimension": 2})
    with pytest.raises(Ex0.UsageError, match="2 dimensions"):
        Ex0.load_volume(file)


def test_lower_dimension_bound_is_inclusive_at_one(tmp_path):
    # FR-5: 1 <= s. A 1-voxel volume is tiny but valid.
    array = np.zeros((1, 1, 1), dtype=np.uint8)
    file = tmp_path / "atom.nrrd"
    write_nrrd(file, array)
    assert Ex0.load_volume(file).shape == (1, 1, 1)


def test_upper_dimension_bound_is_exclusive_at_2_to_the_10(tmp_path):
    # FR-5: 1 <= s < 2^10 — 1024 is rejected naming the axis and the limit;
    # 1023 passes.
    file = tmp_path / "too_big.nrrd"
    write_nrrd(file, np.zeros((1, 1, 1024), dtype=np.uint8))
    with pytest.raises(Ex0.UsageError) as excinfo:
        Ex0.load_volume(file)
    message = str(excinfo.value)
    assert "sz=1024" in message
    assert "1 <= s < 2^10" in message

    write_nrrd(file, np.zeros((1, 1, 1023), dtype=np.uint8))
    assert Ex0.load_volume(file).shape == (1, 1, 1023)


def test_axis_mapping_pins_M_x_y_z_to_disk_order(tmp_path):
    # FR-3: the array is M[x][y][z] with sx, sy, sz = its shape, axis 0 = x.
    # A hand-written ASCII NRRD lists values fastest-axis-first (NRRD §1.4:
    # axis 0 = x varies fastest), so disk value k corresponds to
    # M[x][y][z] = v[x][y][z] == 100*x + 10*y + z.
    content = (
        "NRRD0001\n"
        "type: short\n"
        "dimension: 3\n"
        "sizes: 2 3 4\n"
        "encoding: ascii\n"
        "\n"
        + " ".join(str(100 * x + 10 * y + z) for z in range(4) for y in range(3) for x in range(2))
    )
    file = tmp_path / "order.nrrd"
    file.write_text(content, encoding="ascii")
    volume = Ex0.load_volume(file)
    assert volume.shape == (2, 3, 4)
    for x in range(2):
        for y in range(3):
            for z in range(4):
                assert volume[x, y, z] == 100 * x + 10 * y + z


def test_world_space_header_fields_are_ignored(tmp_path):
    # FR-6 / ADR-0002: 'space', 'space origin', 'space directions' are
    # deliberately unused — the loader neither applies nor errors on them; the
    # volume comes back as raw voxel-index values, unaffected by world space.
    directions = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.7]])
    header = {
        "space": "left-posterior-superior",
        "space origin": np.asarray([8.6, -13.9, 11.7]),
        "space directions": directions,
        "space dimension": 3,
    }
    array = np.array([[[7, 9], [4, 2]], [[1, 3], [5, 6]]], dtype=np.uint8)
    file = tmp_path / "spaced.nrrd"
    write_nrrd(file, array, header=header)
    loaded = Ex0.load_volume(file)
    assert loaded.tolist() == array.tolist()  # values used as-is (index space)


def test_missing_and_directory_paths_are_rejected(tmp_path):
    # FR-2: problem file named in a one-line error. The loader-level contract
    # is the same message shape the CLI surfaces.
    with pytest.raises(Ex0.UsageError, match="no such file"):
        Ex0.load_volume(tmp_path / "absent.nrrd")
    with pytest.raises(Ex0.UsageError, match="is a directory"):
        Ex0.load_volume(tmp_path)


def test_garbage_file_is_rejected_with_a_named_problem(tmp_path):
    file = tmp_path / "garbage.nrrd"
    file.write_text("this is not an nrrd at all")
    with pytest.raises(Ex0.UsageError, match=str(file.name)):
        Ex0.load_volume(file)


def test_provided_corpus_volume_loads():
    # The lecturer's encouraged starter file (short, 256x256x130) passes the
    # dtype contract (tolerance) and the dimension bound, in index space.
    volume = Ex0.load_volume("assets/nrrd/MRHead_1.nrrd")
    assert volume.shape == (256, 256, 130)
    assert volume.dtype == np.dtype("short")
