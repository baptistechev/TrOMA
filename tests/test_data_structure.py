import numpy as np
import pytest

from troma.core.structure import DitString
from troma.core.data_structure import (
    belongs_to_cylinder_set,
    create_cylinder_set_indicator,
    kronecker_develop,
)


# ---------------------------------------------------------------------------
# DitString construction (replaces integer_to_dit_string)
# ---------------------------------------------------------------------------

def test_from_integer_basic_r_convention():
    out = DitString.from_integer(5, length=4, dimension=2)
    assert isinstance(out, DitString)
    assert out.length == 4
    assert out.dimension == 2
    assert out.tolist() == [0, 1, 0, 1]


def test_from_integer_basic_l_convention():
    out = DitString.from_integer(5, length=4, dimension=2, convention="L")
    assert out.tolist() == [1, 0, 1, 0]


def test_from_integer_roundtrip():
    for n in range(8):
        ds = DitString.from_integer(n, length=3, dimension=2)
        assert ds.to_integer() == n


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"integer": -1, "length": 3, "dimension": 2}, ValueError),
        ({"integer": 2.2, "length": 3, "dimension": 2}, TypeError),
        ({"integer": 1, "length": 3, "dimension": 1}, ValueError),
        ({"integer": 1, "length": 3, "dimension": 2.0}, TypeError),
        ({"integer": 1, "length": 2.5, "dimension": 2}, TypeError),
        ({"integer": 8, "length": 3, "dimension": 2}, ValueError),
        ({"integer": 1, "length": 0, "dimension": 2}, ValueError),
    ],
)
def test_from_integer_invalid_inputs(kwargs, exc):
    with pytest.raises(exc):
        DitString.from_integer(**kwargs)


# ---------------------------------------------------------------------------
# DitString.to_integer (replaces dit_string_to_integer)
# ---------------------------------------------------------------------------

def test_to_integer_r_convention():
    ds = DitString([0, 1, 0, 1], dimension=2)
    assert ds.to_integer(convention="R") == 5


def test_to_integer_l_convention():
    ds = DitString([1, 0, 1, 0], dimension=2)
    assert ds.to_integer(convention="L") == 5


def test_to_integer_ternary():
    ds = DitString([1, 0, 2], dimension=3)
    assert ds.to_integer() == 1 * 9 + 0 * 3 + 2


def test_to_integer_empty():
    ds = DitString([], dimension=2)
    assert ds.to_integer() == 0


# ---------------------------------------------------------------------------
# DitString.to_computational_basis (replaces dit_string_to_computational_basis)
# ---------------------------------------------------------------------------

def test_to_computational_basis_binary():
    out = DitString([0, 1, 0], dimension=2).to_computational_basis()
    assert out == [[1, 0], [0, 1], [1, 0]]


def test_to_computational_basis_ternary():
    out = DitString([0, 2, 1], dimension=3).to_computational_basis()
    assert out == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]


def test_to_computational_basis_length_matches():
    ds = DitString([0, 1, 0, 1], dimension=2)
    out = ds.to_computational_basis()
    assert len(out) == ds.length
    assert all(len(v) == ds.dimension for v in out)


# ---------------------------------------------------------------------------
# create_cylinder_set_indicator
# ---------------------------------------------------------------------------

def test_create_cylinder_set_indicator_binary_shape_and_content():
    out = create_cylinder_set_indicator(fixed_dit_positions=[0, 2], set_size=4, dit_dimension=2)
    assert len(out) == 4
    assert out[0] == [[1, 0], [1, 1], [1, 0], [1, 1]]
    assert out[-1] == [[0, 1], [1, 1], [0, 1], [1, 1]]


@pytest.mark.parametrize(
    "args,exc",
    [
        (([0], -1, 2), ValueError),
        (([0], 4.2, 2), TypeError),
        (([0], 4, 1), ValueError),
        (([0.5], 4, 2), TypeError),
        (([-1], 4, 2), ValueError),
        (([0, 0], 4, 2), ValueError),
        (([4], 4, 2), ValueError),
    ],
)
def test_create_cylinder_set_indicator_invalid_inputs(args, exc):
    with pytest.raises(exc):
        create_cylinder_set_indicator(*args)


# ---------------------------------------------------------------------------
# kronecker_develop
# ---------------------------------------------------------------------------

def test_kronecker_develop_binary_r_and_l():
    cylinder_set = [[1, 0], [0, 1]]
    out_r = kronecker_develop(cylinder_set, dit_dimension=2, convention="R")
    out_l = kronecker_develop(cylinder_set, dit_dimension=2, convention="L")
    assert out_r.tolist() == [0, 1, 0, 0]
    assert out_l.tolist() == [0, 0, 1, 0]


@pytest.mark.parametrize(
    "args,exc",
    [
        ((10, 2, "R"), TypeError),
        (([[1, 0]], 1, "R"), ValueError),
        (([[1, 0]], 2, "X"), ValueError),
        (([[1, 0], 5], 2, "R"), TypeError),
        (([[1, 0], [1, 0, 0]], 2, "R"), ValueError),
        (([[1, 2]], 2, "R"), ValueError),
    ],
)
def test_kronecker_develop_invalid_inputs(args, exc):
    with pytest.raises(exc):
        kronecker_develop(*args)


# ---------------------------------------------------------------------------
# belongs_to_cylinder_set
# ---------------------------------------------------------------------------

def test_belongs_to_cylinder_set_member():
    element = [[1, 0], [0, 1]]
    cylinder_set = [[1, 0], [1, 1]]
    assert belongs_to_cylinder_set(element, cylinder_set) is True


def test_belongs_to_cylinder_set_not_member():
    element = [[0, 1], [0, 1]]
    cylinder_set = [[1, 0], [1, 1]]
    assert belongs_to_cylinder_set(element, cylinder_set) is False


def test_belongs_to_cylinder_set_all_free():
    element = [[1, 0], [0, 1]]
    cylinder_set = [[1, 1], [1, 1]]
    assert belongs_to_cylinder_set(element, cylinder_set) is True


@pytest.mark.parametrize(
    "args,exc",
    [
        ((10, [[1, 0]]), TypeError),
        (([[1, 0]], 10), TypeError),
        (([[1, 0]], [5]), TypeError),
        (([[1, 0]], [[1, 2]]), ValueError),
        (([[1, 0]], [[1, 0, 0]]), ValueError),
    ],
)
def test_belongs_to_cylinder_set_invalid_inputs(args, exc):
    with pytest.raises(exc):
        belongs_to_cylinder_set(*args)
