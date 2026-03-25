import numpy as np
import pytest

from troma.data_structure import (
    belongs_to_cylinder_set,
    create_cylinder_set_indicator,
    dit_string_to_computational_basis,
    dit_string_to_integer,
    integer_to_dit_string,
    kronecker_develop,
)


def test_integer_to_dit_string_basic_r_convention():
    out = integer_to_dit_string(5, dit_dimension=2, dit_string_length=4, convention="R")
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [0, 1, 0, 1]


def test_integer_to_dit_string_basic_l_convention():
    out = integer_to_dit_string(5, dit_dimension=2, dit_string_length=4, convention="L")
    assert out.tolist() == [1, 0, 1, 0]


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"integer": -1, "dit_dimension": 2, "dit_string_length": 3}, ValueError),
        ({"integer": 2.2, "dit_dimension": 2, "dit_string_length": 3}, TypeError),
        ({"integer": 1, "dit_dimension": 1, "dit_string_length": 3}, ValueError),
        ({"integer": 1, "dit_dimension": 2.0, "dit_string_length": 3}, TypeError),
        ({"integer": 1, "dit_dimension": 2, "dit_string_length": -1}, ValueError),
        ({"integer": 1, "dit_dimension": 2, "dit_string_length": 2.5}, TypeError),
        ({"integer": 1, "dit_dimension": 2, "dit_string_length": 3, "convention": "X"}, ValueError),
        ({"integer": 8, "dit_dimension": 2, "dit_string_length": 3}, ValueError),
        ({"integer": 1, "dit_dimension": 2, "dit_string_length": 0}, ValueError),
    ],
)
def test_integer_to_dit_string_invalid_inputs(kwargs, exc):
    with pytest.raises(exc):
        integer_to_dit_string(**kwargs)


def test_dit_string_to_integer_roundtrip():
    value = dit_string_to_integer([0, 1, 0, 1], dit_dimension=2, convention="R")
    assert value == 5


@pytest.mark.parametrize(
    "args,exc",
    [
        ((10, 2, "R"), TypeError),
        (([0, 1], 1, "R"), ValueError),
        (([0, 1], 2, "X"), ValueError),
        (([0, 1.2], 2, "R"), TypeError),
        (([0, 2], 2, "R"), ValueError),
    ],
)
def test_dit_string_to_integer_invalid_inputs(args, exc):
    with pytest.raises(exc):
        dit_string_to_integer(*args)


def test_dit_string_to_computational_basis_binary():
    out = dit_string_to_computational_basis([0, 1, 0], dit_dimension=2)
    assert out == [[1, 0], [0, 1], [1, 0]]


def test_dit_string_to_computational_basis_ternary():
    out = dit_string_to_computational_basis([0, 2, 1], dit_dimension=3)
    assert out == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]


@pytest.mark.parametrize(
    "args,exc",
    [
        ((10, 2), TypeError),
        (([0, 1], 1), ValueError),
        (([0, object()], 2), TypeError),
        (([0, 2], 2), ValueError),
    ],
)
def test_dit_string_to_computational_basis_invalid_inputs(args, exc):
    with pytest.raises(exc):
        dit_string_to_computational_basis(*args)


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
