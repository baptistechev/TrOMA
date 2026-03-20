import importlib
import sys

import numpy as np
import pytest

import myopt.data_structure as data_structure_module

# abstract.py uses "from data_structure import ..."; expose package module under that name for tests.
sys.modules.setdefault("data_structure", data_structure_module)
abstract = importlib.import_module("myopt.abstract")


@pytest.mark.parametrize(
    "name,value,exc",
    [
        ("x", 1.2, TypeError),
        ("x", True, TypeError),
        ("x", 0, ValueError),
    ],
)
def test_validate_positive_int_raises(name, value, exc):
    with pytest.raises(exc):
        abstract._validate_positive_int(name, value)


def test_validate_constraint_dict_type_raises():
    with pytest.raises(TypeError):
        abstract._validate_constraint_dict([(0, 1)], dit_string_length=3, dit_dimension=2)


@pytest.mark.parametrize(
    "constraint,exc",
    [
        ({"0": 1}, TypeError),
        ({-1: 1}, ValueError),
        ({3: 1}, ValueError),
        ({0: "1"}, TypeError),
        ({0: -1}, ValueError),
        ({0: 2}, ValueError),
    ],
)
def test_validate_constraint_dict_bad_entries_raise(constraint, exc):
    with pytest.raises(exc):
        abstract._validate_constraint_dict(constraint, dit_string_length=3, dit_dimension=2)


@pytest.mark.parametrize(
    "constraint,exc",
    [
        (5, TypeError),
        ([0, 1], ValueError),
        ([0, "1", 1], TypeError),
        ([0, -1, 1], ValueError),
        ([0, 2, 1], ValueError),
    ],
)
def test_validate_full_assignment_raises(constraint, exc):
    with pytest.raises(exc):
        abstract._validate_full_assignment(constraint, dit_string_length=3, dit_dimension=2)


def test_compute_marginal_bad_function_input_dits_type_raises():
    with pytest.raises(TypeError):
        abstract.compute_marginal(123, [1.0], {0: 0})


def test_compute_marginal_bad_function_values_type_raises():
    with pytest.raises(TypeError):
        abstract.compute_marginal([[0, 1]], 123, {0: 0})


def test_compute_marginal_length_mismatch_raises():
    with pytest.raises(ValueError):
        abstract.compute_marginal([[0, 1]], [1.0, 2.0], {0: 0})


def test_compute_marginal_bad_state_type_raises():
    with pytest.raises(TypeError):
        abstract.compute_marginal([5, [0, 1]], [1.0, 2.0], {0: 0})


def test_compute_marginal_inconsistent_state_length_raises():
    with pytest.raises(ValueError):
        abstract.compute_marginal([[0, 1], [1]], [1.0, 2.0], {0: 0})


def test_compute_marginal_non_integer_state_value_raises():
    with pytest.raises(TypeError):
        abstract.compute_marginal([[0, "1"], [1, 0]], [1.0, 2.0], {0: 0})


def test_compute_marginal_non_numeric_value_raises():
    with pytest.raises(TypeError):
        abstract.compute_marginal([[0, 1], [1, 0]], [1.0, "2.0"], {0: 0})


def test_compute_marginal_invalid_constraints_type_raises():
    with pytest.raises(TypeError):
        abstract.compute_marginal([[0, 1], [1, 0]], [1.0, 2.0], "invalid")


def test_compute_marginal_constraint_index_out_of_range_raises():
    with pytest.raises(ValueError):
        abstract.compute_marginal([[0, 1], [1, 0]], [1.0, 2.0], {2: 1})


def test_compute_marginal_full_assignment_wrong_length_raises():
    with pytest.raises(ValueError):
        abstract.compute_marginal([[0, 1], [1, 0]], [1.0, 2.0], [1, 0, 1])


def test_compute_marginal_full_assignment_bad_value_raises():
    with pytest.raises(ValueError):
        abstract.compute_marginal([[0, 1], [1, 0]], [1.0, 2.0], [2, 0])


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"dit_string_length": 4.5, "interaction_size": 2, "dit_dimension": 2}, TypeError),
        ({"dit_string_length": 4, "interaction_size": 0, "dit_dimension": 2}, ValueError),
        ({"dit_string_length": 4, "interaction_size": 5, "dit_dimension": 2}, ValueError),
        ({"dit_string_length": 4, "interaction_size": 2, "dit_dimension": 1}, ValueError),
    ],
)
def test_constraints_for_nearest_neighbors_interactions_raises(kwargs, exc):
    with pytest.raises(exc):
        abstract.constraints_for_nearest_neighbors_interactions(**kwargs)


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"dit_string_length": "4", "interaction_size": 2, "dit_dimension": 2}, TypeError),
        ({"dit_string_length": 4, "interaction_size": -1, "dit_dimension": 2}, ValueError),
        ({"dit_string_length": 4, "interaction_size": 6, "dit_dimension": 2}, ValueError),
        ({"dit_string_length": 4, "interaction_size": 2, "dit_dimension": True}, TypeError),
    ],
)
def test_constraints_for_all_interactions_raises(kwargs, exc):
    with pytest.raises(exc):
        abstract.constraints_for_all_interactions(**kwargs)


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"index": -1, "dit_constraints": [], "dit_string_length": 3, "dit_dimension": 2}, ValueError),
        ({"index": 0.5, "dit_constraints": [], "dit_string_length": 3, "dit_dimension": 2}, TypeError),
        ({"index": 0, "dit_constraints": {}, "dit_string_length": 3, "dit_dimension": 2}, TypeError),
        ({"index": 8, "dit_constraints": [], "dit_string_length": 3, "dit_dimension": 2}, ValueError),
        ({"index": 0, "dit_constraints": [{"0": 1}], "dit_string_length": 3, "dit_dimension": 2}, TypeError),
        ({"index": 0, "dit_constraints": [{3: 1}], "dit_string_length": 3, "dit_dimension": 2}, ValueError),
    ],
)
def test_reconstruct_structured_matrix_column_raises(kwargs, exc):
    with pytest.raises(exc):
        abstract.reconstruct_structured_matrix_column(**kwargs)


def test_reconstruct_structured_matrix_column_happy_path():
    out = abstract.reconstruct_structured_matrix_column(
        index=5,
        dit_constraints=[{0: 1}, {1: 1}, {2: 1}, {0: 1, 2: 1}],
        dit_string_length=3,
        dit_dimension=2,
    )
    assert out == [1, 0, 1, 1]


def test_compute_marginal_happy_path_list_of_constraints():
    out = abstract.compute_marginal(
        function_input_dits=[[0, 0], [0, 1], [1, 0], [1, 1]],
        function_values=[1.0, 2.0, 3.0, 4.0],
        dit_constraints=[{0: 0}, {1: 1}],
    )
    assert out == [3.0, 6.0]


def test_compute_marginal_numpy_constraints_happy_path():
    out = abstract.compute_marginal(
        function_input_dits=[[0, 0], [0, 1], [1, 0], [1, 1]],
        function_values=[1.0, 2.0, 3.0, 4.0],
        dit_constraints=np.array([[0, 1], [1, 0]]),
    )
    assert out == [2.0, 3.0]
