from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ..core.data_structure import integer_to_dit_string
from .._validation import ensure_int, ensure_sequence, ensure_dict


def _validate_constraint_dict(constraint: dict, dit_string_length: int, dit_dimension: int) -> None:
    ensure_dict("constraint", constraint)
    for dit_idx, dit_val in constraint.items():
        ensure_int("constraint index", dit_idx)
        if int(dit_idx) < 0 or int(dit_idx) >= dit_string_length:
            raise ValueError("Constraint index out of range for dit string length.")
        ensure_int("constraint value", dit_val)
        if int(dit_val) < 0 or int(dit_val) >= dit_dimension:
            raise ValueError("Constraint values must be in [0, dit_dimension - 1].")


def _validate_full_assignment(
    constraint: list | tuple | np.ndarray,
    dit_string_length: int,
    dit_dimension: int,
) -> None:
    ensure_sequence("full assignment", constraint)
    if len(constraint) != dit_string_length:
        raise ValueError("A full dit assignment must have length dit_string_length.")
    for dit_val in constraint:
        ensure_int("full-assignment value", dit_val)
        if int(dit_val) < 0 or int(dit_val) >= dit_dimension:
            raise ValueError("Full-assignment values must be in [0, dit_dimension - 1].")


def compute_marginal(
    function_input_dits: list | tuple | np.ndarray,
    function_values: list | tuple | np.ndarray,
    dit_constraints: dict | list | tuple | np.ndarray,
) -> float | list[float]:
    """
    Compute marginals of a sparse spectrum on one or several dit constraints.

    Supported constraint formats
    ----------------------------
    - single dict: ``{dit_index: dit_value, ...}`` -> returns ``float``
    - single full dit assignment: ``[0, 1, 2, ...]`` -> returns ``float``
    - list of constraints (dicts or full dit assignments) -> returns ``list[float]``

    Parameters
    ----------
    function_input_dits : list or ndarray
        List of function inputs, each given in dit representation.
    function_values : list or ndarray
        Function values corresponding to each input.
    dit_constraints : dict | sequence | list[dict | sequence]
        One constraint or several constraints.

    Returns
    -------
    float or list of float
        Sum(s) of values for states matching each provided constraint.
    """
    from numbers import Integral, Real

    if not isinstance(function_input_dits, (list, tuple, np.ndarray)):
        raise TypeError("function_input_dits must be a sequence of dit sequences.")
    if not isinstance(function_values, (list, tuple, np.ndarray)):
        raise TypeError("function_values must be a sequence of numeric values.")
    if len(function_input_dits) != len(function_values):
        raise ValueError("function_input_dits and function_values must have the same length.")
    if len(function_input_dits) == 0:
        if isinstance(dit_constraints, (list, tuple, np.ndarray)) and len(dit_constraints) > 0 and not all(np.isscalar(x) for x in dit_constraints):
            return [0.0 for _ in dit_constraints]
        return 0.0

    dit_string_length = len(function_input_dits[0])
    if dit_string_length == 0:
        raise ValueError("Each element of function_input_dits must be a non-empty dit sequence.")

    for state in function_input_dits:
        if not isinstance(state, (list, tuple, np.ndarray)):
            raise TypeError("Each state in function_input_dits must be a sequence.")
        if len(state) != dit_string_length:
            raise ValueError("All states in function_input_dits must have the same length.")
        for dit_val in state:
            ensure_int("state value", dit_val)

    for value in function_values:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError("Values in function_values must be numeric.")

    max_dit_value = max(int(v) for state in function_input_dits for v in state)
    dit_dimension = max_dit_value + 1

    def _single_marginal(constraint: Any) -> float:
        if isinstance(constraint, dict):
            _validate_constraint_dict(constraint, dit_string_length, dit_dimension)
            items = list(constraint.items())
            filtered_vals = [
                v for s, v in zip(function_input_dits, function_values)
                if all(int(s[dit_idx]) == int(dit_val) for dit_idx, dit_val in items)
            ]
            return float(np.sum(filtered_vals) if filtered_vals else 0.0)

        _validate_full_assignment(constraint, dit_string_length, dit_dimension)
        dit_values = list(constraint)
        filtered_vals = [
            v for s, v in zip(function_input_dits, function_values)
            if len(s) == len(dit_values) and all(int(s[idx]) == int(dit_val) for idx, dit_val in enumerate(dit_values))
        ]
        return float(np.sum(filtered_vals) if filtered_vals else 0.0)

    if isinstance(dit_constraints, dict):
        return _single_marginal(dit_constraints)

    if isinstance(dit_constraints, np.ndarray):
        if dit_constraints.ndim == 1:
            return _single_marginal(dit_constraints.tolist())
        return [_single_marginal(c) for c in dit_constraints.tolist()]

    if isinstance(dit_constraints, (list, tuple)):
        if not dit_constraints:
            return []
        if all(np.isscalar(x) for x in dit_constraints):
            return _single_marginal(dit_constraints)
        return [_single_marginal(c) for c in dit_constraints]

    raise TypeError(
        "dit_constraints must be a dict, a full dit assignment, or a list of constraints."
    )


def constraints_for_nearest_neighbors_interactions(
    dit_string_length: int,
    interaction_size: int,
    dit_dimension: int = 2,
) -> list[dict[int, int]]:
    """
    Compute constraints for nearest-neighbor interactions.

    Parameters
    ----------
    dit_string_length : int
        Length of the dit string.
    interaction_size : int
        Size of the interaction window.
    dit_dimension : int, optional
        Number of possible dit values. Default is 2.

    Returns
    -------
    list of dict
        List of dit constraints as {position: value} dicts.
    """
    dit_string_length = ensure_int("dit_string_length", dit_string_length, min_value=1)
    interaction_size = ensure_int("interaction_size", interaction_size, min_value=1)
    dit_dimension = ensure_int("dit_dimension", dit_dimension, min_value=1)
    if interaction_size > dit_string_length:
        raise ValueError("interaction_size must be <= dit_string_length.")

    all_dits_constraints: list[dict[int, int]] = []
    constrained_dits_indices = [
        tuple(range(i, i + interaction_size))
        for i in range(dit_string_length - interaction_size + 1)
    ]
    for constraint_dits in constrained_dits_indices:
        for constraint_values in itertools.product(range(dit_dimension), repeat=interaction_size):
            all_dits_constraints.append(dict(zip(constraint_dits, constraint_values)))
    return all_dits_constraints


def constraints_for_all_interactions(
    dit_string_length: int,
    interaction_size: int,
    dit_dimension: int = 2,
) -> list[dict[int, int]]:
    """
    Compute constraints for all (non-consecutive) interactions.

    Parameters
    ----------
    dit_string_length : int
        Length of the dit string.
    interaction_size : int
        Size of the interaction.
    dit_dimension : int, optional
        Number of possible dit values. Default is 2.

    Returns
    -------
    list of dict
        List of dit constraints as {position: value} dicts.
    """
    dit_string_length = ensure_int("dit_string_length", dit_string_length, min_value=1)
    interaction_size = ensure_int("interaction_size", interaction_size, min_value=1)
    dit_dimension = ensure_int("dit_dimension", dit_dimension, min_value=1)
    if interaction_size > dit_string_length:
        raise ValueError("interaction_size must be <= dit_string_length.")

    all_dits_constraints: list[dict[int, int]] = []
    constrained_dits_indices = itertools.combinations(range(dit_string_length), interaction_size)
    for constraint_dits in constrained_dits_indices:
        for constraint_values in itertools.product(range(dit_dimension), repeat=interaction_size):
            all_dits_constraints.append(dict(zip(constraint_dits, constraint_values)))
    return all_dits_constraints


def reconstruct_structured_matrix_column(
    index: int,
    dit_constraints: list[dict],
    dit_string_length: int,
    dit_dimension: int = 2,
) -> np.ndarray:
    """
    Reconstruct a column of the structured cylinder-set indicator matrix.

    Parameters
    ----------
    index : int
        Index of the dit string.
    dit_constraints : list of dict
        Dit constraints as {position: value} dicts.
    dit_string_length : int
        Length of the dit string.
    dit_dimension : int, optional
        Number of possible dit values. Default is 2.

    Returns
    -------
    np.ndarray
        Boolean column vector indicating which constraints the dit string satisfies.
    """
    index = ensure_int("index", index, min_value=0)
    dit_string_length = ensure_int("dit_string_length", dit_string_length, min_value=1)
    dit_dimension = ensure_int("dit_dimension", dit_dimension, min_value=1)
    ensure_sequence("dit_constraints", dit_constraints)

    max_index = dit_dimension ** dit_string_length
    if index >= max_index:
        raise ValueError("index out of range for provided dit_string_length and dit_dimension.")

    for constraint in dit_constraints:
        _validate_constraint_dict(constraint, dit_string_length, dit_dimension)

    dit_str = integer_to_dit_string(index, dit_string_length, dit_dimension=dit_dimension)
    column: list[int] = []
    for constraint in dit_constraints:
        for dit_idx, dit_val in enumerate(dit_str):
            if dit_idx in constraint:
                if int(dit_val) != constraint[dit_idx]:
                    column.append(0)
                    break
        else:
            column.append(1)
    return np.array(column)
