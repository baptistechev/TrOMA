import itertools
from numbers import Integral, Real

import numpy as np
# from .data_structure import integer_to_dit_string
from data_structure import integer_to_dit_string


def _validate_positive_int(name, value, *, min_value=1):
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")


def _validate_constraint_dict(constraint, dit_string_length, dit_dimension):
    if not isinstance(constraint, dict):
        raise TypeError("Each constraint must be a dict or a full dit assignment sequence.")
    for dit_idx, dit_val in constraint.items():
        if not isinstance(dit_idx, Integral) or isinstance(dit_idx, bool):
            raise TypeError("Constraint indices must be integers.")
        if dit_idx < 0 or dit_idx >= dit_string_length:
            raise ValueError("Constraint index out of range for dit string length.")
        if not isinstance(dit_val, Integral) or isinstance(dit_val, bool):
            raise TypeError("Constraint values must be integers.")
        if dit_val < 0 or dit_val >= dit_dimension:
            raise ValueError("Constraint values must be in [0, dit_dimension - 1].")


def _validate_full_assignment(constraint, dit_string_length, dit_dimension):
    if not isinstance(constraint, (list, tuple, np.ndarray)):
        raise TypeError("A full dit assignment must be a sequence of integers.")
    if len(constraint) != dit_string_length:
        raise ValueError("A full dit assignment must have length dit_string_length.")
    for dit_val in constraint:
        if not isinstance(dit_val, Integral) or isinstance(dit_val, bool):
            raise TypeError("Full-assignment values must be integers.")
        if dit_val < 0 or dit_val >= dit_dimension:
            raise ValueError("Full-assignment values must be in [0, dit_dimension - 1].")

def compute_marginal(function_input_dits, function_values, dit_constraints):
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
        List of function inputs, each input is given in dits representation.
    function_values : list or ndarray
        Image of the function associated with each input.
    dit_constraints : dict | sequence | list[dict | sequence]
        One constraint or several constraints.

    Returns
    -------
    float or list of float
        Sum(s) of values for states matching each provided constraint.
    """

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
            if not isinstance(dit_val, Integral) or isinstance(dit_val, bool):
                raise TypeError("State values in function_input_dits must be integers.")

    for value in function_values:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError("Values in function_values must be numeric.")

    max_dit_value = max(int(v) for state in function_input_dits for v in state)
    dit_dimension = max_dit_value + 1

    def _single_marginal(constraint):
        # Sparse constraint: only constrained indices are given.
        if isinstance(constraint, dict):
            _validate_constraint_dict(constraint, dit_string_length, dit_dimension)
            items = list(constraint.items())
            filtered_vals = [
                v for s, v in zip(function_input_dits, function_values)
                if all(int(s[dit_idx]) == int(dit_val) for dit_idx, dit_val in items)
            ]
            return float(np.sum(filtered_vals) if filtered_vals else 0.0)

        # Full dit assignment: all indices are constrained.
        _validate_full_assignment(constraint, dit_string_length, dit_dimension)
        dit_values = list(constraint)
        filtered_vals = [
            v for s, v in zip(function_input_dits, function_values)
            if len(s) == len(dit_values) and all(int(s[idx]) == int(dit_val) for idx, dit_val in enumerate(dit_values))
        ]
        return float(np.sum(filtered_vals) if filtered_vals else 0.0)

    # Single sparse constraint.
    if isinstance(dit_constraints, dict):
        return _single_marginal(dit_constraints)

    # Handle ndarray explicitly (to avoid ambiguous iteration behavior).
    if isinstance(dit_constraints, np.ndarray):
        if dit_constraints.ndim == 1:
            return _single_marginal(dit_constraints.tolist())
        return [_single_marginal(c) for c in dit_constraints.tolist()]

    # Handle list/tuple input.
    if isinstance(dit_constraints, (list, tuple)):
        if not dit_constraints:
            return []

        # If it looks like one full dit assignment (list/tuple of scalars), return one float.
        if all(np.isscalar(x) for x in dit_constraints):
            return _single_marginal(dit_constraints)

        # Otherwise, interpret as multiple constraints.
        return [_single_marginal(c) for c in dit_constraints]

    raise TypeError(
        "dit_constraints must be a dict, a full dit assignment, or a list of constraints."
    )

def constraints_for_nearest_neighbors_interactions(dit_string_length, interaction_size, dit_dimension=2):
    """ 
    Compute constraints for nearest neighbor interactions and bit values, for a given interaction size.

    Parameters
    ----------
    dit_string_length : int
        The length of the dit string (number of dits in the string).
    interaction_size : int
        The size of the interaction (number of nearest neighbors to consider).
    dit_dimension : int, optional
        The dimension of the dits (number of possible values for each dit), by default 2.
        
    Returns
    -------
    list
        List of all dit constraints corresponding to each nearest neighbor interaction and bit value combination.
    """

    _validate_positive_int("dit_string_length", dit_string_length)
    _validate_positive_int("interaction_size", interaction_size)
    _validate_positive_int("dit_dimension", dit_dimension)
    if interaction_size > dit_string_length:
        raise ValueError("interaction_size must be <= dit_string_length.")

    all_dits_constraints = []
    
    # Generate all combinations of nearest neighbors and their values
    constrained_dits_indices = [tuple(range(dits_indices, dits_indices + interaction_size)) for dits_indices in range(dit_string_length - interaction_size + 1)]
    for constraint_dits in constrained_dits_indices:
        for constraint_values in itertools.product(range(dit_dimension), repeat=interaction_size):
            all_dits_constraints += [dict(zip(constraint_dits, constraint_values))]
    return all_dits_constraints


def constraints_for_all_interactions(dit_string_length, interaction_size, dit_dimension=2):
    """ 
    Compute constraints for all interactions and bit values, for a given interaction size.

    Parameters
    ----------
    dit_string_length : int
        The length of the dit string (number of dits in the string).
    interaction_size : int
        The size of the interaction (number of neighbors to consider).
    dit_dimension : int, optional
        The dimension of the dits (number of possible values for each dit), by default 2.
        
    Returns
    -------
    list
        List of all dit constraints corresponding to each interaction and bit value combination.
    """

    _validate_positive_int("dit_string_length", dit_string_length)
    _validate_positive_int("interaction_size", interaction_size)
    _validate_positive_int("dit_dimension", dit_dimension)
    if interaction_size > dit_string_length:
        raise ValueError("interaction_size must be <= dit_string_length.")

    all_dits_constraints = []
    
    # Generate all combinations of neighbors and their values
    constrained_dits_indices = itertools.combinations(range(dit_string_length), interaction_size)
    for constraint_dits in constrained_dits_indices:
        for constraint_values in itertools.product(range(dit_dimension), repeat=interaction_size):
            all_dits_constraints += [dict(zip(constraint_dits, constraint_values))]
    return all_dits_constraints

# def reconstruct_structured_matrix_column(index, cylinder_set_list,dit_dimension=2):
#     """
#     Check to which cylinder sets an element belongs.
#     Reconstruct a column of a structured matrix made of cylinder set indicators, given the element index.
    
#     Parameters
#     ----------
#     index : int
#         The index of the element to check.
#     cylinder_set_list : list
#         List of cylinder set full indicators to check.
        
#     Returns
#     -------
#     list of bool
#         The column of the structured matrix corresponding to this index, i.e. a list indicating whether the element belongs to each cylinder set or not.
#     """
#     length = len(cylinder_set_list[0])  # Assuming all cylinder sets have the same length
#     dit_str = integer_to_dit_string(index, length, dit_dimension=dit_dimension)
#     element = dit_string_to_computational_basis(dit_str, dit_dimension=dit_dimension)
    
#     return [belongs_to_cylinder_set(element, cyl_set,dit_dimension=dit_dimension) for cyl_set in cylinder_set_list]

def reconstruct_structured_matrix_column(index, dit_constraints, dit_string_length,dit_dimension=2):
    """
    Reconstruct a column of a structured matrix made of cylinder set indicators defined by the dit constraints, and given the element index.
    
    Parameters
    ----------
    index : int
        The index of the element to check.
    dit_constraints : list of dict
        List of dit constraints given as dictionaries to check.
        
    Returns
    -------
    list of bool
        The column of the structured matrix corresponding to this index, i.e. a list indicating whether the element satisfies each dit constraint or not.
    """

    _validate_positive_int("index", index, min_value=0)
    _validate_positive_int("dit_string_length", dit_string_length)
    _validate_positive_int("dit_dimension", dit_dimension)
    if not isinstance(dit_constraints, (list, tuple)):
        raise TypeError("dit_constraints must be a list or tuple of dictionaries.")

    max_index = dit_dimension ** dit_string_length
    if index >= max_index:
        raise ValueError("index out of range for provided dit_string_length and dit_dimension.")

    for constraint in dit_constraints:
        _validate_constraint_dict(constraint, dit_string_length, dit_dimension)

    dit_str = integer_to_dit_string(index, dit_string_length, dit_dimension=dit_dimension)
    column = []
    for constraint in dit_constraints:
        for dit_idx, dit_val in enumerate(dit_str):
            if dit_idx in constraint:
                if int(dit_val) != constraint[dit_idx]:
                    column.append(0)
                    break
        else:
            column.append(1)
    return column