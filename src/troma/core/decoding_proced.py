from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ..problem_sketch import ProblemSketch
from ..sketch_map import ConstraintSketchMap
from ..sketch_map import ExplicitSketchMap
from ..optimization import optimizer as optimizer_api
from .._validation import _Validator


def _column_vector_to_array(vec: Any) -> np.ndarray:
    return np.asarray(vec).reshape(-1)


def matchingpursuit_explicit(
    problem_sketch: ProblemSketch,
    iteration_number: int,
    step: float | None = None,
    optimizer: Any | None = None,
) -> np.ndarray:
    """
    Perform matching pursuit to find a sparse solution to the linear system defined by the sketch matrix and the marginals.

    Parameters
    ----------
    problem_sketch : ProblemSketch
        Problem sketch containing sketch values and an ExplicitSketchMap.
    iteration_number : int
        The number of iterations to perform.
    step : float, optional
        The step size. If None, an adaptive step size is used.
    optimizer : Optimizer, optional
        Instantiated optimizer. If None, a brute-force optimizer is used.

    Returns
    -------
    np.ndarray
        2D array where each row is [column_index, coefficient].
    """
    _Validator.ensure_instance("problem_sketch", problem_sketch, ProblemSketch)
    sketch_map = problem_sketch.sketch_map
    _Validator.ensure_instance("sketch_map", sketch_map, ExplicitSketchMap)
    marginals = problem_sketch.sketch_values
    _Validator.ensure_not_none("marginals", marginals)
    sketch = sketch_map.map
    if not hasattr(sketch, "__getitem__"):
        raise TypeError("problem_sketch.sketch_map.map must be an indexable matrix-like object.")
    iteration_number = _Validator.ensure_int("iteration_number", iteration_number, min_value=1)
    _Validator.ensure_optional_real("step", step)

    if optimizer is None:
        optimizer = optimizer_api.get_optimizer("brute_force_max")
    elif not hasattr(optimizer, "optimize"):
        raise TypeError("optimizer must implement an optimize(*args, **kwargs) method.")

    r = copy.deepcopy(marginals)
    selections = []

    for _ in range(iteration_number):
        residue_sketch = problem_sketch.update_sketch(r)
        t = optimizer.optimize(residue_sketch)
        At = _column_vector_to_array(sketch[:, t])

        if step is None:
            norm_sq = np.dot(At, At)
            alpha = np.dot(r, At) / norm_sq if norm_sq != 0 else 0.0
        else:
            alpha = step

        r -= alpha * At
        selections.append((t, alpha))

    return np.array([[idx, coeff] for idx, coeff in selections])


def matchingpursuit_abstract(
    problem_sketch: ProblemSketch,
    iteration_number: int,
    step: float | None = None,
    optimizer: Any | None = None,
) -> np.ndarray:
    """
    Perform matching pursuit using an abstract (implicit) sketch representation.

    Parameters
    ----------
    problem_sketch : ProblemSketch
        Problem sketch containing sketch values and a ConstraintSketchMap.
    iteration_number : int
        The number of iterations.
    step : float, optional
        The step size. If None, adaptive.
    optimizer : Optimizer, optional
        Instantiated optimizer. If None, a spin-chain NN optimizer is used.

    Returns
    -------
    np.ndarray
        2D array where each row is [column_index, coefficient].
    """
    _Validator.ensure_instance("problem_sketch", problem_sketch, ProblemSketch)
    sketch_map = problem_sketch.sketch_map
    _Validator.ensure_instance("sketch_map", sketch_map, ConstraintSketchMap)
    marginals = problem_sketch.sketch_values
    _Validator.ensure_not_none("marginals", marginals)
    _Validator.ensure_sequence("dit_constraints", sketch_map.map)

    iteration_number = _Validator.ensure_int("iteration_number", iteration_number, min_value=1)
    _Validator.ensure_optional_real("step", step)

    if optimizer is None:
        optimizer = optimizer_api.get_optimizer("spin_chain_nn_max")
    elif not hasattr(optimizer, "optimize"):
        raise TypeError("optimizer must implement an optimize(*args, **kwargs) method.")

    r = copy.deepcopy(marginals)
    selections = []

    for _ in range(iteration_number):
        residue_sketch = problem_sketch.update_sketch(r)
        t = optimizer.optimize(residue_sketch)

        At = sketch_map.reconstruct_structured_matrix_column(t)

        if step is None:
            norm_sq = np.dot(At, At)
            alpha = np.dot(r, At) / norm_sq if norm_sq != 0 else 0.0
        else:
            alpha = step

        r -= alpha * At
        selections.append((t, alpha))

    return np.array([[idx, coeff] for idx, coeff in selections])
