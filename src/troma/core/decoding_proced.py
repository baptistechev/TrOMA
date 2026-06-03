from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np

from ..problem_sketch import ProblemSketch
from ..sketch_map import ConstraintSketchMap
from ..sketch_map import ExplicitSketchMap
from ..optimization import optimizer as optimizer_api
from .._validation import _Validator
from .structure import DitString
from .post_processing import greedy_2_bit_swap


def _extract_optimizer_metadata(result: Any, selected_index: int) -> dict[str, Any] | None:
    if not hasattr(result, "final_parameters"):
        return None

    return {
        "selected_index": int(selected_index),
        "final_parameters": np.asarray(result.final_parameters, dtype=float).copy(),
        "gammas": tuple(float(gamma) for gamma in getattr(result, "gammas", ())),
        "betas": tuple(float(beta) for beta in getattr(result, "betas", ())),
        "number_layers": int(getattr(result, "number_layers", 0)),
        "circuit_depth": int(getattr(result, "circuit_depth", 0)),
        "solver_steps": int(getattr(result, "solver_steps", 0)),
        "objective_evaluations": int(getattr(result, "objective_evaluations", 0)),
    }


def _column_vector_to_array(vec: Any) -> np.ndarray:
    return np.asarray(vec).reshape(-1)


def matchingpursuit_explicit(
    problem_sketch: ProblemSketch,
    iteration_number: int,
    step: float | None = None,
    optimizer: Any | None = None,
    post_processing: str | None = None,
    return_optimizer_metadata: bool = False,
    verbose: bool = False,
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
    post_processing : str or None, optional
        Name of a post-processing function to apply to the optimizer solution
        at each iteration.  Supported values: ``"2_bit_swap"``.
    return_optimizer_metadata : bool, optional
        If True, collect per-iteration optimizer metadata when available.
    verbose : bool, optional
        If True, print per-iteration information about the post-processing
        outcome. Default is False.

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

    dit_string_length = int(getattr(problem_sketch, "restricted_problem_size", problem_sketch.problem_size))
    dit_dimension = int(getattr(problem_sketch, "restricted_problem_dimension", problem_sketch.problem_dimension))

    r = copy.deepcopy(marginals)
    selections = []
    optimizer_metadata = [] if return_optimizer_metadata else None

    for _ in range(iteration_number):

        residue_sketch = problem_sketch.update_sketch(r)
        if not any(residue_sketch.sketch_values):
            warnings.warn("Early stop: residue reached zero.", stacklevel=2)
            break
        residue_sketch = problem_sketch.update_sketch(r)
        optimizer_result = optimizer.optimize(
            residue_sketch,
            verbose=verbose,
            return_metadata=return_optimizer_metadata,
        )
        t = int(optimizer_result)
        if post_processing is None:
            pass
        elif post_processing == "2_bit_swap":
            candidate = DitString.from_integer(t, dit_string_length, dit_dimension)
            t = greedy_2_bit_swap(candidate, problem_sketch, verbose=verbose).to_integer()
        else:
            raise ValueError(
                f"Unknown post_processing '{post_processing}'. Supported values: '2_bit_swap'."
            )
        At = _column_vector_to_array(sketch[:, t])

        if step is None:
            norm_sq = np.dot(At, At)
            alpha = np.dot(r, At) / norm_sq if norm_sq != 0 else 0.0
        else:
            alpha = step

        r -= alpha * At
        selections.append((t, alpha))
        if optimizer_metadata is not None:
            optimizer_metadata.append(_extract_optimizer_metadata(optimizer_result, t))

    solution = np.array([[idx, coeff] for idx, coeff in selections])
    if optimizer_metadata is None:
        return solution
    return {"raw": solution, "optimizer_metadata": optimizer_metadata}


def matchingpursuit_abstract(
    problem_sketch: ProblemSketch,
    iteration_number: int,
    step: float | None = None,
    optimizer: Any | None = None,
    post_processing: str | None = None,
    return_optimizer_metadata: bool = False,
    verbose: bool = False,
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
    post_processing : str or None, optional
        Name of a post-processing function to apply to the optimizer solution
        at each iteration.  Supported values: ``"2_bit_swap"``.
    return_optimizer_metadata : bool, optional
        If True, collect per-iteration optimizer metadata when available.
    verbose : bool, optional
        If True, print per-iteration information about the post-processing
        outcome. Default is False.

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

    dit_string_length = int(getattr(problem_sketch, "restricted_problem_size", problem_sketch.problem_size))
    dit_dimension = int(getattr(problem_sketch, "restricted_problem_dimension", problem_sketch.problem_dimension))

    r = copy.deepcopy(marginals)
    selections = []
    optimizer_metadata = [] if return_optimizer_metadata else None

    for _ in range(iteration_number):
        residue_sketch = problem_sketch.update_sketch(r)
        if not any(residue_sketch.sketch_values):
            warnings.warn("Early stop: residue reached zero.", stacklevel=2)
            break
        optimizer_result = optimizer.optimize(
            residue_sketch,
            verbose=verbose,
            return_metadata=return_optimizer_metadata,
        )
        t = int(optimizer_result)
        if post_processing is None:
            pass
        elif post_processing == "2_bit_swap":
            candidate = DitString.from_integer(t, dit_string_length, dit_dimension)
            t = greedy_2_bit_swap(candidate, problem_sketch, verbose=verbose).to_integer()
        else:
            raise ValueError(
                f"Unknown post_processing '{post_processing}'. Supported values: '2_bit_swap'."
            )

        At = sketch_map.reconstruct_structured_matrix_column(t)

        if step is None:
            norm_sq = np.dot(At, At)
            alpha = np.dot(r, At) / norm_sq if norm_sq != 0 else 0.0
        else:
            alpha = step

        r -= alpha * At
        selections.append((t, alpha))
        if optimizer_metadata is not None:
            optimizer_metadata.append(_extract_optimizer_metadata(optimizer_result, t))

    solution = np.array([[idx, coeff] for idx, coeff in selections])
    if optimizer_metadata is None:
        return solution
    return {"raw": solution, "optimizer_metadata": optimizer_metadata}
