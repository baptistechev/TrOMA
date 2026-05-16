from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ..sketchs import abstract as ab
from ..optimization import optimizer as optimizer_api
from .._validation import ensure_int, ensure_sequence


def _column_vector_to_array(vec: Any) -> np.ndarray:
    return np.asarray(vec).reshape(-1)


def matchingpursuit_explicit(
    marginals: list[float] | np.ndarray,
    sketch: np.ndarray,
    iteration_number: int,
    step: float | None = None,
    optimizer: Any | None = None,
) -> np.ndarray:
    """
    Perform matching pursuit to find a sparse solution to the linear system defined by the sketch matrix and the marginals.

    Parameters
    ----------
    marginals : list of float
        The marginals of the function defined on the full spectrum of dit strings.
    sketch : 2D numpy array
        The sketch matrix.
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
    if not hasattr(sketch, "__getitem__"):
        raise TypeError("sketch must be an indexable matrix-like object.")
    iteration_number = ensure_int("iteration_number", iteration_number, min_value=1)
    if step is not None and not isinstance(step, (int, float)):
        raise TypeError("step must be a real number or None.")

    if optimizer is None:
        optimizer = optimizer_api.get_optimizer("brute_force_max")
    elif not hasattr(optimizer, "optimize"):
        raise TypeError("optimizer must implement an optimize(*args, **kwargs) method.")

    r = copy.deepcopy(marginals)
    selections = []

    for _ in range(iteration_number):
        t = optimizer.optimize(r, sketch=sketch)
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
    marginals: list[float] | np.ndarray,
    dit_constraints: list[dict],
    dit_string_length: int,
    iteration_number: int,
    step: float | None = None,
    interaction_size: int = 2,
    dit_dimension: int = 2,
    optimizer: Any | None = None,
) -> np.ndarray:
    """
    Perform matching pursuit using an abstract (implicit) sketch representation.

    Parameters
    ----------
    marginals : list of float
        The marginals of the function.
    dit_constraints : list of dict
        The dit constraints.
    dit_string_length : int
        The length of the dit strings.
    iteration_number : int
        The number of iterations.
    step : float, optional
        The step size. If None, adaptive.
    interaction_size : int, optional
        The interaction size. Default is 2.
    dit_dimension : int, optional
        The dit dimension. Default is 2.
    optimizer : Optimizer, optional
        Instantiated optimizer. If None, a spin-chain NN optimizer is used.

    Returns
    -------
    np.ndarray
        2D array where each row is [column_index, coefficient].
    """
    ensure_sequence("dit_constraints", dit_constraints)
    dit_string_length = ensure_int("dit_string_length", dit_string_length, min_value=1)
    iteration_number = ensure_int("iteration_number", iteration_number, min_value=1)
    interaction_size = ensure_int("interaction_size", interaction_size, min_value=1)
    dit_dimension = ensure_int("dit_dimension", dit_dimension, min_value=2)
    if interaction_size > dit_string_length:
        raise ValueError("interaction_size must be <= dit_string_length.")
    if step is not None and not isinstance(step, (int, float)):
        raise TypeError("step must be a real number or None.")

    if optimizer is None:
        optimizer = optimizer_api.get_optimizer("spin_chain_nn_max")
    elif not hasattr(optimizer, "optimize"):
        raise TypeError("optimizer must implement an optimize(*args, **kwargs) method.")

    r = copy.deepcopy(marginals)
    selections = []

    for _ in range(iteration_number):
        t = optimizer.optimize(
            r,
            dit_constraints=dit_constraints,
            dit_string_length=dit_string_length,
            interaction_size=interaction_size,
            dit_dimension=dit_dimension,
            bit_constraints=dit_constraints,
            bit_string_length=dit_string_length,
        )

        At = ab.reconstruct_structured_matrix_column(t, dit_constraints, dit_string_length, dit_dimension)

        if step is None:
            norm_sq = np.dot(At, At)
            alpha = np.dot(r, At) / norm_sq if norm_sq != 0 else 0.0
        else:
            alpha = step

        r -= alpha * At
        selections.append((t, alpha))

    return np.array([[idx, coeff] for idx, coeff in selections])
