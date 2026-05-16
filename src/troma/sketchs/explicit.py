from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ..core import data_structure as ds
from .._validation import ensure_int


def compute_marginal(
    full_sprectrum_values: list[float] | np.ndarray,
    sketch_function: np.ndarray,
) -> list[float]:
    """
    Compute marginals by multiplying the explicit sketch matrix by the full-spectrum value vector.

    Parameters
    ----------
    full_sprectrum_values : list of float or np.ndarray
        Values of the function over the full spectrum, in lexicographic order.
    sketch_function : np.ndarray
        Sketch matrix (rows = constraints, columns = dit strings).

    Returns
    -------
    list of float
        Marginal values, one per sketch row.
    """
    if not isinstance(full_sprectrum_values, (list, tuple, np.ndarray)):
        raise TypeError("full_sprectrum_values must be a sequence of numeric values.")
    if not hasattr(sketch_function, "__matmul__"):
        raise TypeError("sketch_function must support matrix multiplication (@).")
    values = np.asarray(full_sprectrum_values)
    if hasattr(sketch_function, "shape") and len(sketch_function.shape) >= 2:
        if sketch_function.shape[1] != values.shape[0]:
            raise ValueError("full_sprectrum_values length must match sketch_function column count.")
    y = sketch_function @ values
    return np.asarray(y).flatten().tolist()


def nearest_neighbors_interactions_sketch(
    dit_string_length: int,
    interaction_size: int,
    dit_dimension: int = 2,
) -> np.ndarray:
    """
    Build the explicit sketch matrix for nearest-neighbor interactions.

    Parameters
    ----------
    dit_string_length : int
        Length of the dit strings.
    interaction_size : int
        Size of the interaction window.
    dit_dimension : int, optional
        Number of possible dit values. Default is 2.

    Returns
    -------
    np.ndarray
        Sketch matrix of shape (n_constraints, dit_dimension**dit_string_length).
    """
    dit_string_length = ensure_int("dit_string_length", dit_string_length, min_value=1)
    interaction_size = ensure_int("interaction_size", interaction_size, min_value=1)
    dit_dimension = ensure_int("dit_dimension", dit_dimension, min_value=2)
    if interaction_size > dit_string_length:
        raise ValueError("interaction_size cannot be greater than dit_string_length.")

    cylinder_sets = []
    constrained_dits_indices = [
        tuple(range(i, i + interaction_size))
        for i in range(dit_string_length - interaction_size + 1)
    ]
    for constraint_dits in constrained_dits_indices:
        cylinder_sets += ds.create_cylinder_set_indicator(constraint_dits, dit_string_length, dit_dimension)

    A = [ds.kronecker_develop(s) for s in cylinder_sets]
    return np.array(A)


def all_interactions_sketch(
    dit_string_length: int,
    interaction_size: int,
    dit_dimension: int = 2,
) -> np.ndarray:
    """
    Build the explicit sketch matrix for all (non-consecutive) interactions.

    Parameters
    ----------
    dit_string_length : int
        Length of the dit strings.
    interaction_size : int
        Size of the interaction.
    dit_dimension : int, optional
        Number of possible dit values. Default is 2.

    Returns
    -------
    np.ndarray
        Sketch matrix of shape (n_constraints, dit_dimension**dit_string_length).
    """
    dit_string_length = ensure_int("dit_string_length", dit_string_length, min_value=1)
    interaction_size = ensure_int("interaction_size", interaction_size, min_value=1)
    dit_dimension = ensure_int("dit_dimension", dit_dimension, min_value=2)
    if interaction_size > dit_string_length:
        raise ValueError("interaction_size cannot be greater than dit_string_length.")

    cylinder_sets = []
    constrained_dits_indices = itertools.combinations(range(dit_string_length), interaction_size)
    for constraint_dits in constrained_dits_indices:
        cylinder_sets += ds.create_cylinder_set_indicator(constraint_dits, dit_string_length, dit_dimension)

    A = [ds.kronecker_develop(s) for s in cylinder_sets]
    return np.array(A)


def random_sketch(
    dit_string_length: int,
    m: int,
    dit_dimension: int = 2,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a random Gaussian sketch matrix.

    Parameters
    ----------
    dit_string_length : int
        Length of dit strings.
    m : int
        Number of sketch rows (measurements).
    dit_dimension : int, optional
        Dimension of each dit. Default is 2.
    random_state : int, np.random.Generator, or None, optional
        Seed or Generator for reproducibility.

    Returns
    -------
    np.ndarray
        Random matrix of shape (m, dit_dimension**dit_string_length) with i.i.d. N(0, 1/m) entries.
    """
    dit_string_length = ensure_int("dit_string_length", dit_string_length, min_value=1)
    m = ensure_int("m", m, min_value=1)
    dit_dimension = ensure_int("dit_dimension", dit_dimension, min_value=2)

    n = int(dit_dimension) ** int(dit_string_length)

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    return rng.standard_normal((m, n)) / np.sqrt(m)
