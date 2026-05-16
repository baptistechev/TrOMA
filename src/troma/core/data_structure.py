from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .._validation import ensure_int, ensure_str, ensure_unique_items
from .structure import DitString


def create_cylinder_set_indicator(
    fixed_dit_positions: Iterable[int] | np.ndarray,
    set_size: int,
    dit_dimension: int = 2,
) -> list[list[list[int]]]:
    """Build factorized cylinder-set indicators for fixed dit positions.

    Parameters
    ----------
    fixed_dit_positions : Iterable[int] | np.ndarray
        Positions constrained in the cylinder set.
    set_size : int
        Total dit-string size.
    dit_dimension : int, optional
        Dit base, by default 2.

    Returns
    -------
    list[list[list[int]]]
        List of factorized cylinder indicators, one per fixed-position assignment.
    """
    set_size = ensure_int("set_size", set_size)
    dit_dimension = ensure_int("dit_dimension", dit_dimension)
    if set_size < 0:
        raise ValueError("set_size must be >= 0.")
    if dit_dimension < 2:
        raise ValueError("dit_dimension must be >= 2.")

    fixed_positions_raw = list(fixed_dit_positions)
    for pos in fixed_positions_raw:
        ensure_int("fixed_dit_positions element", pos)
    fixed_positions = [int(pos) for pos in fixed_positions_raw]
    ensure_unique_items("fixed_dit_positions", fixed_positions)

    for position in fixed_positions:
        if position < 0 or position >= set_size:
            raise ValueError("All fixed_dit_positions indices must be in [0, set_size-1].")

    list_cylinder_sets: list[list[list[int]]] = []
    n_positions = len(fixed_positions)
    for config in range(dit_dimension ** n_positions):
        dits = DitString.from_integer(config, n_positions, dit_dimension)
        cylinder_set = [[1] * dit_dimension for _ in range(set_size)]
        for position, dit_value in zip(fixed_positions, dits):
            cp_vector = [0] * dit_dimension
            cp_vector[int(dit_value)] = 1
            cylinder_set[position] = cp_vector
        list_cylinder_sets.append(cylinder_set)
    return list_cylinder_sets


def kronecker_develop(
    cylinder_set: Iterable[Iterable[int]] | np.ndarray,
    dit_dimension: int = 2,
    convention: str = "R",
) -> np.ndarray:
    """Expand a factorized cylinder indicator with Kronecker products.

    Parameters
    ----------
    cylinder_set : Iterable[Iterable[int]] | np.ndarray
        Sequence of local binary indicator vectors.
    dit_dimension : int, optional
        Local vector length / dit base, by default 2.
    convention : str, optional
        Multiplication order: ``"R"`` or ``"L"``.

    Returns
    -------
    np.ndarray
        Full developed indicator vector.
    """
    dit_dimension = ensure_int("dit_dimension", dit_dimension)
    ensure_str("convention", convention)
    if dit_dimension < 2:
        raise ValueError("dit_dimension must be >= 2.")
    if convention not in ("R", "L"):
        raise ValueError("convention must be 'R' or 'L'.")

    vectors = [list(vector) for vector in cylinder_set]
    for vector in vectors:
        if len(vector) != dit_dimension:
            raise ValueError("Each vector in cylinder_set must have length dit_dimension.")
        if any(value not in (0, 1) for value in vector):
            raise ValueError("cylinder_set vectors must be binary.")

    if convention == "L":
        vectors = vectors[::-1]

    developed = np.array([1], dtype=int)
    for vector in vectors:
        developed = np.kron(developed, np.array(vector, dtype=int))
    return developed


def belongs_to_cylinder_set(
    element: Iterable[Iterable[int]] | np.ndarray,
    cylinder_set: Iterable[Iterable[int]] | np.ndarray,
    dit_dimension: int = 2,
) -> bool:
    """Check whether an element belongs to a factorized cylinder set.

    Parameters
    ----------
    element : Iterable[Iterable[int]] | np.ndarray
        Candidate element in one-hot factorized representation.
    cylinder_set : Iterable[Iterable[int]] | np.ndarray
        Factorized cylinder indicator representation.
    dit_dimension : int, optional
        Dit base, by default 2.

    Returns
    -------
    bool
        ``True`` if the element satisfies all fixed coordinates.
    """
    dit_dimension = ensure_int("dit_dimension", dit_dimension)
    if dit_dimension < 2:
        raise ValueError("dit_dimension must be >= 2.")

    element_vectors = [list(vector) for vector in element]
    cylinder_vectors = [list(vector) for vector in cylinder_set]

    if len(element_vectors) != len(cylinder_vectors):
        raise ValueError("element and cylinder_set must have the same length.")

    wildcard = [1] * dit_dimension
    for basis_element, basis_cylinder in zip(element_vectors, cylinder_vectors):
        if len(basis_element) != dit_dimension or len(basis_cylinder) != dit_dimension:
            raise ValueError("Each basis vector must have length dit_dimension.")
        if any(value not in (0, 1) for value in basis_element):
            raise ValueError("element vectors must be binary.")
        if any(value not in (0, 1) for value in basis_cylinder):
            raise ValueError("cylinder_set vectors must be binary.")
        if basis_cylinder != wildcard and basis_element != basis_cylinder:
            return False
    return True
