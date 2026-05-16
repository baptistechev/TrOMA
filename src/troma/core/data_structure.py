from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .._validation import ensure_int, ensure_str, ensure_unique_items, ensure_sequence


def integer_to_dit_string(
    integer: int,
    dit_string_length: int,
    dit_dimension: int = 2,
    convention: str = "R",
) -> np.ndarray:
    """Encode an integer into a fixed-length dit string.

    Parameters
    ----------
    integer : int
        Integer to encode.
    dit_string_length : int
        Target length of the output dit string.
    dit_dimension : int, optional
        Dit base (for example 2 for binary, 3 for ternary), by default 2.
    convention : str, optional
        Ordering convention. ``"R"`` returns most-significant dit first,
        ``"L"`` returns least-significant dit first.

    Returns
    -------
    np.ndarray
        Encoded dit string as an integer numpy array.
    """
    integer = ensure_int("integer", integer)
    dit_dimension = ensure_int("dit_dimension", dit_dimension)
    dit_string_length = ensure_int("dit_string_length", dit_string_length)
    ensure_str("convention", convention)
    if dit_dimension < 2:
        raise ValueError("dit_dimension must be >= 2.")
    if dit_string_length < 0:
        raise ValueError("dit_string_length must be >= 0.")
    if integer < 0:
        raise ValueError("integer must be >= 0.")
    if convention not in ("R", "L"):
        raise ValueError("convention must be 'R' or 'L'.")
    if integer > 0 and dit_string_length == 0:
        raise ValueError("dit_string_length must be positive to encode a non-zero integer.")
    if dit_string_length > 0 and integer >= dit_dimension ** dit_string_length:
        raise ValueError("integer cannot be represented with the given dit_dimension and dit_string_length.")

    values: list[int] = []
    current = int(integer)
    while current > 0:
        current, rem = divmod(current, int(dit_dimension))
        values.append(int(rem))

    values += [0] * (int(dit_string_length) - len(values))
    arr = np.array(values, dtype=int)
    return arr[::-1] if convention == "R" else arr


def dit_string_to_integer(
    dit_string: Iterable[int] | np.ndarray | str,
    dit_dimension: int = 2,
    convention: str = "R",
) -> int:
    """Decode a dit string into its integer value.

    Parameters
    ----------
    dit_string : Iterable[int] | np.ndarray | str
        Input dit sequence. Strings must contain only digits.
    dit_dimension : int, optional
        Dit base, by default 2.
    convention : str, optional
        Ordering convention matching :func:`integer_to_dit_string`.

    Returns
    -------
    int
        Decoded integer.
    """
    dit_dimension = ensure_int("dit_dimension", dit_dimension)
    ensure_str("convention", convention)
    if dit_dimension < 2:
        raise ValueError("dit_dimension must be >= 2.")
    if convention not in ("R", "L"):
        raise ValueError("convention must be 'R' or 'L'.")

    if isinstance(dit_string, str):
        if len(dit_string) == 0:
            values: list[int] = []
        elif not dit_string.isdigit():
            raise ValueError("When dit_string is a string, it must contain only digits.")
        else:
            values = [int(ch) for ch in dit_string]
    else:
        raw = list(dit_string)
        for v in raw:
            ensure_int("dit_string element", v)
        values = [int(v) for v in raw]

    for value in values:
        if value < 0 or value > dit_dimension - 1:
            raise ValueError("Each value in dit_string must be in [0, dit_dimension-1].")

    basis = np.power(dit_dimension, np.arange(len(values)), dtype=float)
    if convention == "R":
        basis = basis[::-1]
    return int(np.dot(np.array(values, dtype=int), basis))


def dit_string_to_computational_basis(
    dit_string: Iterable[int] | np.ndarray | str,
    dit_dimension: int = 2,
) -> list[list[int]]:
    """Convert a dit string into one-hot computational basis vectors.

    Parameters
    ----------
    dit_string : Iterable[int] | np.ndarray | str
        Input dit sequence.
    dit_dimension : int, optional
        Dit base, by default 2.

    Returns
    -------
    list[list[int]]
        One-hot vectors for each dit value.
    """
    dit_dimension = ensure_int("dit_dimension", dit_dimension)
    if dit_dimension < 2:
        raise ValueError("dit_dimension must be >= 2.")

    if isinstance(dit_string, str):
        raw_values = list(dit_string)
    else:
        raw_values = list(dit_string)

    cp_representation: list[list[int]] = []
    for raw_value in raw_values:
        if isinstance(raw_value, str):
            if len(raw_value) != 1 or not raw_value.isdigit():
                raise ValueError("Each dit value must be an int or a single digit string.")
            value = int(raw_value)
        else:
            value = int(raw_value)
        if value < 0 or value > dit_dimension - 1:
            raise ValueError("Each dit value must be in [0, dit_dimension-1].")
        cp_vector = [0] * dit_dimension
        cp_vector[value] = 1
        cp_representation.append(cp_vector)
    return cp_representation


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
        dits = integer_to_dit_string(
            config,
            dit_string_length=n_positions,
            dit_dimension=dit_dimension,
            convention="R",
        )
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
        Multiplication order convention (``"R"`` or ``"L"``).

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
        ``True`` if the element satisfies all fixed coordinates, otherwise ``False``.
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
