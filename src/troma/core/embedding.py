from __future__ import annotations

from typing import Any

import numpy as np

from .._validation import (
    ensure_int,
    ensure_unique_items,
    ensure_vector_collection,
    ensure_sequence,
    ensure_dict,
)


def _validate_spectrum_vectors(spectrum: Any, name: str) -> list:
    vectors = ensure_vector_collection(name, spectrum)
    if vectors:
        vector_length = len(vectors[0])
        for vector in vectors:
            if len(vector) != vector_length:
                raise ValueError(f"All vectors in {name} must have the same length.")
            for value in vector:
                ensure_int(f"{name} value", value)
    return vectors


def _to_dit_string(vector: list[int] | np.ndarray) -> np.ndarray:
    return np.array([int(v) for v in vector])


def _validate_dit_restrictions(dit_restrictions: Any) -> list[int]:
    items = ensure_sequence("dit_restrictions", dit_restrictions)
    result = [ensure_int("dit_restrictions item", idx, min_value=0) for idx in items]
    if not result:
        raise ValueError("dit_restrictions must contain at least one index.")
    ensure_unique_items("dit_restrictions", result)
    return result


def _validate_dit_value_restrictions(dit_value_restrictions: Any) -> list[int]:
    items = ensure_sequence("dit_value_restrictions", dit_value_restrictions)
    result = [ensure_int("dit_value_restrictions item", v, min_value=0) for v in items]
    if len(result) < 2:
        raise ValueError("dit_value_restrictions must contain at least two values.")
    ensure_unique_items("dit_value_restrictions", result)
    return result


def _validate_additional_dits(additional_dits: Any) -> list[int]:
    if additional_dits is None:
        return []
    items = ensure_sequence("additional_dits", additional_dits)
    result = [ensure_int("additional_dits item", v, min_value=0) for v in items]
    ensure_unique_items("additional_dits", result)
    return result


def _validate_dimension_mapping(dimension_mapping: Any) -> dict[int, int] | None:
    if dimension_mapping is None:
        return None
    ensure_dict("dimension_mapping", dimension_mapping)
    for original_val, new_val in dimension_mapping.items():
        ensure_int("dimension_mapping key", original_val, min_value=0)
        ensure_int("dimension_mapping value", new_val, min_value=0)
    return dimension_mapping


def _validate_additional_dits_val(additional_dits_val: Any) -> int:
    return ensure_int("additional_dits_val", additional_dits_val, min_value=0)


def spectrum_restriction(
    spectrum_dit: list[list[int]],
    dit_restrictions: list[int] | None,
    dit_value_restrictions: list[int] | None,
) -> list[np.ndarray]:
    """
    Given a spectrum dit strings, limit the space to specified dits and specific values for these dits.

    Parameters
    ----------
    spectrum_dit : list of list of int
        The original spectrum represented as a list of dit strings.
    dit_restrictions : list of int
        The indices of the dits to restrict.
    dit_value_restrictions : list of int
        The values that the restricted dits should take.

    Returns
    -------
    list of numpy arrays
        The spectrum restricted to the specified dits and values.

    Example
    -------
    >>> spectrum_bin = [ [0,0,0,0,0,0], [0,0,2,0,2,2], [0,2,2,0,0,0], [0,2,2,0,2,2] ]
    >>> dit_restrictions = [1,2,4,5]
    >>> dit_value_restrictions = [0, 2]
    >>> spectrum_restriction(spectrum_bin, dit_restrictions, dit_value_restrictions)
    [ [0,0,0,0], [0,1,1,0], [1,1,0,0], [1,1,1,1] ]
    """
    spectrum_dit = _validate_spectrum_vectors(spectrum_dit, "spectrum_dit")
    if dit_restrictions is not None:
        dit_restrictions = _validate_dit_restrictions(dit_restrictions)
    if dit_value_restrictions is not None:
        dit_value_restrictions = _validate_dit_value_restrictions(dit_value_restrictions)

    new_spectrum_dit = [np.array(s, copy=True) for s in spectrum_dit]
    if dit_restrictions is not None:
        new_spectrum_dit = [s[dit_restrictions] for s in new_spectrum_dit]
    if dit_value_restrictions is not None:
        for i, val in enumerate(dit_value_restrictions):
            for s in new_spectrum_dit:
                s[s == val] = i
    return [_to_dit_string(s) for s in new_spectrum_dit]


def spectrum_embedding(
    spectrum_dit: list[list[int]],
    additional_dits: list[int] | None,
    dimension_mapping: dict[int, int] | None,
    additional_dits_val: int = 0,
) -> list[np.ndarray]:
    """
    Embed a spectrum into a larger space by adding additional dits and additional dimensions.

    Parameters
    ----------
    spectrum_dit : list of list of int
        The original spectrum represented as a list of dit strings.
    additional_dits : list of int
        The indices where the additional dits should be inserted.
    dimension_mapping : dict
        A mapping from the original dit values to the new dit values in the larger-dimensional space.

    Returns
    -------
    list of numpy arrays
        The spectrum embedded into the larger space.

    Example
    -------
    >>> spectrum_bin = [ [0,0,0,0], [0,1,1,0], [1,1,0,0], [1,1,1,1] ]
    >>> additional_dits = [0,3]
    >>> dimension_mapping = {0: 0, 1: 2}
    >>> spectrum_embedding(spectrum_bin, additional_dits, dimension_mapping)
    [ [0,0,0,0,0,0], [0,0,2,0,2,2], [0,2,2,0,0,0], [0,2,2,0,2,2] ]
    """
    spectrum_dit = _validate_spectrum_vectors(spectrum_dit, "spectrum_dit")
    additional_dits = _validate_additional_dits(additional_dits)
    dimension_mapping = _validate_dimension_mapping(dimension_mapping)
    additional_dits_val = _validate_additional_dits_val(additional_dits_val)

    new_spectrum_dit = []
    for s in spectrum_dit:
        s = np.array(s, copy=True)
        for pos in additional_dits:
            s = np.insert(s, pos, additional_dits_val)
        new_spectrum_dit.append(s)

    if dimension_mapping is not None:
        for original_val, new_val in dimension_mapping.items():
            for s in new_spectrum_dit:
                s[s == original_val] = new_val

    return [_to_dit_string(s) for s in new_spectrum_dit]


def reverse_spectrum_restriction(
    spectrum_dits: list[list[int]],
    original_size: int,
    dit_restrictions: list[int] | None,
    dit_value_restrictions: list[int] | None,
    additional_dits_val: int = 0,
) -> list[np.ndarray]:
    """
    Reverse the spectrum restrictions by reintroducing the original dits and values through embeddings.

    Parameters
    ----------
    spectrum_dits : list of list of int
        The restricted spectrum represented as a list of dit strings.
    original_size : int
        The number of dits in the original spectrum before restriction.
    dit_restrictions : list of int
        The indices of the dits that were restricted.
    dit_value_restrictions : list of int
        The original values that the restricted dits should take.

    Returns
    -------
    list of numpy arrays
        The original spectrum with the restricted dits and values reintroduced.

    Example
    -------
    >>> restricted_spectrum_bin = [ [0,0,0,0], [0,1,1,0], [1,1,0,0], [1,1,1,1] ]
    >>> dit_restrictions = [1,2,4,5]
    >>> dit_value_restrictions = [0, 2]
    >>> reverse_spectrum_restriction(restricted_spectrum_bin, 6, dit_restrictions, dit_value_restrictions)
    [ [0,0,0,0,0,0], [0,0,2,0,2,2], [0,2,2,0,0,0], [0,2,2,0,2,2] ]
    """
    original_size = ensure_int("original_size", original_size, min_value=1)
    additional_dits_val = _validate_additional_dits_val(additional_dits_val)
    spectrum_dits = _validate_spectrum_vectors(spectrum_dits, "spectrum_dits")

    if dit_restrictions is not None:
        dit_restrictions = _validate_dit_restrictions(dit_restrictions)
        for index in dit_restrictions:
            if index >= original_size:
                raise ValueError("dit_restrictions items must be less than original_size.")
    if dit_value_restrictions is not None:
        dit_value_restrictions = _validate_dit_value_restrictions(dit_value_restrictions)

    if dit_restrictions is None:
        complementary_dits = []
    else:
        complementary_dits = [i for i in range(original_size) if i not in dit_restrictions]

    if dit_value_restrictions is None:
        return spectrum_embedding(
            spectrum_dits,
            additional_dits=complementary_dits,
            dimension_mapping=None,
            additional_dits_val=additional_dits_val,
        )

    dit_mapping = {k: v for k, v in enumerate(dit_value_restrictions)}

    return spectrum_embedding(
        spectrum_dits,
        additional_dits=complementary_dits,
        dimension_mapping=dit_mapping,
        additional_dits_val=additional_dits_val,
    )
