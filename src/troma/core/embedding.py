from __future__ import annotations

from typing import Any

import numpy as np

from .structure import DitString
from .._validation import (
    ensure_int,
    ensure_unique_items,
    ensure_instance,
    ensure_dict,
)


def _validate_dit_string_list(spectrum: Any, name: str) -> list[DitString]:
    """Validate that spectrum is a list of DitString instances with consistent length/dimension."""
    if not isinstance(spectrum, (list, tuple)):
        raise TypeError(f"{name} must be a list of DitString instances.")
    for i, s in enumerate(spectrum):
        ensure_instance(f"{name}[{i}]", s, DitString)
    if len(spectrum) > 1:
        ref_length = spectrum[0].length
        ref_dim = spectrum[0].dimension
        for i, s in enumerate(spectrum[1:], 1):
            if s.length != ref_length:
                raise ValueError(f"All DitString objects in {name} must have the same length.")
            if s.dimension != ref_dim:
                raise ValueError(f"All DitString objects in {name} must have the same dimension.")
    return list(spectrum)


def _validate_dit_restrictions(dit_restrictions: Any) -> list[int]:
    if not isinstance(dit_restrictions, (list, tuple, np.ndarray)):
        raise TypeError("dit_restrictions must be a sequence.")
    result = [ensure_int("dit_restrictions item", idx, min_value=0) for idx in dit_restrictions]
    if not result:
        raise ValueError("dit_restrictions must contain at least one index.")
    ensure_unique_items("dit_restrictions", result)
    return result


def _validate_dit_value_restrictions(dit_value_restrictions: Any) -> list[int]:
    if not isinstance(dit_value_restrictions, (list, tuple, np.ndarray)):
        raise TypeError("dit_value_restrictions must be a sequence.")
    result = [ensure_int("dit_value_restrictions item", v, min_value=0) for v in dit_value_restrictions]
    if len(result) < 2:
        raise ValueError("dit_value_restrictions must contain at least two values.")
    ensure_unique_items("dit_value_restrictions", result)
    return result


def _validate_additional_dits(additional_dits: Any) -> list[int]:
    if additional_dits is None:
        return []
    if not isinstance(additional_dits, (list, tuple, np.ndarray)):
        raise TypeError("additional_dits must be a sequence.")
    result = [ensure_int("additional_dits item", v, min_value=0) for v in additional_dits]
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
    spectrum_dit: list[DitString],
    dit_restrictions: list[int] | None,
    dit_value_restrictions: list[int] | None,
) -> list[DitString]:
    """Restrict a spectrum to selected dit positions and remap their values.

    Parameters
    ----------
    spectrum_dit : list[DitString]
        The original spectrum. All DitString objects must share the same
        ``length`` and ``dimension``.
    dit_restrictions : list[int] or None
        Indices of the dit positions to keep. ``None`` keeps all positions.
    dit_value_restrictions : list[int] or None
        Values to remap: original value ``dit_value_restrictions[i]`` is
        replaced by ``i``. ``None`` keeps original values.

    Returns
    -------
    list[DitString]
        The restricted spectrum. Each DitString has ``dimension`` equal to
        ``len(dit_value_restrictions)`` when remapping is applied, otherwise
        the same as the input.

    Example
    -------
    >>> spectrum = [DitString([0,0,2,0,2,2], dimension=3), ...]
    >>> spectrum_restriction(spectrum, [1,2,4,5], [0,2])
    [DitString([0,0,1,1], dimension=2), ...]
    """
    spectrum_dit = _validate_dit_string_list(spectrum_dit, "spectrum_dit")
    input_dimension = spectrum_dit[0].dimension if spectrum_dit else 2

    if dit_restrictions is not None:
        dit_restrictions = _validate_dit_restrictions(dit_restrictions)
    if dit_value_restrictions is not None:
        dit_value_restrictions = _validate_dit_value_restrictions(dit_value_restrictions)

    new_spectrum = [np.array(s.tolist(), copy=True) for s in spectrum_dit]
    if dit_restrictions is not None:
        new_spectrum = [s[dit_restrictions] for s in new_spectrum]
    if dit_value_restrictions is not None:
        for i, val in enumerate(dit_value_restrictions):
            for s in new_spectrum:
                s[s == val] = i

    output_dimension = len(dit_value_restrictions) if dit_value_restrictions is not None else input_dimension
    return [DitString(arr.tolist(), dimension=output_dimension) for arr in new_spectrum]


def spectrum_embedding(
    spectrum_dit: list[DitString],
    additional_dits: list[int] | None,
    dimension_mapping: dict[int, int] | None,
    additional_dits_val: int = 0,
) -> list[DitString]:
    """Embed a spectrum into a larger space.

    Parameters
    ----------
    spectrum_dit : list[DitString]
        The original spectrum. All DitString objects must share the same
        ``length`` and ``dimension``.
    additional_dits : list[int] or None
        Indices at which to insert extra dits (filled with ``additional_dits_val``).
    dimension_mapping : dict or None
        Maps original dit values to new dit values in a larger-dimensional space.
    additional_dits_val : int, optional
        Value inserted at the additional dit positions. Default is 0.

    Returns
    -------
    list[DitString]
        The embedded spectrum.

    Example
    -------
    >>> spectrum = [DitString([0,0,0,0], dimension=2), ...]
    >>> spectrum_embedding(spectrum, [0, 3], {0: 0, 1: 2})
    [DitString([0,0,0,0,0,0], dimension=3), ...]
    """
    spectrum_dit = _validate_dit_string_list(spectrum_dit, "spectrum_dit")
    input_dimension = spectrum_dit[0].dimension if spectrum_dit else 2
    additional_dits = _validate_additional_dits(additional_dits)
    dimension_mapping = _validate_dimension_mapping(dimension_mapping)
    additional_dits_val = _validate_additional_dits_val(additional_dits_val)

    new_spectrum = []
    for s in spectrum_dit:
        arr = np.array(s.tolist(), copy=True)
        for pos in additional_dits:
            arr = np.insert(arr, pos, additional_dits_val)
        new_spectrum.append(arr)

    if dimension_mapping is not None:
        for original_val, new_val in dimension_mapping.items():
            for arr in new_spectrum:
                arr[arr == original_val] = new_val

    output_dimension = (
        max(dimension_mapping.values()) + 1
        if dimension_mapping
        else input_dimension
    )
    return [DitString(arr.tolist(), dimension=output_dimension) for arr in new_spectrum]


def reverse_spectrum_restriction(
    spectrum_dits: list[DitString],
    original_size: int,
    dit_restrictions: list[int] | None,
    dit_value_restrictions: list[int] | None,
    additional_dits_val: int = 0,
) -> list[DitString]:
    """Re-embed a restricted spectrum back into the original (larger) space.

    Parameters
    ----------
    spectrum_dits : list[DitString]
        The restricted spectrum. All DitString objects must share the same
        ``length`` and ``dimension``.
    original_size : int
        Number of dits in the original pre-restriction spectrum.
    dit_restrictions : list[int] or None
        The dit indices that were kept during the forward restriction pass.
    dit_value_restrictions : list[int] or None
        The original values that were remapped during the forward pass.
    additional_dits_val : int, optional
        Value to insert at the non-restricted (complementary) positions. Default is 0.

    Returns
    -------
    list[DitString]
        The re-embedded spectrum with ``length == original_size``.

    Example
    -------
    >>> restricted = [DitString([0,0,0,0], dimension=2), ...]
    >>> reverse_spectrum_restriction(restricted, 6, [1,2,4,5], [0,2])
    [DitString([0,0,0,0,0,0], dimension=3), ...]
    """
    original_size = ensure_int("original_size", original_size, min_value=1)
    additional_dits_val = _validate_additional_dits_val(additional_dits_val)
    spectrum_dits = _validate_dit_string_list(spectrum_dits, "spectrum_dits")

    if dit_restrictions is not None:
        dit_restrictions = _validate_dit_restrictions(dit_restrictions)
        for index in dit_restrictions:
            if index >= original_size:
                raise ValueError("dit_restrictions items must be less than original_size.")
    if dit_value_restrictions is not None:
        dit_value_restrictions = _validate_dit_value_restrictions(dit_value_restrictions)

    complementary_dits = (
        [i for i in range(original_size) if i not in dit_restrictions]
        if dit_restrictions is not None
        else []
    )

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
