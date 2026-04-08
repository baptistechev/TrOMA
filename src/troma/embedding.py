import numpy as np

from ._validation import (
    ensure_callable,
    ensure_iterable,
    ensure_int,
    ensure_unique_items,
    ensure_vector_collection,
)


def _validate_spectrum_vectors(spectrum, name):
    vectors = ensure_vector_collection(name, spectrum)
    if vectors:
        vector_length = len(vectors[0])
        for vector in vectors:
            if len(vector) != vector_length:
                raise ValueError(f"All vectors in {name} must have the same length.")
            for value in vector:
                ensure_int(f"{name} value", value)
    return vectors


def _validate_dit_restrictions(dit_restrictions):
    ensure_iterable("dit_restrictions", dit_restrictions)
    dit_restrictions = [
        ensure_int("dit_restrictions item", index, min_value=0)
        for index in dit_restrictions
    ]
    if not dit_restrictions:
        raise ValueError("dit_restrictions must contain at least one index.")
    ensure_unique_items("dit_restrictions", dit_restrictions)
    return dit_restrictions


def _validate_dit_value_restrictions(dit_value_restrictions):
    ensure_iterable("dit_value_restrictions", dit_value_restrictions)
    dit_value_restrictions = [
        ensure_int("dit_value_restrictions item", value, min_value=0)
        for value in dit_value_restrictions
    ]
    if len(dit_value_restrictions) < 2:
        raise ValueError("dit_value_restrictions must contain at least two values.")
    ensure_unique_items("dit_value_restrictions", dit_value_restrictions)
    return dit_value_restrictions


def _validate_additional_dits(additional_dits):
    if additional_dits is None:
        return []
    ensure_iterable("additional_dits", additional_dits)
    additional_dits = [
        ensure_int("additional_dits item", value, min_value=0)
        for value in additional_dits
    ]
    ensure_unique_items("additional_dits", additional_dits)
    return additional_dits


def _validate_dimension_mapping(dimension_mapping):
    if dimension_mapping is None:
        return None
    if not isinstance(dimension_mapping, dict):
        raise TypeError("dimension_mapping must be a dict or None.")
    for original_val, new_val in dimension_mapping.items():
        ensure_int("dimension_mapping key", original_val, min_value=0)
        ensure_int("dimension_mapping value", new_val, min_value=0)
    return dimension_mapping


def _validate_additional_dits_val(additional_dits_val):
    return ensure_int("additional_dits_val", additional_dits_val, min_value=0)


def spectrum_restriction(spectrum_dit, dit_restrictions, dit_value_restrictions):
    """
    Given a spectrum dit strings, limit the space to specified dits and specific values for these dits.

    Parameters
    ----------
    spectrum_bin : list of list of int
        The original spectrum represented as a list of dit strings.
    dit_restrictions : list of int
        The indices of the dits to restrict.
    dit_value_restrictions : list of int
        The values that the restricted dits should take.
    
    Returns
    -------
    list of list of int
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
    return new_spectrum_dit

def spectrum_embedding(spectrum_dit, additional_dits, dimension_mapping, additional_dits_val=0):
    """
    Embed a spectrum into a larger space by adding additional dits and additional dimensions.

    Parameters
    ----------
    spectrum_dit : list of list of int
        The original spectrum represented as a list of dit strings.
    additional_dits : list of int
        The indices where the additional dits should be inserted.
    dimension_mapping : dict
        A mapping from the original dit values to the new dit values in the larger-dimensinal space.
    
    Returns
    -------
    list of list of int
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
    
    return new_spectrum_dit

def reverse_spectrum_restriction(spectrum_dits, original_size, dit_restrictions, dit_value_restrictions, additional_dits_val=0):
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
    list of list of int
        The original spectrum with the restricted dits and values reintroduced.

    Example
    -------
    >>> restricted_spectrum_bin = [ [0,0,0,0], [0,1,1,0], [1,1,0,0], [1,1,1,1] ]
    >>> dit_restrictions = [1,2,4,5]
    >>> dit_value_restrictions = [0, 2]

    >>> reverse_spectrum_restriction(restricted_spectrum_bin, original_size=6, dit_restrictions=dit_restrictions, dit_value_restrictions=dit_value_restrictions)
    [ [0,0,0,0,0,0], [0,0,2,0,2,2], [0,2,2,0,0,0], [0,2,2,0,2,2] ]
    """

    ensure_int("original_size", original_size, min_value=1)
    additional_dits_val = _validate_additional_dits_val(additional_dits_val)
    spectrum_dits = _validate_spectrum_vectors(spectrum_dits, "spectrum_dits")

    if dit_restrictions is not None:
        dit_restrictions = _validate_dit_restrictions(dit_restrictions)
        for index in dit_restrictions:
            if index >= original_size:
                raise ValueError(
                    "dit_restrictions items must be less than original_size."
                )
    if dit_value_restrictions is not None:
        dit_value_restrictions = _validate_dit_value_restrictions(dit_value_restrictions)

    # Treat missing restriction inputs as no-ops.
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

    # Embed the spectrum into the larger space
    return spectrum_embedding(
        spectrum_dits,
        additional_dits=complementary_dits,
        dimension_mapping=dit_mapping,
        additional_dits_val=additional_dits_val,
    )

