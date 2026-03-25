import numpy as np
import itertools
from numbers import Integral

from .. import data_structure as ds

def compute_marginal(full_sprectrum_values, sketch_function):
    """
    Compute the marginal of a function defined on the full spectrum of dit strings, given a sketch function that maps
    each dit string to a value.

    Parameters
    ----------
    full_sprectrum_values : list of float
        The values of the function defined on the full spectrum of dit strings. The order of the values should correspond
        to the order of the dit strings in the full spectrum.
    sketch_function : function
        A function that takes a dit string (as a list of integers) as input and returns a float value. This function is used to compute the marginal.

    Returns
    -------
    float
        The computed marginal value.
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

def nearest_neighbors_interactions_sketch(dit_string_length, interaction_size, dit_dimension=2):
    """
    Generate the sketch matrix corresponding to nearest neighbors interactions for a given dit string length, interaction size, and dit dimension.

    Parameters
    ----------
    dit_string_length : int
        The length of the dit strings.
    interaction_size : int
        The size of the interactions (i.e., the number of neighboring dits involved in each interaction).
    dit_dimension : int, optional
        The dimension of the dits (i.e., the number of possible values each dit can take). The default is 2.

    Returns
    -------
    np.ndarray
        The sketch matrix corresponding to nearest neighbors interactions.
    """

    if not isinstance(dit_string_length, int) or dit_string_length <= 0:
        raise ValueError("dit_string_length must be a positive integer.")
    if not isinstance(interaction_size, int) or interaction_size <= 0:
        raise ValueError("interaction_size must be a positive integer.")
    if interaction_size > dit_string_length:
        raise ValueError("interaction_size cannot be greater than dit_string_length.")
    if not isinstance(dit_dimension, int) or dit_dimension <= 1:
        raise ValueError("dit_dimension must be an integer greater than 1.")

    cylinder_sets = []
    constrained_dits_indices = [tuple(range(dits_indices, dits_indices + interaction_size)) for dits_indices in range(dit_string_length - interaction_size + 1)]
    for constraint_dits in constrained_dits_indices:
        cylinder_sets += ds.create_cylinder_set_indicator(constraint_dits, dit_string_length, dit_dimension)

    A = []
    for set in cylinder_sets:
        A.append(ds.kronecker_develop(set,))

    return np.array(A)

def all_interactions_sketch(dit_string_length, interaction_size, dit_dimension=2):
    """
    Generate the sketch matrix corresponding to all interactions for a given dit string length, interaction size, and dit dimension.

    Parameters
    ----------
    dit_string_length : int
        The length of the dit strings.
    interaction_size : int
        The size of the interactions (i.e., the number of neighboring dits involved in each interaction).
    dit_dimension : int, optional
        The dimension of the dits (i.e., the number of possible values each dit can take). The default is 2.
    
    Returns
    -------
    np.ndarray
        The sketch matrix corresponding to all interactions.
    """

    if not isinstance(dit_string_length, int) or dit_string_length <= 0:
        raise ValueError("dit_string_length must be a positive integer.")
    if not isinstance(interaction_size, int) or interaction_size <= 0:
        raise ValueError("interaction_size must be a positive integer.")
    if interaction_size > dit_string_length:
        raise ValueError("interaction_size cannot be greater than dit_string_length.")
    if not isinstance(dit_dimension, int) or dit_dimension <= 1:
        raise ValueError("dit_dimension must be an integer greater than 1.")

    cylinder_sets = []
    constrained_dits_indices = itertools.combinations(range(dit_string_length), interaction_size)
    for constraint_dits in constrained_dits_indices:
        cylinder_sets += ds.create_cylinder_set_indicator(constraint_dits, dit_string_length, dit_dimension)

    A = []
    for set in cylinder_sets:
        A.append(ds.kronecker_develop(set,))

    return np.array(A)


def random_sketch(dit_string_length, m, dit_dimension=2, random_state=None):
    """
    Generate a random Gaussian sketch matrix typical of compressive sensing.

    Parameters
    ----------
    dit_string_length : int
        Length of dit strings.
    m : int
        Number of sketch rows (measurements).
    dit_dimension : int, optional
        Dimension of each dit, by default 2.
    random_state : int | np.random.Generator | None, optional
        Seed or numpy Generator for reproducibility. If None, use NumPy default RNG.

    Returns
    -------
    np.ndarray
        Random matrix of shape ``(m, dit_dimension ** dit_string_length)`` with
        i.i.d. entries sampled from ``N(0, 1/m)``.
    """

    if not isinstance(dit_string_length, Integral) or isinstance(dit_string_length, bool) or dit_string_length <= 0:
        raise ValueError("dit_string_length must be a positive integer.")
    if not isinstance(m, Integral) or isinstance(m, bool) or m <= 0:
        raise ValueError("m must be a positive integer.")
    if not isinstance(dit_dimension, Integral) or isinstance(dit_dimension, bool) or dit_dimension <= 1:
        raise ValueError("dit_dimension must be an integer greater than 1.")

    n = int(dit_dimension) ** int(dit_string_length)

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    return rng.standard_normal((m, n)) / np.sqrt(m)