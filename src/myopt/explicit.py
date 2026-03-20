import data_structure as ds
import numpy as np
import itertools

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
    y = sketch_function* np.matrix(full_sprectrum_values).T
    
    return np.array(y).flatten().tolist() 

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
    np.matrix
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

    return np.matrix(A)

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
    np.matrix
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

    return np.matrix(A)