import numpy as np
# from .data_structure import belongs_to_cylinder_set
from data_structure import integer_to_dit_string,dit_string_to_computational_basis,belongs_to_cylinder_set

def compute_marginal(function_input_dits, function_values, dit_constraints):
    """
    Compute the marginals of a sparse spectrum on specified dit constraints.
    This gives the generalized moment of the function computed by a structured matrix made of cylinder set indicators.
    
    Parameters
    ----------
    function_input_dits : list or ndarray
        List of function inputs, each input is given in dits representation 
    function_values : list or ndarray
        Image of the function associated with each input.
    dit_constraints : dict or list of tuples
        Constraints on which dits must have which values.
        - If dict: {dit_index: dit_value, ...}
        - If list of tuples: [(dit_index, dit_value), ...]
        
    Returns
    -------
    float
        A generalized moment of the spectrum, i.e. the sum of values for states that match the specified dit constraints.
        
    Examples
    --------
    >>> # Compute marginal for pair of dits d0=0 and d1=2
    >>> compute_marginal(spectrum_bin, spectrum_val, {0: 0, 1: 2})
    
    >>> # Compute marginal for triplet with specific values
    >>> compute_marginal(spectrum_bin, spectrum_val, [(1, 1), (3, 0), (5, 1)])
    """
    # Convert dict constraints to comparable format if needed
    if isinstance(dit_constraints, dict):
        constraints = list(dit_constraints.items())
    else:
        constraints = list(dit_constraints)
    
    # Filter states matching all constraints
    filtered_vals = [
        v for s, v in zip(function_input_dits, function_values)
        if all(s[dit_idx] == dit_val for dit_idx, dit_val in constraints)
    ]
    
    return float(np.sum(filtered_vals) if filtered_vals else 0.0)

# def compute_nearest_neighbors_marginals
# def compute_all_uplets_marginals
# abstract and naive versions -- naive can be more general moments than marginals

def reconstruct_structured_matrix_column(index, cylinder_set_list,dit_dimension=2):
    """
    Check to which cylinder sets an element belongs.
    Reconstruct a column of a structured matrix made of cylinder set indicators, given the element index.
    
    Parameters
    ----------
    index : int
        The index of the element to check.
    cylinder_set_list : list
        List of cylinder set full indicators to check.
        
    Returns
    -------
    list of bool
        The column of the structured matrix corresponding to this index, i.e. a list indicating whether the element belongs to each cylinder set or not.
    """
    length = len(cylinder_set_list[0])  # Assuming all cylinder sets have the same length
    dit_str = integer_to_dit_string(index, length, dit_dimension=dit_dimension)
    element = dit_string_to_computational_basis(dit_str, dit_dimension=dit_dimension)
    
    return [belongs_to_cylinder_set(element, cyl_set) for cyl_set in cylinder_set_list]