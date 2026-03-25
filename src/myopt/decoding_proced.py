import copy
import numpy as np

import optimization as opt
import abstract as ab
from utils import column_vector_to_array

def matchingpursuit_explicit(marginals,sketch,iteration_number,step=None, opt_func=opt.brute_force_max, opt_func_kwargs=None):
    """
    Perform matching pursuit to find a sparse solution to the linear system defined by the sketch matrix and the marginals.

    Parameters
    ----------
    marginals : list of float
        The marginals of the function defined on the full spectrum of dit strings. The order of the values should correspond
        to the order of the dit strings in the full spectrum.
    sketch : 2D numpy array
        The sketch matrix, where each column corresponds to a dit string and each row corresponds to a marginal.
    iteration_number : int
        The number of iterations to perform in the matching pursuit algorithm.
    step : float, optional
        The step size for the matching pursuit algorithm. If None, an adaptive step size is used based on the projection of the residual onto the selected column of the sketch matrix. The default is None
    opt_func : function, optional
        A function that takes a 1D numpy array as input and returns the index of the maximum value. This function is used to select the column of the sketch matrix that best matches the current residual. The default is np.argmax.
    opt_func_kwargs : dict, optional
        Keyword arguments passed to opt_func at each iteration. The default is None.
    
    Returns
    -------
    numpy array
        A 2D numpy array where each row contains the index of a selected column from the sketch matrix and its corresponding coefficient in the sparse solution.
    """

    #Matching Pursuit, initialization
    r = copy.deepcopy(marginals)              
    coeffs = {}

    for _ in range(iteration_number):

        # Find best matching atom
        t = opt_func(r, sketch) if opt_func == opt.brute_force_max else opt_func(r, **(opt_func_kwargs or {}))

        At = column_vector_to_array(sketch[:, t])
        
        if step==None:
            # Adaptive step: projection of residual onto column A_t
            norm_sq = np.dot(At, At)
            alpha = np.dot(r, At) / norm_sq if norm_sq != 0 else 0.0
        else:
            alpha = step
        
        # Update residual
        r -= alpha * At
        coeffs[t] = coeffs.get(t, 0.0) + alpha

    return np.array([[idx, coeff] for idx, coeff in coeffs.items()])


def matchingpursuit_abstract(marginals,dit_constraints,dit_string_length,iteration_number,step=None,interaction_size=2, dit_dimension=2, opt_func=opt.spin_chain_nn_max, opt_func_kwargs=None):
    """
    Perform matching pursuit to find a sparse solution to the linear system defined by the sketch matrix and the marginals, using an abstract representation of the sketch matrix based on nearest neighbors interactions.
    
    Parameters
    ----------
    marginals : list of float
        The marginals of the function defined on the full spectrum of dit strings. The order of the values should correspond
        to the order of the dit strings in the full spectrum.
    dit_constraints : list of tuples
        The constraints on the dit strings, where each tuple represents a constraint on a subset of dits.
    dit_string_length : int
        The length of the dit strings.
    iteration_number : int
        The number of iterations to perform in the matching pursuit algorithm.
    step : float, optional
        The step size for the matching pursuit algorithm. If None, an adaptive step size is used based on the projection of the residual onto the selected column of the sketch matrix. The default is None.
    interaction_size : int, optional
        The size of the interactions between the dits. The default is 2, which corresponds to
        nearest neighbors interactions.
    dit_dimension : int, optional
        The dimension of the dits, i.e., the number of possible values each dit can take. The default is 2.
    opt_func : function, optional
        A function that takes a 1D numpy array as input and returns the index of the maximum value. This function is used to select the column of the sketch matrix that best matches the current
        residual. The default is opt.spin_chain_nn_max, which is a function that finds the optimal configuration of a spin chain with nearest neighbors interactions.
    opt_func_kwargs : dict, optional
        Keyword arguments passed to opt_func at each iteration. The default is None.

    Returns
    -------
    numpy array
        A 2D numpy array where each row contains the index of a selected column from the sketch matrix and its corresponding coefficient in the sparse solution.
    """

    #Matching Pursuit, initialization
    r = copy.deepcopy(marginals)              
    coeffs = {}

    for _ in range(iteration_number):
        
        #Find best matching atom
        if opt_func == opt.spin_chain_nn_max:
            t = opt_func(marginals=r, dit_string_length=dit_string_length, interaction_size=interaction_size, dit_dimension=dit_dimension)
        else:
            t = opt_func(r, **(opt_func_kwargs or {}))
    
        At = ab.reconstruct_structured_matrix_column(t, dit_constraints, dit_string_length, dit_dimension) #column of the sketch matrix corresponding to the configuration conf

        if step==None:
            # Adaptive step: projection of residual onto column A_t
            norm_sq = np.dot(At, At)
            alpha = np.dot(r, At) / norm_sq if norm_sq != 0 else 0.0
        else:
            alpha = step
        
        # Update residual
        r -= alpha * At
        coeffs[t] = coeffs.get(t, 0.0) + alpha

    return np.array([[idx, coeff] for idx, coeff in coeffs.items()])