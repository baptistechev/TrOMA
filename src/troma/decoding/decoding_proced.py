import copy
import numpy as np


from ..sketchs import abstract as ab
from ..optimization import optimizer as optimizer_api

def _column_vector_to_array(vec):
    return np.asarray(vec).reshape(-1)


def matchingpursuit_explicit(marginals, sketch, iteration_number, step=None, optimizer=None):
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
    optimizer : Optimizer, optional
        Instantiated optimizer implementing an ``optimize`` method.
        If None, a default brute-force optimizer is instantiated.
        The matching pursuit context (e.g. ``sketch``) is provided automatically.

    Returns
    -------
    numpy array
        A 2D numpy array where each row contains the index of a selected column from the sketch matrix and its corresponding coefficient in the sparse solution.
    """

    if optimizer is None:
        optimizer = optimizer_api.get_optimizer("brute_force_max")
    elif not hasattr(optimizer, "optimize"):
        raise TypeError("optimizer must implement an optimize(*args, **kwargs) method.")

    #Matching Pursuit, initialization
    r = copy.deepcopy(marginals)
    coeffs = {}

    for _ in range(iteration_number):

        # Find best matching atom
        t = optimizer.optimize(r, sketch=sketch)

        At = _column_vector_to_array(sketch[:, t])

        if step == None:
            # Adaptive step: projection of residual onto column A_t
            norm_sq = np.dot(At, At)
            alpha = np.dot(r, At) / norm_sq if norm_sq != 0 else 0.0
        else:
            alpha = step

        # Update residual
        r -= alpha * At
        coeffs[t] = coeffs.get(t, 0.0) + alpha

    return np.array([[idx, coeff] for idx, coeff in coeffs.items()])


def matchingpursuit_abstract(
    marginals,
    dit_constraints,
    dit_string_length,
    iteration_number,
    step=None,
    interaction_size=2,
    dit_dimension=2,
    optimizer=None,
):
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
    optimizer : Optimizer, optional
        Instantiated optimizer implementing an ``optimize`` method.
        If None, a default spin-chain nearest-neighbor optimizer is instantiated.
        The matching pursuit context (constraints, dimensions) is provided automatically.

    Returns
    -------
    numpy array
        A 2D numpy array where each row contains the index of a selected column from the sketch matrix and its corresponding coefficient in the sparse solution.
    """

    if optimizer is None:
        optimizer = optimizer_api.get_optimizer("spin_chain_nn_max")
    elif not hasattr(optimizer, "optimize"):
        raise TypeError("optimizer must implement an optimize(*args, **kwargs) method.")

    #Matching Pursuit, initialization
    r = copy.deepcopy(marginals)
    coeffs = {}

    for _ in range(iteration_number):

        #Find best matching atom
        t = optimizer.optimize(
            r,
            dit_constraints=dit_constraints,
            dit_string_length=dit_string_length,
            interaction_size=interaction_size,
            dit_dimension=dit_dimension,
            bit_constraints=dit_constraints,
            bit_string_length=dit_string_length,
        )

        At = ab.reconstruct_structured_matrix_column(t, dit_constraints, dit_string_length, dit_dimension) #column of the sketch matrix corresponding to the configuration conf

        if step == None:
            # Adaptive step: projection of residual onto column A_t
            norm_sq = np.dot(At, At)
            alpha = np.dot(r, At) / norm_sq if norm_sq != 0 else 0.0
        else:
            alpha = step

        # Update residual
        r -= alpha * At
        coeffs[t] = coeffs.get(t, 0.0) + alpha

    return np.array([[idx, coeff] for idx, coeff in coeffs.items()])
