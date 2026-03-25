import numpy as np
import copy
import scipy.optimize as scipy_opt

import data_structure as ds
import abstract as ab
from utils import array_to_column_vector

def brute_force_max(marginals,sketch):
    """
    Brute force method to find the value of the dit string that maximizes the sum of the marginals. This method is not efficient for large dit string lengths, but it can be used as a baseline for testing other optimization methods.

    Parameters
    ----------
    marginals : list of float
        The marginals of the function defined on the full spectrum of dit strings. The order of
        the values should correspond to the order of the dit strings in the full spectrum.
    sketch : 2D numpy array
        The sketch matrix, where each column corresponds to a dit string and each row corresponds to a
        marginal.
    
    Returns
    -------
    int
        The index of the dit string that maximizes the sum of the marginals.
    """

    return np.argmax(sketch.T * array_to_column_vector(marginals))

def spin_chain_nn_max(marginals, dit_string_length, interaction_size=2, dit_dimension=2):
    """
    Find the value of the spin chain of length dit_string_length that maximizes the sum of the nearest-neighbor uplets marginals.
    Marginals can contained negative values.
    Spins are of dimension dit_dimension with value from 0 to dit_dimension-1.
    
    Parameters
    ----------
    marginals : list of float
        The marginals of the function defined on the full spectrum of dit strings. The order of
        the values should correspond to the order of the dit strings in the full spectrum.
    dit_string_length : int
        The length of the dit string (i.e., the number of spins in the chain).
    interaction_size : int, optional
        The size of the nearest neighbor pattern. The default is 2.
    dit_dimension : int, optional
        The dimension of the dits (i.e., the number of possible values each dit can take
        ). The values of the dits should be from 0 to dit_dimension-1.
    
    Returns
    -------
    int
        The integer index of the optimal dit string configuration.
    """

    # Number of local interaction windows in the chain.
    n_windows = dit_string_length - interaction_size + 1
    # Reshape flat marginals into a tensor indexed as:
    # (window, spin_1, ..., spin_{interaction_size}).
    marginal_tensor_shape = (n_windows,) + (dit_dimension,) * interaction_size
    marginal_tensor = np.array(marginals).reshape(marginal_tensor_shape)

    if interaction_size == 1:
        spin_chain = np.argmax(
            marginal_tensor.reshape(dit_string_length, dit_dimension),
            axis=1,
        ).tolist()
        return ds.dit_string_to_integer(spin_chain, dit_dimension=dit_dimension)

    # DP state = last (interaction_size - 1) spins.
    state_size = interaction_size - 1
    n_states = dit_dimension ** state_size

    states = [
        tuple(ds.integer_to_dit_string(i, state_size, dit_dimension=dit_dimension))
        for i in range(n_states)
    ]

    # Best cumulative energy ending in each state.
    best_energy_by_state = {state: 0.0 for state in states}
    # For ties on energy, keep the smallest global prefix index.
    best_index_by_state = {
        state: ds.dit_string_to_integer(state, dit_dimension=dit_dimension)
        for state in states
    }
    # Backpointers per window to reconstruct the optimal chain.
    predecessors_by_window = []

    # Forward dynamic programming pass.
    for window_idx in range(n_windows):
        next_best_energy = {}
        next_best_index = {}
        next_predecessor = {}

        for prev_state, prev_energy in best_energy_by_state.items():
            for next_spin in range(dit_dimension):
                next_state = prev_state[1:] + (next_spin,)
                local_energy = marginal_tensor[(window_idx,) + prev_state + (next_spin,)]
                candidate_energy = prev_energy + local_energy
                candidate_index = best_index_by_state[prev_state] * dit_dimension + next_spin

                if (
                    (next_state not in next_best_energy)
                    or (candidate_energy > next_best_energy[next_state])
                    or (
                        candidate_energy == next_best_energy[next_state]
                        and candidate_index < next_best_index[next_state]
                    )
                ):
                    next_best_energy[next_state] = candidate_energy
                    next_best_index[next_state] = candidate_index
                    next_predecessor[next_state] = prev_state

        best_energy_by_state = next_best_energy
        best_index_by_state = next_best_index
        predecessors_by_window.append(next_predecessor)

    # Best final state with deterministic tie-break on global index.
    final_state = max(
        best_energy_by_state,
        key=lambda state: (best_energy_by_state[state], -best_index_by_state[state]),
    )

    state_path = [None] * (n_windows + 1)
    state_path[-1] = final_state

    # Backward pass through stored predecessors.
    for window_idx in range(n_windows, 0, -1):
        state_path[window_idx - 1] = predecessors_by_window[window_idx - 1][state_path[window_idx]]

    # Convert overlapping states into the full spin chain.
    spin_chain = list(state_path[0])
    for window_idx in range(1, n_windows + 1):
        spin_chain.append(int(state_path[window_idx][-1]))

    return ds.dit_string_to_integer(spin_chain, dit_dimension=dit_dimension)

def dual_annealing(marginals, dit_constraints, dit_string_length, dit_dimension=2, opt_func_kwargs=None):
    """
    Use simulated annealing to find the value of the dit string that maximizes the sum of the marginals. This method is more efficient than brute force for large dit string lengths, but it may not always find the optimal solution.

    Parameters
    ----------
    marginals : list of float
        The marginals of the function defined on the full spectrum of dit strings. The order of
        the values should correspond to the order of the dit strings in the full spectrum.
    dit_string_length : int
        The length of the dit strings.
    dit_dimension : int
        The dimension of the dits, i.e., the number of possible values each dit can take.
    opt_func_kwargs : dict, optional
        Keyword arguments passed to dual_annealing. The default is None.
    
    Returns
    -------
    int
        The index of the dit string that maximizes the sum of the marginals.
    """

    bounds = [(0, dit_dimension**dit_string_length - 1)]

    def objective_function(x):
        config_index = int(np.round(x[0]))
        return float(np.dot(- np.asarray(marginals), ab.reconstruct_structured_matrix_column(config_index, dit_constraints=dit_constraints, dit_string_length=dit_string_length, dit_dimension=dit_dimension)))

    result = scipy_opt.dual_annealing(objective_function, bounds, **(opt_func_kwargs or {}))
    return int(np.round(result.x[0]))

def simulated_annealing(marginals, dit_constraints, dit_string_length, dit_dimension=2, max_iter=1000, T0=1.0, alpha=0.99):

    def simulated_annealing_binary(f, n):
        # initial random bitstring
        x = np.random.randint(0, dit_dimension, size=n)
        fx = f(x)

        best_x = copy.deepcopy(x)
        best_fx = fx

        T = T0

        for i in range(max_iter):
            # propose neighbor: flip one bit
            x_new = copy.deepcopy(x)
            idx = np.random.randint(n)
            x_new[idx] ^= 1  # flip bit

            f_new = f(x_new)

            # acceptance rule
            if f_new < fx or np.random.rand() < np.exp((fx - f_new) / T):
                x, fx = x_new, f_new

                if fx < best_fx:
                    best_x, best_fx = copy.deepcopy(x), fx

            # cooling
            T *= alpha

        return best_x, best_fx
    
    def objective_function(config):
        config_index = ds.dit_string_to_integer(config, dit_dimension=dit_dimension)
        return float(np.dot(- np.asarray(marginals), ab.reconstruct_structured_matrix_column(config_index, dit_constraints=dit_constraints, dit_string_length=dit_string_length, dit_dimension=dit_dimension)))
    
    best_config, _ = simulated_annealing_binary(objective_function, dit_string_length)
    return ds.dit_string_to_integer(best_config, dit_dimension=dit_dimension)