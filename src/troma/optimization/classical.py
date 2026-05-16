from __future__ import annotations

import copy
from numbers import Real
from typing import Any

import numpy as np
import scipy.optimize as scipy_opt

from ..sketchs import abstract as ab
from ..core.structure import DitString
from .._validation import ensure_int, ensure_real, ensure_sequence, ensure_optional_dict


def brute_force_max(marginals: list[float] | np.ndarray, sketch: np.ndarray) -> int:
    """
    Brute force maximization: find the dit-string index that maximizes sketch.T @ marginals.

    Parameters
    ----------
    marginals : list of float or np.ndarray
        Marginal values, one per sketch row.
    sketch : np.ndarray
        2D sketch matrix (rows = constraints, columns = dit strings).

    Returns
    -------
    int
        Index of the maximizing dit string.
    """
    if not hasattr(sketch, "T"):
        raise TypeError("sketch must be a 2D array-like matrix.")
    marginals = np.asarray(marginals)
    if hasattr(sketch, "shape") and len(sketch.shape) >= 2 and sketch.shape[0] != marginals.shape[0]:
        raise ValueError("marginals length must match sketch row count.")
    return int(np.argmax(sketch.T @ marginals))


def spin_chain_nn_max(
    marginals: list[float] | np.ndarray,
    dit_string_length: int,
    interaction_size: int = 2,
    dit_dimension: int = 2,
) -> int:
    """
    Dynamic-programming nearest-neighbor maximization on a dit spin chain.

    Parameters
    ----------
    marginals : list of float or np.ndarray
        Flat marginal values ordered by (window, spin_1, ..., spin_k).
    dit_string_length : int
        Number of dits in the chain.
    interaction_size : int, optional
        Size of nearest-neighbor windows. Default is 2.
    dit_dimension : int, optional
        Number of values each dit can take. Default is 2.

    Returns
    -------
    int
        Index of the optimal dit-string configuration.
    """
    marginals = list(marginals)
    dit_string_length = ensure_int("dit_string_length", dit_string_length, min_value=1)
    interaction_size = ensure_int("interaction_size", interaction_size, min_value=1)
    dit_dimension = ensure_int("dit_dimension", dit_dimension, min_value=2)
    if interaction_size > dit_string_length:
        raise ValueError("interaction_size must be <= dit_string_length.")
    expected_marginals = (dit_string_length - interaction_size + 1) * (dit_dimension ** interaction_size)
    if len(marginals) != expected_marginals:
        raise ValueError(
            "marginals length is inconsistent with dit_string_length, interaction_size and dit_dimension."
        )

    n_windows = dit_string_length - interaction_size + 1
    marginal_tensor_shape = (n_windows,) + (dit_dimension,) * interaction_size
    marginal_tensor = np.array(marginals).reshape(marginal_tensor_shape)

    if interaction_size == 1:
        spin_chain = np.argmax(
            marginal_tensor.reshape(dit_string_length, dit_dimension), axis=1
        ).tolist()
        return DitString(spin_chain, dimension=dit_dimension).to_integer()

    state_size = interaction_size - 1
    n_states = dit_dimension ** state_size
    states = [
        tuple(DitString.from_integer(i, state_size, dit_dimension))
        for i in range(n_states)
    ]

    best_energy_by_state: dict = {state: 0.0 for state in states}
    best_index_by_state: dict = {
        state: DitString(list(state), dimension=dit_dimension).to_integer()
        for state in states
    }
    predecessors_by_window: list = []

    for window_idx in range(n_windows):
        next_best_energy: dict = {}
        next_best_index: dict = {}
        next_predecessor: dict = {}

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

    final_state = max(
        best_energy_by_state,
        key=lambda state: (best_energy_by_state[state], -best_index_by_state[state]),
    )

    state_path: list = [None] * (n_windows + 1)
    state_path[-1] = final_state
    for window_idx in range(n_windows, 0, -1):
        state_path[window_idx - 1] = predecessors_by_window[window_idx - 1][state_path[window_idx]]

    spin_chain = list(state_path[0])
    for window_idx in range(1, n_windows + 1):
        spin_chain.append(int(state_path[window_idx][-1]))

    return DitString(spin_chain, dimension=dit_dimension).to_integer()


def dual_annealing(
    marginals: list[float] | np.ndarray,
    dit_constraints: list[dict],
    dit_string_length: int,
    dit_dimension: int = 2,
    opt_func_kwargs: dict | None = None,
) -> int:
    """
    Dual-annealing maximization over the dit-string space.

    Parameters
    ----------
    marginals : list of float or np.ndarray
        Marginal values.
    dit_constraints : list of dict
        Dit constraints.
    dit_string_length : int
        Length of the dit strings.
    dit_dimension : int, optional
        Dit base. Default is 2.
    opt_func_kwargs : dict, optional
        Extra keyword arguments for scipy.optimize.dual_annealing.

    Returns
    -------
    int
        Index of the best dit string found.
    """
    marginals = list(marginals)
    ensure_sequence("dit_constraints", dit_constraints)
    dit_string_length = ensure_int("dit_string_length", dit_string_length, min_value=1)
    dit_dimension = ensure_int("dit_dimension", dit_dimension, min_value=2)
    ensure_optional_dict("opt_func_kwargs", opt_func_kwargs)
    if len(marginals) != len(dit_constraints):
        raise ValueError("marginals and dit_constraints must have the same length.")

    bounds = [(0, dit_dimension**dit_string_length - 1)]

    def objective_function(x: np.ndarray) -> float:
        config_index = int(np.round(x[0]))
        return float(np.dot(
            -np.asarray(marginals),
            ab.reconstruct_structured_matrix_column(
                config_index, dit_constraints=dit_constraints,
                dit_string_length=dit_string_length, dit_dimension=dit_dimension,
            ),
        ))

    result = scipy_opt.dual_annealing(objective_function, bounds, **(opt_func_kwargs or {}))
    return int(np.round(result.x[0]))


def simulated_annealing(
    marginals: list[float] | np.ndarray,
    dit_constraints: list[dict],
    dit_string_length: int,
    dit_dimension: int = 2,
    max_iter: int = 1000,
    T0: float = 1.0,
    alpha: float = 0.99,
) -> int:
    """
    Simulated-annealing maximization over the dit-string space.

    Parameters
    ----------
    marginals : list of float or np.ndarray
        Marginal values.
    dit_constraints : list of dict
        Dit constraints.
    dit_string_length : int
        Length of the dit strings.
    dit_dimension : int, optional
        Dit base. Default is 2.
    max_iter : int, optional
        Maximum iterations. Default is 1000.
    T0 : float, optional
        Initial temperature. Default is 1.0.
    alpha : float, optional
        Cooling rate in (0, 1). Default is 0.99.

    Returns
    -------
    int
        Index of the best dit string found.
    """
    marginals = list(marginals)
    ensure_sequence("dit_constraints", dit_constraints)
    dit_string_length = ensure_int("dit_string_length", dit_string_length, min_value=1)
    dit_dimension = ensure_int("dit_dimension", dit_dimension, min_value=2)
    max_iter = ensure_int("max_iter", max_iter, min_value=1)
    T0 = ensure_real("T0", T0)
    if T0 <= 0:
        raise ValueError("T0 must be > 0.")
    alpha = ensure_real("alpha", alpha)
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1).")
    if len(marginals) != len(dit_constraints):
        raise ValueError("marginals and dit_constraints must have the same length.")

    def _sa_binary(f: Any, n: int) -> tuple[np.ndarray, float]:
        x = np.random.randint(0, dit_dimension, size=n)
        fx = f(x)
        best_x = copy.deepcopy(x)
        best_fx = fx
        T = T0
        for _ in range(max_iter):
            x_new = copy.deepcopy(x)
            idx = np.random.randint(n)
            x_new[idx] ^= 1
            f_new = f(x_new)
            if f_new < fx or np.random.rand() < np.exp((fx - f_new) / T):
                x, fx = x_new, f_new
                if fx < best_fx:
                    best_x, best_fx = copy.deepcopy(x), fx
            T *= alpha
        return best_x, best_fx

    def objective_function(config: np.ndarray) -> float:
        config_index = DitString(list(config), dimension=dit_dimension).to_integer()
        return float(np.dot(
            -np.asarray(marginals),
            ab.reconstruct_structured_matrix_column(
                config_index, dit_constraints=dit_constraints,
                dit_string_length=dit_string_length, dit_dimension=dit_dimension,
            ),
        ))

    best_config, _ = _sa_binary(objective_function, dit_string_length)
    return DitString(list(best_config), dimension=dit_dimension).to_integer()
