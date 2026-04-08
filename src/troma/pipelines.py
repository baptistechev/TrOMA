import numpy as np

from . import data_structure as ds
from . import sketchs
from .modeling import mcco_modeling, restricted_mcco_modeling
from .decoding.matching_pursuit import matching_pursuit
from .optimization.optimizer import get_optimizer
from ._validation import (
    ensure_callable as _ensure_callable,
    ensure_int as _ensure_int,
    ensure_iterable as _ensure_iterable,
    ensure_unique_items as _ensure_unique_items,
)
from .embedding import reverse_spectrum_restriction


def _validate_dit_restrictions(dit_restrictions, dit_string_length):
    _ensure_iterable("dit_restrictions", dit_restrictions)
    dit_restrictions = [
        _ensure_int("dit_restrictions item", index, min_value=0, max_value=dit_string_length - 1)
        for index in dit_restrictions
    ]
    if not dit_restrictions:
        raise ValueError("dit_restrictions must contain at least one index.")
    _ensure_unique_items("dit_restrictions", dit_restrictions)
    return dit_restrictions


def _validate_dit_value_restrictions(dit_value_restrictions, dit_dimension):
    _ensure_iterable("dit_value_restrictions", dit_value_restrictions)
    dit_value_restrictions = [
        _ensure_int("dit_value_restrictions item", value, min_value=0, max_value=dit_dimension - 1)
        for value in dit_value_restrictions
    ]
    if len(dit_value_restrictions) < 2:
        raise ValueError("dit_value_restrictions must contain at least two values.")
    _ensure_unique_items("dit_value_restrictions", dit_value_restrictions)
    return dit_value_restrictions


def _validate_additional_dits_val(additional_dits_val, dit_dimension):
    if not isinstance(additional_dits_val, int):
        raise TypeError("additional_dits_val must be an integer.")
    if additional_dits_val < 0 or additional_dits_val >= dit_dimension:
        raise ValueError(
            "additional_dits_val must be between 0 and dit_dimension - 1."
        )
    return additional_dits_val


def _get_optimizer(optimizer, optimizer_name):
    if optimizer is not None:
        if not hasattr(optimizer, "optimize"):
            raise TypeError("optimizer must implement an optimize(*args, **kwargs) method.")
        return optimizer
    if optimizer_name is not None and not isinstance(optimizer_name, str):
        raise TypeError("optimizer_name must be a string or None.")
    return get_optimizer(optimizer_name) if optimizer_name is not None else None


def _run_mcco_matching_pursuit(
    spectrum_dits,
    spectrum_val,
    dit_string_length,
    interaction_size,
    dit_dimension,
    iteration_number,
    step,
    optimizer,
    optimizer_name,
):
    constraints = sketchs.ConstraintSketch.build_nearest_neighbors_sketch(
        dit_string_length,
        interaction_size,
        dit_dimension,
    )
    y = sketchs.ConstraintSketch.compute_marginal((spectrum_dits, spectrum_val), constraints)

    optimizer = _get_optimizer(optimizer, optimizer_name)
    solution = matching_pursuit(
        "abstract",
        y,
        constraints,
        dit_string_length,
        iteration_number=iteration_number,
        step=step,
        interaction_size=interaction_size,
        dit_dimension=dit_dimension,
        optimizer=optimizer,
    )

    solution_pos = solution[:, 0].astype(int) if solution.size > 0 else np.array([], dtype=int)
    solution_val = solution[:, 1] if solution.size > 0 else np.array([], dtype=solution.dtype)
    return constraints, y, solution_pos, solution_val


def solve_via_mcco(
    objective_function,
    number_samples,
    dit_string_length,
    interaction_size,
    iteration_number=5,
    step=None,
    threshold_parameter="Auto",
    dit_dimension=2,
    optimizer=None,
    optimizer_name="spin_chain_nn_max",
):
    """Run the full MCCO pipeline : modeling + sketching + matching pursuit.

    This helper orchestrates:
    1) ``mcco_modeling`` to build a sparse problem representation,
    2) constraint sketch construction and marginal computation,
    3) abstract matching pursuit reconstruction.

    Parameters
    ----------
    objective_function : Callable
        Function evaluated on dit strings.
    number_samples : int
        Number of sampled configurations used by MCCO.
    dit_string_length : int
        Number of dits/spins.
    interaction_size : int
        Local interaction size for nearest-neighbor constraints.
    iteration_number : int, optional
        Number of matching-pursuit iterations.
    step : float or None, optional
        Matching-pursuit step size. If ``None``, adaptive step is used.
    threshold_parameter : float, str or None, optional
        Threshold applied in ``mcco_modeling`` (``"Auto"`` uses 90th percentile).
    dit_dimension : int, optional
        Dit alphabet size.
    optimizer : Optimizer or None, optional
        Instantiated optimizer object. If ``None``, ``optimizer_name`` is used.
    optimizer_name : str or None, optional
        Name passed to ``get_optimizer`` when ``optimizer`` is not provided.

    Returns
    -------
    dict
        Pipeline outputs with keys:
        ``spectrum_pos``, ``spectrum_val``, ``spectrum_dits``,
        ``constraints``, ``y``, ``solution``.
    """
    _ensure_callable("objective_function", objective_function)
    number_samples = _ensure_int("number_samples", number_samples, min_value=1)
    dit_string_length = _ensure_int("dit_string_length", dit_string_length, min_value=1)
    interaction_size = _ensure_int("interaction_size", interaction_size, min_value=1)
    iteration_number = _ensure_int("iteration_number", iteration_number, min_value=1)
    dit_dimension = _ensure_int("dit_dimension", dit_dimension, min_value=2)

    if interaction_size > dit_string_length:
        raise ValueError("interaction_size must be <= dit_string_length.")
    if step is not None and not isinstance(step, (int, float)):
        raise TypeError("step must be a real number or None.")
    if threshold_parameter is not None and threshold_parameter != "Auto" and not isinstance(threshold_parameter, (int, float)):
        raise TypeError("threshold_parameter must be a real number, 'Auto', or None.")
    if optimizer is not None and not hasattr(optimizer, "optimize"):
        raise TypeError("optimizer must implement an optimize(*args, **kwargs) method.")
    if optimizer is None and optimizer_name is not None and not isinstance(optimizer_name, str):
        raise TypeError("optimizer_name must be a string or None.")

    spectrum_pos, spectrum_val, spectrum_dits = mcco_modeling(
        objective_function,
        number_samples,
        dit_string_length,
        threshold_parameter=threshold_parameter,
        dit_dimension=dit_dimension,
    )

    constraints, y, solution_pos, solution_val = _run_mcco_matching_pursuit(
        spectrum_dits,
        spectrum_val,
        dit_string_length,
        interaction_size,
        dit_dimension,
        iteration_number,
        step,
        optimizer,
        optimizer_name,
    )

    return {
        "spectrum_pos": spectrum_pos,
        "spectrum_val": spectrum_val,
        "spectrum_dits": spectrum_dits,
        "constraints": constraints,
        "y": y,
        "solution_pos": solution_pos,
        "solution_val": solution_val,
    }

def embedding_and_solve_via_mcco(
    objective_function,
    dit_string_length,
    dit_restrictions,
    dit_value_restrictions,
    number_samples,
    interaction_size,
    additional_dits_val=0,
    iteration_number=5,
    step=None,
    threshold_parameter="Auto",
    dit_dimension=2,
    optimizer=None,
    optimizer_name="spin_chain_nn_max",
):
    """Restrict an embedded problem, solve it with MCCO, and lift results back.

    This helper builds a reduced problem by keeping only the dits listed in
    ``dit_restrictions`` and allowing only the values listed in
    ``dit_value_restrictions``. It then runs :func:`solve_via_mcco` on that
    reduced space and maps the recovered spectrum and solution indices back to
    the original dit space with :func:`reverse_spectrum_restriction`.

    Parameters
    ----------
    objective_function : Callable
        Objective defined on the original dit space.
    dit_string_length : int
        Number of dits in the original problem.
    dit_restrictions : iterable of int
        Indices of the original dits kept in the reduced problem.
    dit_value_restrictions : iterable of int
        Original dit values allowed in the reduced problem. Their order defines
        the reduced alphabet mapping ``0..k-1 -> dit_value_restrictions``.
    number_samples : int
        Number of sampled configurations for the reduced problem.
    interaction_size : int
        Local interaction size for the reduced problem.
    iteration_number : int, optional
        Number of matching-pursuit iterations.
    step : float or None, optional
        Matching-pursuit step size. If ``None``, adaptive step is used.
    threshold_parameter : float, str or None, optional
        Threshold applied in ``mcco_modeling``.
    dit_dimension : int, optional
        Dit alphabet size of the original problem.
    optimizer : Optimizer or None, optional
        Instantiated optimizer object. If ``None``, ``optimizer_name`` is used.
    optimizer_name : str or None, optional
        Name passed to ``get_optimizer`` when ``optimizer`` is not provided.

    Returns
    -------
    dict
        Pipeline outputs with keys:
        ``spectrum_val``, ``spectrum_dits``, ``solution_pos``, ``solution_val``.
    """

    _ensure_callable("objective_function", objective_function)
    number_samples = _ensure_int("number_samples", number_samples, min_value=1)
    dit_string_length = _ensure_int("dit_string_length", dit_string_length, min_value=1)
    interaction_size = _ensure_int("interaction_size", interaction_size, min_value=1)
    iteration_number = _ensure_int("iteration_number", iteration_number, min_value=1)
    dit_dimension = _ensure_int("dit_dimension", dit_dimension, min_value=2)
    additional_dits_val = _validate_additional_dits_val(additional_dits_val, dit_dimension)

    if step is not None and not isinstance(step, (int, float)):
        raise TypeError("step must be a real number or None.")
    if threshold_parameter is not None and threshold_parameter != "Auto" and not isinstance(threshold_parameter, (int, float)):
        raise TypeError("threshold_parameter must be a real number, 'Auto', or None.")
    if optimizer is not None and not hasattr(optimizer, "optimize"):
        raise TypeError("optimizer must implement an optimize(*args, **kwargs) method.")
    if optimizer is None and optimizer_name is not None and not isinstance(optimizer_name, str):
        raise TypeError("optimizer_name must be a string or None.")

    if dit_restrictions is not None:
        dit_restrictions = _validate_dit_restrictions(dit_restrictions, dit_string_length)
    if dit_value_restrictions is not None:
        dit_value_restrictions = _validate_dit_value_restrictions(dit_value_restrictions, dit_dimension)

    restricted_space_size = len(dit_restrictions) if dit_restrictions is not None else dit_string_length
    if interaction_size > restricted_space_size:
        raise ValueError("interaction_size must be <= restricted_space_size.")

    spectrum_pos, spectrum_val, spectrum_dits = restricted_mcco_modeling(
        objective_function,
        number_samples,
        dit_string_length,
        threshold_parameter=threshold_parameter,
        dit_dimension=dit_dimension,
        dit_restrictions=dit_restrictions,
        dit_value_restrictions=dit_value_restrictions,
        additional_dits_val=additional_dits_val,
    )

    restricted_space_size = len(dit_restrictions) if dit_restrictions is not None else dit_string_length
    restricted_space_dimension = len(dit_value_restrictions) if dit_value_restrictions is not None else dit_dimension

    constraints, y, solution_pos, solution_val = _run_mcco_matching_pursuit(
        spectrum_dits,
        spectrum_val,
        restricted_space_size,
        interaction_size,
        restricted_space_dimension,
        iteration_number,
        step,
        optimizer,
        optimizer_name,
    )

    solution_dits = [
        ds.integer_to_dit_string(
            pos,
            dit_string_length=restricted_space_size,
            dit_dimension=restricted_space_dimension,
        )
        for pos in solution_pos
    ]

    solution_emb = reverse_spectrum_restriction(
        solution_dits,
        original_size=dit_string_length,
        dit_restrictions=dit_restrictions,
        dit_value_restrictions=dit_value_restrictions,
        additional_dits_val=additional_dits_val
    )

    emb_pos = [ds.dit_string_to_integer(s, dit_dimension=dit_dimension) for s in solution_emb]

    return {
        "spectrum_val": spectrum_val,
        "spectrum_dits": solution_emb,
        "solution_pos": emb_pos,
        "solution_val": solution_val,
    }