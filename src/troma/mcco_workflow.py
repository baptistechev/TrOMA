from . import sketchs
from .modeling import mcco_modeling
from .decoding.matching_pursuit import matching_pursuit
from .optimization.optimizer import get_optimizer
from ._validation import ensure_callable as _ensure_callable, ensure_int as _ensure_int


def solve_via_mcco(
    objective_function,
    number_samples,
    dit_string_length,
    interaction_size,
    iteration_number=5,
    step=None,
    thereshold_parameter="Auto",
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
    thereshold_parameter : float, str or None, optional
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
        ``spectrum_pos``, ``spectrum_val``, ``spectrum_bin``,
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
    if thereshold_parameter is not None and thereshold_parameter != "Auto" and not isinstance(thereshold_parameter, (int, float)):
        raise TypeError("thereshold_parameter must be a real number, 'Auto', or None.")
    if optimizer is not None and not hasattr(optimizer, "optimize"):
        raise TypeError("optimizer must implement an optimize(*args, **kwargs) method.")
    if optimizer is None and optimizer_name is not None and not isinstance(optimizer_name, str):
        raise TypeError("optimizer_name must be a string or None.")

    spectrum_pos, spectrum_val, spectrum_bin = mcco_modeling(
        objective_function,
        number_samples,
        dit_string_length,
        thereshold_parameter=thereshold_parameter,
        dit_dimension=dit_dimension,
    )

    sketch_builder = sketchs.ConstraintSketch
    constraints = sketch_builder.build_nearest_neighbors_sketch(
        dit_string_length,
        interaction_size,
        dit_dimension,
    )
    y = sketch_builder.compute_marginal((spectrum_bin, spectrum_val), constraints)

    if optimizer is None and optimizer_name is not None:
        optimizer = get_optimizer(optimizer_name)

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

    return {
        "spectrum_pos": spectrum_pos,
        "spectrum_val": spectrum_val,
        "spectrum_bin": spectrum_bin,
        "constraints": constraints,
        "y": y,
        "solution": solution,
    }