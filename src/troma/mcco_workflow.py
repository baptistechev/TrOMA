from . import sketchs
from .modeling import mcco_modeling
from .decoding.matching_pursuit import matching_pursuit
from .optimization.optimizer import get_optimizer


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
    """Run the full MCCO + compressive-sensing resolution pipeline.

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