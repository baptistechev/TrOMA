from qiskit import transpile

from ._quantum_map import compute_hamiltonian, create_qaoa_circ


# M4 model coefficients fitted on 140 ibm_marrakesh experiments.
# usage_estimation (quantum seconds) = _A0 + _A1*duration_µs + _A2*shots + _A3*duration_µs*shots
# 5-fold CV: R²=0.841, MAE=0.239 s
_A0 = 1.541374
_A1 = 0.020265
_A2 = 0.000150
_A3 = 0.000018


def estimate_matching_pursuit_qpu_cost(
    constraints,
    y,
    bit_string_length: int,
    optimizer,
    matching_pursuit_iterations: int = 1,
    verbose: bool = True,
) -> dict:
    """Estimate the total QPU time for a QAOA-based matching-pursuit run.

    Per-circuit QPU cost is predicted by a linear model (M4) fitted on 140
    ibm_marrakesh experiments (5-fold CV R²=0.841, MAE=0.239 s):

        usage_estimation = A0 + A1·duration_µs + A2·shots + A3·duration_µs·shots

    where ``duration_µs`` is obtained from ``QuantumCircuit.estimate_duration``
    with ``unit="µ"``.

    Parameters
    ----------
    constraints:
        Constraint sketch passed to ``compute_hamiltonian``.
    y:
        Marginals passed to ``compute_hamiltonian``.
    bit_string_length:
        Number of qubits.
    optimizer:
        A bound QAOA optimizer produced by ``bind_optimizer("qaoa", ...)``.
        The following keyword arguments must have been pre-bound:
        ``sampler``, ``number_shots``, ``number_layers``.
        ``optimizer_options["maxiter"]`` is used as the expected number of
        objective evaluations when present.
    matching_pursuit_iterations:
        Number of matching-pursuit iterations. Defaults to 1.

    Returns
    -------
    dict with keys:
        ``circuits_per_qaoa_run``, ``seconds_per_circuit``,
        ``qaoa_run_seconds``, ``total_seconds``.
    """
    kwargs = optimizer._default_kwargs
    sampler = kwargs["sampler"]
    number_shots = kwargs["number_shots"]
    num_layers = kwargs["number_layers"]
    optimizer_options = kwargs.get("optimizer_options") or {}
    expected_objective_evaluations = optimizer_options.get("maxiter", 100)

    backend = getattr(sampler, "mode", None)
    if backend is None:
        backend = getattr(sampler, "_backend", None)
    if backend is None:
        backend_getter = getattr(sampler, "backend", None)
        if callable(backend_getter):
            backend = backend_getter()
    if backend is None:
        raise ValueError("Could not resolve a backend from the bound sampler.")

    ham_data = compute_hamiltonian(constraints, y, bit_string_length=bit_string_length)
    qaoa_circuit = create_qaoa_circ(ham_data, num_qubits=bit_string_length, num_layers=num_layers)
    transpiled = transpile(qaoa_circuit, backend=backend, optimization_level=1)

    duration_us = transpiled.estimate_duration(target=backend.target, unit="µ")
    circuits_per_qaoa_run = expected_objective_evaluations + 1
    seconds_per_circuit = round(_A0 + _A1 * duration_us + _A2 * number_shots + _A3 * duration_us * number_shots)
    seconds_per_circuit = max(0.0, float(seconds_per_circuit))
    qaoa_run_seconds = seconds_per_circuit * circuits_per_qaoa_run
    total_seconds = qaoa_run_seconds * matching_pursuit_iterations

    if verbose:
        print(f"Estimated circuits per QAOA run: {circuits_per_qaoa_run}")
        print(f"  Estimated duration per circuit: {seconds_per_circuit:.3f} s")
        print(f"  Estimated quantum time per QAOA run: {qaoa_run_seconds:.3f} s")
        print(f"  Estimated quantum time for full matching pursuit: {total_seconds:.3f} s")

    return {
        "circuits_per_qaoa_run": circuits_per_qaoa_run,
        "seconds_per_circuit": seconds_per_circuit,
        "qaoa_run_seconds": qaoa_run_seconds,
        "total_seconds": total_seconds,
    }
