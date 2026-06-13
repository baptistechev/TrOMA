from __future__ import annotations

from typing import Any

import numpy as np

import neal
import scipy.optimize as sk_opt
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qamomile.optimization.qaoa import QAOAConverter
from qamomile.optimization.aoa import AOAConverter
from qamomile.qiskit import QiskitTranspiler

from .qamomile_addon import IBMRuntimeExecutor, AerLocalExecutor
from ..problem_sketch import ProblemSketch, RestrictedProblemSketch
from ..sketch_map import ConstraintSketchMap
from ..core.structure import DitString, VariationalOptimizationResult
from ..core.embedding import reverse_spectrum_restriction
from .._validation import _Validator


def _extract_circuit_stats(executable: Any, backend: Any) -> tuple[int, int, int | None]:
    """Return (logical_depth, transpiled_depth, transpiled_gate_count).

    ``logical_depth`` is the depth of qamomile's circuit as built (still using
    high-level multi-qubit gates such as ``RZZ``/XY-mixer rotations, which
    ``AerLocalExecutor`` runs natively without decomposition). The other two
    figures are obtained by running the same circuit through
    ``generate_preset_pass_manager(optimization_level=3, backend=backend)`` —
    the pass manager used by the qiskit-native optimizers in
    :mod:`troma.optimization.quantum_native` — so depth/gate-count are
    comparable across both code paths. This transpilation is only used for
    reporting; the circuit actually executed is unchanged.
    """
    circuit = None
    if hasattr(executable, "get_first_circuit"):
        circuit = executable.get_first_circuit()
    elif hasattr(executable, "quantum_circuit"):
        circuit = executable.quantum_circuit

    if circuit is None or not hasattr(circuit, "depth"):
        return 0, 0, None

    logical_depth = int(circuit.depth())

    try:
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
        isa_circuit = pm.run(circuit)
        return logical_depth, int(isa_circuit.depth()), int(isa_circuit.size())
    except Exception:
        return logical_depth, logical_depth, None


def _extract_solver_steps(result: Any) -> int:
    steps = getattr(result, "nit", None)
    if steps is not None:
        return int(steps)

    evaluations = getattr(result, "nfev", None)
    if evaluations is not None:
        return int(evaluations)

    return 0


def digital_annealing(problem_sketch: ProblemSketch, number_iter: int = 1000) -> int:
    """
    Perform digital annealing to find a solution to the optimization problem defined by the marginals.
    Only work for QUBO problems, i.e., when marginals are defined on nearest neighbor pairs of bits.

    Parameters
    ----------
    problem_sketch : ProblemSketch
        Problem sketch containing nearest-neighbor binary marginals.
    number_iter : int, optional
        The number of iterations for the digital annealing algorithm. The default is 1000.

    Returns
    -------
    int
        The index of the dit string that maximizes the sum of the marginals.
    """
    number_iter = _Validator.ensure_int("number_iter", number_iter, min_value=1)
    if problem_sketch.sketch_values is None:
        raise ValueError("problem_sketch.sketch_values must be defined before optimization.")

    marginals = list(problem_sketch.sketch_values)
    bit_string_length = int(
        problem_sketch.restricted_problem_size
        if isinstance(problem_sketch, RestrictedProblemSketch)
        else problem_sketch.problem_size
    )

    problem_dimension = (
        problem_sketch.restricted_problem_dimension
        if isinstance(problem_sketch, RestrictedProblemSketch)
        else problem_sketch.problem_dimension
    )
    if problem_dimension != 2:
        raise ValueError("digital_annealing currently supports binary (dimension=2) problems only.")
    if len(marginals) == 0 or len(marginals) % 4 != 0:
        raise ValueError("marginals length must be a positive multiple of 4 for nearest-neighbor QUBO marginals.")

    #Define spin-chain Ising Hamiltonian from the problem sketch
    n = len(marginals)//4 + 1
    if n != bit_string_length:
        raise ValueError("marginals are inconsistent with problem_sketch size for nearest-neighbor QUBO.")
    H = problem_sketch.to_hamiltonian()

    #Convert the Hamiltonian to the format required by the neal library
    h = {i: - H.terms.get((i,), 0.0) for i in range(n)}
    J = {(i, i+1): - H.terms.get((i, i+1), 0.0) for i in range(n-1)}

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_ising(h, J, num_reads=number_iter)
    config = [int((1-j)/2) for i,j in sampleset.first.sample.items()]
    return DitString(config, dimension=2).to_integer()


def _validate_variational_inputs(
    problem_sketch: ProblemSketch,
    number_layers: int,
    number_shots: int,
    method: str,
    optimizer_options: dict | None,
    sampler_options: dict | None,
) -> tuple[list, int, int, int]:
    if problem_sketch.sketch_values is None:
        raise ValueError("problem_sketch.sketch_values must be defined before optimization.")

    marginals = list(problem_sketch.sketch_values)
    bit_string_length = int(
        problem_sketch.restricted_problem_size
        if isinstance(problem_sketch, RestrictedProblemSketch)
        else problem_sketch.problem_size
    )
    bit_constraints = problem_sketch.sketch_map.map
    _Validator.ensure_instance("problem_sketch.sketch_map", problem_sketch.sketch_map, ConstraintSketchMap)
    if len(bit_constraints) == 0:
        raise ValueError("bit_constraints must be non-empty.")
    if len(marginals) != len(bit_constraints):
        raise ValueError("marginals and bit_constraints must have the same length.")
    bit_string_length = _Validator.ensure_int("bit_string_length", bit_string_length, min_value=1)
    number_layers = _Validator.ensure_int("number_layers", number_layers, min_value=1)
    number_shots = _Validator.ensure_int("number_shots", number_shots, min_value=1)
    _Validator.ensure_str("method", method)
    _Validator.ensure_optional_dict("optimizer_options", optimizer_options)
    _Validator.ensure_optional_dict("sampler_options", sampler_options)
    return marginals, bit_string_length, number_layers, number_shots


def _build_runtime_sampler(
    backend: Any,
    number_shots: int,
    sampler_options: dict | None,
) -> Any:
    """Build a RuntimeSamplerV2 for IBM Quantum hardware / cloud backends."""
    sampler_options_dict = dict(sampler_options or {})
    max_execution_time = sampler_options_dict.get("max_execution_time")
    if max_execution_time is not None:
        max_execution_time = _Validator.ensure_int(
            "sampler_options['max_execution_time']",
            max_execution_time,
            min_value=1,
        )

    runtime_options = {"default_shots": number_shots}
    runtime_options.update(sampler_options_dict)

    sampler = RuntimeSamplerV2(mode=backend, options=runtime_options)
    sampler.options.default_shots = number_shots
    if max_execution_time is not None:
        sampler.options.max_execution_time = max_execution_time
    return sampler


def _build_executor(
    backend: Any | None,
    number_shots: int,
    sampler_options: dict | None,
    verbose: bool = False,
) -> tuple[Any, Any]:
    """Return (executor, backend), choosing AerLocalExecutor for local AerSimulator
    backends and IBMRuntimeExecutor (via SamplerV2) for everything else."""
    if backend is None or isinstance(backend, AerSimulator):
        if backend is None:
            backend = AerSimulator()
        return AerLocalExecutor(backend, verbose=verbose), backend

    sampler = _build_runtime_sampler(backend, number_shots, sampler_options)
    return IBMRuntimeExecutor(sampler, backend), backend


def _select_best_by_real_cost(sample_set: Any, problem_sketch: ProblemSketch) -> tuple[int, int]:
    best_val: float | None = None
    best_index: int | None = None
    n_evaluations = 0

    for sample in sample_set.samples:
        dit_string = DitString(list(sample.values()))

        if isinstance(problem_sketch, RestrictedProblemSketch):
            full = reverse_spectrum_restriction(
                [dit_string],
                original_size=problem_sketch.problem_size,
                dit_restrictions=problem_sketch.restriction.dit_restrictions,
                dit_value_restrictions=problem_sketch.restriction.dit_value_restrictions,
                additional_dits_val=problem_sketch.restriction.additional_dits_val,
            )
            eval_str = full[0]
        else:
            eval_str = dit_string

        val = float(problem_sketch.objective_function(np.asarray(eval_str)))
        n_evaluations += 1
        if best_val is None or val > best_val:
            best_val = val
            best_index = dit_string.to_integer('R')

    return best_index, n_evaluations


def _run_variational(
    converter: QAOAConverter,
    executable,
    my_executor: IBMRuntimeExecutor,
    number_layers: int,
    number_shots: int,
    method: str,
    optimizer_options: dict | None,
    x0: np.ndarray | None = None,
    return_metadata: bool = False,
    problem_sketch: ProblemSketch | None = None,
    backend: Any | None = None,
) -> int:
    feasibility_fn = getattr(problem_sketch, "feasibility_function", None)

    def cost_fn(params):
        gammas = list(params[:number_layers])
        betas = list(params[number_layers:])
        job = executable.sample(
            my_executor,
            shots=number_shots,
            bindings={"gammas": gammas, "betas": betas},
        )
        result = job.result()
        decoded = converter.decode_to_binary_sampleset(result)

        if feasibility_fn is not None:
            feasible_energies, feasible_occurrences = [], []
            for s, e, o in zip(decoded.samples, decoded.energy, decoded.num_occurrences):
                if feasibility_fn(np.array(list(s.values()))):
                    feasible_energies.append(e)
                    feasible_occurrences.append(o)
            if feasible_energies:
                e_arr = np.array(feasible_energies)
                o_arr = np.array(feasible_occurrences)
                return -float(e_arr @ o_arr / o_arr.sum())
            # All shots leaked out of feasible subspace — return neutral signal
            return 0.0

        return -decoded.energy_mean()

    number_parameters = 2 * number_layers
    bounds = np.array([[-np.pi, np.pi]] * number_parameters, dtype=float)
    res = sk_opt.minimize(
        cost_fn,
        x0=np.ones(number_parameters) if x0 is None else x0,
        bounds=bounds,
        method=method,
        options=dict(optimizer_options or {}),
    )

    final_parameters = np.asarray(res.x, dtype=float)
    gammas_opt = list(res.x[:number_layers])
    betas_opt = list(res.x[number_layers:])
    sample_result = executable.sample(
        my_executor,
        shots=number_shots,
        bindings={"gammas": gammas_opt, "betas": betas_opt},
    ).result()
    sample_set = converter.decode(sample_result)

    num_occurrences = getattr(sample_set, 'num_occurrences', None)
    final_sample_distribution: dict[int, int] = {}
    for i, sample in enumerate(sample_set.samples):
        idx = DitString(list(sample.values())).to_integer('R')
        count = int(num_occurrences[i]) if num_occurrences is not None else 1
        final_sample_distribution[idx] = final_sample_distribution.get(idx, 0) + count

    if problem_sketch is not None and hasattr(problem_sketch, 'objective_function'):
        best_index, truth_evals = _select_best_by_real_cost(sample_set, problem_sketch)
    else:
        max_idx = sample_set.energy.index(max(sample_set.energy))
        best = sample_set.samples[max_idx]
        best_index = DitString(best.values()).to_integer('R')
        truth_evals = 0

    if not return_metadata:
        return best_index

    circuit_depth, transpiled_circuit_depth, transpiled_gate_count = _extract_circuit_stats(
        executable, backend
    )

    return VariationalOptimizationResult(
        best_index,
        final_parameters=final_parameters,
        gammas=gammas_opt,
        betas=betas_opt,
        number_layers=number_layers,
        circuit_depth=circuit_depth,
        transpiled_circuit_depth=transpiled_circuit_depth,
        transpiled_gate_count=transpiled_gate_count,
        solver_steps=_extract_solver_steps(res),
        objective_evaluations=int(getattr(res, "nfev", _extract_solver_steps(res))),
        truth_objective_evaluations=truth_evals,
        final_sample_distribution=final_sample_distribution,
    )


def QAOA(
    problem_sketch: ProblemSketch,
    number_layers: int = 4,
    method: str = "COBYLA",
    backend: Any | None = None,
    number_shots: int = 4096,
    optimizer_options: dict | None = None,
    sampler_options: dict | None = None,
    verbose: bool = False,
    pretrain: bool = False,
    pretrain_options: dict | None = None,
    x0: np.ndarray | None = None,
    return_metadata: bool = False,
) -> int:
    """Optimize a sketch with QAOA.

    By default this returns a plain ``int``. If ``return_metadata=True``, the
    return value is an int-compatible object exposing ``final_parameters``,
    ``gammas``, ``betas``, ``circuit_depth``, ``solver_steps``, and
    ``objective_evaluations``.
    """
    _, _, number_layers, number_shots = _validate_variational_inputs(
        problem_sketch, number_layers, number_shots, method, optimizer_options, sampler_options
    )

    my_executor, backend = _build_executor(backend, number_shots, sampler_options, verbose=verbose)

    hubo_model = problem_sketch.to_hubo()

    converter = QAOAConverter(hubo_model)
    converter.spin_model = converter.spin_model.normalize_by_abs_max()

    executable = converter.transpile(QiskitTranspiler(), p=number_layers)

    if x0 is None and pretrain:
        from ._quantum_pre_training import pretrain_qaoa_parameters
        hamiltonian = problem_sketch.to_hamiltonian()
        pretrain_opts = {k: v for k, v in (pretrain_options or {}).items() if k != "verbose"}
        x0 = pretrain_qaoa_parameters(hamiltonian, number_layers, verbose=verbose, **pretrain_opts)

    return _run_variational(
        converter,
        executable,
        my_executor,
        number_layers,
        number_shots,
        method,
        optimizer_options,
        x0=x0,
        return_metadata=return_metadata,
        problem_sketch=problem_sketch,
        backend=backend,
    )


def AOA(
    problem_sketch: ProblemSketch,
    number_layers: int = 4,
    method: str = "COBYLA",
    backend: Any | None = None,
    number_shots: int = 4096,
    initial_state: str = "dicke",
    hamming_weight: int = 1,
    mixer: str = "ring",
    pair_indices_mixer: np.ndarray | None = None,
    block_size: int | None = None,
    optimizer_options: dict | None = None,
    sampler_options: dict | None = None,
    verbose: bool = False,
    pretrain: bool = False,
    pretrain_options: dict | None = None,
    x0: np.ndarray | None = None,
    return_metadata: bool = False,
) -> int:
    """Perform optimization using the Adaptive Optimization Algorithm (AOA) from the Qamomile library.
     See https://arxiv.org/abs/2211.13227 for more details on the algorithm and its implementation.

    Parameters
    ----------
    problem_sketch : ProblemSketch
        The problem sketch to optimize.
    number_layers : int, optional
        The number of layers (p) for the AOA ansatz. Defaults to 4
    method : str, optional
        The classical optimization method to use for optimizing the AOA parameters. Defaults to "COBYLA".
    backend : Any, optional
        The backend to use for the quantum execution. Defaults to None, which uses the AerSimulator.
    number_shots : int, optional
        The number of shots to use for each quantum execution. Defaults to 4096.
    initial_state : str, optional
        The type of initial state to use for the AOA ansatz. Defaults to "dicke". Other options include "uniform" and "zero".
    hamming_weight : int, optional
        The Hamming weight of the states in the initial state superposition. Only relevant if initial_state is "dicke". Defaults to 1.
    mixer : str, optional
        The type of mixer to use for the AOA ansatz. Defaults to "ring". Other options is "fully-connected".
    pair_indices_mixer : np.ndarray, optional
        An array of shape (n_pairs, 2) specifying the pairs of qubits to apply the mixer Hamiltonian if mixer is None.
    block_size : int, optional
        The block size on which we apply the state prepatation and the XY mixer.
    optimizer_options : dict, optional
        Additional options to pass to the classical optimizer. Defaults to None.
    sampler_options : dict, optional
        Additional options to pass to the quantum sampler (e.g. max_execution_time for RuntimeSampler backends). Defaults to None.
    verbose : bool, optional
        Whether to print additional information during the optimization process. Defaults to False.
    x0 : np.ndarray, optional
        Initial parameters for the classical optimizer (gammas followed by betas, length 2 * number_layers).
        Takes precedence over pretrain if both are provided. Defaults to None, which uses all-ones.
    return_metadata : bool, optional
        If True, return an int-compatible result object exposing optimization
        metadata. If False, return only the best index. Defaults to False.

    Returns
    -------
    int | VariationalOptimizationResult
        The best dit-string index. When ``return_metadata=True``, metadata is
        available through ``final_parameters``, ``gammas``, ``betas``,
        ``circuit_depth``, ``solver_steps``, and ``objective_evaluations``.
    """
    _, _, number_layers, number_shots = _validate_variational_inputs(
        problem_sketch, number_layers, number_shots, method, optimizer_options, sampler_options
    )
    hamming_weight = _Validator.ensure_int("hamming_weight", hamming_weight, min_value=1)
    _Validator.ensure_str("initial_state", initial_state)
    _Validator.ensure_str("mixer", mixer)

    my_executor, backend = _build_executor(backend, number_shots, sampler_options, verbose=verbose)

    hubo_model = problem_sketch.to_hubo()

    converter = AOAConverter(hubo_model)
    converter.spin_model = converter.spin_model.normalize_by_abs_max()

    executable = converter.transpile(
        QiskitTranspiler(),
        p=number_layers,
        initial_state=initial_state,
        hamming_weight=hamming_weight,
        mixer=mixer,
        pair_indices_mixer=pair_indices_mixer,
        block_size=block_size,
    )

    if x0 is None and pretrain:
        from ._quantum_pre_training import pretrain_qaoa_parameters
        hamiltonian = problem_sketch.to_hamiltonian()
        pretrain_opts = {k: v for k, v in (pretrain_options or {}).items() if k != "verbose"}
        x0 = pretrain_qaoa_parameters(hamiltonian, number_layers, verbose=verbose, **pretrain_opts)

    return _run_variational(
        converter,
        executable,
        my_executor,
        number_layers,
        number_shots,
        method,
        optimizer_options,
        x0=x0,
        return_metadata=return_metadata,
        problem_sketch=problem_sketch,
        backend=backend,
    )
