from __future__ import annotations

from typing import Any

import numpy as np

import neal
import scipy.optimize as sk_opt
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2
from qiskit import transpile
from qamomile.optimization.qaoa import QAOAConverter
from qamomile.optimization.aoa import AOAConverter
from qamomile.qiskit import QiskitTranspiler

from .qamomile_addon import IBMRuntimeExecutor
from ..problem_sketch import ProblemSketch, RestrictedProblemSketch
from ..sketch_map import ConstraintSketchMap
from ._quantum_map import create_qaoa_circ as _create_qaoa_circ
from ..core.structure import DitString
from .._validation import _Validator


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


def _build_sampler(
    backend: Any | None,
    number_shots: int,
    sampler_options: dict | None,
) -> tuple[SamplerV2, Any]:
    if backend is None:
        backend = AerSimulator()

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

    sampler = SamplerV2(mode=backend, options=runtime_options)
    sampler.options.default_shots = number_shots
    if max_execution_time is not None:
        sampler.options.max_execution_time = max_execution_time
    return sampler, backend


def _run_variational(
    converter: QAOAConverter,
    executable,
    my_executor: IBMRuntimeExecutor,
    number_layers: int,
    number_shots: int,
    method: str,
    optimizer_options: dict | None,
) -> int:
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
        return -decoded.energy_mean()

    number_parameters = 2 * number_layers
    bounds = np.array([[-np.pi, np.pi]] * number_parameters, dtype=float)
    res = sk_opt.minimize(
        cost_fn,
        x0=np.ones(number_parameters),
        bounds=bounds,
        method=method,
        options=dict(optimizer_options or {}),
    )

    gammas_opt = list(res.x[:number_layers])
    betas_opt = list(res.x[number_layers:])
    sample_result = executable.sample(
        my_executor,
        shots=number_shots,
        bindings={"gammas": gammas_opt, "betas": betas_opt},
    ).result()
    sample_set = converter.decode(sample_result)

    max_idx = sample_set.energy.index(max(sample_set.energy))
    best = sample_set.samples[max_idx]
    return DitString(best.values()).to_integer('L')


def QAOA(
    problem_sketch: ProblemSketch,
    number_layers: int = 4,
    method: str = "COBYLA",
    backend: Any | None = None,
    number_shots: int = 4096,
    optimizer_options: dict | None = None,
    sampler_options: dict | None = None,
) -> int:
    """
    """
    _, _, number_layers, number_shots = _validate_variational_inputs(
        problem_sketch, number_layers, number_shots, method, optimizer_options, sampler_options
    )

    sampler, backend = _build_sampler(backend, number_shots, sampler_options)

    hubo_model = problem_sketch.to_hubo()

    converter = QAOAConverter(hubo_model)
    converter.spin_model = converter.spin_model.normalize_by_abs_max()
    
    executable = converter.transpile(QiskitTranspiler(), p=number_layers)
    my_executor = IBMRuntimeExecutor(sampler, backend)

    return _run_variational(converter, executable, my_executor, number_layers, number_shots, method, optimizer_options)


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
) -> int:
    """
    """
    _, _, number_layers, number_shots = _validate_variational_inputs(
        problem_sketch, number_layers, number_shots, method, optimizer_options, sampler_options
    )
    hamming_weight = _Validator.ensure_int("hamming_weight", hamming_weight, min_value=1)
    _Validator.ensure_str("initial_state", initial_state)
    _Validator.ensure_str("mixer", mixer)

    sampler, backend = _build_sampler(backend, number_shots, sampler_options)

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
    my_executor = IBMRuntimeExecutor(sampler, backend)

    return _run_variational(converter, executable, my_executor, number_layers, number_shots, method, optimizer_options)
