"""
Classical pre-training of QAOA/AOA parameters.

Workflow
--------
1. Grid scan at depth p=1 using EfficientDepthOneEvaluator (analytical, no circuit execution).
2. INTERP warm-start: broadcast p=1 optimal values to p layers with a linear schedule.
3. Simulator refinement: COBYLA on AerSimulator via the same qamomile path used by QAOA/AOA.
4. Return a parameter vector in TrOMA layout: [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}].

Requires: qaoa-training-pipeline  (pip install qaoa-training-pipeline)
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as sk_opt
from qiskit_aer import AerSimulator
from qamomile.optimization.qaoa import QAOAConverter
from qamomile.qiskit import QiskitTranspiler

from .qamomile_addon import AerLocalExecutor
from ..core.structure import Hamiltonian

# Circuits above this qubit count are skipped by default in _simulator_refinement.
# Statevector memory scales as 2^n: 25 qubits ≈ 512 MB, 26 ≈ 1 GB.
_MAX_SIM_QUBITS = 25


def _hamiltonian_to_sparse_pauli_op(hamiltonian: Hamiltonian):
    """Convert a TrOMA Hamiltonian to a Qiskit SparsePauliOp.

    Coefficients are divided by max|coeff| so the scale matches what
    QAOAConverter.spin_model.normalize_by_abs_max() produces — the BINARY↔SPIN
    round-trip (to_hubo → QAOAConverter) preserves ZZ coefficients, so both
    normalizations are equivalent.

    Qiskit Pauli strings are little-endian: rightmost character = qubit 0.
    """
    from qiskit.quantum_info import SparsePauliOp

    non_zero = {k: v for k, v in hamiltonian.terms.items() if k and not np.isclose(v, 0.0)}
    if not non_zero:
        raise ValueError("Hamiltonian has no non-zero terms — cannot pre-train parameters.")

    max_abs = max(abs(c) for c in non_zero.values())
    n = hamiltonian.num_qubits
    paulis: list[tuple[str, complex]] = []
    for qubits, coeff in non_zero.items():
        chars = ["I"] * n
        for q in qubits:
            chars[n - 1 - q] = "Z"
        paulis.append(("".join(chars), complex(coeff / max_abs)))

    return SparsePauliOp.from_list(paulis)


def _grid_scan_p1(cost_op, num_grid_points: int) -> tuple[float, float]:
    """Run depth-1 analytical grid scan. Returns (beta_opt, gamma_opt).

    qaoa_training_pipeline convention: optimized_params = [beta, gamma].
    """
    try:
        from qaoa_training_pipeline.training import DepthOneScanTrainer
        from qaoa_training_pipeline.evaluation import EfficientDepthOneEvaluator
    except ImportError as exc:
        raise ImportError(
            "qaoa-training-pipeline is required for QAOA pre-training. "
            "Install it with:  pip install qaoa-training-pipeline"
        ) from exc

    trainer = DepthOneScanTrainer(EfficientDepthOneEvaluator())
    result = trainer.train(cost_op, num_points=num_grid_points)
    beta_opt, gamma_opt = result["optimized_params"]
    return float(beta_opt), float(gamma_opt)


def _interp_warm_start(beta1: float, gamma1: float, p: int) -> np.ndarray:
    """INTERP initialization for depth-p QAOA from p=1 optimal parameters.

    Gammas increase linearly gamma1/p → gamma1 (phase separator grows with depth).
    Betas decrease linearly beta1 → beta1/p (mixer weakens with depth).
    For p=1 this recovers [gamma1, beta1] exactly.

    Returns x0 in TrOMA layout: [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}].
    """
    t = np.linspace(1.0 / p, 1.0, p)
    gammas = gamma1 * t
    betas = beta1 * (1.0 - t + 1.0 / p)
    return np.concatenate([gammas, betas])


def _simulator_refinement(
    hamiltonian: Hamiltonian,
    x0: np.ndarray,
    number_layers: int,
    number_shots: int,
    sim_method: str,
    max_iter: int = 50,
    force_simulator: bool = False,
    device: str = "CPU",
    num_threads: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Refine the INTERP warm start on a local AerSimulator via COBYLA.

    Uses the same qamomile path as QAOA/AOA: to_hubo → QAOAConverter →
    normalize_by_abs_max → transpile → sample → decode.

    Skipped when hamiltonian.num_qubits > _MAX_SIM_QUBITS unless
    force_simulator=True. Statevector memory scales as 2^n, so large circuits
    would exhaust RAM. Pass force_simulator=True only when you know the
    simulation fits in memory (or when using "matrix_product_state").

    device="GPU" requires qiskit-aer-gpu and a CUDA-capable card.
    num_threads controls CPU parallelism (None = Aer default = all cores).
    """
    if not force_simulator and hamiltonian.num_qubits > _MAX_SIM_QUBITS:
        if verbose:
            print(
                f"[pretrain] Simulator refinement skipped: circuit has "
                f"{hamiltonian.num_qubits} qubits (> {_MAX_SIM_QUBITS}). "
                f"Pass force_simulator=True to override."
            )
        return x0

    if verbose:
        print(
            f"[pretrain] Running AerSimulator refinement "
            f"(method={sim_method}, device={device}, "
            f"max_iter={max_iter}, shots={number_shots}) ..."
        )

    sim_options: dict = {"method": sim_method, "device": device}
    if num_threads is not None:
        sim_options["max_parallel_threads"] = num_threads
    sim = AerSimulator(**sim_options)
    my_executor = AerLocalExecutor(sim)

    converter = QAOAConverter(hamiltonian.to_hubo())
    converter.spin_model = converter.spin_model.normalize_by_abs_max()
    executable = converter.transpile(QiskitTranspiler(), p=number_layers)

    def cost_fn(params: np.ndarray) -> float:
        gammas = list(params[:number_layers])
        betas = list(params[number_layers:])
        result = executable.sample(
            my_executor,
            shots=number_shots,
            bindings={"gammas": gammas, "betas": betas},
        ).result()
        return -converter.decode_to_binary_sampleset(result).energy_mean()

    res = sk_opt.minimize(
        cost_fn, x0=x0, method="COBYLA", tol=1e-2, options={"maxiter": max_iter}
    )

    if verbose:
        print(
            f"[pretrain] Refinement done after {res.nfev} evaluations. "
            f"Energy: {-res.fun:.6f}"
        )

    return res.x


def pretrain_qaoa_parameters(
    hamiltonian: Hamiltonian,
    number_layers: int,
    number_shots: int = 1024,
    num_grid_points: int = 20,
    sim_method: str = "statevector",
    max_sim_iter: int = 50,
    force_simulator: bool = False,
    device: str = "CPU",
    num_threads: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Pre-train QAOA/AOA parameters classically before the main QPU optimization.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Problem Hamiltonian (from problem_sketch.to_hamiltonian()).
    number_layers : int
        Target QAOA depth p.
    number_shots : int
        Shots per COBYLA evaluation during simulator refinement.
    num_grid_points : int
        Grid density for the p=1 scan (evaluates num_grid_points² points).
    sim_method : str
        AerSimulator simulation method. "statevector" is exact up to ~25 qubits;
        "matrix_product_state" scales to larger circuits.
    max_sim_iter : int
        Maximum COBYLA iterations for simulator refinement. 50 is sufficient
        for a warm start.
    force_simulator : bool
        If False (default), skip simulator refinement when the circuit has more
        than _MAX_SIM_QUBITS (25) qubits and return the INTERP warm start
        directly. Set True to run regardless — useful with "matrix_product_state"
        or GPU backends that handle larger circuits.
    device : str
        "CPU" (default) or "GPU". GPU requires qiskit-aer-gpu and CUDA.
    num_threads : int | None
        CPU thread count passed to AerSimulator. None lets Aer use all cores.
    verbose : bool
        Print progress messages at each stage.

    Returns
    -------
    np.ndarray
        Shape (2*number_layers,) in TrOMA layout:
        [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}].
    """
    if verbose:
        print(
            f"[pretrain] Starting p=1 grid scan "
            f"({num_grid_points}² = {num_grid_points**2} points) ..."
        )

    cost_op = _hamiltonian_to_sparse_pauli_op(hamiltonian)
    beta1, gamma1 = _grid_scan_p1(cost_op, num_grid_points)

    if verbose:
        print(f"[pretrain] Grid scan done: β={beta1:.4f}, γ={gamma1:.4f}")

    x0 = _interp_warm_start(beta1, gamma1, number_layers)

    if verbose:
        gammas = np.array2string(x0[:number_layers], precision=4, separator=", ")
        betas = np.array2string(x0[number_layers:], precision=4, separator=", ")
        print(f"[pretrain] INTERP warm start (p={number_layers}): γ={gammas}, β={betas}")

    result = _simulator_refinement(
        hamiltonian, x0, number_layers, number_shots, sim_method,
        max_iter=max_sim_iter,
        force_simulator=force_simulator,
        device=device,
        num_threads=num_threads,
        verbose=verbose,
    )

    if verbose:
        params = np.array2string(result, precision=4, separator=", ")
        print(f"[pretrain] Pre-training complete. Initial parameters: {params}")

    return result
