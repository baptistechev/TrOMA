"""Tests for the Qiskit-native QAOA/AOA optimizers (troma.optimization.quantum_native)."""

import numpy as np
import pytest

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit import transpile as qiskit_transpile
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator

from qopt_best_practices.circuit_library import annotated_qaoa_ansatz
from qopt_best_practices.transpilation import generate_preset_qaoa_pass_manager
from qopt_best_practices.transpilation.annotated_transpilation_passes import UnrollBoxes

from troma import CombinatorialProblem, ConstraintSketchMap
from troma.problem_sketch import CombinatorialProblemSketch
from troma.core.structure import Hamiltonian
from troma.optimization import bind_optimizer, list_optimizers
from troma.optimization.quantum_native import (
    AOA,
    QAOA,
    annotated_aoa_ansatz,
    dicke_preparation,
    generate_preset_aoa_pass_manager,
    _build_initial_state,
    _normalized_terms,
    _resolve_mixer_pairs,
    _terms_to_cost_operator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exact_dicke_statevector(n: int, k: int) -> np.ndarray:
    """Exact |D^n_k> amplitudes in Qiskit (little-endian) ordering."""
    dim = 2**n
    amps = np.zeros(dim, dtype=complex)
    hits = [i for i in range(dim) if bin(i).count("1") == k]
    for i in hits:
        amps[i] = 1.0
    return amps / np.sqrt(len(hits))


def _line_backend(n: int, seed: int = 0) -> GenericBackendV2:
    cmap = CouplingMap([(i, i + 1) for i in range(n - 1)])
    return GenericBackendV2(
        num_qubits=n, coupling_map=cmap, basis_gates=["x", "sx", "rz", "cz"], seed=seed
    )


def _counts_to_probs(counts: dict) -> dict:
    total = sum(counts.values())
    return {k.replace(" ", ""): v / total for k, v in counts.items()}


def _total_variation(counts_a: dict, counts_b: dict) -> float:
    pa, pb = _counts_to_probs(counts_a), _counts_to_probs(counts_b)
    keys = set(pa) | set(pb)
    return 0.5 * sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys)


def _assign_by_name(circuit: QuantumCircuit, values: dict) -> QuantumCircuit:
    mapping = {p: values[p.name] for p in circuit.parameters}
    return circuit.assign_parameters(mapping)


def _aer_counts(circuit: QuantumCircuit, shots: int, seed: int = 1234) -> dict:
    sim = AerSimulator(seed_simulator=seed)
    runnable = qiskit_transpile(circuit, sim)
    return sim.run(runnable, shots=shots).result().get_counts()


_TEST_TERMS = {
    (0, 1): 0.8,
    (1, 2): -1.0,
    (2, 3): 0.6,
    (3, 4): -0.7,
    (0, 4): 0.9,
    (1, 3): -0.5,
    (2,): 0.3,
}


def _nn_problem_sketch(marginals: list[float], n: int = 3) -> CombinatorialProblemSketch:
    problem = CombinatorialProblem(lambda _: 0.0, problem_size=n, problem_dimension=2)
    sketch_map = ConstraintSketchMap(sketch_length=n, interaction_size=2, sketch_dimension=2)
    sketch_map.build_from_nearest_neighbors()
    return CombinatorialProblemSketch(problem, sketch_map, sketch_values=list(marginals))


# ---------------------------------------------------------------------------
# Dicke state preparation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,k", [(2, 1), (3, 1), (4, 1), (4, 2), (5, 2), (6, 3), (4, 4), (3, 0)])
def test_dicke_preparation_exact(n, k):
    sv = Statevector(dicke_preparation(n, k)).data
    expected = _exact_dicke_statevector(n, k)
    assert np.allclose(sv, expected, atol=1e-10)


def test_dicke_blockwise():
    init = _build_initial_state(6, "dicke", hamming_weight=1, block_size=3)
    sv = Statevector(init)
    block = Statevector(dicke_preparation(3, 1))
    expected = block.tensor(block)  # block 1 (qubits 3-5) ⊗ block 0 (qubits 0-2)
    assert np.allclose(sv.data, expected.data, atol=1e-10)


def test_single_basis_state():
    init = _build_initial_state(4, "single_basis_state", hamming_weight=1, block_size=2)
    sv = Statevector(init).data
    # ones on the last qubit of each block: qubits 1 and 3 -> index 0b1010 = 10
    expected = np.zeros(16, dtype=complex)
    expected[0b1010] = 1.0
    assert np.allclose(sv, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Mixer schedules
# ---------------------------------------------------------------------------


def test_ring_pairs_single_block():
    pairs = _resolve_mixer_pairs(5, "ring", None, None)
    assert pairs == [(0, 1), (2, 3), (1, 2), (3, 4), (4, 0)]


def test_ring_pairs_blocks():
    pairs = _resolve_mixer_pairs(4, "ring", None, 2)
    assert pairs == [(0, 1), (2, 3)]  # block of 2: no wrap-around


def test_fully_connected_pairs_cover_all():
    pairs = _resolve_mixer_pairs(4, "fully-connected", None, None)
    assert sorted(tuple(sorted(p)) for p in pairs) == [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
    ]


def test_explicit_pairs_validation():
    with pytest.raises(ValueError):
        _resolve_mixer_pairs(4, "ring", np.array([[0, 0]]), None)
    with pytest.raises(ValueError):
        _resolve_mixer_pairs(4, "ring", np.array([[0, 7]]), None)


# ---------------------------------------------------------------------------
# Swap-strategy routing correctness (counts equivalence)
# ---------------------------------------------------------------------------

_SHOTS_EQUIV = 60_000
_TV_TOL = 0.06


def test_aoa_routed_counts_match_unrouted():
    """The swap-routed AOA circuit must sample the same distribution as the
    unrouted ansatz — validates AOASwapToFinalMapping (mixer relabelling and
    measurement permutation)."""
    n = 5
    terms = _normalized_terms(Hamiltonian(terms=dict(_TEST_TERMS), num_qubits=n))
    cost_op = _terms_to_cost_operator(terms, n)

    ansatz = annotated_aoa_ansatz(cost_op, reps=1, hamming_weight=2, mixer="ring")
    ansatz.measure_all()
    values = {"γ[0]": 0.37, "β[0]": 0.81}

    reference = PassManager([UnrollBoxes()]).run(ansatz)
    ref_counts = _aer_counts(_assign_by_name(reference, values), _SHOTS_EQUIV, seed=11)

    backend = _line_backend(n)
    pm = generate_preset_aoa_pass_manager(
        backend,
        SwapStrategy.from_line(list(range(n))),
        edge_coloring={(i, i + 1): i % 2 for i in range(n - 1)},
    )
    routed = pm.run(ansatz)
    routed_counts = _aer_counts(_assign_by_name(routed, values), _SHOTS_EQUIV, seed=22)

    assert _total_variation(ref_counts, routed_counts) < _TV_TOL


def test_aoa_routed_preserves_hamming_weight():
    """XY mixer + Dicke init are Hamming-weight preserving; the routed circuit
    must only produce weight-k strings (strong check of measurement remap)."""
    n = 5
    k = 2
    terms = _normalized_terms(Hamiltonian(terms=dict(_TEST_TERMS), num_qubits=n))
    cost_op = _terms_to_cost_operator(terms, n)

    ansatz = annotated_aoa_ansatz(cost_op, reps=1, hamming_weight=k, mixer="ring")
    ansatz.measure_all()

    backend = _line_backend(n)
    pm = generate_preset_aoa_pass_manager(
        backend,
        SwapStrategy.from_line(list(range(n))),
        edge_coloring={(i, i + 1): i % 2 for i in range(n - 1)},
    )
    routed = pm.run(ansatz)
    counts = _aer_counts(
        _assign_by_name(routed, {"γ[0]": 0.9, "β[0]": -0.4}), 20_000, seed=7
    )
    for key in counts:
        assert key.replace(" ", "").count("1") == k


def test_aoa_mixer_uses_xx_plus_yy_gate():
    terms = _normalized_terms(Hamiltonian(terms=dict(_TEST_TERMS), num_qubits=5))
    cost_op = _terms_to_cost_operator(terms, 5)
    ansatz = annotated_aoa_ansatz(cost_op, reps=1, hamming_weight=1, mixer="ring")
    unrolled = PassManager([UnrollBoxes()]).run(ansatz)
    ops = unrolled.count_ops()
    assert ops.get("xx_plus_yy", 0) == 5  # one per ring pair, single 2q gate each


def test_aoa_routed_partial_pair_mixer_counts_match():
    """Explicit mixer pairs covering only a subset of qubits (and long-range)
    must survive routing — regression test for box/global index alignment."""
    n = 4
    terms = _normalized_terms(
        Hamiltonian(terms={(0, 1): 0.8, (1, 2): -1.0, (2, 3): 0.6, (0, 3): -0.4}, num_qubits=n)
    )
    cost_op = _terms_to_cost_operator(terms, n)
    pairs = np.array([[0, 2]])  # partial coverage, non-adjacent on the line

    ansatz = annotated_aoa_ansatz(
        cost_op, reps=1, hamming_weight=1, pair_indices_mixer=pairs
    )
    ansatz.measure_all()
    values = {"γ[0]": 0.55, "β[0]": -0.7}

    reference = PassManager([UnrollBoxes()]).run(ansatz)
    ref_counts = _aer_counts(_assign_by_name(reference, values), 30_000, seed=55)

    backend = _line_backend(n)
    pm = generate_preset_aoa_pass_manager(
        backend,
        SwapStrategy.from_line(list(range(n))),
        edge_coloring={(i, i + 1): i % 2 for i in range(n - 1)},
    )
    routed = pm.run(ansatz)
    routed_counts = _aer_counts(_assign_by_name(routed, values), 30_000, seed=66)

    assert _total_variation(ref_counts, routed_counts) < _TV_TOL


def test_qaoa_routed_counts_match_unrouted_two_layers():
    n = 5
    terms = _normalized_terms(Hamiltonian(terms=dict(_TEST_TERMS), num_qubits=n))
    cost_op = _terms_to_cost_operator(terms, n)

    ansatz = annotated_qaoa_ansatz(cost_op, reps=2)
    ansatz.measure_all()
    values = {"γ[0]": 0.3, "γ[1]": -0.5, "β[0]": 0.7, "β[1]": 0.2}

    reference = PassManager([UnrollBoxes()]).run(ansatz)
    ref_counts = _aer_counts(_assign_by_name(reference, values), _SHOTS_EQUIV, seed=33)

    backend = _line_backend(n)
    pm = generate_preset_qaoa_pass_manager(
        backend,
        SwapStrategy.from_line(list(range(n))),
        edge_coloring={(i, i + 1): i % 2 for i in range(n - 1)},
    )
    routed = pm.run(ansatz)
    routed_counts = _aer_counts(_assign_by_name(routed, values), _SHOTS_EQUIV, seed=44)

    assert _total_variation(ref_counts, routed_counts) < _TV_TOL


# ---------------------------------------------------------------------------
# End-to-end optimizers
# ---------------------------------------------------------------------------


def test_qaoa_native_end_to_end_line_backend():
    # Marginals favour [0, 0, 0] -> index 0.
    sketch = _nn_problem_sketch([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    result = QAOA(
        sketch,
        number_layers=1,
        backend=_line_backend(3),
        number_shots=4096,
        optimizer_options={"maxiter": 20},
    )
    assert int(result) == 0


def test_aoa_native_end_to_end():
    # Marginals favour [0, 0, 1] -> index 1; with hamming_weight=1 the
    # reachable strings are exactly {[1,0,0], [0,1,0], [0,0,1]} and the
    # final decode picks the max-energy sample, so the result is robust.
    sketch = _nn_problem_sketch([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    result = AOA(
        sketch,
        number_layers=1,
        hamming_weight=1,
        backend=_line_backend(3),
        number_shots=4096,
        optimizer_options={"maxiter": 15},
    )
    assert int(result) == 1


def test_aoa_native_result_metadata():
    sketch = _nn_problem_sketch([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    result = AOA(
        sketch,
        number_layers=1,
        hamming_weight=1,
        number_shots=1024,
        optimizer_options={"maxiter": 5},
    )
    assert result.number_layers == 1
    assert result.circuit_depth > 0
    assert result.transpiled_circuit_depth == result.circuit_depth
    assert result.transpiled_gate_count is not None and result.transpiled_gate_count > 0
    assert result.job_id is None
    assert result.objective_evaluations > 0
    assert len(result.gammas) == 1 and len(result.betas) == 1
    assert sum(result.final_sample_distribution.values()) == 1024


def test_aoa_swap_rejects_multilayer():
    sketch = _nn_problem_sketch([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="number_layers=1"):
        AOA(sketch, number_layers=2, use_swap_strategy=True)


def test_aoa_multilayer_without_swap():
    sketch = _nn_problem_sketch([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    result = AOA(
        sketch,
        number_layers=2,
        hamming_weight=1,
        use_swap_strategy=False,
        number_shots=2048,
        optimizer_options={"maxiter": 10},
    )
    assert int(result) == 1


def test_registry_entries():
    names = list_optimizers()
    assert "qaoa_native" in names and "aoa_native" in names
    optimizer = bind_optimizer(
        "aoa_native",
        number_layers=1,
        hamming_weight=1,
        number_shots=1024,
        optimizer_options={"maxiter": 5},
    )
    sketch = _nn_problem_sketch([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert optimizer.optimize(sketch) == 1
