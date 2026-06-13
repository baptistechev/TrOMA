"""Qiskit-native QAOA and AOA optimizers with swap-strategy transpilation.

This module re-implements the variational optimizers of
:mod:`troma.optimization.quantum` directly on top of Qiskit and the
``qopt-best-practices`` library, replacing the qamomile circuit/execution
stack. Circuits are built with the annotated (boxed) ansatz format, routed
with a line swap strategy (Weidenfeller et al., Quantum 6, 870, 2022),
transpiled once, and sampled with parameter re-binding at every optimizer
iteration.

On real QPU backends (``backend.simulator`` is ``False``) the highest-fidelity
linear qubit chain is selected automatically with ``BackendEvaluator`` and
used as the transpiler initial layout.

AOA notes
---------
The XY mixer is a two-local operator, which the upstream annotated passes do
not support (``AnnotatedSwapToFinalMapping`` raises ``NotImplementedError``).
:class:`AOASwapToFinalMapping` extends that pass to relabel mixer gates of any
arity through the virtual permutation left by the cost-layer swap network.
This is only valid for a single layer (``number_layers=1``): with more layers
the permutation must be tracked through the mixer into the next cost layer,
which is an open problem upstream. ``AOA`` therefore rejects
``number_layers > 1`` when ``use_swap_strategy=True``.

Conventions (matching the qamomile-based implementation):

* cost layer: ``exp(-i * gamma * H_C)`` with ``H_C`` the TrOMA Ising
  Hamiltonian normalized by its max-abs coefficient;
* XY mixer per pair ``(i, j)``: ``exp(-i * beta/2 * (X_i X_j + Y_i Y_j))``;
* ring mixer: parity-ordered adjacent pairs per block plus a wrap-around
  pair when the block has more than two qubits;
* Dicke initial state: Bartschi-Eidenbenz split & cyclic shift construction,
  per block, with the ``|1>`` qubits starting at the end of each block;
* result: the sampled bitstring with maximum Hamiltonian energy, encoded as
  an integer with variable 0 as the most significant dit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.optimize as sk_opt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import Measure, RYGate, XXPlusYYGate
from qiskit.circuit.library.standard_gates.equivalence_library import _sel
from qiskit.dagcircuit import DAGCircuit, DAGOutNode
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis.evolution import LieTrotter
from qiskit.transpiler import Layout, PassManager, TransformationPass, generate_preset_pass_manager
from qiskit.transpiler.passes import BasisTranslator, UnrollCustomDefinitions
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy
from qiskit_aer import AerSimulator

from qopt_best_practices.circuit_library import annotated_qaoa_ansatz
from qopt_best_practices.circuit_library.annotated_qaoa_ansatz import (
    CostLayerAnnotation,
    InitStateAnnotation,
    MixerAnnotation,
)
from qopt_best_practices.circuit_library.pauli_evolution import PauliEvolutionGate
from qopt_best_practices.transpilation import generate_preset_qaoa_pass_manager
from qopt_best_practices.transpilation.annotated_transpilation_passes import (
    AnnotatedPrepareCostLayer,
    AnnotatedCommuting2qGateRouter,
    SynthesizeAndSimplifyCostLayer,
    UnrollBoxes,
)

from .quantum import _validate_variational_inputs, _build_runtime_sampler
from ..core.structure import DitString, Hamiltonian, VariationalOptimizationResult
from ..core.embedding import reverse_spectrum_restriction
from ..problem_sketch import ProblemSketch, RestrictedProblemSketch
from .._validation import _Validator


# ---------------------------------------------------------------------------
# Hamiltonian conversion
# ---------------------------------------------------------------------------


def _normalized_terms(hamiltonian: Hamiltonian) -> dict[tuple[int, ...], float]:
    """Return non-empty Hamiltonian terms divided by the max-abs coefficient.

    Matches the scale produced by ``spin_model.normalize_by_abs_max()`` in the
    qamomile-based implementation, so optimizer parameter landscapes are
    comparable between the two paths.
    """
    non_zero = {
        tuple(k): float(v)
        for k, v in hamiltonian.terms.items()
        if k and not np.isclose(v, 0.0)
    }
    if not non_zero:
        raise ValueError(
            "Hamiltonian has no non-zero terms — nothing to optimize. "
            "This typically means the residue sketch is fully recovered."
        )
    max_abs = max(abs(c) for c in non_zero.values())
    return {k: v / max_abs for k, v in non_zero.items()}


def _terms_to_cost_operator(terms: dict[tuple[int, ...], float], num_qubits: int) -> SparsePauliOp:
    """Build the Ising-Z cost operator. Qiskit Pauli strings are little-endian."""
    paulis: list[tuple[str, complex]] = []
    for qubits, coeff in terms.items():
        chars = ["I"] * num_qubits
        for q in qubits:
            chars[num_qubits - 1 - q] = "Z"
        paulis.append(("".join(chars), complex(coeff)))
    return SparsePauliOp.from_list(paulis)


def _ensure_quadratic(terms: dict[tuple[int, ...], float]) -> None:
    high_order = [t for t in terms if len(t) > 2]
    if high_order:
        raise ValueError(
            f"Swap-strategy routing supports at most quadratic Hamiltonians; "
            f"found higher-order terms {high_order[:3]}{'...' if len(high_order) > 3 else ''}. "
            f"Pass use_swap_strategy=False to fall back to standard transpilation."
        )


# ---------------------------------------------------------------------------
# Initial states (Dicke / uniform / basis state)
# ---------------------------------------------------------------------------


def dicke_preparation(num_qubits: int, hamming_weight: int) -> QuantumCircuit:
    """Deterministic Dicke state |D^n_k> preparation.

    Bartschi & Eidenbenz construction (arXiv:1904.07358): starting from
    ``|0^(n-k) 1^k>`` (ones on the last ``k`` qubits), apply split & cyclic
    shift unitaries SCS_l for ``l = n, ..., k+1``. Each SCS_l consists of one
    two-qubit block (s=1) and ``k-1`` three-qubit blocks (s=2..k), where block
    ``s`` rotates between "ones block of length s stays" and "ones block
    shifts left by one" with angle ``2*acos(sqrt(s/l))``.
    """
    n = _Validator.ensure_int("num_qubits", num_qubits, min_value=1)
    k = _Validator.ensure_int("hamming_weight", hamming_weight, min_value=0)
    if k > n:
        raise ValueError(f"hamming_weight ({k}) cannot exceed num_qubits ({n}).")

    qc = QuantumCircuit(n, name=f"dicke_{n}_{k}")
    for q in range(n - k, n):
        qc.x(q)
    if k == 0 or k == n:
        return qc

    # The unitary must prepare |D^l_j> for every weight j <= k on each prefix,
    # so the SCS cascade runs down to the 2-qubit prefix with min(k, l-1)
    # blocks per step.
    for l in range(n, 1, -1):
        for s in range(1, min(k, l - 1) + 1):
            theta = 2 * np.arccos(np.sqrt(s / l))
            a, c = l - s - 1, l - 1
            qc.cx(a, c)
            if s == 1:
                qc.cry(theta, c, a)
            else:
                qc.append(RYGate(theta).control(2, annotated=True), [l - s, c, a])
            qc.cx(a, c)
    return qc


def _build_initial_state(
    num_qubits: int,
    initial_state: str,
    hamming_weight: int,
    block_size: int | None,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    if initial_state == "uniform":
        qc.h(range(num_qubits))
        return qc

    block = num_qubits if block_size is None else block_size
    if num_qubits % block != 0:
        raise ValueError(
            f"num_qubits ({num_qubits}) must be divisible by block_size ({block})."
        )
    if not 0 <= hamming_weight <= block:
        raise ValueError(f"Require 0 <= hamming_weight ({hamming_weight}) <= block_size ({block}).")
    num_blocks = num_qubits // block

    if initial_state == "single_basis_state":
        for b in range(num_blocks):
            start = b * block
            for q in range(start + block - hamming_weight, start + block):
                qc.x(q)
        return qc

    if initial_state == "dicke":
        dicke = dicke_preparation(block, hamming_weight)
        for b in range(num_blocks):
            qc.compose(dicke, qubits=range(b * block, (b + 1) * block), inplace=True)
        return qc

    raise ValueError(
        f"Unknown initial_state {initial_state!r}. "
        f"Expected 'dicke', 'uniform' or 'single_basis_state'."
    )


# ---------------------------------------------------------------------------
# XY mixer schedule (mirrors qamomile AOAConverter semantics)
# ---------------------------------------------------------------------------


def _resolve_mixer_pairs(
    num_qubits: int,
    mixer: str,
    pair_indices_mixer: np.ndarray | None,
    block_size: int | None,
) -> list[tuple[int, int]]:
    if pair_indices_mixer is not None:
        arr = np.asarray(pair_indices_mixer, dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("pair_indices_mixer must have shape (num_pairs, 2).")
        if (arr < 0).any() or (arr >= num_qubits).any():
            raise ValueError(f"pair_indices_mixer contains indices outside [0, {num_qubits - 1}].")
        if (arr[:, 0] == arr[:, 1]).any():
            raise ValueError("pair_indices_mixer contains self-pairs.")
        return [(int(i), int(j)) for i, j in arr]

    block = num_qubits if block_size is None else block_size
    if block <= 1:
        raise ValueError("block_size must be greater than 1.")
    if num_qubits % block != 0:
        raise ValueError(
            f"num_qubits ({num_qubits}) must be divisible by block_size ({block})."
        )
    num_blocks = num_qubits // block

    pairs: list[tuple[int, int]] = []
    if mixer == "ring":
        for b in range(num_blocks):
            start = b * block
            for i in range(0, block - 1, 2):
                pairs.append((start + i, start + i + 1))
            for i in range(1, block - 1, 2):
                pairs.append((start + i, start + i + 1))
            if block > 2:
                pairs.append((start + block - 1, start))
        return pairs

    if mixer == "fully-connected":
        all_pairs: list[tuple[int, int]] = []
        for b in range(num_blocks):
            start = b * block
            for i in range(block):
                for j in range(i + 1, block):
                    all_pairs.append((start + i, start + j))
        # Greedy partition into non-overlapping batches, then flatten.
        partitions: list[list[tuple[int, int]]] = []
        used: list[set[int]] = []
        for left, right in all_pairs:
            for partition, used_nodes in zip(partitions, used):
                if left not in used_nodes and right not in used_nodes:
                    partition.append((left, right))
                    used_nodes.update((left, right))
                    break
            else:
                partitions.append([(left, right)])
                used.append({left, right})
        return [pair for partition in partitions for pair in partition]

    raise ValueError(f"Unknown mixer {mixer!r}. Expected 'ring' or 'fully-connected'.")


def annotated_aoa_ansatz(
    cost_operator: SparsePauliOp,
    reps: int = 1,
    initial_state: str = "dicke",
    hamming_weight: int = 1,
    mixer: str = "ring",
    pair_indices_mixer: np.ndarray | None = None,
    block_size: int | None = None,
    name: str = "AOA",
) -> QuantumCircuit:
    """Annotated (boxed) AOA ansatz: Dicke/basis/uniform init, Ising cost
    layer, XY mixer. Compatible with the qopt-best-practices annotated
    transpilation passes via :func:`generate_preset_aoa_pass_manager`.

    The XY mixer uses ``XXPlusYYGate(2*beta)`` per pair, which equals
    ``exp(-i*beta/2*(XX+YY))`` — the same gate as qamomile's
    ``xy_pair_rotation`` — and transpiles to two 2-qubit native gates even
    with symbolic parameters (unlike a Trotterized RXX+RYY pair, which the
    transpiler cannot consolidate symbolically).

    Every box is widened to the full register with ``noop`` so that qubit
    indices inside box subcircuits coincide with global indices, which the
    swap-strategy passes rely on.
    """
    n = cost_operator.num_qubits
    init = _build_initial_state(n, initial_state, hamming_weight, block_size)
    pairs = _resolve_mixer_pairs(n, mixer, pair_indices_mixer, block_size)

    gammas = ParameterVector("γ", reps)
    betas = ParameterVector("β", reps)
    evolution = LieTrotter()

    circuit = QuantumCircuit(n, name=name)
    with circuit.box(annotations=(InitStateAnnotation(),)):
        circuit.noop(*range(n))
        circuit.compose(init, inplace=True)

    for layer in range(1, reps + 1):
        with circuit.box(annotations=(CostLayerAnnotation(layer),)):
            circuit.noop(*range(n))
            cost_gate = PauliEvolutionGate(cost_operator, gammas[layer - 1], synthesis=evolution)
            circuit.compose(cost_gate.definition, inplace=True)
        with circuit.box(annotations=(MixerAnnotation(layer),)):
            circuit.noop(*range(n))
            for i, j in pairs:
                circuit.append(
                    XXPlusYYGate(2 * betas[layer - 1], 0, label="(XX+YY)"), [i, j]
                )
    return circuit


# ---------------------------------------------------------------------------
# AOA swap-strategy transpilation
# ---------------------------------------------------------------------------


class AOASwapToFinalMapping(TransformationPass):
    """Absorb redundant SWAPs and permute mixer/measurements for AOA.

    Port of ``qopt_best_practices`` ``AnnotatedSwapToFinalMapping`` extended
    to mixers of any gate arity: every mixer gate is relabelled through the
    virtual permutation left by the cost-layer swap network, instead of
    raising ``NotImplementedError`` for two-local mixers. Mixer pairs may end
    up non-adjacent after relabelling; the downstream routing stage of the
    preset pass manager inserts the necessary SWAPs.

    Only single-layer (reps=1) ansaetze are supported: tracking the
    permutation through a two-local mixer into a second cost layer is not
    implemented.
    """

    def run(self, dag: DAGCircuit):
        qmap = self.property_set["virtual_permutation_layout"]
        num_layers = 0

        for node in dag.topological_op_nodes():
            if node.op.name != "box":
                continue

            annotation = node.op.annotations[0]
            layer_index = int(annotation.payload)
            layer_name = annotation.namespace
            num_layers = max(num_layers, layer_index)

            box_circuit = node.op.params[0]
            box_dag = circuit_to_dag(box_circuit)
            if "cost_layer" in layer_name:
                if layer_index != 1:
                    raise NotImplementedError(
                        "AOASwapToFinalMapping supports a single cost layer (reps=1)."
                    )
                # Remove trailing SWAPs and absorb them into the virtual layout.
                while True:
                    swaps_removed = False
                    for box_node in box_dag.topological_op_nodes():
                        if box_node.op.name == "swap":
                            successors = list(box_dag.successors(box_node))
                            if all(isinstance(s, DAGOutNode) for s in successors):
                                qmap.swap(box_node.qargs[0], box_node.qargs[1])
                                box_dag.remove_op_node(box_node)
                                swaps_removed = True
                    if not swaps_removed:
                        break
                node.op.params[0] = dag_to_circuit(box_dag)

            elif "mixer" in layer_name:
                # Relabel every mixer gate (any arity) through the permutation.
                new_dag = box_dag.copy_empty_like()
                inverse_layout = qmap.get_physical_bits()
                for box_node in box_dag.topological_op_nodes():
                    new_qargs = [
                        new_dag.qubits[inverse_layout[qubit._index]._index]
                        for qubit in box_node.qargs
                    ]
                    new_dag.apply_operation_back(box_node.op, qargs=new_qargs)
                node.op.params[0] = dag_to_circuit(new_dag)

        # Permute final measurements.
        measure_nodes = [node for node in dag.op_nodes() if isinstance(node.op, Measure)]
        if len(measure_nodes) > 0:
            for node in measure_nodes:
                dag.remove_op_node(node)
            for cidx in range(dag.num_qubits()):
                qubit = (
                    qmap.get_physical_bits().get(cidx, cidx)
                    if num_layers % 2 == 1
                    else dag.qubits[cidx]
                )
                dag.apply_operation_back(Measure(), [qubit], [dag.clbits[cidx]])
        return dag


def generate_preset_aoa_pass_manager(
    backend,
    swap_strategy: SwapStrategy,
    edge_coloring: dict[tuple[int, int], int] | None = None,
    initial_layout: Layout | None = None,
):
    """Staged pass manager for single-layer AOA circuits built with
    :func:`annotated_aoa_ansatz`. Mirrors
    ``generate_preset_qaoa_pass_manager`` with the two-local-mixer-capable
    :class:`AOASwapToFinalMapping`.
    """
    pre_init = PassManager(
        [
            AnnotatedPrepareCostLayer(),
            AnnotatedCommuting2qGateRouter(swap_strategy, edge_coloring),
            AOASwapToFinalMapping(),
            SynthesizeAndSimplifyCostLayer(basis_gates=["x", "cx", "sx", "rz", "id"]),
            UnrollBoxes(),
        ]
    )
    post_init = PassManager(
        [
            UnrollCustomDefinitions(_sel, basis_gates=backend.operation_names, min_qubits=3),
            BasisTranslator(_sel, target_basis=backend.operation_names, min_qubits=3),
        ]
    )
    staged_pm = generate_preset_pass_manager(3, backend, initial_layout=initial_layout)
    staged_pm.pre_init = pre_init
    staged_pm.post_init = post_init
    return staged_pm


def _generate_unrolled_pass_manager(backend, initial_layout: Layout | None = None):
    """Standard preset pass manager that first unrolls annotation boxes."""
    staged_pm = generate_preset_pass_manager(3, backend, initial_layout=initial_layout)
    staged_pm.pre_init = PassManager([UnrollBoxes()])
    return staged_pm


# ---------------------------------------------------------------------------
# Backend handling
# ---------------------------------------------------------------------------


def _is_real_qpu(backend: Any) -> bool:
    return not getattr(backend, "simulator", True)


def _find_best_line(backend: Any, num_qubits: int) -> list[int] | None:
    """Highest-fidelity linear qubit chain via BackendEvaluator (real QPU)."""
    try:
        from qopt_best_practices.qubit_selection import BackendEvaluator

        path, fidelity, num_subsets = BackendEvaluator(backend).evaluate(num_qubits)
        print(
            f"[quantum_native] Selected qubit chain "
            f"(fidelity={fidelity:.4f}, {num_subsets} candidates): {path}"
        )
        return list(path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[quantum_native] Qubit-chain selection failed ({exc}); using default layout.")
        return None


def _transpile_ansatz(
    ansatz: QuantumCircuit,
    backend: Any,
    *,
    algorithm: str,
    use_swap_strategy: bool,
    initial_layout: list[int] | None = None,
    verbose: bool = False,
) -> QuantumCircuit:
    n = ansatz.num_qubits
    layout = None
    if initial_layout is not None:
        layout = Layout.from_intlist(list(initial_layout), ansatz.qregs[0])
    elif _is_real_qpu(backend):
        path = _find_best_line(backend, n)
        if path is not None:
            layout = Layout.from_intlist(path, ansatz.qregs[0])

    if use_swap_strategy:
        strategy = SwapStrategy.from_line(list(range(n)))
        coloring = {(i, i + 1): i % 2 for i in range(n - 1)}
        if algorithm == "qaoa":
            pm = generate_preset_qaoa_pass_manager(
                backend, strategy, edge_coloring=coloring, initial_layout=layout
            )
        else:
            pm = generate_preset_aoa_pass_manager(
                backend, strategy, edge_coloring=coloring, initial_layout=layout
            )
    else:
        pm = _generate_unrolled_pass_manager(backend, initial_layout=layout)

    isa_circuit = pm.run(ansatz)
    if verbose:
        two_q_depth = isa_circuit.depth(lambda x: x.operation.num_qubits == 2)
        print(
            f"[quantum_native] ISA circuit: depth={isa_circuit.depth()}, "
            f"2q-depth={two_q_depth}, ops={dict(isa_circuit.count_ops())}"
        )
    return isa_circuit


def _make_runner(backend: Any | None, number_shots: int, sampler_options: dict | None):
    """Return (runner, backend, execution_info) where runner(circuit, shots) -> counts dict."""
    if backend is None:
        backend = AerSimulator()

    execution_info = {"last_job_id": None}

    def _job_id(job: Any) -> str | None:
        value = getattr(job, "job_id", None)
        if callable(value):
            value = value()
        return None if value is None else str(value)

    is_runtime = False
    try:
        from qiskit_ibm_runtime import IBMBackend

        is_runtime = isinstance(backend, IBMBackend)
    except ImportError:
        pass

    if is_runtime:
        sampler = _build_runtime_sampler(backend, number_shots, sampler_options)

        def runner(circuit, shots):
            job = sampler.run([circuit], shots=shots)
            if _is_real_qpu(backend):
                execution_info["last_job_id"] = _job_id(job)
            data = job.result()[0].data
            reg_name = next(iter(data))
            return getattr(data, reg_name).get_counts()

    else:

        def runner(circuit, shots):
            job = backend.run(circuit, shots=shots)
            if _is_real_qpu(backend):
                execution_info["last_job_id"] = _job_id(job)
            return job.result().get_counts()

    return runner, backend, execution_info


# ---------------------------------------------------------------------------
# Energy decoding and the variational loop
# ---------------------------------------------------------------------------


def _key_to_bits(key: str, num_vars: int) -> list[int]:
    """Counts key (Qiskit little-endian) -> bits indexed by variable."""
    clean = key.replace(" ", "")
    return [int(clean[num_vars - 1 - i]) for i in range(num_vars)]


def _energy_of_bits(bits: list[int], terms: dict[tuple[int, ...], float]) -> float:
    spins = [1 - 2 * b for b in bits]
    energy = 0.0
    for term, coeff in terms.items():
        prod = 1
        for q in term:
            prod *= spins[q]
        energy += coeff * prod
    return energy


def _mean_energy(counts: dict[str, int], terms: dict, num_vars: int) -> float:
    total = sum(counts.values())
    acc = 0.0
    for key, cnt in counts.items():
        acc += cnt * _energy_of_bits(_key_to_bits(key, num_vars), terms)
    return acc / total


def _split_parameters(circuit: QuantumCircuit, number_layers: int):
    gammas, betas = [], []
    for prm in circuit.parameters:
        if prm.name.startswith("γ"):
            gammas.append(prm)
        elif prm.name.startswith("β"):
            betas.append(prm)
        else:
            raise RuntimeError(f"Unexpected circuit parameter {prm.name!r}.")
    gammas.sort(key=lambda p: p.index)
    betas.sort(key=lambda p: p.index)
    if len(gammas) != number_layers or len(betas) != number_layers:
        raise RuntimeError(
            f"Expected {number_layers} gamma/beta parameters, "
            f"found {len(gammas)}/{len(betas)}."
        )
    return gammas, betas


def _select_best_by_real_cost(
    final_counts: dict[str, int],
    terms: dict[tuple[int, ...], float],
    num_vars: int,
    problem_sketch: ProblemSketch | None,
) -> tuple[int, int]:
    """Pick the best feasible bitstring from the final counts by real objective value.

    Mirrors :func:`troma.optimization.quantum._select_best_by_real_cost`: every
    distinct bitstring sampled from the optimized circuit is decoded, mapped
    back to the full problem space if restricted, and evaluated with
    ``problem_sketch.objective_function``. Bitstrings rejected by
    ``problem_sketch.feasibility_function`` (if any) are skipped. Candidates
    are visited in order of decreasing Hamiltonian energy, so ties in
    objective value fall back to the highest-energy (most-sampled-favoured)
    bitstring. Falls back to the maximum-Hamiltonian-energy bitstring if no
    ``problem_sketch`` is given or none of the sampled bitstrings are feasible.
    """
    objective_function = getattr(problem_sketch, "objective_function", None)
    if objective_function is not None:
        feasibility_function = getattr(problem_sketch, "feasibility_function", None)
        best_val: float | None = None
        best_index: int | None = None
        n_evaluations = 0

        keys_by_energy = sorted(
            final_counts, key=lambda k: _energy_of_bits(_key_to_bits(k, num_vars), terms), reverse=True
        )
        for key in keys_by_energy:
            dit_string = DitString(_key_to_bits(key, num_vars), dimension=2)

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

            eval_arr = np.asarray(eval_str)
            if feasibility_function is not None and not feasibility_function(eval_arr):
                continue

            val = float(objective_function(eval_arr))
            n_evaluations += 1
            if best_val is None or val > best_val:
                best_val = val
                best_index = dit_string.to_integer('R')

        if best_index is not None:
            return best_index, n_evaluations

    best_key = max(final_counts, key=lambda k: _energy_of_bits(_key_to_bits(k, num_vars), terms))
    return DitString(_key_to_bits(best_key, num_vars), dimension=2).to_integer(), 0


def _run_variational_native(
    isa_circuit: QuantumCircuit,
    runner,
    execution_info: dict[str, Any] | None,
    terms: dict[tuple[int, ...], float],
    num_vars: int,
    number_layers: int,
    number_shots: int,
    method: str,
    optimizer_options: dict | None,
    x0: np.ndarray | None = None,
    problem_sketch: ProblemSketch | None = None,
) -> VariationalOptimizationResult:
    gamma_params, beta_params = _split_parameters(isa_circuit, number_layers)

    def bind(params: np.ndarray) -> QuantumCircuit:
        mapping = {p: params[i] for i, p in enumerate(gamma_params)}
        mapping.update({p: params[number_layers + i] for i, p in enumerate(beta_params)})
        return isa_circuit.assign_parameters(mapping)

    def cost_fn(params: np.ndarray) -> float:
        counts = runner(bind(params), number_shots)
        return -_mean_energy(counts, terms, num_vars)

    number_parameters = 2 * number_layers
    bounds = np.array([[-np.pi, np.pi]] * number_parameters, dtype=float)
    res = sk_opt.minimize(
        cost_fn,
        x0=np.ones(number_parameters) if x0 is None else np.asarray(x0, dtype=float),
        bounds=bounds,
        method=method,
        options=dict(optimizer_options or {}),
    )

    final_counts = runner(bind(res.x), number_shots)
    best_index, truth_evals = _select_best_by_real_cost(final_counts, terms, num_vars, problem_sketch)

    return VariationalOptimizationResult(
        best_index,
        final_parameters=res.x,
        gammas=list(res.x[:number_layers]),
        betas=list(res.x[number_layers:]),
        number_layers=number_layers,
        circuit_depth=isa_circuit.depth(),
        transpiled_circuit_depth=isa_circuit.depth(),
        transpiled_gate_count=isa_circuit.size(),
        job_id=None if execution_info is None else execution_info.get("last_job_id"),
        solver_steps=int(getattr(res, "nit", 0) or 0),
        objective_evaluations=int(getattr(res, "nfev", 0) or 0),
        truth_objective_evaluations=truth_evals,
        final_sample_distribution=final_counts,
    )


# ---------------------------------------------------------------------------
# Public optimizers
# ---------------------------------------------------------------------------


def QAOA(
    problem_sketch: ProblemSketch,
    number_layers: int = 4,
    method: str = "COBYLA",
    backend: Any | None = None,
    number_shots: int = 4096,
    use_swap_strategy: bool = True,
    initial_layout: list[int] | None = None,
    optimizer_options: dict | None = None,
    sampler_options: dict | None = None,
    verbose: bool = False,
    pretrain: bool = False,
    pretrain_options: dict | None = None,
) -> int:
    """Qiskit-native QAOA with swap-strategy routing of the cost layer.

    Parameters mirror :func:`troma.optimization.quantum.QAOA`, plus:

    use_swap_strategy : bool, optional
        Route the cost layer with a line swap strategy and edge coloring
        (requires an at-most-quadratic Hamiltonian). Defaults to True.
    initial_layout : list[int], optional
        Physical qubits to map the ansatz onto. On real QPU backends, if not
        provided, the best linear chain is selected automatically with
        ``BackendEvaluator``.

    Returns
    -------
    int
        Index of the sampled configuration with maximum Hamiltonian energy
        (a :class:`VariationalOptimizationResult`, which subclasses int).
    """
    _, _, number_layers, number_shots = _validate_variational_inputs(
        problem_sketch, number_layers, number_shots, method, optimizer_options, sampler_options
    )

    hamiltonian = problem_sketch.to_hamiltonian()
    terms = _normalized_terms(hamiltonian)
    if use_swap_strategy:
        _ensure_quadratic(terms)
    cost_op = _terms_to_cost_operator(terms, hamiltonian.num_qubits)

    ansatz = annotated_qaoa_ansatz(cost_op, reps=number_layers)
    ansatz.measure_all()

    runner, backend, execution_info = _make_runner(backend, number_shots, sampler_options)
    isa_circuit = _transpile_ansatz(
        ansatz,
        backend,
        algorithm="qaoa",
        use_swap_strategy=use_swap_strategy,
        initial_layout=initial_layout,
        verbose=verbose,
    )

    x0 = None
    if pretrain:
        from ._quantum_pre_training import pretrain_qaoa_parameters

        x0 = pretrain_qaoa_parameters(hamiltonian, number_layers, **dict(pretrain_options or {}))

    return _run_variational_native(
        isa_circuit, runner, execution_info, terms, hamiltonian.num_qubits,
        number_layers, number_shots, method, optimizer_options, x0=x0,
        problem_sketch=problem_sketch,
    )


def AOA(
    problem_sketch: ProblemSketch,
    number_layers: int = 1,
    method: str = "COBYLA",
    backend: Any | None = None,
    number_shots: int = 4096,
    initial_state: str = "dicke",
    hamming_weight: int = 1,
    mixer: str = "ring",
    pair_indices_mixer: np.ndarray | None = None,
    block_size: int | None = None,
    use_swap_strategy: bool = True,
    initial_layout: list[int] | None = None,
    optimizer_options: dict | None = None,
    sampler_options: dict | None = None,
    verbose: bool = False,
    pretrain: bool = False,
    pretrain_options: dict | None = None,
) -> int:
    """Qiskit-native AOA (XY mixer) with swap-strategy routing of the cost layer.

    Parameters mirror :func:`troma.optimization.quantum.AOA`, plus
    ``use_swap_strategy`` and ``initial_layout`` (see :func:`QAOA`).

    Notes
    -----
    With ``use_swap_strategy=True`` only ``number_layers=1`` is supported:
    the cost-layer swap network permutes the qubits, and re-aligning a
    two-local XY mixer across multiple layers is not implemented (open
    problem upstream in qopt-best-practices). Use ``use_swap_strategy=False``
    for multi-layer AOA.

    Returns
    -------
    int
        Index of the sampled configuration with maximum Hamiltonian energy
        (a :class:`VariationalOptimizationResult`, which subclasses int).
    """
    _, _, number_layers, number_shots = _validate_variational_inputs(
        problem_sketch, number_layers, number_shots, method, optimizer_options, sampler_options
    )
    hamming_weight = _Validator.ensure_int("hamming_weight", hamming_weight, min_value=1)
    _Validator.ensure_str("initial_state", initial_state)
    _Validator.ensure_str("mixer", mixer)

    if use_swap_strategy and number_layers != 1:
        raise ValueError(
            "AOA with use_swap_strategy=True supports number_layers=1 only: "
            "the cost-layer swap network permutes the qubits and re-aligning "
            "the two-local XY mixer across layers is not implemented. "
            "Pass use_swap_strategy=False for multi-layer AOA."
        )

    hamiltonian = problem_sketch.to_hamiltonian()
    terms = _normalized_terms(hamiltonian)
    if use_swap_strategy:
        _ensure_quadratic(terms)
    cost_op = _terms_to_cost_operator(terms, hamiltonian.num_qubits)

    ansatz = annotated_aoa_ansatz(
        cost_op,
        reps=number_layers,
        initial_state=initial_state,
        hamming_weight=hamming_weight,
        mixer=mixer,
        pair_indices_mixer=pair_indices_mixer,
        block_size=block_size,
    )
    ansatz.measure_all()

    runner, backend, execution_info = _make_runner(backend, number_shots, sampler_options)
    isa_circuit = _transpile_ansatz(
        ansatz,
        backend,
        algorithm="aoa",
        use_swap_strategy=use_swap_strategy,
        initial_layout=initial_layout,
        verbose=verbose,
    )

    x0 = None
    if pretrain:
        from ._quantum_pre_training import pretrain_qaoa_parameters

        x0 = pretrain_qaoa_parameters(hamiltonian, number_layers, **dict(pretrain_options or {}))

    return _run_variational_native(
        isa_circuit, runner, execution_info, terms, hamiltonian.num_qubits,
        number_layers, number_shots, method, optimizer_options, x0=x0,
        problem_sketch=problem_sketch,
    )
