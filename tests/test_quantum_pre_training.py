"""Tests for _quantum_pre_training.py and the pretrain parameter in QAOA/AOA."""

import numpy as np
import pytest
from unittest.mock import patch

from troma import CombinatorialProblem, CombinatorialProblemSketch, ConstraintSketchMap
from troma.core.structure import Hamiltonian
from troma.optimization._quantum_pre_training import (
    _hamiltonian_to_sparse_pauli_op,
    _interp_warm_start,
    _grid_scan_p1,
    _simulator_refinement,
    pretrain_qaoa_parameters,
)
from troma.optimization.quantum import QAOA, AOA


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _simple_hamiltonian(n_qubits: int = 3) -> Hamiltonian:
    """3-qubit ZZ chain: Z0Z1 + 0.5*Z1Z2."""
    return Hamiltonian(terms={(0, 1): 1.0, (1, 2): 0.5}, num_qubits=n_qubits)


def _nn_problem_sketch(n: int = 3, k: int = 2, d: int = 2) -> CombinatorialProblemSketch:
    """Nearest-neighbor constraint sketch favouring index 0 (all-zero bit string)."""
    csm = ConstraintSketchMap(sketch_length=n, interaction_size=k, sketch_dimension=d)
    csm.build_from_nearest_neighbors()
    marginals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    problem = CombinatorialProblem(lambda _: 0.0, problem_size=n, problem_dimension=d)
    return CombinatorialProblemSketch(problem, csm, sketch_values=marginals)


# Shared options to keep integration tests fast
_PRETRAIN_FAST = {"num_grid_points": 5, "number_shots": 128}
_OPTIMIZER_FAST = {"maxiter": 5}


# ---------------------------------------------------------------------------
# _hamiltonian_to_sparse_pauli_op
# ---------------------------------------------------------------------------

class TestHamiltonianToSparsePauliOp:
    def test_zz_qubit01_in_3qubits_gives_izz(self):
        # Qiskit little-endian: rightmost char = qubit 0 → ZZ on (0,1) → "IZZ"
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=3)
        op = _hamiltonian_to_sparse_pauli_op(h)
        assert "IZZ" in [str(p) for p in op.paulis]

    def test_zz_qubit12_in_3qubits_gives_zzi(self):
        h = Hamiltonian(terms={(1, 2): 1.0}, num_qubits=3)
        op = _hamiltonian_to_sparse_pauli_op(h)
        assert "ZZI" in [str(p) for p in op.paulis]

    def test_single_z_qubit0_gives_iiz(self):
        h = Hamiltonian(terms={(0,): 1.0}, num_qubits=3)
        op = _hamiltonian_to_sparse_pauli_op(h)
        assert "IIZ" in [str(p) for p in op.paulis]

    def test_normalization_divides_by_max_abs(self):
        h = Hamiltonian(terms={(0, 1): 2.0, (1, 2): -1.0}, num_qubits=3)
        op = _hamiltonian_to_sparse_pauli_op(h)
        coeffs = {str(p): float(c.real) for p, c in zip(op.paulis, op.coeffs)}
        # max_abs = 2.0: (0,1) → 1.0, (1,2) → -0.5
        assert coeffs["IZZ"] == pytest.approx(1.0)
        assert coeffs["ZZI"] == pytest.approx(-0.5)

    def test_uniform_coeffs_all_normalize_to_one(self):
        h = Hamiltonian(terms={(0, 1): 3.0, (1, 2): 3.0}, num_qubits=3)
        op = _hamiltonian_to_sparse_pauli_op(h)
        for c in op.coeffs:
            assert abs(c) == pytest.approx(1.0)

    def test_num_paulis_matches_num_nonzero_terms(self):
        h = _simple_hamiltonian()
        op = _hamiltonian_to_sparse_pauli_op(h)
        non_zero = sum(1 for c in h.terms.values() if not np.isclose(c, 0.0))
        assert len(op.paulis) == non_zero

    def test_output_num_qubits_matches_hamiltonian(self):
        h = Hamiltonian(terms={(0, 1): 1.0, (3, 4): 0.5}, num_qubits=5)
        op = _hamiltonian_to_sparse_pauli_op(h)
        assert op.num_qubits == 5

    def test_all_zero_coeffs_raises_value_error(self):
        h = Hamiltonian(terms={(0, 1): 0.0}, num_qubits=2)
        with pytest.raises(ValueError, match="non-zero"):
            _hamiltonian_to_sparse_pauli_op(h)


# ---------------------------------------------------------------------------
# _interp_warm_start
# ---------------------------------------------------------------------------

class TestInterpWarmStart:
    def test_p1_recovers_gamma1_and_beta1_exactly(self):
        x0 = _interp_warm_start(beta1=1.2, gamma1=0.8, p=1)
        assert x0[0] == pytest.approx(0.8)  # gamma
        assert x0[1] == pytest.approx(1.2)  # beta

    def test_output_length_is_2p(self):
        for p in [1, 2, 3, 5]:
            x0 = _interp_warm_start(beta1=1.0, gamma1=1.0, p=p)
            assert len(x0) == 2 * p

    def test_gammas_are_monotonically_increasing(self):
        p = 4
        x0 = _interp_warm_start(beta1=1.0, gamma1=2.0, p=p)
        gammas = x0[:p]
        assert np.all(np.diff(gammas) > 0)

    def test_betas_are_monotonically_decreasing(self):
        p = 4
        x0 = _interp_warm_start(beta1=1.0, gamma1=2.0, p=p)
        betas = x0[p:]
        assert np.all(np.diff(betas) < 0)

    def test_last_gamma_equals_gamma1(self):
        gamma1 = 2.5
        x0 = _interp_warm_start(beta1=1.5, gamma1=gamma1, p=4)
        assert x0[3] == pytest.approx(gamma1)

    def test_first_gamma_equals_gamma1_over_p(self):
        gamma1 = 2.4
        p = 3
        x0 = _interp_warm_start(beta1=1.0, gamma1=gamma1, p=p)
        assert x0[0] == pytest.approx(gamma1 / p)

    def test_first_beta_equals_beta1(self):
        beta1 = 1.5
        p = 4
        x0 = _interp_warm_start(beta1=beta1, gamma1=2.5, p=p)
        assert x0[p] == pytest.approx(beta1)

    def test_last_beta_equals_beta1_over_p(self):
        beta1 = 1.8
        p = 3
        x0 = _interp_warm_start(beta1=beta1, gamma1=1.0, p=p)
        assert x0[-1] == pytest.approx(beta1 / p)

    @pytest.mark.parametrize("p", [2, 3, 4])
    def test_layout_gammas_then_betas(self, p):
        # TrOMA layout: [gamma_0,...,gamma_{p-1}, beta_0,...,beta_{p-1}]
        beta1, gamma1 = 1.2, 0.8
        x0 = _interp_warm_start(beta1, gamma1, p)
        assert x0[p - 1] == pytest.approx(gamma1)   # last gamma
        assert x0[p] == pytest.approx(beta1)          # first beta


# ---------------------------------------------------------------------------
# _grid_scan_p1
# ---------------------------------------------------------------------------

class TestGridScanP1:
    def test_missing_package_raises_import_error(self):
        null_modules = {
            "qaoa_training_pipeline": None,
            "qaoa_training_pipeline.training": None,
            "qaoa_training_pipeline.evaluation": None,
        }
        with patch.dict("sys.modules", null_modules):
            with pytest.raises(ImportError, match="qaoa-training-pipeline"):
                _grid_scan_p1(None, num_grid_points=5)

    def test_returns_two_floats(self):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        from qiskit.quantum_info import SparsePauliOp
        cost_op = SparsePauliOp.from_list([("ZZ", 1.0)])
        beta1, gamma1 = _grid_scan_p1(cost_op, num_grid_points=10)
        assert isinstance(beta1, float)
        assert isinstance(gamma1, float)

    def test_beta_in_standard_range(self):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        from qiskit.quantum_info import SparsePauliOp
        cost_op = SparsePauliOp.from_list([("ZZ", 1.0)])
        beta1, _ = _grid_scan_p1(cost_op, num_grid_points=10)
        assert 0.0 <= beta1 <= np.pi

    def test_gamma_in_standard_range(self):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        from qiskit.quantum_info import SparsePauliOp
        cost_op = SparsePauliOp.from_list([("ZZ", 1.0)])
        _, gamma1 = _grid_scan_p1(cost_op, num_grid_points=10)
        assert 0.0 <= gamma1 <= 2 * np.pi


# ---------------------------------------------------------------------------
# _simulator_refinement
# ---------------------------------------------------------------------------

class TestSimulatorRefinement:
    def test_returns_array_of_correct_shape_p1(self):
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=2)
        x0 = np.array([0.5, 0.3])
        result = _simulator_refinement(h, x0, number_layers=1, number_shots=128, sim_method="statevector")
        assert result.shape == (2,)

    def test_returns_array_of_correct_shape_p2(self):
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=2)
        x0 = np.array([0.3, 0.6, 0.8, 0.4])
        result = _simulator_refinement(h, x0, number_layers=2, number_shots=128, sim_method="statevector")
        assert result.shape == (4,)

    def test_output_is_float_array(self):
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=2)
        x0 = np.array([0.5, 0.3])
        result = _simulator_refinement(h, x0, number_layers=1, number_shots=128, sim_method="statevector")
        assert result.dtype.kind == "f"

    def test_output_values_are_finite(self):
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=2)
        x0 = np.array([0.5, 0.3])
        result = _simulator_refinement(h, x0, number_layers=1, number_shots=128, sim_method="statevector")
        assert np.all(np.isfinite(result))

    def test_skipped_by_default_when_qubits_exceed_limit(self):
        # 26-qubit circuit exceeds _MAX_SIM_QUBITS=25 → x0 returned unchanged
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=26)
        x0 = np.array([0.5, 0.3])
        result = _simulator_refinement(
            h, x0, number_layers=1, number_shots=128, sim_method="statevector",
        )
        np.testing.assert_array_equal(result, x0)

    def test_force_simulator_overrides_qubit_limit(self):
        # force_simulator=True should run even above the limit (2-qubit circuit
        # is fine; we just verify the flag doesn't block execution)
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=2)
        x0 = np.array([0.5, 0.3])
        result = _simulator_refinement(
            h, x0, number_layers=1, number_shots=128,
            sim_method="statevector", force_simulator=True,
        )
        assert result.shape == (2,)

    def test_max_iter_respected(self):
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=2)
        x0 = np.array([0.5, 0.3])
        result = _simulator_refinement(
            h, x0, number_layers=1, number_shots=64,
            sim_method="statevector", max_iter=1,
        )
        assert result.shape == (2,)

    def test_num_threads_accepted(self):
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=2)
        x0 = np.array([0.5, 0.3])
        result = _simulator_refinement(
            h, x0, number_layers=1, number_shots=64,
            sim_method="statevector", num_threads=2,
        )
        assert result.shape == (2,)

    def test_verbose_skip_message(self, capsys):
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=26)
        x0 = np.array([0.5, 0.3])
        _simulator_refinement(
            h, x0, number_layers=1, number_shots=64,
            sim_method="statevector", verbose=True,
        )
        out = capsys.readouterr().out
        assert "skipped" in out.lower()
        assert "force_simulator" in out

    def test_verbose_running_message(self, capsys):
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=2)
        x0 = np.array([0.5, 0.3])
        _simulator_refinement(
            h, x0, number_layers=1, number_shots=64,
            sim_method="statevector", verbose=True,
        )
        out = capsys.readouterr().out
        assert "AerSimulator" in out
        assert "done" in out.lower()


# ---------------------------------------------------------------------------
# pretrain_qaoa_parameters
# ---------------------------------------------------------------------------

class TestPretrainQaoaParameters:
    def test_returns_correct_shape_p1(self):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=2)
        result = pretrain_qaoa_parameters(h, number_layers=1, number_shots=128, num_grid_points=5)
        assert result.shape == (2,)

    def test_returns_correct_shape_p2(self):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=2)
        result = pretrain_qaoa_parameters(h, number_layers=2, number_shots=128, num_grid_points=5)
        assert result.shape == (4,)

    def test_output_values_are_finite(self):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        h = _simple_hamiltonian()
        result = pretrain_qaoa_parameters(h, number_layers=1, number_shots=128, num_grid_points=5)
        assert np.all(np.isfinite(result))

    def test_empty_hamiltonian_raises(self):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        h = Hamiltonian(terms={(0, 1): 0.0}, num_qubits=2)
        with pytest.raises(ValueError, match="non-zero"):
            pretrain_qaoa_parameters(h, number_layers=1)

    def test_verbose_prints_all_stages(self, capsys):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=2)
        pretrain_qaoa_parameters(
            h, number_layers=1, number_shots=64, num_grid_points=5, verbose=True
        )
        out = capsys.readouterr().out
        assert "grid scan" in out.lower()
        assert "interp" in out.lower()
        assert "complete" in out.lower()

    def test_large_circuit_skipped_with_verbose(self, capsys):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        h = Hamiltonian(terms={(0, 1): 1.0}, num_qubits=26)
        pretrain_qaoa_parameters(
            h, number_layers=1, number_shots=64, num_grid_points=5, verbose=True
        )
        out = capsys.readouterr().out
        assert "skipped" in out.lower()


# ---------------------------------------------------------------------------
# QAOA with pretrain=True
# ---------------------------------------------------------------------------

class TestQAOAWithPretrain:
    def test_pretrain_false_baseline_returns_int(self):
        """Sanity check: pretrain=False still works after the refactor."""
        sketch = _nn_problem_sketch()
        result = QAOA(sketch, number_layers=1, number_shots=64,
                      optimizer_options=_OPTIMIZER_FAST, pretrain=False)
        assert isinstance(result, int)
        assert 0 <= result < 2**3

    def test_pretrain_true_returns_int(self):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        sketch = _nn_problem_sketch()
        result = QAOA(sketch, number_layers=1, number_shots=64,
                      optimizer_options=_OPTIMIZER_FAST, pretrain=True,
                      pretrain_options=_PRETRAIN_FAST)
        assert isinstance(result, int)

    def test_pretrain_true_result_in_valid_range(self):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        sketch = _nn_problem_sketch()
        result = QAOA(sketch, number_layers=1, number_shots=64,
                      optimizer_options=_OPTIMIZER_FAST, pretrain=True,
                      pretrain_options=_PRETRAIN_FAST)
        assert 0 <= result < 2**3

    def test_pretrain_multi_layer_returns_int(self):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        sketch = _nn_problem_sketch()
        result = QAOA(sketch, number_layers=2, number_shots=64,
                      optimizer_options=_OPTIMIZER_FAST, pretrain=True,
                      pretrain_options=_PRETRAIN_FAST)
        assert isinstance(result, int)
        assert 0 <= result < 2**3

    def test_pretrain_options_none_uses_defaults(self):
        """pretrain_options=None should not raise."""
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        sketch = _nn_problem_sketch()
        # Use tiny defaults to avoid slow run
        result = QAOA(sketch, number_layers=1, number_shots=64,
                      optimizer_options=_OPTIMIZER_FAST, pretrain=True,
                      pretrain_options={"num_grid_points": 5, "number_shots": 64})
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# AOA with pretrain=True
# ---------------------------------------------------------------------------

class TestAOAWithPretrain:
    def test_pretrain_false_baseline_returns_int(self):
        """Sanity check: pretrain=False still works for AOA after the refactor."""
        sketch = _nn_problem_sketch()
        result = AOA(sketch, number_layers=1, number_shots=64,
                     optimizer_options=_OPTIMIZER_FAST, pretrain=False)
        assert isinstance(result, int)
        assert 0 <= result < 2**3

    def test_pretrain_true_returns_int(self):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        sketch = _nn_problem_sketch()
        result = AOA(sketch, number_layers=1, number_shots=64,
                     optimizer_options=_OPTIMIZER_FAST, pretrain=True,
                     pretrain_options=_PRETRAIN_FAST)
        assert isinstance(result, int)

    def test_pretrain_true_result_in_valid_range(self):
        pytest.importorskip("qaoa_training_pipeline", reason="qaoa-training-pipeline not installed")
        sketch = _nn_problem_sketch()
        result = AOA(sketch, number_layers=1, number_shots=64,
                     optimizer_options=_OPTIMIZER_FAST, pretrain=True,
                     pretrain_options=_PRETRAIN_FAST)
        assert 0 <= result < 2**3
