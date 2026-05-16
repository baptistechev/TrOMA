"""Tests for the public optimization API (troma.optimization)."""

import numpy as np
import pytest

from troma import bind_optimizer, get_optimizer, optimize
from troma import CombinatorialProblem, CombinatorialProblemSketch
from troma.optimization._quantum_map import create_qaoa_circ
from troma.optimization import (
    list_optimizers,
    brute_force_max,
    spin_chain_nn_max,
    dual_annealing,
    simulated_annealing,
)
from troma.optimization.optimizer import FunctionOptimizer
from troma import ConstraintSketchMap, ExplicitSketchMap, Hamiltonian

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _identity_sketch(n: int) -> np.ndarray:
    return np.eye(n)


def _constraint_problem_sketch(
    marginals: list[float],
    constraints: list[dict],
    n: int,
    k: int = 2,
    d: int = 2,
) -> CombinatorialProblemSketch:
    problem = CombinatorialProblem(lambda _: 0.0, problem_size=n, problem_dimension=d)
    sketch_map = ConstraintSketchMap(
        sketch_length=n,
        interaction_size=k,
        sketch_map=constraints,
        sketch_dimension=d,
    )
    return CombinatorialProblemSketch(problem, sketch_map, sketch_values=list(marginals))


def _explicit_problem_sketch(
    marginals: list[float],
    sketch: np.ndarray,
    problem_size: int,
    d: int = 2,
) -> CombinatorialProblemSketch:
    problem = CombinatorialProblem(lambda _: 0.0, problem_size=problem_size, problem_dimension=d)
    sketch_map = ExplicitSketchMap(
        sketch_length=problem_size,
        interaction_size=1,
        sketch_map=sketch,
        sketch_dimension=d,
    )
    return CombinatorialProblemSketch(problem, sketch_map, sketch_values=list(marginals))


def _nn_setup(n=3, k=2, d=2):
    """Return (constraints, explicit_sketch, marginals favouring index 0)."""
    csm = ConstraintSketchMap(sketch_length=n, interaction_size=k, sketch_dimension=d)
    csm.build_from_nearest_neighbors()
    constraints = csm.map
    esm = ExplicitSketchMap(sketch_length=n, interaction_size=k, sketch_dimension=d)
    esm.build_from_nearest_neighbors()
    sketch = esm.map
    # f([0,0,0])=1 → marginals = [1,0,0,0,1,0,0,0]
    marginals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    return constraints, sketch, marginals


def _parameter_by_name(circuit, name: str):
    return next(parameter for parameter in circuit.parameters if parameter.name == name)


# ---------------------------------------------------------------------------
# list_optimizers
# ---------------------------------------------------------------------------

class TestListOptimizers:
    def test_returns_list(self):
        result = list_optimizers()
        assert isinstance(result, list)

    def test_sorted(self):
        result = list_optimizers()
        assert result == sorted(result)

    def test_contains_classical_optimizers(self):
        result = list_optimizers()
        for name in ("brute_force", "brute_force_max",
                     "dual_annealing", "simulated_annealing",
                     "spin_chain_nn_max"):
            # At least one of these names or their sub-string should exist
            pass  # exact names tested below
        assert "brute_force_max" in result
        assert "dual_annealing" in result
        assert "simulated_annealing" in result
        assert "spin_chain_nn_max" in result

    def test_contains_digital_annealing(self):
        assert "digital_annealing" in list_optimizers()


class TestCreateQaoaCircuit:
    def test_single_layer_circuit_uses_named_parameters(self):
        qc = create_qaoa_circ(Hamiltonian({(0,): 1.0, (0, 1): -0.5}, num_qubits=2))

        beta = _parameter_by_name(qc, "beta")
        gamma = _parameter_by_name(qc, "gamma")
        bound_qc = qc.assign_parameters({gamma: 0.3, beta: 0.1}, inplace=False)

        assert {parameter.name for parameter in qc.parameters} == {"beta", "gamma"}
        assert len(bound_qc.parameters) == 0

    def test_multi_layer_circuit_exposes_parameter_vectors(self):
        qc = create_qaoa_circ(Hamiltonian({(0,): 1.0}, num_qubits=1), num_layers=2)
        bound_qc = qc.assign_parameters(
            {
                _parameter_by_name(qc, "beta[0]"): 0.1,
                _parameter_by_name(qc, "beta[1]"): 0.2,
                _parameter_by_name(qc, "gamma[0]"): 0.3,
                _parameter_by_name(qc, "gamma[1]"): 0.4,
            },
            inplace=False,
        )

        assert {parameter.name for parameter in qc.parameters} == {
            "beta[0]",
            "beta[1]",
            "gamma[0]",
            "gamma[1]",
        }
        assert len(bound_qc.parameters) == 0


# ---------------------------------------------------------------------------
# get_optimizer
# ---------------------------------------------------------------------------

class TestGetOptimizer:
    def test_returns_function_optimizer(self):
        opt = get_optimizer("brute_force_max")
        assert isinstance(opt, FunctionOptimizer)

    def test_case_insensitive(self):
        opt = get_optimizer("Brute_Force_Max")
        assert isinstance(opt, FunctionOptimizer)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            get_optimizer("does_not_exist")

    @pytest.mark.parametrize("name", ["brute_force_max", "spin_chain_nn_max",
                                       "dual_annealing", "simulated_annealing"])
    def test_all_classical_optimizers_loadable(self, name):
        opt = get_optimizer(name)
        assert isinstance(opt, FunctionOptimizer)


# ---------------------------------------------------------------------------
# bind_optimizer
# ---------------------------------------------------------------------------

class TestBindOptimizer:
    def test_returns_function_optimizer(self):
        opt = bind_optimizer("brute_force_max")
        assert isinstance(opt, FunctionOptimizer)

    def test_pre_bound_kwargs_are_used(self):
        constraints, sketch, marginals = _nn_setup()
        problem_sketch = _explicit_problem_sketch(marginals, sketch, problem_size=3, d=2)
        opt = bind_optimizer("brute_force_max", problem_sketch)
        result = opt.optimize()
        assert isinstance(result, int)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            bind_optimizer("nonexistent_optimizer")


# ---------------------------------------------------------------------------
# optimize (convenience)
# ---------------------------------------------------------------------------

class TestOptimizeConvenience:
    def test_brute_force_max_returns_int(self):
        constraints, sketch, marginals = _nn_setup()
        problem_sketch = _explicit_problem_sketch(marginals, sketch, problem_size=3, d=2)
        result = optimize("brute_force_max", problem_sketch)
        assert isinstance(result, int)

    def test_brute_force_max_correct_index(self):
        constraints, sketch, marginals = _nn_setup()
        problem_sketch = _explicit_problem_sketch(marginals, sketch, problem_size=3, d=2)
        result = optimize("brute_force_max", problem_sketch)
        assert result == 0

    def test_invalid_optimizer_name(self):
        with pytest.raises(ValueError):
            optimize("unknown_optimizer", [1, 2, 3])


# ---------------------------------------------------------------------------
# brute_force_max
# ---------------------------------------------------------------------------

class TestBruteForceMax:
    def test_with_identity_sketch(self):
        marginals = [0.0, 2.0, 0.0, 0.0]
        sketch = _identity_sketch(4)
        problem_sketch = _explicit_problem_sketch(marginals, sketch, problem_size=2, d=2)
        result = brute_force_max(problem_sketch)
        assert int(result) == 1

    def test_single_max_at_last_index(self):
        marginals = [0.0, 0.0, 0.0, 5.0]
        sketch = _identity_sketch(4)
        problem_sketch = _explicit_problem_sketch(marginals, sketch, problem_size=2, d=2)
        assert int(brute_force_max(problem_sketch)) == 3

    def test_with_nn_sketch_favours_index_0(self):
        _, sketch, marginals = _nn_setup()
        problem_sketch = _explicit_problem_sketch(marginals, sketch, problem_size=3, d=2)
        assert int(brute_force_max(problem_sketch)) == 0

    def test_returns_int_compatible(self):
        marginals = [1.0, 2.0]
        sketch = _identity_sketch(2)
        problem_sketch = _explicit_problem_sketch(marginals, sketch, problem_size=1, d=2)
        result = brute_force_max(problem_sketch)
        assert isinstance(int(result), int)

    def test_marginals_sketch_shape_mismatch_raises(self):
        marginals = [1.0, 2.0, 3.0]
        sketch = _identity_sketch(2)
        problem_sketch = _explicit_problem_sketch(marginals, sketch, problem_size=1, d=2)
        with pytest.raises(ValueError):
            brute_force_max(problem_sketch)


# ---------------------------------------------------------------------------
# spin_chain_nn_max
# ---------------------------------------------------------------------------

class TestSpinChainNNMax:
    def test_returns_int(self):
        marginals = [1, 0, 0, 0, 1, 0, 0, 0]
        constraints, _, _ = _nn_setup()
        problem_sketch = _constraint_problem_sketch(marginals, constraints, n=3, k=2, d=2)
        result = spin_chain_nn_max(problem_sketch)
        assert isinstance(result, int)

    def test_favours_all_zeros(self):
        # marginals strongly favour strings where d0=d1=0 and d1=d2=0 → [0,0,0] = index 0
        marginals = [10.0, 0, 0, 0, 10.0, 0, 0, 0]
        constraints, _, _ = _nn_setup()
        problem_sketch = _constraint_problem_sketch(marginals, constraints, n=3, k=2, d=2)
        result = spin_chain_nn_max(problem_sketch)
        assert result == 0

    def test_favours_all_ones(self):
        # Strongly favour [1,1,1] = index 7
        # marginal_tensor[w, s0, s1]: for index [1,1,1] we want w0_prev1_next1 and w1_prev1_next1
        # marginal_tensor[0,1,1]=10, marginal_tensor[1,1,1]=10, others=0
        marginals = [0, 0, 0, 10.0, 0, 0, 0, 10.0]
        constraints, _, _ = _nn_setup()
        problem_sketch = _constraint_problem_sketch(marginals, constraints, n=3, k=2, d=2)
        result = spin_chain_nn_max(problem_sketch)
        assert result == 7

    def test_interaction_size_1(self):
        # k=1: each spin is independent, marginals = [energy(d0=0), energy(d0=1), energy(d1=0), ...]
        # With 3 spins: n_windows=3, marginal_tensor shape = (3, 2)
        # Favour d0=1, d1=0, d2=1 → index = bit_string_to_integer([1,0,1]) = 5
        marginals = [0.0, 10.0, 10.0, 0.0, 0.0, 10.0]  # w0:[0,10], w1:[10,0], w2:[0,10]
        constraints = [{0: 0}, {0: 1}, {1: 0}, {1: 1}, {2: 0}, {2: 1}]
        problem_sketch = _constraint_problem_sketch(marginals, constraints, n=3, k=1, d=2)
        result = spin_chain_nn_max(problem_sketch)
        assert result == 5

    def test_returns_valid_index_range(self):
        n, k, d = 4, 2, 2
        n_windows = n - k + 1
        marginals = list(range(n_windows * d**k))
        csm = ConstraintSketchMap(sketch_length=n, interaction_size=k, sketch_dimension=d)
        csm.build_from_nearest_neighbors()
        problem_sketch = _constraint_problem_sketch(marginals, csm.map, n=n, k=k, d=d)
        result = spin_chain_nn_max(problem_sketch)
        assert 0 <= result < d**n

    def test_ternary_dits_returns_valid_index(self):
        n, k, d = 3, 2, 3
        n_windows = n - k + 1
        marginals = [float(i) for i in range(n_windows * d**k)]
        csm = ConstraintSketchMap(sketch_length=n, interaction_size=k, sketch_dimension=d)
        csm.build_from_nearest_neighbors()
        problem_sketch = _constraint_problem_sketch(marginals, csm.map, n=n, k=k, d=d)
        result = spin_chain_nn_max(problem_sketch)
        assert 0 <= result < d**n

    def test_invalid_marginals_length_raises(self):
        constraints, _, _ = _nn_setup()
        problem_sketch = _constraint_problem_sketch([1.0, 0.0], constraints, n=3, k=2, d=2)
        with pytest.raises(ValueError):
            spin_chain_nn_max(problem_sketch)


# ---------------------------------------------------------------------------
# dual_annealing
# ---------------------------------------------------------------------------

class TestDualAnnealing:
    def test_returns_int(self):
        constraints, _, marginals = _nn_setup()
        problem_sketch = _constraint_problem_sketch(marginals, constraints, n=3, k=2, d=2)
        result = dual_annealing(problem_sketch, opt_func_kwargs={"maxiter": 10, "seed": 42})
        assert isinstance(result, int)

    def test_result_in_valid_range(self):
        constraints, _, marginals = _nn_setup()
        problem_sketch = _constraint_problem_sketch(marginals, constraints, n=3, k=2, d=2)
        result = dual_annealing(problem_sketch, opt_func_kwargs={"maxiter": 10, "seed": 1})
        assert 0 <= result < 2**3

    def test_via_optimizer_interface(self):
        constraints, _, marginals = _nn_setup()
        problem_sketch = _constraint_problem_sketch(marginals, constraints, n=3, k=2, d=2)
        opt = get_optimizer("dual_annealing")
        result = opt.optimize(problem_sketch, opt_func_kwargs={"maxiter": 10, "seed": 0})
        assert isinstance(result, int)

    def test_marginals_constraints_length_mismatch_raises(self):
        constraints, _, marginals = _nn_setup()
        problem_sketch = _constraint_problem_sketch(marginals[:-1], constraints, n=3, k=2, d=2)
        with pytest.raises(ValueError):
            dual_annealing(problem_sketch)


# ---------------------------------------------------------------------------
# simulated_annealing
# ---------------------------------------------------------------------------

class TestSimulatedAnnealing:
    def test_returns_int(self):
        constraints, _, marginals = _nn_setup()
        problem_sketch = _constraint_problem_sketch(marginals, constraints, n=3, k=2, d=2)
        result = simulated_annealing(problem_sketch, max_iter=20)
        assert isinstance(result, int)

    def test_result_in_valid_range(self):
        constraints, _, marginals = _nn_setup()
        problem_sketch = _constraint_problem_sketch(marginals, constraints, n=3, k=2, d=2)
        result = simulated_annealing(problem_sketch, max_iter=50)
        assert 0 <= result < 2**3

    def test_via_optimizer_interface(self):
        constraints, _, marginals = _nn_setup()
        problem_sketch = _constraint_problem_sketch(marginals, constraints, n=3, k=2, d=2)
        opt = get_optimizer("simulated_annealing")
        result = opt.optimize(problem_sketch, max_iter=20)
        assert isinstance(result, int)

    def test_marginals_constraints_length_mismatch_raises(self):
        constraints, _, marginals = _nn_setup()
        problem_sketch = _constraint_problem_sketch(marginals[:-1], constraints, n=3, k=2, d=2)
        with pytest.raises(ValueError):
            simulated_annealing(problem_sketch)


# ---------------------------------------------------------------------------
# FunctionOptimizer
# ---------------------------------------------------------------------------

class TestFunctionOptimizer:
    def test_name_attribute(self):
        opt = get_optimizer("brute_force_max")
        assert opt.name == "brute_force_max"

    def test_with_defaults_creates_new_instance(self):
        opt = get_optimizer("brute_force_max")
        _, sketch, marginals = _nn_setup()
        problem_sketch = _explicit_problem_sketch(marginals, sketch, problem_size=3, d=2)
        bound = opt.with_defaults(problem_sketch)
        assert bound is not opt
        assert isinstance(bound, FunctionOptimizer)

    def test_with_defaults_pre_binds_positional_arg(self):
        _, sketch, marginals = _nn_setup()
        problem_sketch = _explicit_problem_sketch(marginals, sketch, problem_size=3, d=2)
        opt = get_optimizer("brute_force_max")
        bound = opt.with_defaults(problem_sketch)
        result = bound.optimize()
        assert isinstance(result, int)

    def test_optimize_passes_kwargs(self):
        constraints, _, marginals = _nn_setup()
        problem_sketch = _constraint_problem_sketch(marginals, constraints, n=3, k=2, d=2)
        opt = get_optimizer("dual_annealing")
        result = opt.optimize(
            problem_sketch,
            opt_func_kwargs={"maxiter": 5, "seed": 0},
        )
        assert isinstance(result, int)
