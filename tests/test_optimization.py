"""Tests for the public optimization API (troma.optimization)."""

import numpy as np
import pytest

from troma import bind_optimizer, get_optimizer, optimize
from troma.optimization import (
    list_optimizers,
    brute_force_max,
    spin_chain_nn_max,
    dual_annealing,
    simulated_annealing,
)
from troma.optimization.optimizer import FunctionOptimizer
from troma.sketchs import nearest_neighbors_interactions_sketch
from troma.sketchs import constraints_for_nearest_neighbors_interactions

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _identity_sketch(n: int) -> np.ndarray:
    return np.eye(n)


def _nn_setup(n=3, k=2, d=2):
    """Return (constraints, explicit_sketch, marginals favouring index 0)."""
    constraints = constraints_for_nearest_neighbors_interactions(n, k, d)
    sketch = nearest_neighbors_interactions_sketch(n, k, d)
    # f([0,0,0])=1 → marginals = [1,0,0,0,1,0,0,0]
    marginals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    return constraints, sketch, marginals


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
        # Pre-bind sketch as a keyword argument
        opt = bind_optimizer("brute_force_max", sketch=sketch)
        result = opt.optimize(marginals)
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
        result = optimize("brute_force_max", marginals, sketch)
        assert isinstance(result, int)

    def test_brute_force_max_correct_index(self):
        constraints, sketch, marginals = _nn_setup()
        result = optimize("brute_force_max", marginals, sketch)
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
        result = brute_force_max(marginals, sketch)
        assert int(result) == 1

    def test_single_max_at_last_index(self):
        marginals = [0.0, 0.0, 0.0, 5.0]
        sketch = _identity_sketch(4)
        assert int(brute_force_max(marginals, sketch)) == 3

    def test_with_nn_sketch_favours_index_0(self):
        _, sketch, marginals = _nn_setup()
        assert int(brute_force_max(marginals, sketch)) == 0

    def test_returns_int_compatible(self):
        marginals = [1.0, 2.0]
        sketch = _identity_sketch(2)
        result = brute_force_max(marginals, sketch)
        assert isinstance(int(result), int)

    def test_marginals_sketch_shape_mismatch_raises(self):
        marginals = [1.0, 2.0, 3.0]
        sketch = _identity_sketch(2)
        with pytest.raises(ValueError):
            brute_force_max(marginals, sketch)


# ---------------------------------------------------------------------------
# spin_chain_nn_max
# ---------------------------------------------------------------------------

class TestSpinChainNNMax:
    def test_returns_int(self):
        marginals = [1, 0, 0, 0, 1, 0, 0, 0]
        result = spin_chain_nn_max(marginals, 3, interaction_size=2, dit_dimension=2)
        assert isinstance(result, int)

    def test_favours_all_zeros(self):
        # marginals strongly favour strings where d0=d1=0 and d1=d2=0 → [0,0,0] = index 0
        marginals = [10.0, 0, 0, 0, 10.0, 0, 0, 0]
        result = spin_chain_nn_max(marginals, 3, 2, 2)
        assert result == 0

    def test_favours_all_ones(self):
        # Strongly favour [1,1,1] = index 7
        # marginal_tensor[w, s0, s1]: for index [1,1,1] we want w0_prev1_next1 and w1_prev1_next1
        # marginal_tensor[0,1,1]=10, marginal_tensor[1,1,1]=10, others=0
        marginals = [0, 0, 0, 10.0, 0, 0, 0, 10.0]
        result = spin_chain_nn_max(marginals, 3, 2, 2)
        assert result == 7

    def test_interaction_size_1(self):
        # k=1: each spin is independent, marginals = [energy(d0=0), energy(d0=1), energy(d1=0), ...]
        # With 3 spins: n_windows=3, marginal_tensor shape = (3, 2)
        # Favour d0=1, d1=0, d2=1 → index = bit_string_to_integer([1,0,1]) = 5
        marginals = [0.0, 10.0, 10.0, 0.0, 0.0, 10.0]  # w0:[0,10], w1:[10,0], w2:[0,10]
        result = spin_chain_nn_max(marginals, 3, interaction_size=1, dit_dimension=2)
        assert result == 5

    def test_returns_valid_index_range(self):
        n, k, d = 4, 2, 2
        n_windows = n - k + 1
        marginals = list(range(n_windows * d**k))
        result = spin_chain_nn_max(marginals, n, k, d)
        assert 0 <= result < d**n

    def test_ternary_dits_returns_valid_index(self):
        n, k, d = 3, 2, 3
        n_windows = n - k + 1
        marginals = [float(i) for i in range(n_windows * d**k)]
        result = spin_chain_nn_max(marginals, n, k, d)
        assert 0 <= result < d**n

    def test_invalid_marginals_length_raises(self):
        with pytest.raises(ValueError):
            spin_chain_nn_max([1.0, 0.0], 3, 2, 2)


# ---------------------------------------------------------------------------
# dual_annealing
# ---------------------------------------------------------------------------

class TestDualAnnealing:
    def test_returns_int(self):
        constraints, _, marginals = _nn_setup()
        result = dual_annealing(marginals, constraints, dit_string_length=3, dit_dimension=2,
                                opt_func_kwargs={"maxiter": 10, "seed": 42})
        assert isinstance(result, int)

    def test_result_in_valid_range(self):
        constraints, _, marginals = _nn_setup()
        result = dual_annealing(marginals, constraints, dit_string_length=3, dit_dimension=2,
                                opt_func_kwargs={"maxiter": 10, "seed": 1})
        assert 0 <= result < 2**3

    def test_via_optimizer_interface(self):
        constraints, _, marginals = _nn_setup()
        opt = get_optimizer("dual_annealing")
        result = opt.optimize(marginals, constraints, dit_string_length=3, dit_dimension=2,
                              opt_func_kwargs={"maxiter": 10, "seed": 0})
        assert isinstance(result, int)

    def test_marginals_constraints_length_mismatch_raises(self):
        constraints, _, marginals = _nn_setup()
        with pytest.raises(ValueError):
            dual_annealing(marginals[:-1], constraints, dit_string_length=3, dit_dimension=2)


# ---------------------------------------------------------------------------
# simulated_annealing
# ---------------------------------------------------------------------------

class TestSimulatedAnnealing:
    def test_returns_int(self):
        constraints, _, marginals = _nn_setup()
        result = simulated_annealing(marginals, constraints, dit_string_length=3,
                                     dit_dimension=2, max_iter=20)
        assert isinstance(result, int)

    def test_result_in_valid_range(self):
        constraints, _, marginals = _nn_setup()
        result = simulated_annealing(marginals, constraints, dit_string_length=3,
                                     dit_dimension=2, max_iter=50)
        assert 0 <= result < 2**3

    def test_via_optimizer_interface(self):
        constraints, _, marginals = _nn_setup()
        opt = get_optimizer("simulated_annealing")
        result = opt.optimize(marginals, constraints, dit_string_length=3,
                              dit_dimension=2, max_iter=20)
        assert isinstance(result, int)

    def test_marginals_constraints_length_mismatch_raises(self):
        constraints, _, marginals = _nn_setup()
        with pytest.raises(ValueError):
            simulated_annealing(marginals[:-1], constraints, dit_string_length=3, dit_dimension=2)


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
        bound = opt.with_defaults(marginals)
        assert bound is not opt
        assert isinstance(bound, FunctionOptimizer)

    def test_with_defaults_pre_binds_positional_arg(self):
        _, sketch, marginals = _nn_setup()
        opt = get_optimizer("brute_force_max")
        bound = opt.with_defaults(marginals)
        result = bound.optimize(sketch)
        assert isinstance(result, int)

    def test_optimize_passes_kwargs(self):
        constraints, _, marginals = _nn_setup()
        opt = get_optimizer("dual_annealing")
        result = opt.optimize(
            marginals,
            constraints,
            dit_string_length=3,
            dit_dimension=2,
            opt_func_kwargs={"maxiter": 5, "seed": 0},
        )
        assert isinstance(result, int)
