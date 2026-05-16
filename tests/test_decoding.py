"""Tests for decoding-related APIs."""

import numpy as np
import pytest

from troma import (
    CombinatorialProblem,
    CombinatorialProblemSketch,
    bind_matching_pursuit,
    get_matching_pursuit,
    matching_pursuit,
)
from troma.matching_pursuit import (
    FunctionMatchingPursuit,
    list_matching_pursuits,
)
from troma.core.decoding_proced import matchingpursuit_explicit, matchingpursuit_abstract
from troma import ConstraintSketchMap, ExplicitSketchMap
from troma.optimization.optimizer import get_optimizer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _explicit_setup():
    """Return (marginals, sketch, problem_sketch) for f([0,0,0])=1 on n=3, k=2, d=2."""
    sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
    sm.build_from_nearest_neighbors()
    sketch = sm.map
    marginals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    problem = CombinatorialProblem(lambda _: 0.0, problem_size=3, problem_dimension=2)
    problem_sketch = CombinatorialProblemSketch(problem=problem, sketch_map=sm, sketch_values=marginals)
    return marginals, sketch, problem_sketch


def _abstract_setup():
    """Return (marginals, constraints, problem_sketch) for f([0,0,0])=1 on n=3, k=2, d=2."""
    sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
    sm.build_from_nearest_neighbors()
    constraints = sm.map
    marginals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    problem = CombinatorialProblem(lambda _: 0.0, problem_size=3, problem_dimension=2)
    problem_sketch = CombinatorialProblemSketch(problem=problem, sketch_map=sm, sketch_values=marginals)
    return marginals, constraints, problem_sketch


# ---------------------------------------------------------------------------
# list_matching_pursuits
# ---------------------------------------------------------------------------

class TestListMatchingPursuits:
    def test_returns_list(self):
        result = list_matching_pursuits()
        assert isinstance(result, list)

    def test_sorted(self):
        result = list_matching_pursuits()
        assert result == sorted(result)

    def test_contains_expected_backends(self):
        result = list_matching_pursuits()
        assert "explicit" in result
        assert "abstract" in result

    def test_non_empty(self):
        assert len(list_matching_pursuits()) >= 2


# ---------------------------------------------------------------------------
# get_matching_pursuit
# ---------------------------------------------------------------------------

class TestGetMatchingPursuit:
    def test_returns_function_matching_pursuit(self):
        mp = get_matching_pursuit("explicit")
        assert isinstance(mp, FunctionMatchingPursuit)

    def test_abstract_backend_loadable(self):
        mp = get_matching_pursuit("abstract")
        assert isinstance(mp, FunctionMatchingPursuit)

    def test_case_insensitive(self):
        mp = get_matching_pursuit("Explicit")
        assert isinstance(mp, FunctionMatchingPursuit)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            get_matching_pursuit("nonexistent_backend")

    def test_name_attribute_explicit(self):
        mp = get_matching_pursuit("explicit")
        assert mp.name == "explicit"

    def test_name_attribute_abstract(self):
        mp = get_matching_pursuit("abstract")
        assert mp.name == "abstract"


# ---------------------------------------------------------------------------
# bind_matching_pursuit
# ---------------------------------------------------------------------------

class TestBindMatchingPursuit:
    def test_returns_function_matching_pursuit(self):
        mp = bind_matching_pursuit("explicit")
        assert isinstance(mp, FunctionMatchingPursuit)

    def test_pre_bound_args_used_on_run(self):
        _, _, problem_sketch = _explicit_setup()
        mp = bind_matching_pursuit("explicit", problem_sketch)
        result = mp.run(1)  # iteration_number=1
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 2

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            bind_matching_pursuit("bad_backend")


# ---------------------------------------------------------------------------
# matching_pursuit (convenience)
# ---------------------------------------------------------------------------

class TestMatchingPursuitConvenience:
    def test_requires_problem_sketch_instance(self):
        with pytest.raises(TypeError):
            matching_pursuit("abstract", iteration_number=1)


# ---------------------------------------------------------------------------
# matchingpursuit_explicit
# ---------------------------------------------------------------------------

class TestMatchingPursuitExplicit:
    def test_1_iteration_recovers_index_0(self):
        _, _, problem_sketch = _explicit_setup()
        result = matchingpursuit_explicit(problem_sketch, iteration_number=1)
        assert result.shape == (1, 2)
        assert int(result[0, 0]) == 0
        assert result[0, 1] == pytest.approx(1.0)

    def test_returns_2d_array(self):
        _, _, problem_sketch = _explicit_setup()
        result = matchingpursuit_explicit(problem_sketch, iteration_number=2)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_fixed_step(self):
        _, _, problem_sketch = _explicit_setup()
        result = matchingpursuit_explicit(problem_sketch, iteration_number=1, step=1.0)
        assert int(result[0, 0]) == 0
        assert result[0, 1] == pytest.approx(1.0)

    def test_residual_shrinks(self):
        """After recovering the only non-zero component the residual should be zero."""
        marginals, sketch, problem_sketch = _explicit_setup()
        # 1 iteration recovers everything
        result = matchingpursuit_explicit(problem_sketch, iteration_number=1)
        # Reconstruct residual manually
        idx = int(result[0, 0])
        coeff = result[0, 1]
        At = np.asarray(sketch[:, idx]).flatten()
        residual = np.array(marginals) - coeff * At
        assert np.allclose(residual, 0.0, atol=1e-10)

    def test_bad_optimizer_type_raises(self):
        _, _, problem_sketch = _explicit_setup()
        with pytest.raises(TypeError):
            matchingpursuit_explicit(problem_sketch, iteration_number=1, optimizer="bad")

    def test_with_custom_optimizer(self):
        _, _, problem_sketch = _explicit_setup()
        opt = get_optimizer("brute_force_max")
        result = matchingpursuit_explicit(problem_sketch, iteration_number=1, optimizer=opt)
        assert int(result[0, 0]) == 0

    def test_zero_marginals_returns_empty_or_zero_coeffs(self):
        _, _, problem_sketch = _explicit_setup()
        problem_sketch.sketch_values = [0.0] * 8
        result = matchingpursuit_explicit(problem_sketch, iteration_number=1)
        # Coefficients should all be zero
        if result.shape[0] > 0:
            assert np.allclose(result[:, 1], 0.0)

    def test_multiple_iterations(self):
        """More than one iteration should not raise."""
        _, _, problem_sketch = _explicit_setup()
        result = matchingpursuit_explicit(problem_sketch, iteration_number=3)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# matchingpursuit_abstract
# ---------------------------------------------------------------------------

class TestMatchingPursuitAbstract:
    def test_1_iteration_recovers_index_0(self):
        _, _, problem_sketch = _abstract_setup()
        result = matchingpursuit_abstract(problem_sketch, iteration_number=1)
        assert result.shape == (1, 2)
        assert int(result[0, 0]) == 0
        assert result[0, 1] == pytest.approx(1.0)

    def test_returns_2d_array(self):
        _, _, problem_sketch = _abstract_setup()
        result = matchingpursuit_abstract(problem_sketch, iteration_number=2)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_fixed_step(self):
        _, _, problem_sketch = _abstract_setup()
        result = matchingpursuit_abstract(problem_sketch, iteration_number=1, step=1.0)
        assert int(result[0, 0]) == 0
        assert result[0, 1] == pytest.approx(1.0)

    def test_bad_optimizer_type_raises(self):
        _, _, problem_sketch = _abstract_setup()
        with pytest.raises(TypeError):
            matchingpursuit_abstract(problem_sketch, iteration_number=1, optimizer="bad")

    def test_with_custom_optimizer(self):
        _, _, problem_sketch = _abstract_setup()
        opt = get_optimizer("spin_chain_nn_max")
        result = matchingpursuit_abstract(problem_sketch, iteration_number=1, optimizer=opt)
        assert int(result[0, 0]) == 0

    def test_residual_shrinks_after_recovery(self):
        """One iteration on a 1-sparse signal should yield zero residual."""
        marginals, constraints, problem_sketch = _abstract_setup()
        result = matchingpursuit_abstract(problem_sketch, iteration_number=1)
        idx = int(result[0, 0])
        coeff = result[0, 1]
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.map = constraints
        At = sm.reconstruct_structured_matrix_column(idx).astype(float)
        residual = np.array(marginals) - coeff * At
        assert np.allclose(residual, 0.0, atol=1e-10)

    def test_multiple_iterations(self):
        _, _, problem_sketch = _abstract_setup()
        result = matchingpursuit_abstract(problem_sketch, iteration_number=5)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# FunctionMatchingPursuit
# ---------------------------------------------------------------------------

class TestFunctionMatchingPursuit:
    def test_with_defaults_creates_new_instance(self):
        mp = get_matching_pursuit("explicit")
        mp2 = mp.with_defaults()
        assert mp2 is not mp
        assert isinstance(mp2, FunctionMatchingPursuit)

    def test_with_defaults_pre_binds_positional_arg(self):
        _, _, problem_sketch = _explicit_setup()
        mp = get_matching_pursuit("explicit")
        bound = mp.with_defaults(problem_sketch)
        result = bound.run(1)  # iteration_number=1
        assert int(result[0, 0]) == 0

    def test_run_explicit_backend(self):
        _, _, problem_sketch = _explicit_setup()
        mp = get_matching_pursuit("explicit")
        result = mp.run(problem_sketch, 1)
        assert isinstance(result, np.ndarray)

    def test_run_abstract_backend(self):
        _, _, problem_sketch = _abstract_setup()
        mp = get_matching_pursuit("abstract")
        result = mp.run(problem_sketch, iteration_number=1)
        assert isinstance(result, np.ndarray)
