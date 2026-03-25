"""Tests for the public decoding API (troma.decoding)."""

import numpy as np
import pytest

from troma import bind_matching_pursuit, get_matching_pursuit, matching_pursuit
from troma.decoding import (
    list_matching_pursuits,
    matchingpursuit_explicit,
    matchingpursuit_abstract,
)
from troma.decoding.matching_pursuit import FunctionMatchingPursuit
from troma.sketchs import (
    nearest_neighbors_interactions_sketch,
    constraints_for_nearest_neighbors_interactions,
)
from troma.optimization.optimizer import get_optimizer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _explicit_setup():
    """Return (marginals, sketch) for f([0,0,0])=1 on n=3, k=2, d=2."""
    sketch = nearest_neighbors_interactions_sketch(3, 2, 2)
    marginals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    return marginals, sketch


def _abstract_setup():
    """Return (marginals, constraints) for f([0,0,0])=1 on n=3, k=2, d=2."""
    constraints = constraints_for_nearest_neighbors_interactions(3, 2, 2)
    marginals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    return marginals, constraints


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
        marginals, sketch = _explicit_setup()
        mp = bind_matching_pursuit("explicit", marginals, sketch)
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
    def test_explicit_returns_ndarray(self):
        marginals, sketch = _explicit_setup()
        result = matching_pursuit("explicit", marginals, sketch, 1)
        assert isinstance(result, np.ndarray)

    def test_abstract_returns_ndarray(self):
        marginals, constraints = _abstract_setup()
        result = matching_pursuit(
            "abstract", marginals, constraints, 3, iteration_number=1,
            interaction_size=2, dit_dimension=2,
        )
        assert isinstance(result, np.ndarray)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            matching_pursuit("no_such_backend", [], [], 1)


# ---------------------------------------------------------------------------
# matchingpursuit_explicit
# ---------------------------------------------------------------------------

class TestMatchingPursuitExplicit:
    def test_1_iteration_recovers_index_0(self):
        marginals, sketch = _explicit_setup()
        result = matchingpursuit_explicit(marginals, sketch, iteration_number=1)
        assert result.shape == (1, 2)
        assert int(result[0, 0]) == 0
        assert result[0, 1] == pytest.approx(1.0)

    def test_returns_2d_array(self):
        marginals, sketch = _explicit_setup()
        result = matchingpursuit_explicit(marginals, sketch, iteration_number=2)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_fixed_step(self):
        marginals, sketch = _explicit_setup()
        result = matchingpursuit_explicit(marginals, sketch, iteration_number=1, step=1.0)
        assert int(result[0, 0]) == 0
        assert result[0, 1] == pytest.approx(1.0)

    def test_residual_shrinks(self):
        """After recovering the only non-zero component the residual should be zero."""
        marginals, sketch = _explicit_setup()
        r = list(marginals)
        # 1 iteration recovers everything
        result = matchingpursuit_explicit(r, sketch, iteration_number=1)
        # Reconstruct residual manually
        idx = int(result[0, 0])
        coeff = result[0, 1]
        At = np.asarray(sketch[:, idx]).flatten()
        residual = np.array(marginals) - coeff * At
        assert np.allclose(residual, 0.0, atol=1e-10)

    def test_bad_optimizer_type_raises(self):
        marginals, sketch = _explicit_setup()
        with pytest.raises(TypeError):
            matchingpursuit_explicit(marginals, sketch, iteration_number=1, optimizer="bad")

    def test_with_custom_optimizer(self):
        marginals, sketch = _explicit_setup()
        opt = get_optimizer("brute_force_max")
        result = matchingpursuit_explicit(marginals, sketch, iteration_number=1, optimizer=opt)
        assert int(result[0, 0]) == 0

    def test_zero_marginals_returns_empty_or_zero_coeffs(self):
        _, sketch = _explicit_setup()
        marginals = [0.0] * 8
        result = matchingpursuit_explicit(marginals, sketch, iteration_number=1)
        # Coefficients should all be zero
        if result.shape[0] > 0:
            assert np.allclose(result[:, 1], 0.0)

    def test_multiple_iterations(self):
        """More than one iteration should not raise."""
        marginals, sketch = _explicit_setup()
        result = matchingpursuit_explicit(marginals, sketch, iteration_number=3)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# matchingpursuit_abstract
# ---------------------------------------------------------------------------

class TestMatchingPursuitAbstract:
    def test_1_iteration_recovers_index_0(self):
        marginals, constraints = _abstract_setup()
        result = matchingpursuit_abstract(
            marginals, constraints, dit_string_length=3,
            iteration_number=1, interaction_size=2, dit_dimension=2,
        )
        assert result.shape == (1, 2)
        assert int(result[0, 0]) == 0
        assert result[0, 1] == pytest.approx(1.0)

    def test_returns_2d_array(self):
        marginals, constraints = _abstract_setup()
        result = matchingpursuit_abstract(
            marginals, constraints, dit_string_length=3,
            iteration_number=2, interaction_size=2, dit_dimension=2,
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_fixed_step(self):
        marginals, constraints = _abstract_setup()
        result = matchingpursuit_abstract(
            marginals, constraints, dit_string_length=3,
            iteration_number=1, step=1.0, interaction_size=2, dit_dimension=2,
        )
        assert int(result[0, 0]) == 0
        assert result[0, 1] == pytest.approx(1.0)

    def test_bad_optimizer_type_raises(self):
        marginals, constraints = _abstract_setup()
        with pytest.raises(TypeError):
            matchingpursuit_abstract(
                marginals, constraints, dit_string_length=3,
                iteration_number=1, optimizer="bad",
            )

    def test_with_custom_optimizer(self):
        marginals, constraints = _abstract_setup()
        opt = get_optimizer("spin_chain_nn_max")
        result = matchingpursuit_abstract(
            marginals, constraints, dit_string_length=3,
            iteration_number=1, optimizer=opt,
            interaction_size=2, dit_dimension=2,
        )
        assert int(result[0, 0]) == 0

    def test_residual_shrinks_after_recovery(self):
        """One iteration on a 1-sparse signal should yield zero residual."""
        from troma.sketchs import reconstruct_structured_matrix_column
        marginals, constraints = _abstract_setup()
        result = matchingpursuit_abstract(
            marginals, constraints, dit_string_length=3,
            iteration_number=1, interaction_size=2, dit_dimension=2,
        )
        idx = int(result[0, 0])
        coeff = result[0, 1]
        At = reconstruct_structured_matrix_column(idx, constraints, 3, 2).astype(float)
        residual = np.array(marginals) - coeff * At
        assert np.allclose(residual, 0.0, atol=1e-10)

    def test_multiple_iterations(self):
        marginals, constraints = _abstract_setup()
        result = matchingpursuit_abstract(
            marginals, constraints, dit_string_length=3,
            iteration_number=5, interaction_size=2, dit_dimension=2,
        )
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
        marginals, sketch = _explicit_setup()
        mp = get_matching_pursuit("explicit")
        bound = mp.with_defaults(marginals, sketch)
        result = bound.run(1)  # iteration_number=1
        assert int(result[0, 0]) == 0

    def test_run_explicit_backend(self):
        marginals, sketch = _explicit_setup()
        mp = get_matching_pursuit("explicit")
        result = mp.run(marginals, sketch, 1)
        assert isinstance(result, np.ndarray)

    def test_run_abstract_backend(self):
        marginals, constraints = _abstract_setup()
        mp = get_matching_pursuit("abstract")
        result = mp.run(
            marginals, constraints, 3,
            iteration_number=1, interaction_size=2, dit_dimension=2,
        )
        assert isinstance(result, np.ndarray)
