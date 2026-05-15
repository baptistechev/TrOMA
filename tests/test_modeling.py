"""Tests for the high-level CombinatorialProblem pipeline API."""

import numpy as np
import pytest

from troma import (
    CombinatorialProblem,
    ConstraintSketchMap,
    MatchingPursuitResults,
    matching_pursuit,
)
from troma.combinatorial_problem import RestrictedProblem
from troma.core.structure import Restriction


def _sum_of_bits(dit_string):
    return float(sum(dit_string))


def _constant_one(dit_string):
    return 1.0


def _zero(dit_string):
    return 0.0


# ---------------------------------------------------------------------------
# CombinatorialProblem construction
# ---------------------------------------------------------------------------

class TestCombinatorialProblemInit:
    def test_basic_init(self):
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        assert prob.problem_size == 4
        assert prob.problem_dimension == 2

    def test_custom_dimension(self):
        prob = CombinatorialProblem(_sum_of_bits, problem_size=3, problem_dimension=3)
        assert prob.problem_dimension == 3

    def test_stores_objective(self):
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        assert prob.objective_function is _sum_of_bits


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def test_returns_sample(self):
        np.random.seed(0)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        sample = prob.sampling(n_samples=20)
        assert sample is not None

    def test_fills_sample_attribute(self):
        np.random.seed(1)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        prob.sampling(n_samples=20)
        assert len(prob.sample.values) > 0

    def test_sample_values_nonzero(self):
        np.random.seed(2)
        prob = CombinatorialProblem(_constant_one, problem_size=3)
        prob.sampling(n_samples=10)
        assert all(v != 0 for v in prob.sample.values)

    def test_zero_objective_gives_empty_sample(self):
        np.random.seed(3)
        prob = CombinatorialProblem(_zero, problem_size=3)
        prob.sampling(n_samples=10)
        assert len(prob.sample.values) == 0

    def test_dit_strings_have_correct_length(self):
        np.random.seed(4)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=5)
        prob.sampling(n_samples=20)
        for ds in prob.sample.dit_strings:
            assert len(ds) == 5


# ---------------------------------------------------------------------------
# Sketching
# ---------------------------------------------------------------------------

class TestSketching:
    def test_sketching_with_sketch_map(self):
        np.random.seed(5)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        prob.sampling(n_samples=30)
        sm = ConstraintSketchMap(
            sketch_length=4, interaction_size=2, sketch_dimension=2,
            constraints="nearest_neighbors",
        )
        sketch = prob.sketching(sm)
        assert sketch is not None
        assert hasattr(sketch, "sketch_values")

    def test_sketching_with_string_shorthand(self):
        np.random.seed(6)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        prob.sampling(n_samples=30)
        sketch = prob.sketching("nearest_neighbors", interaction_size=2)
        assert sketch is not None

    def test_sketching_string_requires_interaction_size(self):
        np.random.seed(7)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        prob.sampling(n_samples=10)
        with pytest.raises(TypeError):
            prob.sketching("nearest_neighbors")

    def test_sketching_wrong_sketch_map_size_raises(self):
        np.random.seed(8)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        prob.sampling(n_samples=10)
        sm = ConstraintSketchMap(
            sketch_length=3, interaction_size=2, sketch_dimension=2,
            constraints="nearest_neighbors",
        )
        with pytest.raises(ValueError):
            prob.sketching(sm)

    def test_invalid_constraints_type_raises(self):
        np.random.seed(9)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        prob.sampling(n_samples=10)
        with pytest.raises((TypeError, ValueError)):
            prob.sketching(42)


# ---------------------------------------------------------------------------
# matching_pursuit integration
# ---------------------------------------------------------------------------

class TestMatchingPursuitIntegration:
    def test_returns_results_object(self):
        np.random.seed(10)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        prob.sampling(n_samples=40)
        sketch = prob.sketching("nearest_neighbors", interaction_size=2)
        result = matching_pursuit(sketch, iteration_number=3)
        assert isinstance(result, MatchingPursuitResults)

    def test_positions_in_valid_range(self):
        np.random.seed(11)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        prob.sampling(n_samples=40)
        sketch = prob.sketching("nearest_neighbors", interaction_size=2)
        result = matching_pursuit(sketch, iteration_number=3)
        for pos in result.positions:
            assert 0 <= pos < 2**4

    def test_n_lines_equals_iteration_number(self):
        np.random.seed(12)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        prob.sampling(n_samples=40)
        sketch = prob.sketching("nearest_neighbors", interaction_size=2)
        result = matching_pursuit(sketch, iteration_number=5)
        assert result.n_lines == 5

    def test_dit_strings_correct_length(self):
        np.random.seed(13)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        prob.sampling(n_samples=40)
        sketch = prob.sketching("nearest_neighbors", interaction_size=2)
        result = matching_pursuit(sketch, iteration_number=2)
        for ds in result.dit_strings:
            assert len(ds) == 4

    def test_to_dict_has_expected_keys(self):
        np.random.seed(14)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=4)
        prob.sampling(n_samples=40)
        sketch = prob.sketching("nearest_neighbors", interaction_size=2)
        result = matching_pursuit(sketch, iteration_number=2)
        d = result.to_dict()
        assert {"positions", "values", "n_lines", "dit_strings"}.issubset(d)

    def test_requires_problem_sketch_instance(self):
        with pytest.raises(TypeError):
            matching_pursuit("not_a_sketch", iteration_number=1)


# ---------------------------------------------------------------------------
# RestrictedProblem
# ---------------------------------------------------------------------------

class TestRestrictedProblem:
    def test_restrict_returns_restricted_problem(self):
        prob = CombinatorialProblem(_sum_of_bits, problem_size=5)
        restriction = Restriction(dit_restrictions=[1, 2, 3, 4])
        restricted = prob.restrict(restriction)
        assert isinstance(restricted, RestrictedProblem)

    def test_restricted_sampling_and_sketching(self):
        np.random.seed(15)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=5)
        restriction = Restriction(dit_restrictions=[1, 2, 3, 4])
        restricted = prob.restrict(restriction)
        restricted.sampling(n_samples=30)
        sketch = restricted.sketching("nearest_neighbors", interaction_size=2)
        assert sketch is not None

    def test_restricted_matching_pursuit_positions_in_full_range(self):
        np.random.seed(16)
        prob = CombinatorialProblem(_sum_of_bits, problem_size=5)
        restriction = Restriction(dit_restrictions=[1, 2, 3, 4])
        restricted = prob.restrict(restriction)
        restricted.sampling(n_samples=40)
        sketch = restricted.sketching("nearest_neighbors", interaction_size=2)
        result = matching_pursuit(sketch, iteration_number=3)
        for pos in result.positions:
            assert 0 <= pos < 2**5
