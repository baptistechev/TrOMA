"""Tests for the public sketch API (troma.sketchs and troma.ConstraintSketch/ExplicitSketch)."""

import numpy as np
import pytest

from troma import ConstraintSketch, ExplicitSketch
from troma.sketchs import (
    constraint_compute_marginal,
    constraints_for_all_interactions,
    constraints_for_nearest_neighbors_interactions,
    reconstruct_structured_matrix_column,
    explicit_compute_marginal,
    nearest_neighbors_interactions_sketch,
    all_interactions_sketch,
    random_sketch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nn_constraints_3_2_2():
    """Nearest-neighbor constraints for n=3, k=2, d=2 → 8 constraints."""
    return constraints_for_nearest_neighbors_interactions(3, 2, 2)


@pytest.fixture
def nn_sketch_3_2_2():
    """Explicit nearest-neighbor sketch matrix for n=3, k=2, d=2 (8×8)."""
    return nearest_neighbors_interactions_sketch(3, 2, 2)


@pytest.fixture
def sparse_spectrum():
    """Sparse representation of f([0,0,0])=1, all others = 0."""
    input_dits = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
    values = [1.0, 0.0, 0.0, 0.0]
    return input_dits, values


@pytest.fixture
def full_spectrum():
    """Dense spectrum for n=3, d=2: f([0,0,0])=1, others=0."""
    v = [0.0] * 8
    v[0] = 1.0
    return v


# ---------------------------------------------------------------------------
# constraints_for_nearest_neighbors_interactions
# ---------------------------------------------------------------------------

class TestConstraintsNearestNeighbors:
    def test_count(self):
        # n=3, k=2, d=2 → 2 windows × 4 value combinations = 8 constraints
        c = constraints_for_nearest_neighbors_interactions(3, 2, 2)
        assert len(c) == 8

    def test_count_ternary(self):
        # n=3, k=2, d=3 → 2 windows × 9 value combinations = 18 constraints
        c = constraints_for_nearest_neighbors_interactions(3, 2, 3)
        assert len(c) == 18

    def test_all_dicts(self, nn_constraints_3_2_2):
        assert all(isinstance(c, dict) for c in nn_constraints_3_2_2)

    def test_keys_are_consecutive(self, nn_constraints_3_2_2):
        # First 4 constraints should involve positions (0,1)
        first_four = nn_constraints_3_2_2[:4]
        for c in first_four:
            assert set(c.keys()) == {0, 1}
        # Last 4 constraints should involve positions (1,2)
        last_four = nn_constraints_3_2_2[4:]
        for c in last_four:
            assert set(c.keys()) == {1, 2}

    def test_single_window_equals_all_interactions(self):
        # When k == n, there is only one window, same as all interactions
        nn = constraints_for_nearest_neighbors_interactions(3, 3, 2)
        ai = constraints_for_all_interactions(3, 3, 2)
        assert len(nn) == len(ai)

    def test_invalid_interaction_size_too_large(self):
        with pytest.raises(ValueError):
            constraints_for_nearest_neighbors_interactions(3, 4, 2)

    @pytest.mark.parametrize("n,k,d,exc", [
        (-1, 2, 2, ValueError),
        (3, -1, 2, ValueError),
        (1.5, 2, 2, TypeError),
        (3, 2.0, 2, TypeError),
        (3, 2, 1.5, TypeError),
    ])
    def test_invalid_inputs(self, n, k, d, exc):
        with pytest.raises(exc):
            constraints_for_nearest_neighbors_interactions(n, k, d)


# ---------------------------------------------------------------------------
# constraints_for_all_interactions
# ---------------------------------------------------------------------------

class TestConstraintsAllInteractions:
    def test_count(self):
        # n=3, k=2, d=2 → C(3,2)=3 windows × 4 = 12 constraints
        c = constraints_for_all_interactions(3, 2, 2)
        assert len(c) == 12

    def test_covers_all_pairs(self):
        c = constraints_for_all_interactions(3, 2, 2)
        all_key_sets = [frozenset(ci.keys()) for ci in c]
        assert frozenset({0, 1}) in all_key_sets
        assert frozenset({0, 2}) in all_key_sets
        assert frozenset({1, 2}) in all_key_sets

    def test_invalid_interaction_size_too_large(self):
        with pytest.raises(ValueError):
            constraints_for_all_interactions(3, 4, 2)


# ---------------------------------------------------------------------------
# reconstruct_structured_matrix_column
# ---------------------------------------------------------------------------

class TestReconstructColumn:
    def test_index_0_binary(self, nn_constraints_3_2_2):
        # [0,0,0] satisfies {0:0,1:0} and {1:0,2:0} → column = [1,0,0,0,1,0,0,0]
        col = reconstruct_structured_matrix_column(0, nn_constraints_3_2_2, 3, 2)
        assert col.tolist() == [1, 0, 0, 0, 1, 0, 0, 0]

    def test_index_7_binary(self, nn_constraints_3_2_2):
        # [1,1,1] satisfies {0:1,1:1} and {1:1,2:1} → column = [0,0,0,1,0,0,0,1]
        col = reconstruct_structured_matrix_column(7, nn_constraints_3_2_2, 3, 2)
        assert col.tolist() == [0, 0, 0, 1, 0, 0, 0, 1]

    def test_returns_ndarray(self, nn_constraints_3_2_2):
        col = reconstruct_structured_matrix_column(0, nn_constraints_3_2_2, 3, 2)
        assert isinstance(col, np.ndarray)

    def test_length_equals_number_of_constraints(self, nn_constraints_3_2_2):
        col = reconstruct_structured_matrix_column(3, nn_constraints_3_2_2, 3, 2)
        assert len(col) == len(nn_constraints_3_2_2)

    def test_index_out_of_range(self, nn_constraints_3_2_2):
        with pytest.raises(ValueError):
            reconstruct_structured_matrix_column(100, nn_constraints_3_2_2, 3, 2)

    def test_invalid_constraints_type(self):
        with pytest.raises(TypeError):
            reconstruct_structured_matrix_column(0, "bad", 3, 2)


# ---------------------------------------------------------------------------
# constraint_compute_marginal
# ---------------------------------------------------------------------------

class TestConstraintComputeMarginal:
    def test_single_dict_constraint(self, sparse_spectrum):
        input_dits, values = sparse_spectrum
        # All strings with d0=0 → [0,0,0] and [0,0,1] and [0,1,0] → 1.0+0+0 = 1.0
        result = constraint_compute_marginal(input_dits, values, {0: 0})
        assert result == pytest.approx(1.0)

    def test_single_dict_constraint_no_match(self, sparse_spectrum):
        input_dits, values = sparse_spectrum
        result = constraint_compute_marginal(input_dits, values, {0: 1})
        assert result == pytest.approx(0.0)

    def test_list_of_constraints(self, sparse_spectrum):
        input_dits, values = sparse_spectrum
        result = constraint_compute_marginal(input_dits, values, [{0: 0}, {0: 1}])
        assert result == pytest.approx([1.0, 0.0])

    def test_full_assignment_match(self, sparse_spectrum):
        input_dits, values = sparse_spectrum
        result = constraint_compute_marginal(input_dits, values, [0, 0, 0])
        assert result == pytest.approx(1.0)

    def test_full_assignment_no_match(self, sparse_spectrum):
        input_dits, values = sparse_spectrum
        result = constraint_compute_marginal(input_dits, values, [1, 1, 1])
        assert result == pytest.approx(0.0)

    def test_empty_inputs(self):
        result = constraint_compute_marginal([], [], {0: 0})
        assert result == pytest.approx(0.0)

    def test_invalid_function_input_dits_type(self):
        with pytest.raises(TypeError):
            constraint_compute_marginal(42, [1.0], {0: 0})

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            constraint_compute_marginal([[0, 0]], [1.0, 2.0], {0: 0})


# ---------------------------------------------------------------------------
# nearest_neighbors_interactions_sketch
# ---------------------------------------------------------------------------

class TestNearestNeighborsSketch:
    def test_shape(self):
        M = nearest_neighbors_interactions_sketch(3, 2, 2)
        assert M.shape == (8, 8)

    def test_shape_larger(self):
        # n=4, k=2, d=2 → 3 windows × 4 = 12 rows, 2^4=16 columns
        M = nearest_neighbors_interactions_sketch(4, 2, 2)
        assert M.shape == (12, 16)

    def test_returns_ndarray(self):
        M = nearest_neighbors_interactions_sketch(3, 2, 2)
        assert isinstance(M, np.ndarray)

    def test_binary_values(self):
        M = nearest_neighbors_interactions_sketch(3, 2, 2)
        assert set(np.asarray(M).flatten()).issubset({0, 1})

    def test_row_0_matches_constraint_column(self, nn_sketch_3_2_2):
        # Row 0 is indicator of {d0=0,d1=0}: strings [0,0,0]=0, [0,0,1]=1 → [1,1,0,0,0,0,0,0]
        row0 = np.asarray(nn_sketch_3_2_2[0]).flatten().tolist()
        assert row0 == [1, 1, 0, 0, 0, 0, 0, 0]

    def test_invalid_dit_string_length(self):
        with pytest.raises(ValueError):
            nearest_neighbors_interactions_sketch(-1, 2, 2)

    def test_invalid_interaction_size_too_large(self):
        with pytest.raises(ValueError):
            nearest_neighbors_interactions_sketch(2, 3, 2)

    def test_invalid_dit_dimension(self):
        with pytest.raises(ValueError):
            nearest_neighbors_interactions_sketch(3, 2, 1)


# ---------------------------------------------------------------------------
# all_interactions_sketch
# ---------------------------------------------------------------------------

class TestAllInteractionsSketch:
    def test_shape(self):
        # n=3, k=2, d=2 → C(3,2)*4=12 rows, 8 columns
        M = all_interactions_sketch(3, 2, 2)
        assert M.shape == (12, 8)

    def test_returns_ndarray(self):
        M = all_interactions_sketch(3, 2, 2)
        assert isinstance(M, np.ndarray)

    def test_binary_values(self):
        M = all_interactions_sketch(3, 2, 2)
        assert set(np.asarray(M).flatten()).issubset({0, 1})

    def test_invalid_interaction_size(self):
        with pytest.raises(ValueError):
            all_interactions_sketch(3, 4, 2)


# ---------------------------------------------------------------------------
# random_sketch
# ---------------------------------------------------------------------------

class TestRandomSketch:
    def test_shape(self):
        M = random_sketch(3, 5, dit_dimension=2, random_state=0)
        assert M.shape == (5, 8)

    def test_reproducible_with_seed(self):
        M1 = random_sketch(3, 5, random_state=42)
        M2 = random_sketch(3, 5, random_state=42)
        np.testing.assert_array_equal(np.asarray(M1), np.asarray(M2))

    def test_different_seeds_differ(self):
        M1 = random_sketch(3, 10, random_state=0)
        M2 = random_sketch(3, 10, random_state=1)
        assert not np.allclose(np.asarray(M1), np.asarray(M2))

    def test_variance(self):
        # Entries are N(0, 1/m); variance should be close to 1/m
        m = 1000
        M = random_sketch(3, m, random_state=0)
        var = np.asarray(M).var()
        assert var == pytest.approx(1.0 / m, rel=0.1)

    def test_invalid_dit_string_length(self):
        with pytest.raises(ValueError):
            random_sketch(0, 5)

    def test_invalid_m(self):
        with pytest.raises(ValueError):
            random_sketch(3, -1)

    def test_invalid_dit_dimension(self):
        with pytest.raises(ValueError):
            random_sketch(3, 5, dit_dimension=1)

    def test_accepts_numpy_generator(self):
        rng = np.random.default_rng(99)
        M = random_sketch(3, 4, random_state=rng)
        assert M.shape == (4, 8)


# ---------------------------------------------------------------------------
# explicit_compute_marginal
# ---------------------------------------------------------------------------

class TestExplicitComputeMarginal:
    def test_single_nonzero(self, full_spectrum, nn_sketch_3_2_2):
        result = explicit_compute_marginal(full_spectrum, nn_sketch_3_2_2)
        # f([0,0,0])=1: constraints satisfied by [0,0,0] are {0:0,1:0} and {1:0,2:0}
        assert result == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0])

    def test_returns_list(self, full_spectrum, nn_sketch_3_2_2):
        result = explicit_compute_marginal(full_spectrum, nn_sketch_3_2_2)
        assert isinstance(result, list)

    def test_uniform_spectrum(self, nn_sketch_3_2_2):
        # f = 1 everywhere → each constraint sums 2 matching strings → marginal=2
        full = [1.0] * 8
        result = explicit_compute_marginal(full, nn_sketch_3_2_2)
        assert all(v == pytest.approx(2.0) for v in result)

    def test_invalid_full_spectrum_length_raises(self, nn_sketch_3_2_2):
        with pytest.raises(ValueError):
            explicit_compute_marginal([1.0] * 7, nn_sketch_3_2_2)


# ---------------------------------------------------------------------------
# ConstraintSketch (class API)
# ---------------------------------------------------------------------------

class TestConstraintSketch:
    def test_build_nn_sketch_returns_list_of_dicts(self):
        sketch = ConstraintSketch.build_nearest_neighbors_sketch(3, 2, 2)
        assert isinstance(sketch, list)
        assert all(isinstance(c, dict) for c in sketch)

    def test_build_nn_sketch_count(self):
        sketch = ConstraintSketch.build_nearest_neighbors_sketch(3, 2, 2)
        assert len(sketch) == 8

    def test_build_all_sketch_count(self):
        sketch = ConstraintSketch.build_all_interactions_sketch(3, 2, 2)
        assert len(sketch) == 12

    def test_compute_marginal_tuple_input(self):
        sketch = ConstraintSketch.build_nearest_neighbors_sketch(3, 2, 2)
        # Include both 0 and 1 values so that dit_dimension=2 is correctly inferred
        input_dits = [[0, 0, 0], [1, 1, 1]]
        values = [1.0, 0.0]
        result = ConstraintSketch.compute_marginal((input_dits, values), sketch)
        assert result == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0])

    def test_compute_marginals_3_arg_form(self):
        sketch = ConstraintSketch.build_nearest_neighbors_sketch(3, 2, 2)
        input_dits = [[0, 0, 0], [1, 1, 1]]
        values = [1.0, 0.0]
        result = ConstraintSketch.compute_marginals(input_dits, values, sketch)
        assert result == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0])

    def test_compute_marginals_2_arg_form(self):
        sketch = ConstraintSketch.build_nearest_neighbors_sketch(3, 2, 2)
        input_dits = [[0, 0, 0], [1, 1, 1]]
        values = [1.0, 0.0]
        result = ConstraintSketch.compute_marginals((input_dits, values), sketch)
        assert result == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0])

    def test_reconstruct_column_via_class(self):
        sketch = ConstraintSketch.build_nearest_neighbors_sketch(3, 2, 2)
        col = ConstraintSketch.reconstruct_structured_matrix_column(0, sketch, 3, 2)
        assert col.tolist() == [1, 0, 0, 0, 1, 0, 0, 0]


# ---------------------------------------------------------------------------
# ExplicitSketch (class API)
# ---------------------------------------------------------------------------

class TestExplicitSketch:
    def test_build_nn_sketch_shape(self):
        M = ExplicitSketch.build_nearest_neighbors_sketch(3, 2, 2)
        assert M.shape == (8, 8)

    def test_build_all_sketch_shape(self):
        M = ExplicitSketch.build_all_interactions_sketch(3, 2, 2)
        assert M.shape == (12, 8)

    def test_build_returns_ndarray(self):
        M = ExplicitSketch.build_nearest_neighbors_sketch(3, 2, 2)
        assert isinstance(M, np.ndarray)

    def test_compute_marginal_single_nonzero(self):
        M = ExplicitSketch.build_nearest_neighbors_sketch(3, 2, 2)
        full = [0.0] * 8
        full[0] = 1.0
        result = ExplicitSketch.compute_marginal(full, M)
        assert result == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0])

    def test_compute_marginals_alias(self):
        M = ExplicitSketch.build_nearest_neighbors_sketch(3, 2, 2)
        full = [0.0] * 8
        full[0] = 1.0
        r1 = ExplicitSketch.compute_marginal(full, M)
        r2 = ExplicitSketch.compute_marginals(full, M)
        assert r1 == pytest.approx(r2)

    def test_random_sketch_shape(self):
        M = ExplicitSketch.random_sketch(3, 5, dit_dimension=2, random_state=0)
        assert M.shape == (5, 8)

    def test_random_sketch_reproducible(self):
        M1 = ExplicitSketch.random_sketch(3, 5, random_state=7)
        M2 = ExplicitSketch.random_sketch(3, 5, random_state=7)
        np.testing.assert_array_equal(np.asarray(M1), np.asarray(M2))

    def test_nn_and_constraint_marginals_agree(self):
        """Explicit and constraint sketches should give the same marginals."""
        input_dits = [[0, 0, 0], [0, 1, 0], [1, 0, 1]]
        values = [3.0, 1.0, 2.0]

        c_sketch = ConstraintSketch.build_nearest_neighbors_sketch(3, 2, 2)
        c_marginals = ConstraintSketch.compute_marginal((input_dits, values), c_sketch)

        full = [0.0] * 8
        for dit_str, v in zip(input_dits, values):
            idx = sum(d * (2 ** (2 - i)) for i, d in enumerate(dit_str))
            full[idx] = v
        e_sketch = ExplicitSketch.build_nearest_neighbors_sketch(3, 2, 2)
        e_marginals = ExplicitSketch.compute_marginal(full, e_sketch)

        assert c_marginals == pytest.approx(e_marginals, abs=1e-10)
