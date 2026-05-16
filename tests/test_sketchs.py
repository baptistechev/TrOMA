"""Tests for ConstraintSketchMap and ExplicitSketchMap."""

import numpy as np
import pytest

from troma import ConstraintSketchMap, ExplicitSketchMap
from troma.core.structure import DitString

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nn_constraints_3_2_2():
    """Nearest-neighbor constraints for n=3, k=2, d=2 → 8 constraints."""
    sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
    sm.build_from_nearest_neighbors()
    return sm.map


@pytest.fixture
def nn_sketch_3_2_2():
    """Explicit nearest-neighbor sketch matrix for n=3, k=2, d=2 (8×8)."""
    sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2,
                           constraints="nearest_neighbors")
    return sm.map


@pytest.fixture
def sparse_spectrum():
    """Sparse representation of f([0,0,0])=1, all others = 0."""
    input_dits = [
        DitString([0, 0, 0], dimension=2),
        DitString([0, 0, 1], dimension=2),
        DitString([0, 1, 0], dimension=2),
        DitString([1, 0, 0], dimension=2),
    ]
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
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.build_from_nearest_neighbors()
        c = sm.map
        assert len(c) == 8

    def test_count_ternary(self):
        # n=3, k=2, d=3 → 2 windows × 9 value combinations = 18 constraints
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=3)
        sm.build_from_nearest_neighbors()
        c = sm.map
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
        nn_sm = ConstraintSketchMap(sketch_length=3, interaction_size=3, sketch_dimension=2)
        nn_sm.build_from_nearest_neighbors()
        ai_sm = ConstraintSketchMap(sketch_length=3, interaction_size=3, sketch_dimension=2)
        ai_sm.build_from_all_interactions()
        nn = nn_sm.map
        ai = ai_sm.map
        assert len(nn) == len(ai)

    def test_invalid_interaction_size_too_large(self):
        with pytest.raises(ValueError):
            sm = ConstraintSketchMap(sketch_length=3, interaction_size=4, sketch_dimension=2)
            sm.build_from_nearest_neighbors()

    @pytest.mark.parametrize("n,k,d,exc", [
        (-1, 2, 2, ValueError),
        (3, -1, 2, ValueError),
        (1.5, 2, 2, TypeError),
        (3, 2.0, 2, TypeError),
        (3, 2, 1.5, TypeError),
    ])
    def test_invalid_inputs(self, n, k, d, exc):
        with pytest.raises(exc):
            sm = ConstraintSketchMap(sketch_length=n, interaction_size=k, sketch_dimension=d)
            sm.build_from_nearest_neighbors()


# ---------------------------------------------------------------------------
# constraints_for_all_interactions
# ---------------------------------------------------------------------------

class TestConstraintsAllInteractions:
    def test_count(self):
        # n=3, k=2, d=2 → C(3,2)=3 windows × 4 = 12 constraints
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.build_from_all_interactions()
        c = sm.map
        assert len(c) == 12

    def test_covers_all_pairs(self):
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.build_from_all_interactions()
        c = sm.map
        all_key_sets = [frozenset(ci.keys()) for ci in c]
        assert frozenset({0, 1}) in all_key_sets
        assert frozenset({0, 2}) in all_key_sets
        assert frozenset({1, 2}) in all_key_sets

    def test_invalid_interaction_size_too_large(self):
        with pytest.raises(ValueError):
            sm = ConstraintSketchMap(sketch_length=3, interaction_size=4, sketch_dimension=2)
            sm.build_from_all_interactions()


# ---------------------------------------------------------------------------
# reconstruct_structured_matrix_column
# ---------------------------------------------------------------------------

class TestReconstructColumn:
    def test_index_0_binary(self, nn_constraints_3_2_2):
        # [0,0,0] satisfies {0:0,1:0} and {1:0,2:0} → column = [1,0,0,0,1,0,0,0]
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.map = nn_constraints_3_2_2
        col = sm.reconstruct_structured_matrix_column(0)
        assert col.tolist() == [1, 0, 0, 0, 1, 0, 0, 0]

    def test_index_7_binary(self, nn_constraints_3_2_2):
        # [1,1,1] satisfies {0:1,1:1} and {1:1,2:1} → column = [0,0,0,1,0,0,0,1]
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.map = nn_constraints_3_2_2
        col = sm.reconstruct_structured_matrix_column(7)
        assert col.tolist() == [0, 0, 0, 1, 0, 0, 0, 1]

    def test_returns_ndarray(self, nn_constraints_3_2_2):
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.map = nn_constraints_3_2_2
        col = sm.reconstruct_structured_matrix_column(0)
        assert isinstance(col, np.ndarray)

    def test_length_equals_number_of_constraints(self, nn_constraints_3_2_2):
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.map = nn_constraints_3_2_2
        col = sm.reconstruct_structured_matrix_column(3)
        assert len(col) == len(nn_constraints_3_2_2)

    def test_index_out_of_range(self, nn_constraints_3_2_2):
        with pytest.raises(ValueError):
            sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
            sm.map = nn_constraints_3_2_2
            sm.reconstruct_structured_matrix_column(100)

    def test_invalid_constraints_type(self):
        with pytest.raises(TypeError):
            sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
            sm.map = "bad"
            sm.reconstruct_structured_matrix_column(0)


# ---------------------------------------------------------------------------
# constraint_compute_marginal
# ---------------------------------------------------------------------------

class TestConstraintComputeMarginal:
    def test_single_dict_constraint(self, sparse_spectrum):
        input_dits, values = sparse_spectrum
        # All strings with d0=0 → [0,0,0] and [0,0,1] and [0,1,0] → 1.0+0+0 = 1.0
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        result = sm._compute_sparse_marginal(input_dits, values, {0: 0})
        assert result == pytest.approx(1.0)

    def test_single_dict_constraint_no_match(self, sparse_spectrum):
        input_dits, values = sparse_spectrum
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        result = sm._compute_sparse_marginal(input_dits, values, {0: 1})
        assert result == pytest.approx(0.0)

    def test_list_of_constraints(self, sparse_spectrum):
        input_dits, values = sparse_spectrum
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        result = sm._compute_sparse_marginal(input_dits, values, [{0: 0}, {0: 1}])
        assert result == pytest.approx([1.0, 0.0])

    def test_full_assignment_match(self, sparse_spectrum):
        input_dits, values = sparse_spectrum
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        result = sm._compute_sparse_marginal(input_dits, values, [0, 0, 0])
        assert result == pytest.approx(1.0)

    def test_full_assignment_no_match(self, sparse_spectrum):
        input_dits, values = sparse_spectrum
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        result = sm._compute_sparse_marginal(input_dits, values, [1, 1, 1])
        assert result == pytest.approx(0.0)

    def test_empty_inputs(self):
        sm = ConstraintSketchMap(sketch_length=1, interaction_size=1, sketch_dimension=2)
        result = sm._compute_sparse_marginal([], [], {0: 0})
        assert result == pytest.approx(0.0)

    def test_invalid_function_input_dits_type(self):
        with pytest.raises(TypeError):
            sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
            sm._compute_sparse_marginal(42, [1.0], {0: 0})

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            sm = ConstraintSketchMap(sketch_length=2, interaction_size=1, sketch_dimension=2)
            sm._compute_sparse_marginal([DitString([0, 0], dimension=2)], [1.0, 2.0], {0: 0})


# ---------------------------------------------------------------------------
# nearest_neighbors_interactions_sketch
# ---------------------------------------------------------------------------

class TestNearestNeighborsSketch:
    def test_shape(self):
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.build_from_nearest_neighbors()
        M = sm.map
        assert M.shape == (8, 8)

    def test_shape_larger(self):
        # n=4, k=2, d=2 → 3 windows × 4 = 12 rows, 2^4=16 columns
        sm = ExplicitSketchMap(sketch_length=4, interaction_size=2, sketch_dimension=2)
        sm.build_from_nearest_neighbors()
        M = sm.map
        assert M.shape == (12, 16)

    def test_returns_ndarray(self):
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.build_from_nearest_neighbors()
        M = sm.map
        assert isinstance(M, np.ndarray)

    def test_binary_values(self):
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.build_from_nearest_neighbors()
        M = sm.map
        assert set(np.asarray(M).flatten()).issubset({0, 1})

    def test_row_0_matches_constraint_column(self, nn_sketch_3_2_2):
        # Row 0 is indicator of {d0=0,d1=0}: strings [0,0,0]=0, [0,0,1]=1 → [1,1,0,0,0,0,0,0]
        row0 = np.asarray(nn_sketch_3_2_2[0]).flatten().tolist()
        assert row0 == [1, 1, 0, 0, 0, 0, 0, 0]

    def test_invalid_dit_string_length(self):
        with pytest.raises(ValueError):
            sm = ExplicitSketchMap(sketch_length=-1, interaction_size=2, sketch_dimension=2)
            sm.build_from_nearest_neighbors()

    def test_invalid_interaction_size_too_large(self):
        with pytest.raises(ValueError):
            sm = ExplicitSketchMap(sketch_length=2, interaction_size=3, sketch_dimension=2)
            sm.build_from_nearest_neighbors()

    def test_invalid_dit_dimension(self):
        with pytest.raises(ValueError):
            sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=1)
            sm.build_from_nearest_neighbors()


# ---------------------------------------------------------------------------
# all_interactions_sketch
# ---------------------------------------------------------------------------

class TestAllInteractionsSketch:
    def test_shape(self):
        # n=3, k=2, d=2 → C(3,2)*4=12 rows, 8 columns
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.build_from_all_interactions()
        M = sm.map
        assert M.shape == (12, 8)

    def test_returns_ndarray(self):
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.build_from_all_interactions()
        M = sm.map
        assert isinstance(M, np.ndarray)

    def test_binary_values(self):
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.build_from_all_interactions()
        M = sm.map
        assert set(np.asarray(M).flatten()).issubset({0, 1})

    def test_invalid_interaction_size(self):
        with pytest.raises(ValueError):
            sm = ExplicitSketchMap(sketch_length=3, interaction_size=4, sketch_dimension=2)
            sm.build_from_all_interactions()


# ---------------------------------------------------------------------------
# random_sketch
# ---------------------------------------------------------------------------

class TestRandomSketch:
    def test_shape(self):
        sm = ExplicitSketchMap(sketch_length=3, sketch_dimension=2)
        M = sm.random_sketch(5, random_state=0)
        assert M.shape == (5, 8)

    def test_reproducible_with_seed(self):
        sm1 = ExplicitSketchMap(sketch_length=3, sketch_dimension=2)
        M1 = sm1.random_sketch(5, random_state=42)
        sm2 = ExplicitSketchMap(sketch_length=3, sketch_dimension=2)
        M2 = sm2.random_sketch(5, random_state=42)
        np.testing.assert_array_equal(np.asarray(M1), np.asarray(M2))

    def test_different_seeds_differ(self):
        sm1 = ExplicitSketchMap(sketch_length=3, sketch_dimension=2)
        M1 = sm1.random_sketch(10, random_state=0)
        sm2 = ExplicitSketchMap(sketch_length=3, sketch_dimension=2)
        M2 = sm2.random_sketch(10, random_state=1)
        assert not np.allclose(np.asarray(M1), np.asarray(M2))

    def test_variance(self):
        # Entries are N(0, 1/m); variance should be close to 1/m
        m = 1000
        sm = ExplicitSketchMap(sketch_length=3, sketch_dimension=2)
        M = sm.random_sketch(m, random_state=0)
        var = np.asarray(M).var()
        assert var == pytest.approx(1.0 / m, rel=0.1)

    def test_invalid_dit_string_length(self):
        with pytest.raises(ValueError):
            sm = ExplicitSketchMap(sketch_length=0, sketch_dimension=2)
            sm.random_sketch(5)

    def test_invalid_m(self):
        with pytest.raises(ValueError):
            sm = ExplicitSketchMap(sketch_length=3, sketch_dimension=2)
            sm.random_sketch(-1)

    def test_invalid_dit_dimension(self):
        with pytest.raises(ValueError):
            sm = ExplicitSketchMap(sketch_length=3, sketch_dimension=1)
            sm.random_sketch(5)

    def test_accepts_numpy_generator(self):
        rng = np.random.default_rng(99)
        sm = ExplicitSketchMap(sketch_length=3, sketch_dimension=2)
        M = sm.random_sketch(4, random_state=rng)
        assert M.shape == (4, 8)


# ---------------------------------------------------------------------------
# explicit_compute_marginal
# ---------------------------------------------------------------------------

class TestExplicitComputeMarginal:
    def test_single_nonzero(self, full_spectrum, nn_sketch_3_2_2):
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.map = nn_sketch_3_2_2
        result = sm.compute_marginal(full_spectrum)
        # f([0,0,0])=1: constraints satisfied by [0,0,0] are {0:0,1:0} and {1:0,2:0}
        assert result == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0])

    def test_returns_list(self, full_spectrum, nn_sketch_3_2_2):
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.map = nn_sketch_3_2_2
        result = sm.compute_marginal(full_spectrum)
        assert isinstance(result, list)

    def test_uniform_spectrum(self, nn_sketch_3_2_2):
        # f = 1 everywhere → each constraint sums 2 matching strings → marginal=2
        full = [1.0] * 8
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
        sm.map = nn_sketch_3_2_2
        result = sm.compute_marginal(full)
        assert all(v == pytest.approx(2.0) for v in result)

    def test_invalid_full_spectrum_length_raises(self, nn_sketch_3_2_2):
        with pytest.raises(ValueError):
            sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2)
            sm.map = nn_sketch_3_2_2
            sm.compute_marginal([1.0] * 7)


# ---------------------------------------------------------------------------
# ConstraintSketchMap (class API)
# ---------------------------------------------------------------------------

class TestConstraintSketchMap:
    def test_build_nn_sketch_returns_list_of_dicts(self):
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2, constraints="nearest_neighbors")
        assert isinstance(sm.map, list)
        assert all(isinstance(c, dict) for c in sm.map)

    def test_build_nn_sketch_count(self):
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2, constraints="nearest_neighbors")
        assert len(sm.map) == 8

    def test_build_all_sketch_count(self):
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2, constraints="all_interactions")
        assert len(sm.map) == 12

    def test_compute_marginal_tuple_input(self):
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2, constraints="nearest_neighbors")
        input_dits = [DitString([0, 0, 0], dimension=2), DitString([1, 1, 1], dimension=2)]
        values = [1.0, 0.0]
        result = sm.compute_marginal((input_dits, values))
        assert result == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0])

    def test_compute_marginal_two_arg_form(self):
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2, constraints="nearest_neighbors")
        input_dits = [DitString([0, 0, 0], dimension=2), DitString([1, 1, 1], dimension=2)]
        values = [1.0, 0.0]
        result = sm.compute_marginal(input_dits, values)
        assert result == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0])

    def test_reconstruct_column(self):
        sm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2, constraints="nearest_neighbors")
        col = sm.reconstruct_structured_matrix_column(0)
        assert col.tolist() == [1, 0, 0, 0, 1, 0, 0, 0]


# ---------------------------------------------------------------------------
# ExplicitSketchMap (class API)
# ---------------------------------------------------------------------------

class TestExplicitSketchMap:
    def test_build_nn_sketch_shape(self):
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2, constraints="nearest_neighbors")
        assert sm.map.shape == (8, 8)

    def test_build_all_sketch_shape(self):
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2, constraints="all_interactions")
        assert sm.map.shape == (12, 8)

    def test_build_returns_ndarray(self):
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2, constraints="nearest_neighbors")
        assert isinstance(sm.map, np.ndarray)

    def test_compute_marginal_single_nonzero(self):
        sm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2, constraints="nearest_neighbors")
        full = [0.0] * 8
        full[0] = 1.0
        result = sm.compute_marginal(full)
        assert result == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0])

    def test_random_sketch_shape(self):
        sm = ExplicitSketchMap(sketch_length=3, sketch_dimension=2)
        sm.random_sketch(5, random_state=0)
        assert sm.map.shape == (5, 8)

    def test_random_sketch_reproducible(self):
        sm1 = ExplicitSketchMap(sketch_length=3, sketch_dimension=2)
        sm1.random_sketch(5, random_state=7)
        sm2 = ExplicitSketchMap(sketch_length=3, sketch_dimension=2)
        sm2.random_sketch(5, random_state=7)
        np.testing.assert_array_equal(sm1.map, sm2.map)

    def test_nn_and_constraint_marginals_agree(self):
        """Explicit and constraint sketches should give the same marginals."""
        input_dits = [
            DitString([0, 0, 0], dimension=2),
            DitString([0, 1, 0], dimension=2),
            DitString([1, 0, 1], dimension=2),
        ]
        values = [3.0, 1.0, 2.0]

        csm = ConstraintSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2, constraints="nearest_neighbors")
        c_marginals = csm.compute_marginal(input_dits, values)

        full = [0.0] * 8
        for dit_str, v in zip(input_dits, values):
            idx = sum(d * (2 ** (2 - i)) for i, d in enumerate(dit_str))
            full[idx] = v
        esm = ExplicitSketchMap(sketch_length=3, interaction_size=2, sketch_dimension=2, constraints="nearest_neighbors")
        e_marginals = esm.compute_marginal(full)

        assert c_marginals == pytest.approx(e_marginals, abs=1e-10)
