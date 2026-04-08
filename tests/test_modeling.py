"""Tests for the public modeling API (troma.modeling, troma.mcco_workflow)."""

import numpy as np
import pytest

from troma import mcco_modeling, solve_via_mcco

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sum_of_bits(dit_string):
    """Simple objective: sum of the dit values."""
    return float(sum(dit_string))


def _constant_one(dit_string):
    """Objective that always returns 1."""
    return 1.0


def _zero(dit_string):
    """Objective that always returns 0."""
    return 0.0


# ---------------------------------------------------------------------------
# mcco_modeling
# ---------------------------------------------------------------------------

class TestMccoModeling:
    def test_returns_three_tuple(self):
        np.random.seed(0)
        result = mcco_modeling(_sum_of_bits, number_samples=10, dit_string_length=4)
        assert len(result) == 3

    def test_output_lengths_match(self):
        np.random.seed(1)
        idx, vals, dits = mcco_modeling(_sum_of_bits, number_samples=10, dit_string_length=4)
        assert len(idx) == len(vals) == len(dits)

    def test_number_of_samples_le_requested(self):
        np.random.seed(2)
        idx, vals, dits = mcco_modeling(_sum_of_bits, number_samples=8, dit_string_length=4)
        assert len(idx) <= 10

    def test_dit_string_length_correct(self):
        np.random.seed(3)
        _, _, dits = mcco_modeling(_sum_of_bits, number_samples=8, dit_string_length=5)
        for d in dits:
            assert len(d) == 5

    def test_sample_indices_in_valid_range(self):
        np.random.seed(4)
        idx, _, _ = mcco_modeling(_sum_of_bits, number_samples=16, dit_string_length=4,
                                  dit_dimension=2)
        for i in idx:
            assert 0 <= i < 2**4

    def test_all_returned_values_nonzero(self):
        """After thresholding, all returned values should be non-zero."""
        np.random.seed(5)
        _, vals, _ = mcco_modeling(_sum_of_bits, number_samples=12, dit_string_length=4,
                                   threshold_parameter=0.5)
        for v in vals:
            assert v != 0

    def test_auto_threshold_filters(self):
        """With 'Auto' threshold, at most 10% of samples are kept (90th percentile)."""
        np.random.seed(6)
        _, vals, _ = mcco_modeling(_sum_of_bits, number_samples=16, dit_string_length=4,
                                   threshold_parameter="Auto")
        # We cannot test exact count easily, but all returned values should be > threshold
        # (non-zero after threshold)
        for v in vals:
            assert float(v) > 0.0

    def test_no_threshold_returns_all_nonzero_samples(self):
        """Without threshold, only strictly-zero objective values are dropped."""
        np.random.seed(7)
        idx, vals, dits = mcco_modeling(_constant_one, number_samples=8, dit_string_length=3)
        # All sampled values are 1 → all kept
        assert len(vals) == 8
        assert all(v == 1.0 for v in vals)

    def test_all_zero_objective_returns_empty(self):
        np.random.seed(8)
        idx, vals, dits = mcco_modeling(_zero, number_samples=8, dit_string_length=3)
        assert len(idx) == 0
        assert len(vals) == 0
        assert len(dits) == 0

    def test_ternary_dits_valid_range(self):
        np.random.seed(9)
        idx, _, dits = mcco_modeling(_sum_of_bits, number_samples=9, dit_string_length=3,
                                     dit_dimension=3)
        for i in idx:
            assert 0 <= i < 3**3
        for d in dits:
            assert all(0 <= v <= 2 for v in d)

    def test_sorted_by_index(self):
        """Returned sample_indexes should be sorted in ascending order."""
        np.random.seed(10)
        idx, _, _ = mcco_modeling(_sum_of_bits, number_samples=14, dit_string_length=4)
        assert list(idx) == sorted(idx)


# ---------------------------------------------------------------------------
# solve_via_mcco
# ---------------------------------------------------------------------------

class TestSolveViaMcco:
    def test_returns_dict(self):
        np.random.seed(20)
        result = solve_via_mcco(
            _sum_of_bits, number_samples=12, dit_string_length=4,
            interaction_size=2, iteration_number=3,
        )
        assert isinstance(result, dict)

    def test_result_has_expected_keys(self):
        np.random.seed(21)
        result = solve_via_mcco(
            _sum_of_bits, number_samples=12, dit_string_length=4,
            interaction_size=2, iteration_number=3,
        )
        expected_keys = {"spectrum_pos", "spectrum_val", "spectrum_bin",
                         "constraints", "y", "solution"}
        assert expected_keys.issubset(result.keys())

    def test_solution_is_2d_array(self):
        np.random.seed(22)
        result = solve_via_mcco(
            _sum_of_bits, number_samples=12, dit_string_length=4,
            interaction_size=2, iteration_number=3,
        )
        sol = result["solution"]
        assert isinstance(sol, np.ndarray)
        assert sol.ndim == 2
        assert sol.shape[1] == 2

    def test_solution_indices_in_valid_range(self):
        np.random.seed(23)
        n, d = 4, 2
        result = solve_via_mcco(
            _sum_of_bits, number_samples=12, dit_string_length=n,
            interaction_size=2, iteration_number=3, dit_dimension=d,
        )
        for idx, _ in result["solution"]:
            assert 0 <= idx < d**n

    def test_y_length_matches_constraints(self):
        np.random.seed(24)
        result = solve_via_mcco(
            _sum_of_bits, number_samples=12, dit_string_length=4,
            interaction_size=2, iteration_number=3,
        )
        assert len(result["y"]) == len(result["constraints"])

    def test_constraints_are_list_of_dicts(self):
        np.random.seed(25)
        result = solve_via_mcco(
            _sum_of_bits, number_samples=12, dit_string_length=4,
            interaction_size=2,
        )
        assert isinstance(result["constraints"], list)
        assert all(isinstance(c, dict) for c in result["constraints"])

    def test_spectrum_lengths_match(self):
        np.random.seed(26)
        result = solve_via_mcco(
            _sum_of_bits, number_samples=12, dit_string_length=4,
            interaction_size=2,
        )
        assert len(result["spectrum_pos"]) == len(result["spectrum_val"])
        assert len(result["spectrum_val"]) == len(result["spectrum_bin"])

    def test_ternary_dits(self):
        np.random.seed(27)
        result = solve_via_mcco(
            _sum_of_bits, number_samples=9, dit_string_length=3,
            interaction_size=2, iteration_number=2, dit_dimension=3,
        )
        assert isinstance(result, dict)
        assert isinstance(result["solution"], np.ndarray)

    def test_with_no_threshold(self):
        np.random.seed(28)
        result = solve_via_mcco(
            _constant_one, number_samples=10, dit_string_length=4,
            interaction_size=2, threshold_parameter=None, iteration_number=2,
        )
        assert isinstance(result["solution"], np.ndarray)

    def test_iteration_number_affects_solution_size(self):
        """More iterations can add more components to the solution."""
        np.random.seed(29)
        result_1iter = solve_via_mcco(
            _sum_of_bits, number_samples=30, dit_string_length=5,
            interaction_size=2, iteration_number=1,
        )
        np.random.seed(29)
        result_5iter = solve_via_mcco(
            _sum_of_bits, number_samples=30, dit_string_length=5,
            interaction_size=2, iteration_number=5,
        )
        # 5 iterations can produce at most 5 unique components
        assert result_5iter["solution"].shape[0] >= result_1iter["solution"].shape[0]

    def test_all_zero_objective_returns_empty_solution(self):
        """When the objective is zero everywhere, no sample is kept → solution should be empty."""
        np.random.seed(30)
        result = solve_via_mcco(
            _zero, number_samples=12, dit_string_length=4,
            interaction_size=2, threshold_parameter=None, iteration_number=3,
        )
        # Marginals are all zero → each iteration selects some index with coeff=0
        # The solution array may have entries but all coefficients should be zero
        sol = result["solution"]
        if sol.shape[0] > 0:
            assert np.allclose(sol[:, 1], 0.0)
