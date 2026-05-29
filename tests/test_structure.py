"""Tests for DitString and CylinderSet in troma.core.structure."""

import numpy as np
import pytest

from troma.core.structure import CylinderSet, DitString


# ===========================================================================
# DitString
# ===========================================================================

class TestDitStringConstruction:
    def test_basic_binary(self):
        ds = DitString([0, 1, 0, 1], dimension=2)
        assert ds.length == 4
        assert ds.dimension == 2
        assert ds.tolist() == [0, 1, 0, 1]

    def test_inferred_length(self):
        ds = DitString([0, 1, 0])
        assert ds.length == 3

    def test_explicit_length(self):
        ds = DitString([0, 1], length=2, dimension=2)
        assert ds.length == 2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            DitString([0, 1], length=3, dimension=2)

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DitString([0, 2], dimension=2)

    def test_float_element_raises(self):
        with pytest.raises(TypeError):
            DitString([0, 1.5], dimension=2)

    def test_dimension_too_small_raises(self):
        with pytest.raises(ValueError):
            DitString([0], dimension=1)

    def test_ternary(self):
        ds = DitString([0, 1, 2], dimension=3)
        assert ds.tolist() == [0, 1, 2]


class TestDitStringFromInteger:
    def test_r_convention(self):
        ds = DitString.from_integer(5, length=4, dimension=2)
        assert ds.tolist() == [0, 1, 0, 1]
        assert ds.length == 4
        assert ds.dimension == 2

    def test_l_convention(self):
        ds = DitString.from_integer(5, length=4, dimension=2, convention="L")
        assert ds.tolist() == [1, 0, 1, 0]

    def test_roundtrip(self):
        for n in range(8):
            ds = DitString.from_integer(n, length=3, dimension=2)
            assert ds.to_integer() == n

    def test_zero(self):
        ds = DitString.from_integer(0, length=3, dimension=2)
        assert ds.tolist() == [0, 0, 0]

    @pytest.mark.parametrize("kwargs,exc", [
        ({"integer": -1, "length": 3, "dimension": 2}, ValueError),
        ({"integer": 2.2, "length": 3, "dimension": 2}, TypeError),
        ({"integer": 1, "length": 3, "dimension": 1}, ValueError),
        ({"integer": 1, "length": 3, "dimension": 2.0}, TypeError),
        ({"integer": 1, "length": 2.5, "dimension": 2}, TypeError),
        ({"integer": 8, "length": 3, "dimension": 2}, ValueError),
        ({"integer": 1, "length": 0, "dimension": 2}, ValueError),
    ])
    def test_invalid_inputs(self, kwargs, exc):
        with pytest.raises(exc):
            DitString.from_integer(**kwargs)


class TestDitStringToInteger:
    def test_r_convention(self):
        assert DitString([0, 1, 0, 1], dimension=2).to_integer() == 5

    def test_l_convention(self):
        assert DitString([1, 0, 1, 0], dimension=2).to_integer("L") == 5

    def test_ternary(self):
        ds = DitString([1, 0, 2], dimension=3)
        assert ds.to_integer() == 1 * 9 + 0 * 3 + 2

    def test_empty(self):
        assert DitString([], dimension=2).to_integer() == 0


class TestDitStringToComputationalBasis:
    def test_returns_cylinder_set(self):
        cs = DitString([0, 1, 0], dimension=2).to_computational_basis()
        assert isinstance(cs, CylinderSet)

    def test_binary(self):
        cs = DitString([0, 1, 0], dimension=2).to_computational_basis()
        assert cs.vectors == [[1, 0], [0, 1], [1, 0]]

    def test_ternary(self):
        cs = DitString([0, 2, 1], dimension=3).to_computational_basis()
        assert cs.vectors == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]

    def test_dimension_preserved(self):
        ds = DitString([0, 1], dimension=2)
        cs = ds.to_computational_basis()
        assert cs.dimension == ds.dimension
        assert cs.length == ds.length

    def test_no_wildcards(self):
        """All vectors should be one-hot (sum == 1)."""
        cs = DitString([0, 1, 0, 1], dimension=2).to_computational_basis()
        for vec in cs:
            assert sum(vec) == 1


class TestDitStringSequenceProtocol:
    def test_len(self):
        assert len(DitString([0, 1, 0], dimension=2)) == 3

    def test_iter(self):
        assert list(DitString([0, 1, 0], dimension=2)) == [0, 1, 0]

    def test_getitem(self):
        ds = DitString([0, 1, 0], dimension=2)
        assert ds[1] == 1

    def test_tuple(self):
        ds = DitString([0, 1, 0], dimension=2)
        assert tuple(ds) == (0, 1, 0)

    def test_as_numpy(self):
        ds = DitString([0, 1, 0], dimension=2)
        arr = np.asarray(ds)
        assert arr.tolist() == [0, 1, 0]


class TestDitStringHashAndEquality:
    def test_equal_same_content(self):
        assert DitString([0, 1], dimension=2) == DitString([0, 1], dimension=2)

    def test_not_equal_different_values(self):
        assert DitString([0, 1], dimension=2) != DitString([1, 0], dimension=2)

    def test_not_equal_different_dimension(self):
        assert DitString([0, 1, 2], dimension=3) != DitString([0, 1, 2], dimension=4)

    def test_hashable_as_dict_key(self):
        d = {DitString([0, 1], dimension=2): 42}
        assert d[DitString([0, 1], dimension=2)] == 42

    def test_hashable_in_set(self):
        s = {DitString([0, 0], dimension=2), DitString([0, 1], dimension=2)}
        assert len(s) == 2


# ===========================================================================
# CylinderSet
# ===========================================================================

class TestCylinderSetConstruction:
    def test_basic(self):
        cs = CylinderSet([[1, 0], [0, 1]], dimension=2)
        assert cs.length == 2
        assert cs.dimension == 2

    def test_wildcard_allowed(self):
        cs = CylinderSet([[1, 1], [1, 0]], dimension=2)
        assert cs.vectors == [[1, 1], [1, 0]]

    def test_non_binary_raises(self):
        with pytest.raises(ValueError):
            CylinderSet([[1, 2]], dimension=2)

    def test_wrong_vector_length_raises(self):
        with pytest.raises(ValueError):
            CylinderSet([[1, 0, 0]], dimension=2)

    def test_dimension_too_small_raises(self):
        with pytest.raises(ValueError):
            CylinderSet([[1]], dimension=1)

    def test_non_list_vectors_raises(self):
        with pytest.raises(TypeError):
            CylinderSet("bad", dimension=2)


class TestCylinderSetForPositions:
    def test_count_binary(self):
        sets = CylinderSet.for_positions([0, 1], set_size=3, dimension=2)
        assert len(sets) == 4  # 2^2

    def test_count_ternary(self):
        sets = CylinderSet.for_positions([0], set_size=3, dimension=3)
        assert len(sets) == 3

    def test_all_cylinder_sets(self):
        sets = CylinderSet.for_positions([0], set_size=2, dimension=2)
        # pos-0 fixed to 0
        assert sets[0].vectors[0] == [1, 0]
        assert sets[0].vectors[1] == [1, 1]  # wildcard
        # pos-0 fixed to 1
        assert sets[1].vectors[0] == [0, 1]
        assert sets[1].vectors[1] == [1, 1]  # wildcard

    def test_empty_fixed_positions(self):
        sets = CylinderSet.for_positions([], set_size=3, dimension=2)
        assert len(sets) == 1

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError):
            CylinderSet.for_positions([5], set_size=3, dimension=2)

    def test_duplicate_positions_raises(self):
        with pytest.raises(ValueError):
            CylinderSet.for_positions([1, 1], set_size=3, dimension=2)


class TestCylinderSetKroneckerDevelop:
    def test_all_fixed_r(self):
        cs = CylinderSet([[1, 0], [0, 1]], dimension=2)
        out = cs.kronecker_develop(convention="R")
        assert out.tolist() == [0, 1, 0, 0]

    def test_all_fixed_l(self):
        cs = CylinderSet([[1, 0], [0, 1]], dimension=2)
        out = cs.kronecker_develop(convention="L")
        assert out.tolist() == [0, 0, 1, 0]

    def test_wildcard_is_sum(self):
        """A wildcard position contributes all ones — its rows sum to the non-wildcard slice."""
        cs = CylinderSet([[1, 1], [1, 0]], dimension=2)
        out = cs.kronecker_develop()
        # [1,1] ⊗ [1,0] = [1,0,1,0]
        assert out.tolist() == [1, 0, 1, 0]

    def test_invalid_convention_raises(self):
        with pytest.raises(ValueError):
            CylinderSet([[1, 0]], dimension=2).kronecker_develop(convention="X")

    def test_consistency_with_to_computational_basis(self):
        """DitString → computational basis → kronecker_develop should equal indicator at integer."""
        ds = DitString([0, 1, 0], dimension=2)
        idx = ds.to_integer()
        indicator = ds.to_computational_basis().kronecker_develop()
        assert len(indicator) == 2 ** ds.length
        assert indicator[idx] == 1
        assert sum(indicator) == 1


class TestCylinderSetSequenceProtocol:
    def test_len(self):
        cs = CylinderSet([[1, 0], [0, 1], [1, 1]], dimension=2)
        assert len(cs) == 3

    def test_iter(self):
        vecs = [[1, 0], [0, 1]]
        cs = CylinderSet(vecs, dimension=2)
        assert list(cs) == vecs

    def test_getitem(self):
        cs = CylinderSet([[1, 0], [0, 1]], dimension=2)
        assert cs[1] == [0, 1]
