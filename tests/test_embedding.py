import pytest

from troma.core.structure import DitString
from troma.core.embedding import (
    reverse_spectrum_restriction,
    spectrum_embedding,
    spectrum_restriction,
)


def _sample_spectrum() -> list[DitString]:
    return [
        DitString([0, 0, 2, 0, 2, 2], dimension=3),
        DitString([0, 2, 2, 0, 0, 0], dimension=3),
    ]


def test_spectrum_restriction_with_all_none_is_identity_and_no_mutation():
    spectrum = _sample_spectrum()
    original_values = [s.tolist() for s in spectrum]

    result = spectrum_restriction(spectrum, dit_restrictions=None, dit_value_restrictions=None)

    assert [s.tolist() for s in result] == original_values
    assert [s.tolist() for s in spectrum] == original_values
    assert all(out is not src for out, src in zip(result, spectrum))


def test_spectrum_embedding_with_all_none_is_identity_and_no_mutation():
    spectrum = _sample_spectrum()
    original_values = [s.tolist() for s in spectrum]

    result = spectrum_embedding(spectrum, additional_dits=None, dimension_mapping=None)

    assert [s.tolist() for s in result] == original_values
    assert [s.tolist() for s in spectrum] == original_values
    assert all(out is not src for out, src in zip(result, spectrum))


def test_reverse_spectrum_restriction_with_none_inputs_is_identity_and_no_mutation():
    spectrum = _sample_spectrum()
    original_values = [s.tolist() for s in spectrum]

    result = reverse_spectrum_restriction(
        spectrum, original_size=6, dit_restrictions=None, dit_value_restrictions=None
    )

    assert [s.tolist() for s in result] == original_values
    assert [s.tolist() for s in spectrum] == original_values
    assert all(out is not src for out, src in zip(result, spectrum))


def test_reverse_spectrum_restriction_with_none_value_mapping_only_inserts_missing_dits():
    restricted = [DitString([3, 4], dimension=7), DitString([5, 6], dimension=7)]

    result = reverse_spectrum_restriction(
        restricted,
        original_size=4,
        dit_restrictions=[1, 3],
        dit_value_restrictions=None,
    )

    assert [s.tolist() for s in result] == [[0, 3, 0, 4], [0, 5, 0, 6]]


def test_spectrum_restriction_rejects_non_dit_string():
    with pytest.raises(TypeError):
        spectrum_restriction([[0, 1, 2]], None, None)


def test_spectrum_embedding_rejects_non_dit_string():
    with pytest.raises(TypeError):
        spectrum_embedding([[0, 1, 2]], None, None)


def test_reverse_spectrum_restriction_rejects_non_dit_string():
    with pytest.raises(TypeError):
        reverse_spectrum_restriction([[0, 1]], 3, None, None)
