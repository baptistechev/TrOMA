import numpy as np

from troma.embedding import (
    reverse_spectrum_restriction,
    spectrum_embedding,
    spectrum_restriction,
)


def _sample_spectrum():
    return [
        np.array([0, 0, 2, 0, 2, 2]),
        np.array([0, 2, 2, 0, 0, 0]),
    ]


def test_spectrum_restriction_with_all_none_is_identity_and_no_mutation():
    spectrum = _sample_spectrum()
    original = [row.copy() for row in spectrum]

    result = spectrum_restriction(
        spectrum,
        dit_restrictions=None,
        dit_value_restrictions=None,
    )

    assert [row.tolist() for row in result] == [row.tolist() for row in original]
    assert [row.tolist() for row in spectrum] == [row.tolist() for row in original]
    assert all(out is not src for out, src in zip(result, spectrum))


def test_spectrum_embedding_with_all_none_is_identity_and_no_mutation():
    spectrum = _sample_spectrum()
    original = [row.copy() for row in spectrum]

    result = spectrum_embedding(
        spectrum,
        additional_dits=None,
        dimension_mapping=None,
    )

    assert [row.tolist() for row in result] == [row.tolist() for row in original]
    assert [row.tolist() for row in spectrum] == [row.tolist() for row in original]
    assert all(out is not src for out, src in zip(result, spectrum))


def test_reverse_spectrum_restriction_with_none_inputs_is_identity_and_no_mutation():
    spectrum = _sample_spectrum()
    original = [row.copy() for row in spectrum]

    result = reverse_spectrum_restriction(
        spectrum,
        original_size=6,
        dit_restrictions=None,
        dit_value_restrictions=None,
    )

    assert [row.tolist() for row in result] == [row.tolist() for row in original]
    assert [row.tolist() for row in spectrum] == [row.tolist() for row in original]
    assert all(out is not src for out, src in zip(result, spectrum))


def test_reverse_spectrum_restriction_with_none_value_mapping_only_inserts_missing_dits():
    restricted = [np.array([3, 4]), np.array([5, 6])]

    result = reverse_spectrum_restriction(
        restricted,
        original_size=4,
        dit_restrictions=[1, 3],
        dit_value_restrictions=None,
    )

    assert [row.tolist() for row in result] == [[0, 3, 0, 4], [0, 5, 0, 6]]
