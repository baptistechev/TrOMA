from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..core import data_structure as ds
from ..optimization.optimizer import get_optimizer
from .structure import Restriction, Sample
from .embedding import reverse_spectrum_restriction
from .._validation import ensure_callable, ensure_int, ensure_optional_dict


def _basic_sampling(
    number_samples: int,
    dit_string_length: int,
    dit_dimension: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    total_states = dit_dimension ** dit_string_length
    rng = np.random.default_rng()
    sample_indexes = rng.integers(0, total_states, size=number_samples, dtype=np.int64)
    sample_dit_strings = [
        ds.integer_to_dit_string(int(index), dit_string_length, dit_dimension)
        for index in sample_indexes
    ]
    return sample_indexes, sample_dit_strings


def objective_sampling(
    objective_function: Callable,
    number_samples: int,
    dit_string_length: int,
    threshold_parameter: float | str | None = None,
    dit_dimension: int = 2,
    sampling_function: Callable | None = None,
    sampling_args: dict | None = None,
) -> Sample:
    """
    Sample the objective function on a random subset of the search space, and then apply a threshold to the sampled values.

    Parameters
    ----------
    objective_function : Callable
        Function evaluated on dit strings.
    number_samples : int
        Number of configurations sampled.
    dit_string_length : int
        Length of the dit strings.
    threshold_parameter : float or str, optional
        Threshold applied to the sampled values. If "Auto", set to the 90th percentile of non-zero sampled values.
    dit_dimension : int, optional
        Dimension of the dits. Default is 2.
    sampling_function : Callable or None, optional
        Custom sampling function. Must accept (number_samples, dit_string_length, dit_dimension) and return
        (sample_indexes, sample_dit_strings). Defaults to uniform sampling.
    sampling_args : dict, optional
        Extra keyword arguments forwarded to sampling_function. Default is None.

    Returns
    -------
    Sample
        Sample object containing indexes, values, and dit_strings.
    """
    ensure_callable("objective_function", objective_function)
    dit_string_length = ensure_int("dit_string_length", dit_string_length, min_value=1)
    dit_dimension = ensure_int("dit_dimension", dit_dimension, min_value=2)
    number_samples = ensure_int("number_samples", number_samples, min_value=1)

    if sampling_function is None:
        sampling_function = _basic_sampling
    else:
        ensure_callable("sampling_function", sampling_function)

    if threshold_parameter is not None and threshold_parameter != "Auto":
        if not isinstance(threshold_parameter, (int, float, np.integer, np.floating)):
            raise TypeError("threshold_parameter must be a real number, 'Auto', or None.")

    ensure_optional_dict("sampling_args", sampling_args)

    if sampling_args is None:
        sampling_args = {}
    sample_indexes, sample_dit_strings = sampling_function(
        number_samples, dit_string_length, dit_dimension, **sampling_args
    )

    # Convert DitString to numpy array so user objective functions receive array-like input,
    # matching the original contract (objective_function was always called with ndarray).
    sample_values = np.array([objective_function(np.asarray(s)) for s in sample_dit_strings])

    if threshold_parameter == "Auto":
        non_zero = sample_values[sample_values != 0]
        threshold_parameter = np.percentile(non_zero, 90) if non_zero.size > 0 else 0
    if threshold_parameter is not None:
        sample_values[sample_values < threshold_parameter] = 0

    non_zero_triples = [
        (int(i), s, int(v))
        for i, s, v in zip(sample_indexes, sample_dit_strings, sample_values) if v != 0
    ]
    non_zero_triples.sort(key=lambda t: t[0])
    if non_zero_triples:
        out_indexes, out_dit_strings, out_values = zip(*non_zero_triples)
    else:
        out_indexes, out_dit_strings, out_values = [], [], []
    return Sample(
        indexes=list(out_indexes),
        values=list(out_values),
        dit_strings=list(out_dit_strings),
    )


def restricted_objective_sampling(
    objective_function: Callable,
    number_samples: int,
    dit_string_length: int,
    threshold_parameter: float | str | None = None,
    dit_dimension: int = 2,
    restriction: Restriction | None = None,
    sampling_function: Callable = _basic_sampling,
    sampling_args: dict | None = None,
) -> Sample:
    """
    Sample the objective function restricted to a subspace, then apply a threshold.

    Parameters
    ----------
    objective_function : Callable
        Function evaluated on dit strings.
    number_samples : int
        Number of configurations sampled.
    dit_string_length : int
        Length of the dit strings.
    threshold_parameter : float or str, optional
        Threshold applied to the sampled values. If "Auto", set to the 90th percentile of non-zero values.
    dit_dimension : int, optional
        Dimension of the dits. Default is 2.
    restriction : Restriction or None, optional
        Restriction object. If None, no restriction is applied.
    sampling_function : Callable, optional
        Custom sampling function. Defaults to uniform sampling.
    sampling_args : dict, optional
        Extra keyword arguments forwarded to sampling_function. Default is None.

    Returns
    -------
    Sample
        Sample object containing indexes, values, and dit_strings.
    """
    ensure_callable("objective_function", objective_function)
    dit_string_length = ensure_int("dit_string_length", dit_string_length, min_value=1)
    dit_dimension = ensure_int("dit_dimension", dit_dimension, min_value=2)
    number_samples = ensure_int("number_samples", number_samples, min_value=1)

    if sampling_function is None:
        sampling_function = _basic_sampling
    else:
        ensure_callable("sampling_function", sampling_function)

    if threshold_parameter is not None and threshold_parameter != "Auto":
        if not isinstance(threshold_parameter, (int, float, np.integer, np.floating)):
            raise TypeError("threshold_parameter must be a real number, 'Auto', or None.")

    ensure_optional_dict("sampling_args", sampling_args)

    if restriction is None:
        restriction = Restriction()

    if (
        restriction.dit_restrictions is None
        and restriction.dit_value_restrictions is None
    ):
        return objective_sampling(
            objective_function, number_samples, dit_string_length,
            threshold_parameter, dit_dimension, sampling_function, sampling_args,
        )

    restricted_space_size = (
        len(restriction.dit_restrictions)
        if restriction.dit_restrictions is not None
        else dit_string_length
    )
    restricted_space_dimension = (
        len(restriction.dit_value_restrictions)
        if restriction.dit_value_restrictions is not None
        else dit_dimension
    )

    if sampling_args is None:
        sampling_args = {}
    sample_indexes_rest, sample_dit_strings_rest = sampling_function(
        number_samples, restricted_space_size, restricted_space_dimension, **sampling_args
    )

    sample_dit_strings = reverse_spectrum_restriction(
        sample_dit_strings_rest,
        original_size=dit_string_length,
        dit_restrictions=restriction.dit_restrictions,
        dit_value_restrictions=restriction.dit_value_restrictions,
        additional_dits_val=restriction.additional_dits_val,
    )

    # Convert DitString to numpy array so user objective functions receive array-like input,
    # matching the original contract (objective_function was always called with ndarray).
    sample_values = np.array([objective_function(np.asarray(s)) for s in sample_dit_strings])

    if threshold_parameter == "Auto":
        non_zero = sample_values[sample_values != 0]
        threshold_parameter = np.percentile(non_zero, 90) if non_zero.size > 0 else 0
    if threshold_parameter is not None:
        sample_values[sample_values < threshold_parameter] = 0

    non_zero_triples = [
        (int(i), s, int(v))
        for i, s, v in zip(sample_indexes_rest, sample_dit_strings_rest, sample_values) if v != 0
    ]
    non_zero_triples.sort(key=lambda t: t[0])
    if non_zero_triples:
        out_indexes, out_dit_strings, out_values = zip(*non_zero_triples)
    else:
        out_indexes, out_dit_strings, out_values = [], [], []
    return Sample(
        indexes=list(out_indexes),
        values=list(out_values),
        dit_strings=list(out_dit_strings),
    )
