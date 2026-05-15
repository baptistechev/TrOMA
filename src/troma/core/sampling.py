import numpy as np

from ..core import data_structure as ds
from ..optimization.optimizer import get_optimizer
from .._validation import ensure_callable as _ensure_callable, ensure_int as _ensure_int
from .structure import Restriction, Sample
from .embedding import reverse_spectrum_restriction

def _basic_sampling(number_samples, dit_string_length, dit_dimension):
    total_states = dit_dimension ** dit_string_length
    rng = np.random.default_rng()
    sample_indexes = rng.integers(0, total_states, size=number_samples, dtype=np.int64)
    sample_dit_strings = [ds.integer_to_dit_string(int(index), dit_string_length, dit_dimension) for index in sample_indexes]
    return sample_indexes,sample_dit_strings

def objective_sampling(objective_function, number_samples, dit_string_length, threshold_parameter = None, dit_dimension=2, sampling_function=None, sampling_args=None):
    """
    Sample the objective function on a random subset of the search space, and then applying a threshold to the sampled values. 

    Parameters
    ----------
    objective_function : Callable
        Function evaluated on dit strings.
    number_samples : int
        Number of configurations sampled.
    dit_string_length : int
        Length of the dit strings.
    threshold_parameter : float or str, optional
        Threshold applied to the sampled values. If "Auto", the threshold is set to the 90th percentile of the non-zero sampled values.
    dit_dimension : int, optional
        Dimension of the dits. Default is 2.
    sampling_function : callable or None, optional
        Function used for sampling the configurations. It should take number_samples, dit_string_length, dit_dimension as positional arguments and return a tuple of (sample_indexes, sample_dit_strings). Default is a uniform sampling function.
    sampling_args : dict, optional
        Additional keyword arguments to pass to the sampling_function. Default is None.
    
    Returns
    -------
    Sample
        Sample object containing indexes, values, and dit_strings.
    """
    _ensure_callable("objective_function", objective_function)
    dit_string_length = _ensure_int("dit_string_length", dit_string_length, min_value=1)
    dit_dimension = _ensure_int("dit_dimension", dit_dimension, min_value=2)
    number_samples = _ensure_int("number_samples", number_samples, min_value=1)
    if sampling_function is None:
        sampling_function = _basic_sampling
    else:
        _ensure_callable("sampling_function", sampling_function)

    if threshold_parameter is not None and threshold_parameter != "Auto":
        if not isinstance(threshold_parameter, (int, float, np.integer, np.floating)):
            raise TypeError("threshold_parameter must be a real number, 'Auto', or None.")

    if sampling_args is not None and not isinstance(sampling_args, dict):
        raise TypeError("sampling_args must be a dict or None.")

    #Sample from the full space
    if sampling_args is None:
        sampling_args = {}
    sample_indexes , sample_dit_strings = sampling_function( number_samples, dit_string_length, dit_dimension, **sampling_args)

    #Compute the objective function values on the sampled configurations
    sample_values = np.array([objective_function(sample_dit_string) for sample_dit_string in sample_dit_strings])

    #Apply the threshold
    if threshold_parameter == "Auto":
        non_zero_sample_values = sample_values[sample_values != 0]
        if non_zero_sample_values.size == 0:
            threshold_parameter = 0
        else:
            threshold_parameter = np.percentile(non_zero_sample_values, 90)
    if threshold_parameter is not None:
        sample_values[sample_values < threshold_parameter] = 0

    #Return only the non-zero samples, sorted by index
    non_zero = sorted([(int(i), s.tolist(), int(v)) for i, s, v in zip(sample_indexes, sample_dit_strings, sample_values) if v != 0])
    sample_indexes, sample_dit_strings, sample_values = zip(*non_zero) if non_zero else ([], [], [])
    return Sample(
        indexes=list(sample_indexes),
        values=list(sample_values),
        dit_strings=list(sample_dit_strings),
    )

def restricted_objective_sampling(
    objective_function,
    number_samples,
    dit_string_length,
    threshold_parameter = None,
    dit_dimension=2,
    restriction: Restriction | None = None,
    sampling_function=_basic_sampling,
    sampling_args=None,
):
    """
    Sample the objective function on a random subset of the search space, and then applying a threshold to the sampled values. 

    Parameters
    ----------
    objective_function : Callable
        Function evaluated on dit strings.
    number_samples : int
        Number of configurations sampled.
    dit_string_length : int
        Length of the dit strings.
    threshold_parameter : float or str, optional
        Threshold applied to the sampled values. If "Auto", the threshold is set to the 90th percentile of the non-zero sampled values.
    dit_dimension : int, optional
        Dimension of the dits. Default is 2.
    restriction : Restriction | None, optional
        Restriction object holding dit_restrictions, dit_value_restrictions and
        additional_dits_val. If None, no restriction is applied.
    sampling_function : callable, optional
        Function used for sampling the configurations. It should take number_samples, dit_string_length, dit_dimension as positional arguments and return a list of samples dit_strings. Default is a uniform sampling function.
    sampling_args : dict, optional
        Additional keyword arguments to pass to the sampling_function. Default is None.
    
    Returns
    -------
    Sample
        Sample object containing indexes, values, and dit_strings.
    """
    _ensure_callable("objective_function", objective_function)
    dit_string_length = _ensure_int("dit_string_length", dit_string_length, min_value=1)
    dit_dimension = _ensure_int("dit_dimension", dit_dimension, min_value=2)
    number_samples = _ensure_int("number_samples", number_samples, min_value=1)
    if sampling_function is None:
        sampling_function = _basic_sampling
    else:
        _ensure_callable("sampling_function", sampling_function)

    if threshold_parameter is not None and threshold_parameter != "Auto":
        if not isinstance(threshold_parameter, (int, float, np.integer, np.floating)):
            raise TypeError("threshold_parameter must be a real number, 'Auto', or None.")

    if sampling_args is not None and not isinstance(sampling_args, dict):
        raise TypeError("sampling_args must be a dict or None.")

    if restriction is None:
        restriction = Restriction()

    if (
        restriction.dit_restrictions is None
        and restriction.dit_value_restrictions is None
    ):
        return objective_sampling(objective_function, number_samples, dit_string_length, threshold_parameter, dit_dimension, sampling_function, sampling_args)

    #Compute the size of the restricted space
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
    
    #Sample from the restricted space
    if sampling_args is None:
        sampling_args = {}
    sample_indexes_rest, sample_dit_strings_rest = sampling_function(number_samples, restricted_space_size, restricted_space_dimension, **sampling_args)

    #Map back to the full space for calling the objective function.
    sample_dit_strings = reverse_spectrum_restriction(
        sample_dit_strings_rest,
        original_size=dit_string_length,
        dit_restrictions=restriction.dit_restrictions,
        dit_value_restrictions=restriction.dit_value_restrictions,
        additional_dits_val=restriction.additional_dits_val,
    )

    #Compute the objective function values on the sampled configurations
    sample_values = np.array([objective_function(sample_dit_string) for sample_dit_string in sample_dit_strings])

    #Apply the threshold
    if threshold_parameter == "Auto":
        non_zero_sample_values = sample_values[sample_values != 0]
        if non_zero_sample_values.size == 0:
            threshold_parameter = 0
        else:
            threshold_parameter = np.percentile(non_zero_sample_values, 90)
    if threshold_parameter is not None:
        sample_values[sample_values < threshold_parameter] = 0

    #Return only the non-zero samples, sorted by index
    non_zero = sorted([(int(i), s.tolist(), int(v)) for i, s, v in zip(sample_indexes_rest, sample_dit_strings_rest, sample_values) if v != 0])
    sample_indexes_rest, sample_dit_strings_rest, sample_values = zip(*non_zero) if non_zero else ([], [], [])
    return Sample(
        indexes=list(sample_indexes_rest),
        values=list(sample_values),
        dit_strings=list(sample_dit_strings_rest),
    )