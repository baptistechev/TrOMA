import numpy as np

from . import data_structure as ds
from . import sketchs
from .decoding.matching_pursuit import matching_pursuit
from .optimization.optimizer import get_optimizer
from ._validation import ensure_callable as _ensure_callable, ensure_int as _ensure_int
from .embedding import reverse_spectrum_restriction

def _basic_sampling(number_samples, dit_string_length, dit_dimension):
    total_states = dit_dimension ** dit_string_length
    rng = np.random.default_rng()
    sample_indexes = rng.integers(0, total_states, size=number_samples, dtype=np.int64)
    sample_dit_strings = [ds.integer_to_dit_string(int(index), dit_string_length, dit_dimension) for index in sample_indexes]
    return sample_indexes,sample_dit_strings

def mcco_modeling(objective_function, number_samples, dit_string_length, threshold_parameter = None, dit_dimension=2, sampling_function=_basic_sampling, sampling_args=None):
    """
    Create a MCCO model of the objective function seen as a blackbox.
    The model is created by sampling the objective function on a random subset of the search space, and then applying a threshold to the sampled values. 
    The model defined an unconstrained optimization problem, which can be solved by a matching pursuit decoder.

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
    tuple
        sample_indexes : list of int
            List of the indexes of the sampled configurations.
        sample_values : list of float
            List of the values of the objective function on the sampled configurations.
        sample_dit_strings : list of list of int
            List of the sampled configurations as dit strings.
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
    return sample_indexes, sample_values, sample_dit_strings

def restricted_mcco_modeling(objective_function, number_samples, dit_string_length, threshold_parameter = None, dit_dimension=2, dit_restrictions=None, dit_value_restrictions=None, additional_dits_val=0, sampling_function=_basic_sampling, sampling_args=None):
    """
    Create a MCCO model of the objective function seen as a blackbox.
    The model is created by sampling the objective function on a random subset of the search space, and then applying a threshold to the sampled values. 
    The model defined an unconstrained optimization problem, which can be solved by a matching pursuit decoder.

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
    dit_restrictions : list of int, optional
        List of the positions of the dits where we restrict the sampling. If None, no restriction is applied.
    dit_value_restrictions : list of list of int, optional
        List of the allowed dits values to sample from. If None, no restriction is applied.
    additional_dits_val : int, optional
        Value to assign to the trivial dits when calling the objective function. Default is 0.
    sampling_function : callable, optional
        Function used for sampling the configurations. It should take number_samples, dit_string_length, dit_dimension as positional arguments and return a list of samples dit_strings. Default is a uniform sampling function.
    sampling_args : dict, optional
        Additional keyword arguments to pass to the sampling_function. Default is None.
    
    Returns
    -------
    tuple
        sample_indexes : list of int
            List of the indexes of the sampled configurations.
        sample_values : list of float
            List of the values of the objective function on the sampled configurations.
        sample_dit_strings : list of list of int
            List of the sampled configurations as dit strings.
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

    if dit_restrictions is None and dit_value_restrictions is None:
        return mcco_modeling(objective_function, number_samples, dit_string_length, threshold_parameter, dit_dimension, sampling_function, sampling_args)

    #Compute the size of the restricted space
    restricted_space_size = len(dit_restrictions) if dit_restrictions is not None else dit_string_length
    restricted_space_dimension = len(dit_value_restrictions) if dit_value_restrictions is not None else dit_dimension
    
    #Sample from the restricted space
    if sampling_args is None:
        sampling_args = {}
    sample_indexes_rest, sample_dit_strings_rest = sampling_function(number_samples, restricted_space_size, restricted_space_dimension, **sampling_args)

    #Map back to the full space for calling the objective function.
    sample_dit_strings = reverse_spectrum_restriction(
        sample_dit_strings_rest,
        original_size=dit_string_length,
        dit_restrictions=dit_restrictions,
        dit_value_restrictions=dit_value_restrictions,
        additional_dits_val=additional_dits_val
    )

    #Compute the objective function values on the sampled configurations
    sample_values = np.array([objective_function(sample_dit_string) for sample_dit_string in sample_dit_strings])

    #Apply the threshold
    if threshold_parameter == "Auto":
        threshold_parameter = _auto_threshold_from_non_zero(sample_values)
    if threshold_parameter is not None:
        sample_values[sample_values < threshold_parameter] = 0

    #Return only the non-zero samples, sorted by index
    non_zero = sorted([(int(i), s.tolist(), int(v)) for i, s, v in zip(sample_indexes_rest, sample_dit_strings_rest, sample_values) if v != 0])
    sample_indexes_rest, sample_dit_strings_rest, sample_values = zip(*non_zero) if non_zero else ([], [], [])
    return sample_indexes_rest, sample_values, sample_dit_strings_rest