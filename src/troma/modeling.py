import numpy as np

from . import data_structure as ds
from . import sketchs
from .decoding.matching_pursuit import matching_pursuit
from .optimization.optimizer import get_optimizer
from ._validation import ensure_callable as _ensure_callable, ensure_int as _ensure_int

def mcco_modeling(objective_function, number_samples, dit_string_length, thereshold_parameter = None, dit_dimension=2):
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
    thereshold_parameter : float or str, optional
        Threshold applied to the sampled values. If "Auto", the threshold is set to the 90th percentile of the sampled values.
    dit_dimension : int, optional
        Dimension of the dits. Default is 2.
    
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

    total_states = dit_dimension ** dit_string_length
    if number_samples > total_states:
        raise ValueError("number_samples must be <= dit_dimension ** dit_string_length.")
    if thereshold_parameter is not None and thereshold_parameter != "Auto":
        if not isinstance(thereshold_parameter, (int, float, np.integer, np.floating)):
            raise TypeError("thereshold_parameter must be a real number, 'Auto', or None.")

    sample_indexes = np.random.choice(total_states, number_samples, replace=False)
    sample_dit_strings = [ds.integer_to_dit_string(index, dit_string_length, dit_dimension) for index in sample_indexes]
    sample_values = np.array([objective_function(sample_dit_string) for sample_dit_string in sample_dit_strings])

    if thereshold_parameter == "Auto":
        thereshold_parameter = np.percentile(sample_values, 90)

    if thereshold_parameter is not None:
        sample_values[sample_values < thereshold_parameter] = 0

    non_zero = sorted([(int(i), s.tolist(), int(v)) for i, s, v in zip(sample_indexes, sample_dit_strings, sample_values) if v != 0])
    sample_indexes, sample_dit_strings, sample_values = zip(*non_zero) if non_zero else ([], [], [])

    return sample_indexes, sample_values, sample_dit_strings