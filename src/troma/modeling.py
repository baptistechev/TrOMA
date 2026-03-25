import numpy as np

from . import data_structure as ds
from . import sketchs
from .decoding.matching_pursuit import matching_pursuit
from .optimization.optimizer import get_optimizer

def mcco_modeling(objective_function, number_samples, dit_string_length, thereshold_parameter = None, dit_dimension=2):
    """
    """
    sample_indexes = np.random.choice(dit_dimension**dit_string_length, number_samples, replace=False)
    sample_dit_strings = [ds.integer_to_dit_string(index, dit_string_length, dit_dimension) for index in sample_indexes]
    sample_values = np.array([objective_function(sample_dit_string) for sample_dit_string in sample_dit_strings])

    if thereshold_parameter == "Auto":
        thereshold_parameter = np.percentile(sample_values, 90)

    if thereshold_parameter is not None:
        sample_values[sample_values < thereshold_parameter] = 0

    non_zero = sorted([(int(i), s.tolist(), int(v)) for i, s, v in zip(sample_indexes, sample_dit_strings, sample_values) if v != 0])
    sample_indexes, sample_dit_strings, sample_values = zip(*non_zero) if non_zero else ([], [], [])

    return sample_indexes, sample_values, sample_dit_strings