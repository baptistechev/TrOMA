import random as _random
import numpy as np

from troma import integer_to_dit_string

def generate_random_problem(sparsity, dit_string_length, dit_dimension=2):
    n_total = dit_dimension ** dit_string_length
    # Use random.sample with a range object — O(sparsity) memory regardless of n_total.
    spectrum_pos = sorted(_random.sample(range(n_total), sparsity))
    spectrum_values = np.random.rand(sparsity)
    spectrum_bin =  [integer_to_dit_string(pos, dit_string_length, dit_dimension) for pos in spectrum_pos]

    return spectrum_pos,spectrum_values, spectrum_bin