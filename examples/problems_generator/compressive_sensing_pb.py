import numpy as np

from troma import integer_to_dit_string

def generate_random_problem(sparsity, dit_string_length, dit_dimension=2):
    spectrum_pos = sorted(np.random.choice(dit_dimension**dit_string_length, size=sparsity, replace=False).tolist())
    spectrum_values = np.random.rand(sparsity)
    spectrum_bin =  [integer_to_dit_string(pos, dit_string_length, dit_dimension) for pos in spectrum_pos]

    return spectrum_pos,spectrum_values, spectrum_bin