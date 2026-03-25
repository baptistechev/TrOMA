import numpy as np

def array_to_column_vector(lst):
    return np.matrix(lst).T

def column_vector_to_array(vec):
    return np.asarray(vec).reshape(-1)