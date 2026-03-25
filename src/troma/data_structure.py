import numpy as np
import copy
import numbers

def integer_to_dit_string(integer, dit_string_length, dit_dimension=2, convention='R'):
    """
    Encode an integer into a dit_string.

    Parameters
    ----------
    integer : int
            The integer to encode.
    dit_string_length : int
            The length of the resulting dit string.
    dit_dimension : int, optional
            The dimension of the dits (e.g., 2 for binary, 3 for ternary). Default is 2.
    convention : str, optional
            The direction of the encoding. 'R' for right-aligned (default), 'L' for left-aligned.
    
    Returns
    -------
    ndarray
            The dit_string representation of the integer.
    """ 

    if not isinstance(integer, numbers.Integral):
        raise TypeError("integer must be an integer.")
    if integer < 0:
        raise ValueError("integer must be non-negative.")
    if not isinstance(dit_dimension, numbers.Integral):
        raise TypeError("dit_dimension must be an integer.")
    if dit_dimension < 2:
            raise ValueError("dit_dimension must be greater than or equal to 2.")
    if not isinstance(dit_string_length, numbers.Integral):
            raise TypeError("dit_string_length must be an integer.")
    if dit_string_length < 0:
            raise ValueError("dit_string_length must be non-negative.")
    if convention not in ('R', 'L'):
            raise ValueError("convention must be either 'R' or 'L'.")
    if integer >= dit_dimension ** dit_string_length and dit_string_length > 0:
            raise ValueError("integer cannot be represented with the given dit_dimension and dit_string_length.")
    if integer > 0 and dit_string_length == 0:
            raise ValueError("dit_string_length must be positive to encode a non-zero integer.")

    dit_string = []  
    while integer > 0:
        integer, rem = divmod(integer, dit_dimension)
        dit_string.append(rem)
    dit_string += [0] * (dit_string_length - len(dit_string))
    dit_string = np.array(dit_string)
    return dit_string[::-1] if convention == 'R' else dit_string

def dit_string_to_integer(dit_string, dit_dimension=2, convention='R'):
    """
    Decode a dit_string back into an integer.

    Parameters
    ----------
    dit_string : list or ndarray
            The dit_string to decode.
    dit_dimension : int, optional
            The dimension of the dits (e.g., 2 for binary, 3 for ternary). Default is 2.
    convention : str, optional
            The direction of the encoding. 'R' for right-aligned (default), 'L' for left-aligned.

    Returns
    -------
    int
            The decoded integer.
    """

    if not isinstance(dit_string, str) and not hasattr(dit_string, '__iter__'):
            raise TypeError("dit_string must be an iterable of integers or a digit string.")
    if not isinstance(dit_dimension, numbers.Integral):
            raise TypeError("dit_dimension must be an integer.")
    if dit_dimension < 2:
            raise ValueError("dit_dimension must be greater than or equal to 2.")
    if convention not in ('R', 'L'):
            raise ValueError("convention must be either 'R' or 'L'.")

    if isinstance(dit_string, str):
            if len(dit_string) == 0:
                    dit_string = []
            elif not dit_string.isdigit():
                    raise ValueError("When dit_string is a string, it must contain only digits.")
            else:
                    dit_string = [int(character) for character in dit_string]
    else:
            dit_string = list(dit_string)

    for value in dit_string:
            if not isinstance(value, numbers.Integral):
                    raise TypeError("Each value in dit_string must be an integer.")
            if value < 0 or value >= dit_dimension:
                    raise ValueError("Each value in dit_string must be in [0, dit_dimension-1].")

    dit_string_length = len(dit_string)
    number_basis = np.power(dit_dimension, np.arange(dit_string_length), dtype = float)
    if convention == 'R':
        number_basis = number_basis[::-1]
    return int(np.dot(dit_string, number_basis))

def dit_string_to_computational_basis(dit_string, dit_dimension=2):
    """
    Decompose a dit_string into its computational basis representation.

    Parameters
    ----------
    dit_string : iterable of int or str
            The dit_string to decompose.
    dit_dimension : int, optional
            Dit dimension. Default is 2 (binary).

    Returns
    -------
    list
            The computational basis representation of the input dit_string.
    """

    if not hasattr(dit_string, '__iter__'):
        raise TypeError("dit_string must be an iterable of integers.")
    if not isinstance(dit_dimension, numbers.Integral):
        raise TypeError("dit_dimension must be an integer.")
    if dit_dimension < 2:
        raise ValueError("dit_dimension must be an integer >= 2.")

    cp_representation = []
    for value in dit_string:
        if not isinstance(value, (numbers.Integral, str)):
            raise TypeError("Each value in dit_string must be an integer-like value or digit string.")
        dit_value = int(value)
        if dit_value < 0 or dit_value >= dit_dimension:
            raise ValueError("Each dit value must be in [0, dit_dimension-1].")
        cp_vector = [0] * dit_dimension
        cp_vector[dit_value] = 1
        cp_representation.append(cp_vector)
    return cp_representation

def create_cylinder_set_indicator(fixed_dit_positions, set_size, dit_dimension=2):
    """
    Create all facotirzed indicators of the cylinder sets defined by fixing the dit positions for each possible values.

    Parameters
    ----------
    fixed_dit_positions : iterable of int
            Dits which define the cylinder sets.
    set_size : int
            The size of the set (number of dits).
    dit_dimension : int, optional
            Dimension of each dit. Default is 2 (binary).

    Returns
    -------
    list
            A list whose entries are the factorized indicators of the possible cylinder sets defined by fixing the dit positions to each possible values.
    
    Examples
    --------
    >>> create_cylinder_set_indicator([0, 1], 3, 2)
    [[[1, 0], [1, 0], [1, 1]],
     [[1, 0], [0, 1], [1, 1]],
     [[0, 1], [1, 0], [1, 1]],
     [[0, 1], [0, 1], [1, 1]]]
            
    """

    fixed_dit_positions = list(fixed_dit_positions)
    if not isinstance(set_size, numbers.Integral):
            raise TypeError("set_size must be an integer.")
    if set_size < 0:
            raise ValueError("set_size must be non-negative.")
    if not isinstance(dit_dimension, numbers.Integral):
            raise TypeError("dit_dimension must be an integer.")
    if dit_dimension < 2:
            raise ValueError("dit_dimension must be an integer >= 2.")
    if any(not isinstance(position, numbers.Integral) for position in fixed_dit_positions):
            raise TypeError("All fixed_dit_positions indices must be integers.")
    if any((position < 0 or position >= set_size) for position in fixed_dit_positions):
        raise ValueError("All fixed_dit_positions indices must be in [0, set_size-1].")
    if len(set(fixed_dit_positions)) != len(fixed_dit_positions):
        raise ValueError("fixed_dit_positions indices must be unique.")

    list_cylinder_sets = []
    n_positions = len(fixed_dit_positions)
    for config in range(dit_dimension ** n_positions):
        dits = integer_to_dit_string(
            config,
            dit_dimension=dit_dimension,
            dit_string_length=n_positions,
            convention='R'
        )
        cylinder_set = [[1] * dit_dimension for _ in range(set_size)]
        for position, dit_value in zip(fixed_dit_positions, dits):
            cp_vector = [0] * dit_dimension
            cp_vector[int(dit_value)] = 1
            cylinder_set[position] = cp_vector
        list_cylinder_sets.append(cylinder_set)
    return list_cylinder_sets


def kronecker_develop(cylinder_set, dit_dimension=2, convention='R'):
    """
    Developp a computational basis representation or a factorized cylinder set indicator using Kronecker product.

    Parameters
    ----------
    cylinder_set : list
            A cylinder set factorized indicator, represented as a list of computational basis and identity vectors (lists or ndarrays).
    dit_dimension : int, optional
            The dimension of the dits (e.g., 2 for binary, 3 for ternary). Default is 2.
    convention : str, optional
            The direction of the encoding. 'R' for right-aligned (default), 'L' for left-aligned.
            
    Returns
    -------
    ndarray
            The developed indicator of the cylinder set.
    """
    if not hasattr(cylinder_set, '__iter__'):
        raise TypeError("cylinder_set must be an iterable of vectors.")
    cylinder_set = list(cylinder_set)
    if not isinstance(dit_dimension, numbers.Integral):
        raise TypeError("dit_dimension must be an integer.")
    if dit_dimension < 2:
        raise ValueError("dit_dimension must be an integer >= 2.")
    if convention not in ('R', 'L'):
        raise ValueError("convention must be either 'R' or 'L'.")
    if any(not hasattr(vector, '__len__') for vector in cylinder_set):
        raise TypeError("Each element of cylinder_set must be a vector-like object.")
    if not all(len(i) == dit_dimension for i in cylinder_set):
        raise ValueError("Each one-hot vector in the cylinder set must have length equal to dit_dimension.")
    if any(any(value not in (0, 1) for value in vector) for vector in cylinder_set):
        raise ValueError("Each vector in cylinder_set must contain only 0 or 1 values.")
    
    if convention == 'L':
        cylinder_set = cylinder_set[::-1]

    f = np.array([1])
    for i in cylinder_set:
        f = np.kron(f,i)
    return f


def belongs_to_cylinder_set(element,cylinder_set,dit_dimension=2):
    """
    Check if a element belongs to a given cylinder set.

    Parameters
    ----------
    element : list or ndarray
            The element to check, given within its computational basis representation.
    cylinder_set : list
            The cylinder set represented by its factorized indicator.

    Returns
    -------
    bool
            If the element belongs to the cylinder set or not.
    """
    
    if not hasattr(element, '__iter__'):
        raise TypeError("element must be an iterable of integers.")
    if not hasattr(cylinder_set, '__iter__'):
        raise TypeError("cylinder_set must be an iterable of vectors.")
    element = list(element)
    cylinder_set = list(cylinder_set)
    if any(not hasattr(vector, '__len__') for vector in cylinder_set):
        raise TypeError("Each element of cylinder_set must be a vector-like object.")
    if any(any(value not in (0, 1) for value in vector) for vector in cylinder_set):
        raise ValueError("Each vector in cylinder_set must contain only 0 or 1 values.")

    for basis1,basis2 in zip(element,cylinder_set):
        if basis2 != [1 for _ in range(dit_dimension)] and basis1 != basis2:
            return 0
    return 1