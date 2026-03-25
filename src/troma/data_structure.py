import numpy as np

from ._validation import (
        ensure_int as _ensure_int,
        ensure_int_or_digit as _ensure_int_or_digit,
        ensure_iterable as _ensure_iterable,
        ensure_one_of as _ensure_one_of,
        ensure_same_length as _ensure_same_length,
        ensure_unique_items as _ensure_unique_items,
        ensure_vector_collection as _ensure_vector_collection,
)

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

    integer = _ensure_int("integer", integer, min_value=0)
    dit_dimension = _ensure_int("dit_dimension", dit_dimension, min_value=2)
    dit_string_length = _ensure_int("dit_string_length", dit_string_length, min_value=0)
    _ensure_one_of("convention", convention, ('R', 'L'))
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

    dit_dimension = _ensure_int("dit_dimension", dit_dimension, min_value=2)
    _ensure_one_of("convention", convention, ('R', 'L'))
    if not isinstance(dit_string, str):
            _ensure_iterable("dit_string", dit_string)

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
            _ensure_int("Each value in dit_string", value, min_value=0, max_value=dit_dimension-1)

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
    
    _ensure_iterable("dit_string", dit_string, allow_str=True)
    dit_dimension = _ensure_int("dit_dimension", dit_dimension, min_value=2)

    cp_representation = []
    for value in dit_string:
        dit_value = _ensure_int_or_digit(
            "Each dit value",
            value,
            min_value=0,
            max_value=dit_dimension - 1,
        )
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

    _ensure_iterable("fixed_dit_positions", fixed_dit_positions)
    fixed_dit_positions = list(fixed_dit_positions)
    set_size = _ensure_int("set_size", set_size, min_value=0)
    dit_dimension = _ensure_int("dit_dimension", dit_dimension, min_value=2)
    for position in fixed_dit_positions:
        position = _ensure_int("fixed_dit_positions index", position, min_value=0)
        if position >= set_size:
            raise ValueError("All fixed_dit_positions indices must be in [0, set_size-1].")
        _ensure_unique_items("fixed_dit_positions", fixed_dit_positions)

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

    _ensure_iterable("cylinder_set", cylinder_set)
    cylinder_set = _ensure_vector_collection("cylinder_set", cylinder_set)
    dit_dimension = _ensure_int("dit_dimension", dit_dimension, min_value=2)
    _ensure_one_of("convention", convention, ('R', 'L'))
    _ensure_vector_collection(
        "cylinder_set",
        cylinder_set,
        vector_length=dit_dimension,
        binary=True,
    )
    
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

    _ensure_iterable("element", element)
    _ensure_iterable("cylinder_set", cylinder_set)
    element = _ensure_vector_collection("element", element)
    cylinder_set = _ensure_vector_collection("cylinder_set", cylinder_set, binary=True)
    _ensure_same_length("element", element, "cylinder_set", cylinder_set)
    for basis1, basis2 in zip(element, cylinder_set):
        if len(basis1) != len(basis2):
            raise ValueError("Each vector in element must have the same length as the corresponding vector in cylinder_set.")
    for basis1, basis2 in zip(element, cylinder_set):
        basis1 = list(basis1)
        basis2 = list(basis2)
        if basis2 != [1] * len(basis2) and basis1 != basis2:
            return False
    return True