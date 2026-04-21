import numpy as np
from collections import defaultdict
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from .._validation import ensure_int as _ensure_int, ensure_iterable as _ensure_iterable

def compute_hamiltonian(constraints_sketch, marginals, bit_string_length=None):
    """
    Return a Hamiltonian as a dict: {tuple(indices_Z): coefficient}.

    Supported inputs
    ----------------
    - Full patterns: [[1,0], [1,1], [0,1], ...]
    - Constraints: {1: 0, 3: 1, 5: 0}

    Convention
    ----------
    - ()    -> constant term
    - (i,)  -> Z_i
    - (i,j) -> Z_i Z_j
    - (i,j,k,...) -> k-spin interaction

    For constraints:
    - 0 -> [1,0] -> (I + Z)/2
    - 1 -> [0,1] -> (I - Z)/2
    - unspecified positions -> [1,1] -> identity

    Parameters
    ----------
    constraints_sketch : list of patterns
        Each pattern can be either a list of local states (for full patterns) or a dict
        mapping qubit indices to fixed bit values (for constraints).
    marginals : list of float
        The marginals corresponding to each pattern in constraints_sketch. The order of the values should correspond to the order of the patterns in constraints_sketch.
    bit_string_length : int, optional
        The length of the bit strings (i.e., the number of qubits). If None, the function will attempt to infer it from the input patterns. The default is None.
    
    Returns
    -------
    dict
        A dictionary mapping tuples of qubit indices (representing Z operators) to their coefficients in the Hamiltonian. Terms with coefficients close to zero are omitted.
    """
    if not isinstance(constraints_sketch, (list, tuple)):
        raise TypeError("constraints_sketch must be a list or tuple.")
    _ensure_iterable("marginals", marginals)
    if bit_string_length is not None:
        bit_string_length = _ensure_int("bit_string_length", bit_string_length, min_value=1)
    marginals = list(marginals)
    if len(constraints_sketch) != len(marginals):
        raise ValueError("constraints_sketch and marginals must have the same length.")

    def extract_weight(value):
        return float(value[0] if np.ndim(value) > 0 else value)

    def infer_n_qubits(patterns):
        if bit_string_length is not None:
            return int(bit_string_length)

        if len(patterns) == 0:
            raise ValueError("Cannot infer number of qubits from an empty input.")

        first = patterns[0]
        if isinstance(first, dict):
            max_pos = -1
            for pat in patterns:
                if len(pat) > 0:
                    max_pos = max(max_pos, max(int(pos) for pos in pat.keys()))
            if max_pos < 0:
                raise ValueError("Cannot infer number of qubits from empty constraints.")
            return max_pos + 1

        return len(first)

    def fixed_spins_from_pattern(pattern, total_qubits):
        """
        Return a list of (qubit_index, sign) where:
        sign = +1 for (I + Z)/2, sign = -1 for (I - Z)/2.
        """
        fixed = []

        if isinstance(pattern, dict):
            for raw_idx, bit in pattern.items():
                idx = int(raw_idx)
                if idx < 0 or idx >= total_qubits:
                    raise ValueError(f"Qubit index {idx} out of range for n_qubits={total_qubits}.")
                if bit == 0:
                    fixed.append((idx, +1))
                elif bit == 1:
                    fixed.append((idx, -1))
                else:
                    raise ValueError(f"Bit value must be 0 or 1, got {bit} at position {idx}.")
            fixed.sort()
            return fixed

        if len(pattern) != total_qubits:
            raise ValueError(
                f"Pattern length {len(pattern)} does not match n_qubits={total_qubits}."
            )

        for idx, local_state in enumerate(pattern):
            if local_state == [1, 0]:
                fixed.append((idx, +1))
            elif local_state == [0, 1]:
                fixed.append((idx, -1))
            elif local_state == [1, 1]:
                continue
            else:
                raise ValueError(f"Invalid local pattern at qubit {idx}: {local_state}")

        return fixed

    total_qubits = infer_n_qubits(constraints_sketch)
    coeffs = defaultdict(float)

    for constraint, yi in zip(constraints_sketch, marginals):
        weight = extract_weight(yi)
        if np.isclose(weight, 0.0):
            continue

        fixed = fixed_spins_from_pattern(constraint, total_qubits)
        k = len(fixed)
        base_coeff = weight / (2 ** k)

        # Expand product of k factors (I ± Z)/2 into all k-body subsets.
        for mask in range(1 << k):
            z_idx = []
            sign = 1.0

            for bit_pos, (qubit_idx, local_sign) in enumerate(fixed):
                if (mask >> bit_pos) & 1:
                    z_idx.append(qubit_idx)
                    sign *= local_sign

            coeffs[tuple(z_idx)] += base_coeff * sign

    return {term: coef for term, coef in coeffs.items() if not np.isclose(coef, 0.0)}

def create_qaoa_circ(ham_data, num_qubits, num_layers=1):
    """
    Create a parameterized QAOA circuit for a given Hamiltonian.

    Parameters
    ----------
    ham_data : dict
        The Hamiltonian data, where keys are tuples of qubit indices and values are the corresponding coefficients.
    num_qubits : int
        The number of qubits in the circuit.
    num_layers : int, optional
        The number of QAOA layers. The default is 1.

    Returns
    -------
    QuantumCircuit
        The constructed QAOA circuit. For a single layer, the circuit contains
        parameters named ``beta`` and ``gamma`` so values can be bound with
        ``qc.assign_parameters({gamma: g, beta: b})``.
    """
    if not isinstance(ham_data, dict):
        raise TypeError("ham_data must be a dict mapping Z-terms to coefficients.")
    num_qubits = _ensure_int("num_qubits", num_qubits, min_value=1)
    num_layers = _ensure_int("num_layers", num_layers, min_value=1)
    circuit = QuantumCircuit(num_qubits)

    if num_layers == 1:
        beta_parameters = [Parameter("beta")]
        gamma_parameters = [Parameter("gamma")]
    else:
        beta_parameters = list(ParameterVector("beta", num_layers))
        gamma_parameters = list(ParameterVector("gamma", num_layers))

    circuit.metadata = {
        **(circuit.metadata or {}),
        "beta_parameters": tuple(beta_parameters),
        "gamma_parameters": tuple(gamma_parameters),
    }

    def apply_z_term(qubits, angle):
        for left, right in zip(qubits[:-1], qubits[1:]):
            circuit.cx(left, right)
        circuit.rz(angle, qubits[-1])
        for left, right in zip(reversed(qubits[:-1]), reversed(qubits[1:])):
            circuit.cx(left, right)

    for qubit in range(num_qubits):
        circuit.h(qubit)

    for layer in range(num_layers):
        for qubit in range(num_qubits):
            circuit.rx(2 * beta_parameters[layer], qubit)

        for z_term_qubits, coeff in ham_data.items():
            if not z_term_qubits or np.isclose(coeff, 0.0):
                continue
            qubits = list(z_term_qubits)
            apply_z_term(qubits, 2 * gamma_parameters[layer] * coeff)

    circuit.measure_all()
    return circuit