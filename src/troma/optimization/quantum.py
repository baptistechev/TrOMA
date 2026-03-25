import numpy as np
import neal
import scipy.optimize as sk_opt
from qiskit_aer import AerSimulator
from numbers import Real

from ..sketchs import abstract as ab
from .. import data_structure as ds
from ._quantum_map import compute_hamiltonian as _compute_hamiltonian, create_qaoa_circ as _create_qaoa_circ
from .._validation import ensure_int as _ensure_int, ensure_iterable as _ensure_iterable


def digital_annealing(marginals, number_iter=1000):
    """
    Perform digital annealing to find a solution to the optimization problem defined by the marginals.
    Only work for QUBO problems, i.e., when marginals are defined on nearest neighbor pairs of bits.

    Parameters
    ----------
    marginals : list of float
        The marginals of the function defined on the full spectrum of dit strings. The order of the values should correspond
        to the order of the dit strings in the full spectrum.
    number_iter : int, optional
        The number of iterations for the digital annealing algorithm. The default is 1000.
    
    Returns
    -------
    int
        The index of the dit string that maximizes the sum of the marginals.
    """
    _ensure_iterable("marginals", marginals)
    number_iter = _ensure_int("number_iter", number_iter, min_value=1)
    marginals = list(marginals)
    if len(marginals) == 0 or len(marginals) % 4 != 0:
        raise ValueError("marginals length must be a positive multiple of 4 for nearest-neighbor QUBO marginals.")

    #Define spin-chain Ising Hamiltonian from the marginals
    n = len(marginals)//4 + 1
    constraints = ab.constraints_for_nearest_neighbors_interactions(n,2)
    H = _compute_hamiltonian(constraints, marginals, bit_string_length=n)

    #Convert the Hamiltonian to the format required by the neal library
    h = {i: - H.get((i,), 0.0) for i in range(n)}
    J = {(i, i+1): - H.get((i, i+1), 0.0) for i in range(n-1)}

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_ising(h, J, num_reads=number_iter)
    config = [int((1-j)/2) for i,j in sampleset.first.sample.items()]
    return ds.dit_string_to_integer(config)


def QAOA(marginals, bit_constraints, bit_string_length, number_layers=4, method="COBYLA", backend=AerSimulator(), number_shots=4096, random_seed=123):
    """
    Perform Quantum Approximate Optimization Algorithm (QAOA) to find a solution to the optimization problem defined by the marginals and constraints.
    The function assumes that the optimization problem can be mapped to a Hamiltonian defined on qubits, where the terms in the Hamiltonian correspond to the patterns in the constraints_sketch.
    Applicable on bit strings of any length, and can handle arbitrary patterns of constraints, including those that do not correspond to nearest neighbor interactions.

    Parameters
    ----------
    marginals : list of float
        The marginals corresponding to each pattern in constraints_sketch. The order of the values should correspond to the order of the patterns in constraints_sketch.
    bit_constraints : list of patterns
        Each pattern can be either a list of local states (for full patterns) or a dict mapping qubit indices to fixed bit values (for constraints). The order of the patterns should correspond to the order of the marginals in the input.
    bit_string_length : int
        The length of the bit strings (i.e., the number of qubits).
    number_layers : int, optional
        The number of layers in the QAOA circuit. The default is 4.
    method : str, optional
        The optimization method to use for finding the optimal parameters of the QAOA circuit. The default is "COBYLA".
    backend : qiskit provider, optional
        The quantum backend to use for running the QAOA circuit. The default is AerSimulator().
    number_shots : int, optional
        The number of shots to use when running the QAOA circuit. The default is 4096.
    random_seed : int, optional
        The random seed to use for reproducibility when running the QAOA circuit. The default is 123.

    Returns
    ------
    int
        The index of the bit string that maximizes the sum of the marginals, as found by the QAOA algorithm.
    """
    _ensure_iterable("marginals", marginals)
    marginals = list(marginals)
    if not isinstance(bit_constraints, (list, tuple)):
        raise TypeError("bit_constraints must be a list or tuple of constraints.")
    if len(bit_constraints) == 0:
        raise ValueError("bit_constraints must be non-empty.")
    if len(marginals) != len(bit_constraints):
        raise ValueError("marginals and bit_constraints must have the same length.")
    bit_string_length = _ensure_int("bit_string_length", bit_string_length, min_value=1)
    number_layers = _ensure_int("number_layers", number_layers, min_value=1)
    number_shots = _ensure_int("number_shots", number_shots, min_value=1)
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if random_seed is not None:
        _ensure_int("random_seed", random_seed)
    if backend is None or not hasattr(backend, "run"):
        raise TypeError("backend must provide a run(circuit, **kwargs) method.")

    def _run_backend(circuit, shots):
        run_kwargs = {"shots": shots}
        if random_seed is not None:
            run_kwargs["seed_simulator"] = random_seed
        try:
            return backend.run(circuit, **run_kwargs).result().get_counts()
        except TypeError:
            run_kwargs.pop("seed_simulator", None)
            return backend.run(circuit, **run_kwargs).result().get_counts()

    def _get_expectation(ham_data):

        def objective_function(config):
            config_index = ds.dit_string_to_integer(config, convention='L')
            return float(np.dot(- np.asarray(marginals), ab.reconstruct_structured_matrix_column(config_index, dit_constraints=bit_constraints, dit_string_length=bit_string_length)))

        def compute_expectation(counts):
            weighted_sum = 0

            for bitstring, shot_count in counts.items():
                objective_value = objective_function(bitstring)
                weighted_sum += objective_value * shot_count

            total_shots = max(sum(counts.values()), 1)
            return weighted_sum / total_shots

        def run_qaoa_circuit(theta, ham_data):
            circuit = _create_qaoa_circ(theta, ham_data, num_qubits=bit_string_length)
            return _run_backend(circuit, number_shots)

        def execute_circ(theta):
            counts = run_qaoa_circuit(theta, ham_data)
            return compute_expectation(counts)

        return execute_circ

    def sample_best_state(circuit):
        counts = _run_backend(circuit, number_shots)

        def objective_from_bitstring(bitstring):
            config_index = ds.dit_string_to_integer(bitstring, convention='L')
            return float(np.dot(- np.asarray(marginals), ab.reconstruct_structured_matrix_column(config_index, dit_constraints=bit_constraints, dit_string_length=bit_string_length)))

        return min(counts, key=objective_from_bitstring)

    number_parameters = 2 * number_layers
    bounds = np.array([[-np.pi, np.pi]]*number_parameters, dtype=float)

    ham_data = _compute_hamiltonian(bit_constraints,marginals, bit_string_length=bit_string_length)
    expectation = _get_expectation(ham_data)
    res = sk_opt.minimize(expectation, x0=np.ones(number_parameters), bounds=bounds, method=method)

    qc_res = _create_qaoa_circ(res.x,ham_data, num_qubits=bit_string_length)
    best_conf = sample_best_state(qc_res)
    return ds.dit_string_to_integer(best_conf, convention='L')
