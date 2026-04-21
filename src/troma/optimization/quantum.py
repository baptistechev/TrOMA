import numpy as np
from numbers import Real

import neal
import scipy.optimize as sk_opt
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2
from qiskit import transpile

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


def QAOA(marginals, bit_constraints, bit_string_length, number_layers=4, method="COBYLA", sampler=None, number_shots=4096, optimizer_options=None):
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
    sampler : qiskit provider, optional
        The quantum sampler to use for running the QAOA circuit. The default is None, which uses the AerSimulator().
    number_shots : int, optional
        The number of shots to use when running the QAOA circuit. The default is 4096.
    optimizer_options : dict, optional
        Additional keyword options forwarded to ``scipy.optimize.minimize``. This can be used, for example,
        to limit the number of optimizer evaluations with entries such as ``{"maxiter": 30}``.
    
    Returns
    ------
    int
        The index of the bit string that maximizes the sum of the marginals, as found by the QAOA algorithm.
    """

    #Security checks on the inputs
    #-----------------------------
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
    if optimizer_options is not None and not isinstance(optimizer_options, dict):
        raise TypeError("optimizer_options must be a dict or None.")
    #-----------------------------
    

    #Init the sampler if not provided as it is used in the cost function, and set the number of shots
    if sampler is None:
        backend = AerSimulator()
        sampler = SamplerV2(mode=backend, options={"default_shots": number_shots})
    sampler.options.default_shots = number_shots

    def _objective_function(config):
        config_index = ds.dit_string_to_integer(config, convention='L')
        return float(np.dot(- np.asarray(marginals), ab.reconstruct_structured_matrix_column(config_index, dit_constraints=bit_constraints, dit_string_length=bit_string_length)))

    def _bind_qaoa_parameters(circuit,theta):
        """
        Bind the QAOA circuit parameters to specific values of beta and gamma
        """

        beta_parameters = circuit.metadata["beta_parameters"]
        gamma_parameters = circuit.metadata["gamma_parameters"]

        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != 2 * number_layers:
            raise ValueError("theta length must match 2 * number_layers.")

        beta_values = theta[:number_layers]
        gamma_values = theta[number_layers:]
        parameter_map = {
            parameter: float(value)
            for parameter, value in zip(beta_parameters, beta_values)
        }
        parameter_map.update(
            {
                parameter: float(value)
                for parameter, value in zip(gamma_parameters, gamma_values)
            }
        )
        return circuit.assign_parameters(parameter_map, inplace=False)

    def _get_count_from_backend(sampler, circuit):
        job = sampler.run([circuit])
        result = job.result()
        return result[0].data.meas.get_counts()

    def _compute_expectation(counts):

        weighted_sum = 0

        for bitstring, shot_count in counts.items():
            objective_value = _objective_function(bitstring)
            weighted_sum += objective_value * shot_count

        total_shots = max(sum(counts.values()), 1)
        return weighted_sum / total_shots

    def cost_function(theta):
        #Bind the parameter
        binded_circuit = _bind_qaoa_parameters(qaoa_circuit, theta)

        #Run the sampling on backend
        counts = _get_count_from_backend(sampler, binded_circuit)

        #Get the expectation value from counts
        return _compute_expectation(counts)

    def sample_best_state(qaoa_circuit, optimal_theta):
        """
        Return the best state sampled from QAOA with the optimal parameters.
        """
        binded_circuit = _bind_qaoa_parameters(qaoa_circuit, optimal_theta)
        counts = _get_count_from_backend(sampler, binded_circuit)
        return min(counts, key=_objective_function)

    number_parameters = 2 * number_layers
    ham_data = _compute_hamiltonian(bit_constraints,marginals, bit_string_length=bit_string_length)
    qaoa_circuit = _create_qaoa_circ(
                        ham_data,
                        num_qubits=bit_string_length,
                        num_layers=number_layers,
                    )
    qaoa_circuit = transpile(qaoa_circuit, backend=sampler.backend())


    bounds = np.array([[-np.pi, np.pi]]*number_parameters, dtype=float)
    res = sk_opt.minimize(
        cost_function,
        x0=np.ones(number_parameters),
        bounds=bounds,
        method=method,
        options=dict(optimizer_options or {}),
    )

    optimal_theta = res.x
    best_conf = sample_best_state(qaoa_circuit, optimal_theta)

    return ds.dit_string_to_integer(best_conf, convention='L')
