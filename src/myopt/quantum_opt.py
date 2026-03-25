import numpy as np
import neal
from qiskit_aer import AerSimulator
import scipy.optimize as sk_opt

import data_structure as ds
import abstract as ab
import quantum_map as qm

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
    
    #Define spin-chain Ising Hamiltonian from the marginals
    n = len(marginals)//4 + 1
    constraints = ab.constraints_for_nearest_neighbors_interactions(n,2)
    H = qm.compute_hamiltonian(constraints, marginals, bit_string_length=n)

    #Convert the Hamiltonian to the format required by the neal library
    h = {i: - H.get((i,), 0.0) for i in range(n)}
    J = {(i, i+1): - H.get((i, i+1), 0.0) for i in range(n-1)}

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_ising(h, J, num_reads=number_iter)
    config = [int((1-j)/2) for i,j in sampleset.first.sample.items()]
    return ds.dit_string_to_integer(config)


def QAOA(marginals, bit_constraints, bit_string_length, number_layers=3, method="COBYLA", backend=AerSimulator(), number_shots=512, random_seed=123):
    """
    """

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
            circuit = qm.create_qaoa_circ(theta, ham_data, num_qubits=bit_string_length)
            return _run_backend(circuit, number_shots)

        def execute_circ(theta):
            counts = run_qaoa_circuit(theta, ham_data)
            return compute_expectation(counts)

        return execute_circ

    def sample_most_likely_state(circuit):
        counts = _run_backend(circuit, number_shots)
        return max(counts, key=counts.get)
        
    number_parameters = 2 * number_layers
    bounds = np.array([[-np.pi, np.pi]]*number_parameters, dtype=float)

    ham_data = qm.compute_hamiltonian(bit_constraints,marginals, bit_string_length=bit_string_length)
    expectation = _get_expectation(ham_data)
    res = sk_opt.minimize(expectation, x0=np.ones(number_parameters), bounds=bounds, method=method)

    qc_res = qm.create_qaoa_circ(res.x,ham_data, num_qubits=bit_string_length)
    best_conf = sample_most_likely_state(qc_res)
    return ds.dit_string_to_integer(best_conf, convention='L')