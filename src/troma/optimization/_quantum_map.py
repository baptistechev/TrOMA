from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector

from ..core.structure import Hamiltonian
from .._validation import ensure_int


def create_qaoa_circ(
    hamiltonian: Hamiltonian,
    num_layers: int = 1,
) -> QuantumCircuit:
    """
    Create a parameterized QAOA circuit for a given Hamiltonian.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object.
    num_layers : int, optional
        Number of QAOA layers. Default is 1.

    Returns
    -------
    QuantumCircuit
        Parameterized QAOA circuit.
    """
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("hamiltonian must be a Hamiltonian instance.")
    num_qubits = ensure_int("hamiltonian.num_qubits", hamiltonian.num_qubits, min_value=1)
    num_layers = ensure_int("num_layers", num_layers, min_value=1)

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

    def apply_z_term(qubits: list[int], angle: Any) -> None:
        for left, right in zip(qubits[:-1], qubits[1:]):
            circuit.cx(left, right)
        circuit.rz(angle, qubits[-1])
        for left, right in zip(reversed(qubits[:-1]), reversed(qubits[1:])):
            circuit.cx(left, right)

    for qubit in range(num_qubits):
        circuit.h(qubit)

    for layer in range(num_layers):
        for z_term_qubits, coeff in hamiltonian.terms.items():
            if not z_term_qubits or np.isclose(coeff, 0.0):
                continue
            apply_z_term(list(z_term_qubits), 2 * gamma_parameters[layer] * coeff)
        for qubit in range(num_qubits):
            circuit.rx(2 * beta_parameters[layer], qubit)

    circuit.measure_all()
    return circuit
