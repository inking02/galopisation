import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from typing import List, Union
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit import Parameter


def trotter_circuit(
    hamiltonian: SparsePauliOp,
    total_duration: Union[float, Parameter],
    num_trotter_steps: int,
) -> QuantumCircuit:
    """
    Construct the "QuantumCircuit" using the first order Trotter formula.

    Args:
    hamiltonian (SparsePauliOp): The Hamiltonian which governs the evolution.
    total_duration (Union[float, Parameter]): The duration of the complete evolution.
    num_trotter_steps (int): The number of trotter steps.

    Returns:
    QuantumCircuit: The circuit of the Trotter evolution operator
    """
    qreg = QuantumRegister(hamiltonian.paulis.num_qubits, "q")
    circuit = QuantumCircuit(qreg)
    delta_t = total_duration / num_trotter_steps

    for _ in range(num_trotter_steps):
        circuit.append(
            step_circuit(hamiltonian, delta_t).to_gate(label="step_circuit"), qreg
        )

    return circuit


def step_circuit(hamiltonian: SparsePauliOp, delta_t: float) -> QuantumCircuit:
    """
    gives the trotter circuit for one step

    args:
    delta_t: the variation of time for this step of the circuit
    hamiltonian: the hamiltonian represented as a SparsePauliOp

    return: the circuit representing one step of trotterization
    """
    paulis = hamiltonian.paulis
    coeffs = hamiltonian.coeffs
    qreg = QuantumRegister(paulis.num_qubits, "q")
    circuit = QuantumCircuit(qreg)
    for pauli, coeff in zip(paulis, coeffs):
        gate = pauli_circuit(pauli, coeff, delta_t).to_gate(label="pauli_circuit")
        circuit.append(gate, qreg)

    return circuit


def pauli_circuit(pauli: Pauli, coeff: complex, delta_t: float) -> QuantumCircuit:
    """
    build the trotter circuit for one particular pauli

    args:
    pauli: the pauli to use for the circuit
    delta_t: the variation of time for this pauli circuit

    return: the circuit for the troterrization of this particular pauli
    """

    def diag_pauli_circuit(pauli: Pauli, nb_qubits: int) -> QuantumCircuit:
        circuit = QuantumCircuit(nb_qubits)
        for i in range(nb_qubits):
            if pauli.x[i]:
                if pauli.z[i]:
                    circuit.sdg(i)
                circuit.h(i)
        return circuit

    def evolution_circuit(
        nb_qubits: int, coeff: complex, delta_t: float
    ) -> QuantumCircuit:
        phi = np.real(2 * coeff * delta_t)
        circuit = QuantumCircuit(nb_qubits)
        for i in range(nb_qubits - 1):
            circuit.cx(i, i + 1)

        circuit.rz(phi, nb_qubits - 1)

        for i in reversed(range(nb_qubits - 1)):
            circuit.cx(i, i + 1)

        return circuit

    nb_qubits = pauli.num_qubits
    qreg = QuantumRegister(nb_qubits, "q")
    circuit = QuantumCircuit(qreg)

    gate = diag_pauli_circuit(pauli, nb_qubits)

    circuit.append(gate.to_gate(label="diag_circuit"), qreg)
    circuit.append(
        evolution_circuit(nb_qubits, coeff, delta_t).to_gate(label="evolution"), qreg
    )
    circuit.append(gate.inverse().to_gate(label="inv_diag"), qreg)

    return circuit