import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from typing import Union
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

    s_circuit = step_circuit(hamiltonian, delta_t).to_gate(label="step_circuit")
    for _ in range(num_trotter_steps):
        circuit.append(s_circuit, qreg)
            

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
        gate = pauli_evolution_circuit(pauli, coeff, delta_t).to_gate(label="pauli_circuit")
        circuit.append(gate, qreg)

    return circuit

def diag_pauli_circuit(pauli: Pauli) -> QuantumCircuit:
        circuit = QuantumCircuit(pauli.num_qubits)
        for i in range(pauli.num_qubits):
            if pauli.x[i]:
                if pauli.z[i]:
                    circuit.sdg(i)
                circuit.h(i)
        return circuit

def control_circuit(pauli: Pauli) -> QuantumCircuit:
    circuit = QuantumCircuit(pauli.num_qubits)
    cx_positions = np.where(np.logical_or(pauli.x, pauli.z))
    for i in range(len(cx_positions[0])-1):
         circuit.cx(cx_positions[0][i], cx_positions[0][i+1])
    return circuit


def rotation_circuit(
        pauli: Pauli, coeff: complex, delta_t: float
    ) -> QuantumCircuit:
        phi = np.real(2 * coeff * delta_t)
        qreg = QuantumRegister(pauli.num_qubits)
        circuit = QuantumCircuit(qreg)

        c = control_circuit(pauli)
        circuit.append(c.to_gate(label = "control_circuit"), qreg)
        rotation_positions = np.where(np.logical_or(pauli.x, pauli.z))
        circuit.rz(phi, (rotation_positions[0][-1]))
        circuit.append(c.inverse().to_gate(label = "inv_control"), qreg)

        return circuit

def pauli_evolution_circuit(pauli: Pauli, coeff: complex, delta_t: float) -> QuantumCircuit:
    """
    build the trotter circuit for one particular pauli

    args:
    pauli: the pauli to use for the circuit
    delta_t: the variation of time for this pauli circuit

    return: the circuit for the troterrization of this particular pauli
    """

    qreg = QuantumRegister(pauli.num_qubits, "q")
    circuit = QuantumCircuit(qreg)

    gate = diag_pauli_circuit(pauli)

    circuit.append(gate.to_gate(label="diag_circuit"), qreg)
    circuit.append(
        rotation_circuit(pauli, coeff, delta_t).to_gate(label="rotation"), qreg
    )
    circuit.append(gate.inverse().to_gate(label="inv_diag"), qreg)

    return circuit