import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from typing import List, Union
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.primitives import Estimator



def exact_evolution(initial_state: QuantumCircuit, hamiltonian: SparsePauliOp, time_values: NDArray[np.float_], observables: List[SparsePauliOp],):
    """
    Simulate the exact evolution of a quantum system in state "initial_state" under a given
    "hamiltonian" for different "time_values". The result is a series of expected values
    for given "observables".

    Args:
    time_values (NDArray[np.float_]): An array of time values.
    initial_state (QuantumCircuit): The circuit preparing the initial quantum state.
    hamiltonian (SparsePauliOp): The Hamiltonien of the system
    observables (List[SparsePauliOp]): The observable to be measure at each the the
    "time_values".

    Returns:
    NDArray[np.float_]: The expected values of the observable. Should be of shape
    "(len(time_values), len(observables))".
    """
    hamiltonian_matrix = hamiltonian.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)

    #w = np.exp((-1*time_values[:, None])*eigenvalues[None, :])
    w = np.exp(1j*np.einsum("i, j-> ij", time_values, eigenvalues))
    evolution_operators = np.einsum("ik, sk, jk -> sij", eigenvectors, w, np.conjugate(eigenvectors))

    init_state = Statevector(initial_state)
    evolved_states = np.einsum("sij, j -> si", evolution_operators, init_state)
    observables_expected_values = np.einsum("si, mij, sj -> sm", np.conjugate(evolved_states), observables, evolved_states)

    return observables_expected_values

def trotter_evolution(initial_state: QuantumCircuit, hamiltonian: SparsePauliOp, time_values: NDArray[np.float_], observables: List[SparsePauliOp], num_trotter_steps: NDArray[np.int_],):
    """
    Simulate, using Trotterisation, the evolution of a quantum system in state "initial_state"
    under a given "hamiltonian" for different "time_values". The result is a series of
    expected values for given "observables".

    Args:
    time_values (NDArray[np.float_]): An array of time values.
    initial_state (QuantumCircuit): The circuit preparing the initial quantum state.
    hamiltonian (SparsePauliOp): The Hamiltonien of the system
    observables (List[SparsePauliOp]): The observable to be measure at each the the
    "time_values".
    num_trotter_steps: (NDArray[np.int_]): The number of steps of the Trotterization for
    each "time_values".

    Returns:
    NDArray[np.float_]: The expected values of the observable. Should be of shape
    "(len(time_values), len(observables))".
    """
    circuits = []
    observables_list = []
    
    for time_value, num_trotter_step in zip(time_values, num_trotter_steps):
        observables_list = observables_list + observables
        qreg = QuantumRegister(initial_state.num_qubits)
        circuit = QuantumCircuit(qreg)
        circuit.append(initial_state.to_gate(label = "init_state"), qreg)
        circuit.append(trotter_circuit(hamiltonian, time_value, num_trotter_step).to_gate(label = "trotter_circuit"), qreg)
        #fig = circuit.decompose(["init_state", "trotter_circuit", "step_circuit", "pauli_circuit", "diag_circuit", "evolution", "inv_diag"], reps = 4).draw(output = "mpl", style = "iqp")
        #fig.savefig("evolution_circuit")
        for i in range(len(observables)):
            circuits.append(circuit)
    
    estimator = Estimator()
    results = estimator.run(circuits, observables_list).result().values
    
    reshape_tuple = (len(time_values), len(observables))
    observables_expected_values = results.reshape(reshape_tuple)
    return observables_expected_values

def trotter_circuit(hamiltonian: SparsePauliOp, total_duration: Union[float, Parameter], num_trotter_steps: int) -> QuantumCircuit:
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
    delta_t = total_duration/num_trotter_steps
    
    for i in range(num_trotter_steps):
        circuit.append(step_circuit(hamiltonian, delta_t).to_gate(label = "step_circuit"), qreg)
    

    return circuit

def step_circuit(hamiltonian : SparsePauliOp, delta_t: float)-> QuantumCircuit:
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
        gate = pauli_circuit(pauli, coeff, delta_t).to_gate(label = "pauli_circuit")
        circuit.append(gate, qreg)

    return circuit



def pauli_circuit(pauli: Pauli, coeff: complex, delta_t:float)-> QuantumCircuit:
    """
    build the trotter circuit for one particular pauli

    args: 
    pauli: the pauli to use for the circuit
    delta_t: the variation of time for this pauli circuit

    return: the circuit for the troterrization of this particular pauli
    """
    
    def diag_pauli_circuit(pauli: Pauli, nb_qubits: int)-> QuantumCircuit:
        circuit = QuantumCircuit(nb_qubits)
        for i in range(nb_qubits):
            if pauli.x[i]:
                if pauli.z[i]:
                    circuit.sdg(i)
                circuit.h(i)  
        return circuit
    
    def evolution_circuit(nb_qubits: int, coeff: complex, delta_t: float)->QuantumCircuit:
        phi = np.real(2*coeff*delta_t)
        circuit = QuantumCircuit(nb_qubits)
        for i in range(nb_qubits-1):
            circuit.cx(i, i+1)

        circuit.rz(phi, nb_qubits-1)   
        
        for i in reversed(range(nb_qubits-1)):
            circuit.cx(i, i+1)

        return circuit

    nb_qubits = pauli.num_qubits 
    qreg = QuantumRegister(nb_qubits, "q")   
    circuit = QuantumCircuit(qreg)

    gate = diag_pauli_circuit(pauli, nb_qubits)

    circuit.append(gate.to_gate(label = "diag_circuit"), qreg)
    circuit.append(evolution_circuit(nb_qubits, coeff, delta_t).to_gate(label = "evolution"), qreg)
    circuit.append(gate.inverse().to_gate(label = "inv_diag"), qreg)

    return circuit

