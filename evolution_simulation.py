import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from typing import List
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector
from qiskit.primitives import Estimator
from evolution_simulation_utils import trotter_circuit


def exact_evolution(
    initial_state: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    time_values: NDArray[np.float_],
    observables: List[SparsePauliOp],
):
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

    exp_array = np.exp(-1j * np.einsum("i, j-> ij", time_values, eigenvalues))
    evolution_operators = np.einsum(
        "ik, sk, jk -> sij", eigenvectors, exp_array, np.conjugate(eigenvectors)
    )

    init_state = Statevector(initial_state)
    evolved_states = np.einsum("sij, j -> si", evolution_operators, init_state)
    observables_expected_values = np.einsum(
        "si, mij, sj -> sm", np.conjugate(evolved_states), observables, evolved_states
    )

    return observables_expected_values


def trotter_evolution(
    initial_state: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    time_values: NDArray[np.float_],
    observables: List[SparsePauliOp],
    num_trotter_steps: NDArray[np.int_],
):
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

    observables_list = len(num_trotter_steps)*observables

    for time_value, num_trotter_step in zip(time_values, num_trotter_steps):
        qreg = QuantumRegister(initial_state.num_qubits)
        circuit = QuantumCircuit(qreg)
        circuit.append(initial_state.to_gate(label="init_state"), qreg)
        circuit.append(
            trotter_circuit(hamiltonian, time_value, num_trotter_step).to_gate(
                label="trotter_circuit"
            ),
            qreg,
        )
        for _ in range(len(observables)):
            circuits.append(circuit)

    estimator = Estimator()
    results = estimator.run(circuits, observables_list).result().values

    new_shape = (len(time_values), len(observables))
    observables_expected_values = results.reshape(new_shape)
    return observables_expected_values
