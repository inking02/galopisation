import numpy as np
from qiskit import QuantumCircuit
from typing import List, Union
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.primitives import Estimator
def exact_evolution(initial_state: QuantumCircuit, hamiltonian: SparsePauliOp, time_values: NDArray[np.float_], observables: List[SparsePauliOp],):
    """
    Simulate the exact evolution of a quantum system in state ‘initial_state‘ under a given
    ‘hamiltonian‘ for different ‘time_values‘. The result is a series of expected values
    for given ‘observables‘.

    Args:
    time_values (NDArray[np.float_]): An array of time values.
    initial_state (QuantumCircuit): The circuit preparing the initial quantum state.
    hamiltonian (SparsePauliOp): The Hamiltonien of the system
    observables (List[SparsePauliOp]): The observable to be measure at each the the
    ‘time_values‘.

    Returns:
    NDArray[np.float_]: The expected values of the observable. Should be of shape
    ‘(len(time_values), len(observables))‘.
    """

    diag_hamiltonian = None

    evolution_operator = None

    evolution_state = evolution_operator * Statevector(initial_state)
    observables_expected_values = None


    return observables_expected_values

def trotter_evolution(initial_state: QuantumCircuit, hamiltonian: SparsePauliOp, time_values: NDArray[np.float_], observables: List[SparsePauliOp], num_trotter_steps: NDArray[np.int_],):
    """
    Simulate, using Trotterisation, the evolution of a quantum system in state ‘initial_state‘
    under a given ‘hamiltonian‘ for different ‘time_values‘. The result is a series of
    expected values for given ‘observables‘.

    Args:
    time_values (NDArray[np.float_]): An array of time values.
    initial_state (QuantumCircuit): The circuit preparing the initial quantum state.
    hamiltonian (SparsePauliOp): The Hamiltonien of the system
    observables (List[SparsePauliOp]): The observable to be measure at each the the
    ‘time_values‘.
    num_trotter_steps: (NDArray[np.int_]): The number of steps of the Trotterization for
    each ‘time_values‘.

    Returns:
    NDArray[np.float_]: The expected values of the observable. Should be of shape
    ‘(len(time_values), len(observables))‘.
    """

    observables_expected_values = None

    return observables_expected_values

def trotter_circuit(hamiltonian: SparsePauliOp, total_duration: Union[float, Parameter], num_trotter_steps: int,) -> QuantumCircuit:
    """
    Construct the ‘QuantumCircuit‘ using the first order Trotter formula.

    Args:
    hamiltonian (SparsePauliOp): The Hamiltonian which governs the evolution.
    total_duration (Union[float, Parameter]): The duration of the complete evolution.
    num_trotter_steps (int): The number of trotter steps.

    Returns:
    QuantumCircuit: The circuit of the Trotter evolution operator
    """



    circuit = None

    return circuit