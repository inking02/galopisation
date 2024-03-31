import numpy as np
from galoper import exact_evolution
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

nb_qubits = 1


Pauli_list = ["I", "Z"]
coeffs = 1/2*np.ones(2**nb_qubits)
hamiltonian = SparsePauliOp(Pauli_list, coeffs)
initial_state = QuantumCircuit(nb_qubits)
initial_state.h(0)
time_values = [1, 2, 3, 4]
observables = None

test = exact_evolution(initial_state, hamiltonian, time_values, observables)