import numpy as np
from galoper import exact_evolution, pauli_circuit, trotter_circuit
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli

nb_qubits = 1


Pauli_list = ["I", "Z"]
coeffs = 1/2*np.ones(2**nb_qubits)
hamiltonian = SparsePauliOp(Pauli_list, coeffs)
initial_state = QuantumCircuit(nb_qubits)
initial_state.h(0)
time_values = np.arange(0, 3, 1)
X = SparsePauliOp("X", [1])
Y = SparsePauliOp("Y", [1])
Z = SparsePauliOp("Z", [1])
observables = [X, Y, Z]

exact_result = exact_evolution(initial_state, hamiltonian, time_values, observables)
print(exact_result)

pauli_list = ["IZXX", "ZIZI", "XXYY", "ZYXI"]
coeffs = 1/2*np.ones(4)
hamiltonian = SparsePauliOp(pauli_list, coeffs)
time = 15
steps = 5

test = trotter_circuit(hamiltonian, time, steps)
#test = pauli_circuit(Pauli("XYIZ"), complex(0 ,1/2), 0.5)

fig = test.decompose(reps = 3).draw(output = "mpl", style = "iqp")
fig.savefig("trotter_circuit")