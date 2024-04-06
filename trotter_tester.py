import numpy as np
from galoper import exact_evolution, trotter_evolution
from trotter_graphs import afficher_évolution_3D, afficher_evolution_bloch, afficher_evolution_1D
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import Estimator

############# test des fonctions implémentées ###################################################################
nb_qubits = 1
Pauli_list = ["X", "Y"]
coeffs = 1/2*np.ones(2**nb_qubits)
hamiltonian = SparsePauliOp(Pauli_list, coeffs)
initial_state = QuantumCircuit(nb_qubits)
initial_state.h(0)
time_values = np.arange(0, 1, 0.02)

X = SparsePauliOp("X", [1])
Y = SparsePauliOp("Y", [1])
Z = SparsePauliOp("Z", [1])
observables = [X, Y, Z]

num_steps = 100 * np.ones(50, dtype = int)
exact_results = exact_evolution(initial_state, hamiltonian, time_values, observables)
trotter_evolution = trotter_evolution(initial_state, hamiltonian, time_values, observables, num_steps)
#################################################################################################################


###################### test avec les outils de qiskit #########################
#qiskit_results = np.empty((10000, 3))
#for i, time in enumerate(time_values):
    #evolution_gate = PauliEvolutionGate(hamiltonian, time)
    #qreg = QuantumRegister(nb_qubits)
    #circuit = QuantumCircuit(qreg)
    #circuit.append(initial_state, qreg)
    #circuit.append(evolution_gate, qreg)
    #circuits = [circuit, circuit, circuit]
    #estimator = Estimator()
    #result = estimator.run(circuits, observables).result().values
    #qiskit_results[i, :] = result

###############################################################################

#tolerance = 1e-1
#test = exact_results-trotter_evolution
#test_2 = np.abs(test) > tolerance

###################### affichage des résultats pour comparaison ###################################
#print(exact_results)
#print(trotter_evolution)
#print(qiskit_results)
#print(test_2)
###################################################################################################

########################## affichage des résultats sur graphiques #################################
#afficher_évolution_3D(exact_results, time_values)
afficher_evolution_bloch(trotter_evolution)
#afficher_evolution_1D(exact_results, time_values)
###################################################################################################
