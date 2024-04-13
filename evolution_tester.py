import numpy as np
from evolution_simulation import exact_evolution, trotter_evolution
from simulation_graphs import afficher_evolution_bloch, afficher_evolution_2D, comparaison_graph, soustraction_graph
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import Estimator
from numpy.typing import NDArray
from typing import List
from evolution_simulation_utils import pauli_evolution_circuit


def results_validation(exact_results :NDArray[np.complex_], trotter_results : NDArray[np.complex_], tolerance : float)-> NDArray[np.bool_]:
    soustraction_test = trotter_results - exact_results   
    validation = np.abs(soustraction_test)< tolerance

    return validation


############# test des fonctions implémentées ###################################################################
nb_qubits = 1
pas = 0.1
temps_total = 10
nb_valeurs = int(temps_total/pas)
Pauli_list = ["X", "Y"]
coeffs = 1/2*np.ones(2**nb_qubits)
hamiltonian = SparsePauliOp(Pauli_list, coeffs)
initial_state = QuantumCircuit(nb_qubits)
initial_state.h(0)

time_values = np.arange(0, temps_total, pas)
tr_stop = 10

tr_values = 25
tr_time_values = np.linspace(0, tr_stop, num = tr_values)

X = SparsePauliOp("X", [1])
Y = SparsePauliOp("Y", [1])
Z = SparsePauliOp("Z", [1])
observables = [X, Y, Z]

num_steps = 100 * np.ones(tr_values, dtype = int)

exact_results = exact_evolution(initial_state, hamiltonian, time_values, observables)
trotter_results = trotter_evolution(initial_state, hamiltonian, tr_time_values, observables, num_steps)


#validation = results_validation(exact_results, trotter_results, 1e-1)

#print(validation)


comparaison_graph(exact_results, trotter_results, time_values, tr_time_values, len(observables))
#soustraction_graph(exact_results, trotter_results, time_values, 0, 0.01)
#afficher_evolution_2D(trotter_results, time_values, "IIX", "IYI", "ZII", "3_qubits_evolution")


#test = pauli_evolution_circuit(Pauli("IXIX"), 1/2, 1)

#fig = test.decompose(["control_circuit", "inv_control", "rotation", "diag_circuit", "inv_diag"], reps = 3).draw(output = "mpl", style = "iqp")
#fig.savefig("pauli_circuit")


#fig = test.decompose(["control_circuit", "inv_control", "diag_circuit", "inv_diag", "evolution"], reps = 2).draw(output = "mpl", style = "iqp")
#fig.savefig("pauli_circuit")
###################### test avec les outils de qiskit #########################
#qiskit_results = np.empty((500, 3))
#for i, time in enumerate(time_values):
#    evolution_gate = PauliEvolutionGate(hamiltonian, time)
#    qreg = QuantumRegister(nb_qubits)
#    circuit = QuantumCircuit(qreg)
#    circuit.append(initial_state, qreg)
#    circuit.append(evolution_gate, qreg)
#    circuits = [circuit, circuit, circuit]
#    estimator = Estimator()
#    result = estimator.run(circuits, observables).result().values
#    qiskit_results[i, :] = result

###############################################################################


