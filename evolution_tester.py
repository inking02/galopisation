import numpy as np
from galoper import exact_evolution, trotter_evolution
from trotter_graphs import afficher_evolution_3D, afficher_evolution_bloch, afficher_evolution_2D
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import Estimator
from numpy.typing import NDArray
from typing import List


def system_evolution_simulation(hamiltonian : SparsePauliOp, initial_state : QuantumCircuit, time_values : NDArray[np.float_], observables : List[SparsePauliOp], graph_method : str = "", nb_steps : NDArray[np.int_] = np.empty(1), evolution_method : str = "trotter", graph_labels : NDArray[np.str_] = np.empty(4, dtype = str)) -> NDArray[np.complex_]:

    ####### création de l'array qui représente l'évolution du système #####################################
    if evolution_method == "trotter":
        evolution = trotter_evolution(initial_state, hamiltonian, time_values, observables, nb_steps)
    elif evolution_method == "exact":
        evolution = exact_evolution(initial_state, hamiltonian, time_values, observables)
    else:
        print("Erreur dans la simulation de l'évolution...")
        print("Les deux méthodes disponibles sont 'trotter' et 'exact', veuillez entrer une méthode valide")
        return np.empty(1)
    
    if graph_method == "bloch":
        afficher_evolution_bloch(evolution)
    elif graph_method == "3D":
        if (graph_labels != np.empty(4, dtype = str)).all():
            afficher_evolution_3D(evolution, graph_title= graph_labels[0], first_observable=graph_labels[1], second_observable=graph_labels[2], third_observable=graph_labels[3])
        else: 
            afficher_evolution_3D(evolution)
    elif graph_method == "2D":
        if (graph_labels != np.empty(4, dtype = str)).all():
            afficher_evolution_2D(evolution, time_values, graph_title= graph_labels[0], first_observable=graph_labels[1], second_observable=graph_labels[2], third_observable=graph_labels[3])
        else:
            afficher_evolution_2D(evolution, time_values)
    else:
        print("Le graphique n'a pas été créé...")
        print("Les méthodes d'affichage de graphique sont 'bloch', '3D' et '2D'")
    
    return evolution


def results_validation(exact_results :NDArray[np.complex_], trotter_results : NDArray[np.complex_], tolerance : float)-> NDArray[np.bool_]:
    soustraction_test = trotter_results - exact_results   
    validation = np.abs(soustraction_test)< tolerance

    return validation


############# test des fonctions implémentées ###################################################################
nb_qubits = 1
pas = 0.5
temps_total = 10
nb_valeurs = int(temps_total/pas)
Pauli_list = ["X", "Y"]
coeffs = 1/2*np.ones(2**nb_qubits)
hamiltonian = SparsePauliOp(Pauli_list, coeffs)
initial_state = QuantumCircuit(nb_qubits)
initial_state.h(0)
time_values = np.arange(0, temps_total, pas)

X = SparsePauliOp("X", [1])
Y = SparsePauliOp("Y", [1])
Z = SparsePauliOp("Z", [1])
observables = [X, Y, Z]

num_steps = 100 * np.ones(nb_valeurs, dtype = int)

exact_results = system_evolution_simulation(hamiltonian, initial_state, time_values, observables, evolution_method = "exact", graph_method="2D")
trotter_results = system_evolution_simulation(hamiltonian, initial_state, time_values, observables, nb_steps=num_steps, evolution_method="trotter")


validation = results_validation(exact_results, trotter_results, 1e-1)

print(validation)


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


