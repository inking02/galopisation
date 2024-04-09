from qutip import Bloch
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt


def afficher_evolution_bloch(results: NDArray):
    b = Bloch()
    xs = np.array(results[:, 0])
    ys = np.array(results[:, 1])
    zs = np.array(results[:, 2])
    pnts = [xs, ys, zs]
    b.add_points(pnts)
    b.save()


def afficher_evolution_2D(
    results: NDArray,
    time_values: NDArray,
    first_observable: str = "valeur moyenne en x",
    second_observable: str = "valeur moyenne en y",
    third_observable: str = "valeur moyenne en z",
    graph_title: str = "évolution des valeurs moyennes x, y et z",
):
    y1 = np.array(results[:, 0])
    x = time_values
    plt.plot(x, y1, label=first_observable)
    y2 = np.array(results[:, 1])
    plt.plot(x, y2, label=second_observable)
    y3 = np.array(results[:, 2])
    plt.plot(x, y3, label=third_observable)
    plt.title(graph_title)
    plt.legend()
    plt.xlabel("temps en seconde")
    plt.ylabel("valeur moyenne des observables")
    plt.savefig(graph_title)

def comparaison_graph(exact_results: NDArray, trotter_results : NDArray, time_values : NDArray[np.float_], tr_time_values: NDArray[np.float_], observable_column : int):
    y1 = np.array(exact_results[: ,observable_column])
    plt.plot(time_values, y1, label = "observable avec exact_evolution")
    y2 = np.array(trotter_results[:, observable_column])
    plt.plot(tr_time_values, y2, "ob", label = "observable avec trotter_evolution")
    plt.title("comparaison de la valeur moyenne d'un observable entre les résultats exacts et les résultats de la trotterisation")
    plt.legend()
    plt.xlabel("temps en seconde")
    plt.ylabel("valeur moyenne de l'observable")
    plt.savefig("comparaison_graph")

def soustraction_graph(exact_results: NDArray, trotter_results : NDArray, time_values : NDArray[np.float_], observable_column : int, tolerance : float):
    y = np.array((exact_results[: ,observable_column] - trotter_results[:, observable_column])<tolerance, dtype=int)
    plt.plot(time_values, y)
    plt.title("comparaison de la valeur moyenne d'un observable entre les résultats exacts et les résultats de la trotterisation")
    plt.xlabel("temps en seconde")
    plt.ylabel("valeur moyenne de l'observable")
    plt.savefig("soustraction_graph")