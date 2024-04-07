from qutip import Bloch
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt


def afficher_evolution_3D(
    results: NDArray,
    first_observable: str = "valeur moyenne en x",
    second_observable: str = "valeur moyenne en y",
    third_observable: str = "valeur moyenne en z",
    graph_title: str = "évolution des valeurs moyennes x, y et z",
):

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel(first_observable)
    ax.set_ylabel(second_observable)
    ax.set_zlabel(third_observable)
    xs = np.array(results[:, 0])
    ys = np.array(results[:, 1])
    zs = np.array(results[:, 2])
    ax.scatter(xs, ys, zs, color="blue")
    plt.title(graph_title)
    plt.savefig(graph_title)


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
