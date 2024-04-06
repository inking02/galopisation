from qutip import Bloch
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt


def afficher_évolution_3D(results : NDArray, time_values : NDArray):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('valeur moyenne en x')
    ax.set_ylabel('valeur moyenne en Y')
    ax.set_zlabel('valeur moyenne en Z')  
    xs = np.array(results[:, 0], dtype = int)
    ys = np.array(results[:, 1], dtype = int)
    zs = np.array(results[:, 2], dtype = int)

    ax.scatter(xs, ys, zs, color = "blue")
    plt.title("évolution du système selon le temps (s)")
    plt.show()

def afficher_evolution_bloch(results : NDArray, time_values :NDArray):
    b = Bloch()
    xs = np.array(results[:, 0])
    ys = np.array(results[:, 1])
    zs = np.array(results[:, 2])
    pnts = [xs, ys, zs]
    b.add_points(pnts)
    b.show()

def afficher_evolution_2D(results :NDArray, time_values : NDArray):
    x = np.array(results[:, 0])
    y = np.array(results[: ,1])

    plt.plot(x, y)
    plt.show()

def afficher_evolution_1D(results :NDArray, time_values: NDArray):
    y = np.array(results[:, 2])
    x = time_values

    plt.plot(x, y)
    plt.show()