# Task 2:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as grid_spec


def andronov_phase_portrait(alpha):
    x1, x2 = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    u = alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2)
    v = x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)
    plt.quiver(x1, x2, u, v)
    plt.streamplot(x1, x2, u, v)
    plt.title(r'$\alpha$ = ' + str(alpha))
    plt.show()


def andronov_orbit():
    alpha = 1
    dt = .001
    steps = 8000
    # initial points
    point1 = (2.0, 0.0)
    point2 = (0.5, 0.0)

    fig, ax = plt.subplots()

    ax.annotate(xy=point1, s=rf'$x_0$ = ({point1[0]}, {point1[1]})')
    # This is implementation using Euler's method at alpha = 1
    x1, x2 = [point1[0]], [point1[1]]
    for i in range(steps):
        point1 += dt * andronov_hopf(point1, alpha)
        x1.append(point1[0])
        x2.append(point1[1])
    ax.plot(np.array(x1), np.array(x2))
    ax.annotate(xy=point2, s=rf'$x_0$ = ({point2[0]}, {point2[1]})')

    # This is implementation using Euler's method at alpha = 1
    x1, x2 = [point2[0]], [point2[1]]
    for i in range(steps):
        point2 += dt * andronov_hopf(point2, alpha)
        x1.append(point2[0])
        x2.append(point2[1])
    ax.plot(np.array(x1), np.array(x2))

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.title(rf'Orbits in Andronov Hopf system for $\alpha$ = {alpha}')
    plt.show()


def andronov_hopf(point, alpha):
    x1, x2 = point
    u = alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2)
    v = x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)
    return np.array([u, v])


def cusp_bifurcation():
    alpha_range = linspace(-2, 2)
    alpha1s = []
    alpha2s = []

    eqn_roots = []
    for alpha1 in alpha_range:
        for alpha2 in alpha_range:
            for root in np.roots([-1, 0, alpha2, alpha1]):
                if np.isreal(root):
                    alpha1s.append(alpha1)
                    alpha2s.append(alpha2)
                    eqn_roots.append(np.real(root))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(alpha1s, alpha2s, eqn_roots)
    ax.set_xlabel(r'$\alpha_1$')
    ax.set_ylabel(r'$\alpha_2$')
    ax.set_zlabel(r'$x$')
    plt.show()


# Script for plotting
for alpha in [-1, 0, 1]:
    andronov_phase_portrait(alpha)
andronov_orbit()
cusp_bifurcation()
