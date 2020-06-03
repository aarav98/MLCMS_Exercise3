# Task 2:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as grid_spec


def andronov_phase_portrait(alpha):
    """
    This plots the andronov hopf phase portrait for the given alpha.
    :param alpha:
    :return:
    """
    x1, x2 = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    # These are the equations given in the exercise
    u = alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2)
    v = x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)
    plt.quiver(x1, x2, u, v)
    plt.streamplot(x1, x2, u, v)
    plt.title(r'$\alpha$ = ' + str(alpha))
    plt.show()


def andronov_orbit(initial_points=None,
                   alpha=1,
                   dt=.001,
                   steps=8000):
    """
    This plots andronov orbits on the same plot for given initial points
    :param initial_points: List of tuple specifying initial points
    :param alpha:
    :param dt: step size
    :param steps:  number of steps
    :return:
    """
    if initial_points is None:
        initial_points = [(2.0, 0.0), (0.5, 0.0)]
    fig, ax = plt.subplots()

    for point in initial_points:
        ax.annotate(xy=point, s=rf'$x_0$ = ({point[0]}, {point[1]})')
        # This is implementation using Euler's method at alpha = 1
        x1, x2 = [point[0]], [point[1]]
        for i in range(steps):
            point += dt * andronov_hopf(point, alpha)
            x1.append(point[0])
            x2.append(point[1])
        ax.plot(np.array(x1), np.array(x2))

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.title(rf'Orbits in Andronov Hopf system for $\alpha$ = {alpha}')
    plt.show()


def andronov_hopf(point, alpha):
    """
    This is the given equation in exercise 3 task 3 (equation 8)
    :param point:
    :param alpha:
    :return:
    """
    x1, x2 = point
    u = alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2)
    v = x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)
    return np.array([u, v])


def cusp_bifurcation():
    """
    This is a method to visualize cusp bifurcation for 2 parameters alpha1 and alpha2.
    This is using the scatter plot to plot the cusp bifurcation surface.
    :return:
    """
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


if __name__ == '__main__':
    #plot phase portrait for alphas proved in the list
    for alpha in [-1, 0, 1]:
        andronov_phase_portrait(alpha)
    andronov_orbit()
    cusp_bifurcation()
