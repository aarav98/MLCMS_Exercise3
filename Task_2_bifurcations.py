# Task 2:
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec


def bifurcations(alpha_domain=linspace(-2, 8)):
    """
    This method plots the bifurcation diagram for the given equations in Task 2 of exercise 3 over specified
    alpha domain
    :param alpha_domain:
    :return:
    """
    fig, f_axes = plt.subplots(ncols=2, nrows=1, figsize=(14, 6))
    plot(alpha_domain, first_eqn, r'$\alpha - x^2$', f_axes[0])
    plot(alpha_domain, second_eqn, r'$\alpha - 2x^2 - 3$', f_axes[1])
    plt.show()


def plot(domain, equation, equation_text, axes):
    """
    This method plots the bifurcation diagram for the specified equation on the specified alpha domain
    :param domain:
    :param equation:
    :param equation_text:
    :param axes:
    :return:
    """
    pos, neg, unsteady, point = equation(domain)
    axes.plot(domain, pos, 'b-', label='stable equilibrium', linewidth=3)
    axes.plot(domain, neg, 'r--', label='unstable equilibrium', linewidth=3)
    axes.plot(domain, unsteady, 'g--', label='unsteady state', linewidth=3)
    axes.plot([point[0]], [point[1]], 'ok', label='steady state')
    axes.set_xlabel('alpha')
    axes.set_ylabel('x equilibrium points')
    axes.set_title(equation_text)
    axes.legend()


def first_eqn(alpha):
    """
    Given equation alpha - x^2
    :param alpha:
    :return: Returns stable/ unstable equilibrium points along with unsteady state of the given equation
    """
    filtered = np.where(alpha < 0, np.nan, alpha)
    return np.sqrt(filtered), -np.sqrt(filtered), np.where(alpha < 0, 0, np.nan), (0, 0)


def second_eqn(alpha):
    """
    Given equation alpha - 2x^2 - 3
    :param alpha:
    :return: Returns stable/ unstable equilibrium points along with unsteady state
    """
    filtered = np.where(alpha < 3, np.nan, alpha)
    return np.sqrt((filtered - 3) / 2), -np.sqrt((filtered - 3) / 2), np.where(alpha < 3, 0, np.nan), (3, 0)


if __name__ == '__main__':
    bifurcations(alpha_domain=linspace(-2, 8))
