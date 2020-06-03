# Task 1:
import numpy as np
import matplotlib.pyplot as plt


def hyperbolic_equilibrium(
        x_intervals=10,
        y_intervals=10,
        x_min=-1,
        x_max=1,
        y_min=-1,
        y_max=1,
        alpha=1):
    """
    This method plots the figures of hyperbolic equilibrium for 3 vector fields (node, saddle and focus)
    as shown in the Fig. 2.5 from https://www.springer.com/de/book/9780387219066 book.
    :param alpha:
    :param x_intervals:
    :param y_intervals:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :return:
    """
    matrices = [get_node_alpha_matrix(alpha), get_focus_alpha_matrix(alpha), get_saddle_alpha_matrix(alpha)]
    fig1, f1_axes = plt.subplots(ncols=len(matrices), nrows=1, figsize=(7 * len(matrices), 6))
    x, y = np.meshgrid(np.linspace(x_min, x_max, x_intervals), np.linspace(y_min, y_max, y_intervals))
    u, v = np.zeros((x_intervals, y_intervals)), np.zeros((x_intervals, y_intervals))

    for idx, matrix in enumerate(matrices):
        for i in range(x_intervals):
            for j in range(y_intervals):
                xy = np.array([x[i, j], y[i, j]])
                uv = matrix @ xy
                u[i, j] = uv[0]
                v[i, j] = uv[1]
        f1_axes[idx].quiver(x, y, u, v)
        f1_axes[idx].streamplot(x, y, u, v, color='red')
        f1_axes[idx].title.set_text('At alpha = 1.0')
    plt.show()


def get_node_alpha_matrix(alpha=1):
    """
    This returns an alpha parameterized matrix for node vector field (dynamical system)
    that is used to compute the equilibrium
    :param alpha:
    :return:
    """
    return np.array([[alpha, 0], [0, 2 * alpha]])


def get_focus_alpha_matrix(alpha=1):
    """
    This returns an alpha parameterized matrix for focus vector field (dynamical system)
    that is used to compute the equilibrium
    :param alpha:
    :return:
    """
    return np.array([[0 * alpha, -2 * alpha], [alpha, -1 * alpha]])


def get_saddle_alpha_matrix(alpha=1):
    """
    This returns an alpha parameterized matrix for saddle vector field (dynamical system)
    that is used to compute the equilibrium
    :param alpha:
    :return:
    """
    return np.array([[alpha, 0 * alpha], [0 * alpha, -alpha]])


def phase_portrait(
        x_intervals=10,
        y_intervals=10,
        x_min=-1,
        x_max=1,
        y_min=-1,
        y_max=1,
        alphas=None):
    """
    This is a method to plot phase portraits similar to figures given in the Task 1 of exercise 3.
    :param x_intervals:
    :param y_intervals:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param alphas:
    :return:
    """
    if alphas is None:
        alphas = [0.1, 0.5, 2.0, 10.0]
    fig2, f2_axes = plt.subplots(ncols=4, nrows=1, figsize=(28, 6))
    x, y = np.meshgrid(np.linspace(x_min, x_max, x_intervals), np.linspace(y_min, y_max, y_intervals))
    v = -0.25 * x

    for idx, alpha in enumerate(alphas):
        u = alpha * x + alpha * y
        f2_axes[idx].quiver(x, y, u, v)
        f2_axes[idx].streamplot(x, y, u, v, color='red')
        f2_axes[idx].title.set_text('alpha =' + str(alpha))

    plt.show()


if __name__ == '__main__':
    hyperbolic_equilibrium(x_intervals=10,
                           y_intervals=10,
                           x_min=-1,
                           x_max=1,
                           y_min=-1,
                           y_max=1)
    phase_portrait(x_intervals=10,
                   y_intervals=10,
                   x_min=-1,
                   x_max=1,
                   y_min=-1,
                   y_max=1)
