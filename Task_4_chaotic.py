import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fixed_point

logistic = lambda r, x: r * x * (1 - x)


def lorenz(t, y, sigma, beta, rho):
    return np.array([
        sigma * (y[1] - y[0]),
        rho * y[0] - y[1] - (y[0] * y[2]),
        y[0] * y[1] - beta * y[2]
    ])


# PLOT LOGISTIC MAP, f(x) and x
def plot_logistic(x, r):
    # x -> array with numbers between 0 and 1
    # r -> int, typically between 0 and 4

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Logistic Map")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    ax.plot(x, logistic(r, x))
    plt.show()


# COBWEB PLOT TO ANALYSE THE FUNCTION
def plot_cobweb(x, r, it=100):
    # r -> list of r values. Plots different plots for each r in the list.
    # x -> integer

    for i in r:
        x0 = x
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Cobweb Plot " + str(r.index(i) + 1))
        plt.xlabel("x")
        plt.ylabel("f(x)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        t = np.linspace(0, 1)
        ax.plot(t, logistic(i, t), 'k', color="red")
        ax.plot([0, 1], [0, 1], 'k', color="black")
        for j in range(it):
            y0 = logistic(i, x0)
            if j > 0.8 * it:
                ax.plot([x0, x0], [x0, y0], 'k', color="blue")
                ax.plot([x0, y0], [y0, y0], 'k', color="blue")
                ax.plot(x0, y0, 'ok', color="black", ms=10, alpha=(j + 1) / it)
            x0 = y0

        plt.show()


# PLOT THE BIFURCATION DIAGRAM
def plot_bifurcation(x=None, r=None, n=0, rlow=0, rhigh=4):
    # x -> an array of x values between 0 and 1
    # r -> an array of r values, typically between 0 and 4
    # n is an integer to automatically set x and r. if n is set x and r are ignored and new values are generated.

    iterations = 250
    fig, ax1 = plt.subplots(1, 1)
    print(n)
    if n > 0:
        x = np.linspace(0, 1, n)
        r = np.linspace(rlow, rhigh, n)
        ax1.set_xlim(rlow, rhigh)
        # ax1.set_ylim(0.3, 0.7)

    if r is None:
        r = [0, 1, 2, 3]
    if x is None:
        x = [0, 0.5, 1]
    # print (r)
    # print(x)
    ax1.set_title("Bifurcation diagram")
    plt.xlabel("r")
    plt.ylabel('x')

    for i in range(iterations):
        x = logistic(r, x)
        ax1.plot(r, x, ',k', alpha=(i + 1) / iterations)  # alpha value to decrease the intensity of initial x values
    plt.show()


# LORENZ ATTRACTOR
def plot_lorenz_attractor(y, _sigma=None, _beta=None, _rho=None, t=None):
    # sigma, beta, rho -> a list of values. the list sizes must be the same.
    # y -> a list of lists containing x, y, z coordinates

    if _sigma is None:
        _sigma = [10]
    if _beta is None:
        _beta = [8 / 3]
    if t is None:
        t = [0, 1000]
    if _rho is None:
        _rho = [28]

    for sigma, beta, rho in zip(_sigma, _beta, _rho):
        alpha = 1
        # y = [[10, 10, 10], [10 + 10 ** -8, 10, 10]]
        fig = plt.figure()
        ax3 = fig.gca(projection='3d')
        fig.suptitle(rf'Lorenz system with $\sigma$ = {sigma}, $\rho$ = {rho}, $\beta$ = {beta}')
        for y0 in y:
            # print(y0)
            res = solve_ivp(lorenz, t, y0, args=(sigma, beta, rho))
            # t = res.t
            x, y, z = res.y
            lab = str("x = " + str(y0[0]) + ", y = " + str(y0[1]) + ", z = " + str(y0[2]))
            ax3.plot(x, y, z, alpha=alpha, label=lab)
            alpha *= 0.65

        ax3.legend()
        plt.show()


if __name__ == '__main__':
    x = np.linspace(0, 1)
    r = 2
    plot_logistic(x, r)
    x = 0.01
    r = [2.5, 3.5]
    plot_cobweb(x, r)
    n = 1000
    plot_bifurcation(n=n)
    y = [[10, 10, 10], [10 + 10 ** -8, 10, 10]]
    plot_lorenz_attractor(y)
