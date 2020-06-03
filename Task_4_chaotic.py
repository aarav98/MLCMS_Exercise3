
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
x = np.linspace(0, 1)
r = 2
fig, ax = plt.subplots(1, 1)
ax.set_title("Logistic Map")
plt.xlabel("x")
plt.ylabel("f(x)")
ax.plot(x, logistic(r, x))
plt.show()

# COBWEB PLOT TO ANALYSE THE FUNCTION
r = [0.5, 1, 1.5, 2]
n = 20
for i in r:
    x0 = 0.1
    fig, ax = plt.subplots(1, 1)
    ax.set_title("Cobweb Plot "+str(r.index(i)+1))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    t = np.linspace(0, 1)
    ax.plot(t, logistic(i, t), 'k', color="red")
    ax.plot([0, 1], [0, 1], 'k', color="black")
    for j in range(n):
        y0 = logistic(i, x0)
        ax.plot([x0, x0], [x0, y0], 'k', color="blue")
        ax.plot([x0, y0], [y0, y0], 'k', color="blue")
        ax.plot(x0, y0, 'ok',color="black",  ms=10, alpha=(j+1)/n)
        x0 = y0

    plt.show()

# PLOT THE BIFURCATION DIAGRAM
n = 1000
r = np.linspace(0, 4, n)
x = np.linspace(0, 1, n)

iterations = 250

fig, ax1 = plt.subplots(1, 1)
ax1.set_xlim(0, 4)
ax1.set_ylim(-0.2, 1)
ax1.set_title("Bifurcation diagram")
plt.xlabel("r")
plt.ylabel('x')

for i in range(iterations):
    x = logistic(r, x)
    ax1.plot(r, x, ',k', alpha=(i+1)/iterations)


plt.show()

# FIXED POINTS
# fig, ax2 = plt.subplots(1, 1)
# ax2.set_title("Fixed points")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# ax2.plot(r, fixed_point(logistic, x, args=(r,)))
# plt.show()
# print(fixed_point(logistic, x, args=(r,)))

# LORENZ ATTRACTOR
t = [0, 100]

_sigma = [10, 10]
_beta = [8/3, 8/3]
_rho = [28, 0.5]

for sigma, beta, rho in zip(_sigma, _beta, _rho):
    alpha = 1
    y = [[10, 10, 10], [10 + 10 ** -8, 10, 10]]
    fig = plt.figure()
    ax3 = fig.gca(projection='3d')
    fig.suptitle(rf'Lorenz system with $\sigma$ = {sigma}, $\rho$ = {rho}, $\beta$ = {beta}')
    for y0 in y:
        #print(y0)
        res = solve_ivp(lorenz, t, y0, args=(sigma, beta, rho))
        x, y, z = res.y
        lab = str("x = "+str(y0[0])+", y = "+str(y0[1])+", z = "+str(y0[2]))
        ax3.plot(x, y, z, alpha=alpha, label=lab)
        alpha *= 0.65

    ax3.legend()
    plt.show()