# Task 2:
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

fig = plt.figure(figsize=(10, 15))
gs = grid_spec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))


# stable equilibrium
def first_eqn_pos(alpha):
    filtered = np.where(alpha < 0, np.nan, alpha)
    return np.sqrt(filtered)


# unstable equilibrium
def first_eqn_neg(alpha):
    filtered = np.where(alpha < 0, np.nan, alpha)
    return -np.sqrt(filtered)


# unsteady state
def first_eqn_unsteady(alpha):
    return np.where(alpha < 0, 0, np.nan)


# stable equilibrium
def second_eqn_pos(alpha):
    filtered = np.where(alpha < 3, np.nan, alpha)
    return np.sqrt((filtered - 3) / 2)


# unstable equilibrium
def second_eqn_neg(alpha):
    filtered = np.where(alpha < 3, np.nan, alpha)
    return -np.sqrt((filtered - 3) / 2)


# unsteady state
def second_eqn_unsteady(alpha):
    return np.where(alpha < 3, 0, np.nan)


domain = linspace(-2, 8)

ax1.plot(domain, first_eqn_pos(domain), 'b-', label='stable equilibrium', linewidth=3)
ax1.plot(domain, first_eqn_neg(domain), 'r--', label='unstable equilibrium', linewidth=3)
ax1.plot(domain, first_eqn_unsteady(domain), 'g--', label='unsteady state', linewidth=3)
ax1.plot([0], [0], 'ok', label='steady state')
ax1.set_xlabel('alpha')
ax1.set_ylabel('x equilibrium points')
ax1.set_title(r'$\alpha - x^2$')
ax1.legend()

ax2.plot(domain, second_eqn_pos(domain), 'b-', label='stable equilibrium', linewidth=3)
ax2.plot(domain, second_eqn_neg(domain), 'r--', label='unstable equilibrium', linewidth=3)
ax2.plot(domain, second_eqn_unsteady(domain), 'g--', label='unsteady state', linewidth=3)
ax2.plot([3], [0], 'ok', label='steady state')
ax2.set_xlabel('alpha')
ax2.set_ylabel('x equilibrium points')
ax2.set_title(r'$\alpha - 2x^2 - 3$')
ax2.legend()

plt.show()
