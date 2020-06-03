# Task 1:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

# FIGURES FROM THE BOOK
fig1, f1_axes = plt.subplots(ncols=2, nrows=2)
X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
U, V = np.zeros((10, 10)), np.zeros((10, 10))
alpha = 1
# NODE
for i in range(10):
    for j in range(10):
        xy = np.array([X[i, j], Y[i, j]])
        A = np.array([[alpha, 0], [0, 2 * alpha]])
        uv = A @ xy
        U[i, j] = uv[0]
        V[i, j] = uv[1]


f1_axes[0][0].quiver(X, Y, U, V)
f1_axes[0][0].streamplot(X, Y, U, V, color='red')
f1_axes[0][0].title.set_text('Node at alpha = 1.0')


# FOCUS
for i in range(10):
    for j in range(10):
        xy = np.array([X[i, j], Y[i, j]])
        A = np.array([[0*alpha, -2*alpha], [alpha, -1*alpha]])
        uv = A @ xy
        U[i, j] = uv[0]
        V[i, j] = uv[1]


f1_axes[1][0].quiver(X, Y, U, V)
f1_axes[1][0].streamplot(X, Y, U, V, color='red')
f1_axes[1][0].title.set_text('Focus at alpha = 1.0')

# SADDLE
for i in range(10):
    for j in range(10):
        xy = np.array([X[i, j], Y[i, j]])
        A = np.array([[alpha, 0*alpha], [0*alpha, -alpha]])
        uv = A @ xy
        U[i, j] = uv[0]
        V[i, j] = uv[1]


f1_axes[0][1].quiver(X, Y, U, V)
f1_axes[0][1].streamplot(X, Y, U, V, color='red')
f1_axes[0][1].title.set_text('Saddle at alpha = 1.0')

plt.show()


# FIGURES FROM THE EXERCISE SHEET

fig = plt.figure(figsize=(10, 15))
gs = grid_spec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])
x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
Alphas = [0.1, 0.5, 2.0, 10.0]
v = -0.25*x

# alpha = 0.1
ax0 = fig.add_subplot(gs[0, 0])
u = Alphas[0]*x + Alphas[0]*y
ax0.quiver(x, y, u, v)
ax0.streamplot(x, y, u, v, color='red')
ax0.title.set_text('alpha = 0.1')

# alpha = 0.5
ax1 = fig.add_subplot(gs[0, 1])
u = Alphas[1]*x + Alphas[1]*y
ax1.quiver(x, y, u, v)
ax1.streamplot(x, y, u, v, color='red')
ax1.title.set_text('alpha = 0.5')

# alpha = 2.0
ax2 = fig.add_subplot(gs[1, 0])
u = Alphas[2]*x + Alphas[2]*y
ax2.quiver(x, y, u, v)
ax2.streamplot(x, y, u, v, color='blue')
ax2.title.set_text('alpha = 2.0')

# alpha = 10.0
ax3 = fig.add_subplot(gs[1, 1])
u = Alphas[3]*x + Alphas[3]*y
ax3.quiver(x, y, u, v)
ax3.streamplot(x, y, u, v, color='blue')
ax3.title.set_text('alpha = 10.0')

plt.show()
