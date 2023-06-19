import numpy as np
import matplotlib.pyplot as plt

delta = 0.001
x = np.arange(-0.01, 0.005, delta)
y = np.arange(-0.01, 0.005, delta)
X, Y = np.meshgrid(x, y)
_x = 2
b1 = X
b2 = Y
Z = 1 / ((b1 ** 2 + 2 * np.exp(_x) * b1 * b2 + b2 ** 2 * np.exp(2 * _x)))

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Simplest default with labels')

plt.show()
