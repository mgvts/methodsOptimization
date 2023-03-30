import json

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from lab1.tools import Func

with open('../data/3a_count_iter_1_1_3.json', 'r') as file:
    dt = json.loads(file.read())
    d = dt['matrix']

data = np.zeros((200, 200))
vm = 99999
vmax = 0
for i in d:
    vm = min(vm, d[i])
    vmax = max(vmax, d[i])
    x, y = map(float, i.split(', '))
    data[int(x), int(y)] = d[i]

fig, ax = plt.subplots()

c = ax.imshow(data, cmap='summer', interpolation='none', vmin=vm, vmax=vmax, extent=([-100, 100, -100, 100]))  #
fig.colorbar(c, ax=ax)
plt.title(f"Function: {dt['func']}\n Type: {dt['type']}, Alpha: {dt['alpha']}, Eps: {dt['eps']}")

# level lines
def make_data():
    n = 2
    func = Func(n, dt['func'])
    x = np.arange(-100, 100, 0.5)
    y = np.arange(-100, 100, 0.5)
    xgrid, ygrid = np.meshgrid(x, y)

    # wow watafuck
    z = sp.lambdify(func.sp_variables, func.f, 'numpy')


    return xgrid, ygrid, z(xgrid, ygrid)

x, y, z = make_data()

# !!!
cs = plt.contour(x, y, z, levels=20)
plt.clabel(cs)

plt.show()
fig.savefig('../images/3_1_1_3.png')
