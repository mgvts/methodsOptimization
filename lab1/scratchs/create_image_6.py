import json

import matplotlib.pyplot as plt
import numpy as np

with open('../data/count-iter67.json', 'r') as file:
    dt = json.loads(file.read())
    d = dt

data = np.empty((10, 10))
vm = 99999
vmax = 0
for i in d:
    z = sum(i['dichotomy_grag_sr_iter']) / len(i['dichotomy_grag_sr_iter'])
    vm = min(vm, z)
    vmax = max(vmax, z)
    print(z)
    if z != -1:
        data[int(i['n'] / 100), int(i['k'] / 100)] = z
    else:
        data[int(i['n'] / 100), int(i['k'] / 100)] = 800
fig, ax = plt.subplots()

c = ax.imshow(data, cmap='summer', interpolation='bilinear', vmin=vm, vmax=vmax, extent=([0, 1000, 0, 1000]))  #
ax.set_xlabel('n')
ax.set_ylabel('k')
fig.colorbar(c, ax=ax)
# plt.title(f"Function: {dt['func']}\n Type: {dt['type']}, Alpha: {dt['alpha']}, Eps: {dt['eps']}")
plt.show()
fig.savefig('../images/6_dichotomy.png')
