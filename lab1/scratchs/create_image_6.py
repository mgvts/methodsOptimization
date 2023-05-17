import json

import matplotlib.pyplot as plt
import numpy as np

dt = {}
d = []
with open('../data/count-iter67_n2-302_k2-992.json', 'r') as file:
    _ = json.loads(file.read())
    d += _

with open('../data/count-iter67_n302-602.json', 'r') as file:
    _ = json.loads(file.read())
    d += _

with open('../data/count-iter67_n992-692_k2-902.json', 'r') as file:
    _ = json.loads(file.read())
    d += _


data = np.empty((10, 10))
vm = 99999
vmax = 0
for i in d:
    r = i['const_grag_sr_iter'].copy()
    while -1 in r:
        r.pop(-1)
    if len(r) != 0:
        z = sum(r) / len(r)
    else:
        z = 1
    vm = min(vm, z)
    vmax = max(vmax, z)
    print(z)
    if z != -1:
        data[int(i['n'] / 100), int(i['k'] / 100)] = z
    else:
        data[int(i['n'] / 100), int(i['k'] / 100)] = 20000

# for i in range(6):
#     for j in range(9):
#         if data[j, i] == -1:
#             data[i, j]


fig, ax = plt.subplots()
#bilinear
c = ax.imshow(data, cmap='summer', interpolation='bilinear', vmin=vm, vmax=vmax, extent=([0, 1000, 1000, 0]))  #
plt.gca().invert_yaxis()
ax.set_xlabel('n')
ax.set_ylabel('k')
fig.colorbar(c, ax=ax)
plt.show()
fig.savefig('../images/6_const.png')
