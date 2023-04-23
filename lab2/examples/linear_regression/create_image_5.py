import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

import twod_line
from lab2.linear_regression import LinearRegression

mpl.use('TkAgg')
if __name__ == '__main__':
    count = 100
    X, Y = twod_line.generate_linear_regression_2d(count, shift=(2, 2))
    b = np.matrix([10., 10.]).transpose()

    regression = LinearRegression(X, Y, b, count / 2)

    data = [
        {
            'name': "Minibatch CD",
            'points': regression.stochastic_grad_down_points(alpha=0.001, runs=1000, eps=0.0001),
            'c': 'o-g',
            'mec': 'g'
        },
        # {
        #     "name": "Momentum",
        #     'points': regression.momentum_stochastic_grad_down_points(y=0.8, alpha=0.001, runs=1000),
        #     'c': 'o-m',
        #     'mec': 'm'
        # },
        # {
        #     "name": "Nesterov",
        #     'points': regression.nesterov_stochastic_grad_down_points(y=0.9, alpha=0.0001, runs=1000),
        #     'c': 'o-b',
        #     'mec': 'b'
        # },
        {
            "name": "Adagrad",
            'points': regression.adagrad_stochastic_grad_down_points(alpha=5, runs=1000),
            'c': 'v-c',
            'mec': 'c'
        },
        # {
        #     "name": "RMS",
        #     'points': regression.rms_stochastic_grad_down_points(W=10, alpha=0.2, runs=1000),
        #     'c': 'v-r',
        #     'mec': 'r'
        # },
        {
            "name": "Adam",
            'points': regression.adam_stochastic_grad_down_points(b1=0.9, b2=0.9, alpha=0.1, runs=1000),
            'c': 'v-b',
            'mec': 'b'
        }
    ]

    plt.figure(figsize=(12, 5))

    x = np.arange(0, 12, 0.01)
    y = np.arange(-20, 20, 0.01)
    xgrid, ygrid = np.meshgrid(x, y)

    z = 2 * (xgrid - 2) ** 2 + (ygrid - 1) ** 2
    plt.contour(xgrid, ygrid, z, colors='black', linewidths=0.2, levels=100, alpha=0.6)

    title = ''

    for j in data:
        x = []
        y = []
        b = j['points']
        for i in range(0, len(b)):
            x.append(b[i][0, 0])
            y.append(b[i][1, 0])
        plt.plot(x, y, j['c'], alpha=1, label=j['name'], lw=0.5, mec=j['mec'], mew=1, ms=1)
        title += f', {j["name"]}={len(b)}'

    plt.scatter(2, 1, 1, 'b')

    plt.title(f"Start point=(10, 10)" + title)
    plt.legend()
    plt.show()
