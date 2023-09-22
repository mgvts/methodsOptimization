import numpy as np
from matplotlib import pyplot as plt

from lab2.examples.linear_regression import twod_line
from lab2.linear_regression import LinearRegression
from lab4.linear.pytorch_linear_regression import PyTorchLinearRegression

if __name__ == '__main__':
    count = 100
    X, Y = twod_line.generate_linear_regression_2d(count, shift=(2, 2))
    b = np.matrix([10., 10.]).transpose()

    regression = LinearRegression(X, Y, b, count / 2)
    torch_regression = PyTorchLinearRegression(X, Y, b, count / 2)

    data = [
        # {
        #     'name': "Minibatch CD",
        #     'points': regression.stochastic_grad_down_points(alpha=0.001, runs=1000, eps=0.0001),
        #     'c': 'o-g',
        #     'mec': 'g'
        # },
        # {
        #     'name': "Minibatch CD Pytorch",
        #     'torch_points': torch_regression.stochastic_grad_down_points(alpha=0.001, runs=1000, eps=0.0001),
        #     'c': 'o-m',
        #     'mec': 'm'
        # },
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
        # {
        #     "name": "Momentum Pytorch",
        #     'torch_points': torch_regression.momentum_stochastic_grad_down_points(y=0.8, alpha=0.001, runs=1000),
        #     'c': 'v-c',
        #     'mec': 'c'
        # },
        # {
        #     "name": "Nesterov Pytorch",
        #     'torch_points': torch_regression.nesterov_stochastic_grad_down_points(y=0.9, alpha=0.0001, runs=1000),
        #     'c': 'v-r',
        #     'mec': 'r'
        # },
        # {
        #     "name": "Adagrad",
        #     'points': regression.adagrad_stochastic_grad_down_points(alpha=10, runs=1000),
        #     'c': 'v-c',
        #     'mec': 'c'
        # },
        # {
        #     'name': "Adagrad Pytorch",
        #     'torch_points': torch_regression.adagrad_stochastic_grad_down_points(alpha=10, runs=1000, eps=0.0001),
        #     'c': 'o-m',
        #     'mec': 'm'
        # },
        # {
        #     "name": "RMS",
        #     'points': regression.rms_stochastic_grad_down_points(W=4, alpha=0.2, runs=1000),
        #     'c': 'v-r',
        #     'mec': 'r'
        # },
        # {
        #     'name': "RMS Pytorch",
        #     'torch_points': torch_regression.rms_stochastic_grad_down_points(W=0.99, alpha=0.2, runs=1000, eps=0.0001),
        #     'c': 'o-g',
        #     'mec': 'g'
        # },
        {
            "name": "Adam",
            'points': regression.adam_stochastic_grad_down_points(b1=0.9, b2=0.9, alpha=0.11, runs=1000),
            'c': 'v-b',
            'mec': 'b'
        },
        {
            'name': "ADAM Pytorch",
            'torch_points': torch_regression.adam_stochastic_grad_down_points(b1=0.9, b2=0.9, alpha=0.11, runs=1000, eps=0.0001),
            'c': 'o-g',
            'mec': 'g'
        },
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
        if 'torch_points' in j:
            b = j['torch_points']
            print(j)
            for i in range(0, len(b)):
                x.append(b[i][0])
                y.append(b[i][1])
        else:
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
