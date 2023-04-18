import numpy as np
from matplotlib import pyplot as plt

from lab2.linear_regression import LinearRegression
import twod_line
import matplotlib as mpl

mpl.use('TkAgg')
if __name__ == '__main__':
    count = 20
    X, Y = twod_line.generate_linear_regression_2d(count, shift=(2, 4))
    b = np.matrix([10., 10.]).transpose()
    b = LinearRegression(X, Y, b, 5).momentum_stochastic_grad_down(y=0.9, alpha=0.001, runs=1000)
    print(b)
    fig, ax = plt.subplots(1)

    x = np.linspace(0, 10, 100)
    y = b[0, 0] + b[1, 0] * x
    plt.plot(x, y, '-r')

    X = np.delete(X, 0, 1)
    plt.plot(X[:], Y[:], 'o', label='data')

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

    plt.show()
