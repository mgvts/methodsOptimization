import numpy as np
from matplotlib import pyplot as plt

from lab2.linear_regression import LinearRegression
import twod_line
import matplotlib as mpl

mpl.use('TkAgg')
if __name__ == '__main__':
    count = 100
    X, Y = twod_line.generate_linear_regression_2d(count, shift=(2, 2))
    b = np.matrix([10., 10.]).transpose()
    b1 = LinearRegression(X, Y, b, 1).stochastic_grad_down_points(alpha=0.01, runs=100000, eps = 0.0001)
    print(len(b1))

    b2 = LinearRegression(X, Y, b, count / 2).stochastic_grad_down_points(alpha=0.001, runs=100000, eps = 0.0001)
    print(len(b2))

    b3 = LinearRegression(X, Y, b, count).stochastic_grad_down_points(alpha=0.0004, runs=100000, eps = 0.0001)
    print(len(b3))

    x1 = []
    y1 = []

    for i in range(0, len(b1)):
        x1.append(b1[i][0, 0])
        y1.append(b1[i][1, 0])


    x2 = []
    y2 = []

    for i in range(0, len(b2)):
        x2.append(b2[i][0, 0])
        y2.append(b2[i][1, 0])


    x3 = []
    y3 = []

    for i in range(0, len(b3)):
        x3.append(b3[i][0, 0])
        y3.append(b3[i][1, 0])


    plt.figure(figsize=(5, 5))


    x = np.arange(0, 12, 0.05)
    y = np.arange(-10, 12, 0.05)
    xgrid, ygrid = np.meshgrid(x, y)

    z = 2*(xgrid - 2) ** 2 + (ygrid - 1) ** 2
    plt.contour(xgrid, ygrid, z, colors='black', linewidths=0.2, levels=100)

    plt.title(f"Start point=(10, 10), SGD={len(b1)}, Minibatch CD={len(b2)}, CD={len(b3)}")
    plt.plot(x1, y1, 'o-r', alpha=1, label="SGD", lw=0.5, mec='b', mew=1, ms=1)
    plt.plot(x2, y2, '.-g', alpha=1, label="Minibatch CD", lw=0.5, mec='m', mew=1, ms=1)
    plt.plot(x3, y3, '1-k', alpha=1, label="CD", lw=0.8, mec='c', mew=1, ms=1)

    plt.legend()

    # plt.grid(True)
    plt.show()
