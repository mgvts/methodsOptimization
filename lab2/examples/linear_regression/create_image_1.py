import numpy as np
from matplotlib import pyplot as plt

from lab2.linear_regression import LinearRegression
import twod_line
import matplotlib as mpl

mpl.use('TkAgg')
if __name__ == '__main__':
    count = 10
    X, Y = twod_line.generate_linear_regression_2d(count, shift=(2, 2))

    X = np.matrix(
        [[8.5416, 14.584339906038576], [8.2922, 17.078413778150818], [7.699, 23.00911519619328], [4.0431, 59.56759537565088], [3.0378, 69.62568726997998], [0.9025, 90.96851077788254], [0.6705, 93.29517799680318], [0.1951, 98.06759331746257], [0.0692, 99.28014323584033], [0.0251, 99.91427488174314], [0.016, 100.1163934115714], [0.0066, 99.78465061643551]]

    )

    Y = np.matrix(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ).T

    count = len(Y)
    print(count)

    b = np.matrix([2., 10.]).transpose()
    b1 = LinearRegression(X, Y, b, count).stochastic_grad_down_points(alpha=0.00001, runs=1000, eps = 0.001)
    print(len(b1))
    print(b1[-1])

    b2 = LinearRegression(X, Y, b, count / 2).stochastic_grad_down_points(alpha=0.00001, runs=1000, eps = 0.0001)
    print(len(b2))

    b3 = LinearRegression(X, Y, b, count).stochastic_grad_down_points(alpha=0.00001, runs=100, eps = 0.0001)
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


    x = np.arange(0, 12, 0.1)
    y = np.arange(-0.5, 0.5, 0.01)
    xgrid, ygrid = np.meshgrid(x, y)

    z = 0 * xgrid * ygrid
    r = LinearRegression(X, Y, b, count)

    x_cnt = 0
    for i in x:
        y_cnt = 0
        for j in y:
            m = r.get_grad_in_point(np.matrix([i, j]).T)
            z[y_cnt, x_cnt] = abs(m[0, 0] + m[1, 0])
            y_cnt += 1
        x_cnt += 1

    plt.contour(xgrid, ygrid, z, colors='black', linewidths=0.2, levels=100)

    plt.title(f"Start point=(0.1, 0), SGD={len(b1)}, Minibatch CD={len(b2)}, CD={len(b3)}")
    plt.plot(x1, y1, 'o-r', alpha=1, label="SGD", lw=0.5, mec='b', mew=1, ms=1)
    plt.plot(x2, y2, '.-g', alpha=1, label="Minibatch CD", lw=0.5, mec='m', mew=1, ms=1)
    plt.plot(x3, y3, '1-k', alpha=1, label="CD", lw=0.8, mec='c', mew=1, ms=1)

    plt.legend()

    # plt.grid(True)
    plt.show()
