import os
import numpy as np
from matplotlib import pyplot as plt
from lab2.linear_regression import LinearRegression
import matplotlib as mpl

mpl.use('TkAgg')

# x0 * 1 + x1 * c1 + x2
# y = b0 * x0^0 + b1 * x1^1 + b2 * x2
# x2 = x1 ** 2

def generate_polynom_regression(count, coefs=(1, 1), shift=(1, 2)):
    X = np.matrix(np.random.uniform(-4, 4, (count, 1)))
    X = np.column_stack((X, np.square(X)))
    Y = np.zeros((count, 1))
    for i in range(2):
        Y += coefs[i] * X[:, i]
    Y += np.matrix(np.random.uniform(shift[0], shift[1], (count, 1)))

    X = np.column_stack((np.ones((count, 1))[:, 0], X))
    return X, Y


if __name__ == '__main__':
    count = 20
    X, Y = generate_polynom_regression(count, coefs=(1, 4))

    b = np.matrix([10., 10., 10.0]).transpose()
    points = LinearRegression(X, Y, b, 10).stochastic_grad_down_points(alpha=0.001, runs=300)

    X = np.delete(X, 0, 1)
    X = np.delete(X, 1, 1)
    x = np.linspace(-10, 10, 100)

    path = '../../images/temp/'
    for i in os.listdir(path):
        os.remove(f'{path}{i}')

    for i in range(0, len(points), 10):
        fig = plt.figure(figsize=(6, 6))

        y = points[i][0, 0] + points[i][1, 0] * x + points[i][2, 0] * x * x
        plt.plot(x, y, '-r')
        plt.plot(X[:], Y[:], 'o', label='data')

        plt.xlim([-5, 5])
        plt.ylim([0, 22])

        plt.savefig(rf'../../images/temp/image_{i}.png')
        plt.close()

    # fig = plt.figure(figsize=(6, 6))
    #
    # y = points[-1][0, 0] + points[-1][1, 0] * x + points[-1][2, 0] * x * x
    # plt.plot(x, y, '-r')
    # plt.plot(X[:], Y[:], 'o', label='data')
    #
    # plt.xlim([-5, 5])
    # plt.ylim([0, 22])
    #
    # plt.show()