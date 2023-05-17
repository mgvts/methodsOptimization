import random

import numpy as np

from lab3.util import show_image


def generate_no_linear_regression(count: int, x=(-10, 10), noise=(0, 0), arg_count=1, func=lambda x: x ** 0.5) -> (
        np.matrix, np.matrix):
    X = np.matrix(np.random.uniform(x[0], x[1], (count, arg_count)))
    Y = np.zeros((count, 1))
    for i in range(count):
        Y[i, 0] = func(*X[i].tolist()[0])
    Y += np.matrix(np.random.uniform(noise[0], noise[1], (count, 1)))
    return X, Y


if __name__ == '__main__':
    X, Y = generate_no_linear_regression(200, func=lambda i: 1 /(np.exp(i)))
    show_image(X, Y)
