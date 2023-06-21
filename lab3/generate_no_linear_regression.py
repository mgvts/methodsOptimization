import numpy as np

from lab3.util import show_image


def generate_no_linear_regression(count: int, x=(0, 10), noise=(0, 0), arg_count=1, func=lambda x: x ** 0.5) -> (
        np.matrix, np.matrix):
    X = np.matrix(np.random.uniform(x[0], x[1], (count, arg_count)))
    Y = np.zeros((count, 1))
    for i in range(count):
        Y[i, 0] = func(*X[i].tolist()[0])
    Y += np.matrix(np.random.uniform(noise[0], noise[1], (count, 1)))
    return X, Y


def generate_first_case(b1=0.1, b2=0.01):
    """
    func: y = 1 / (b1 + b2 * e^x)
    b1 = 0.1, b2 = 0.01
    """
    return generate_no_linear_regression(50, func=lambda i: 1 / (b1 + b2 * np.exp(i)), x=(-5, 10))


def generate_second_case():
    """
    func: y = b1 * x / (b2 + x)
    b1 = 10, b2 = 1
    """
    return generate_no_linear_regression(25, func=lambda i: 10 * i / (1 + i))


if __name__ == '__main__':
    X, Y = generate_first_case()
    print(X, Y)
    show_image(X, Y)
