import numpy as np
from lab2.linear_regression import LinearRegression

def generate_linear_regression_nd(count, x=(0, 10), shift=(0, 0), coefs=(1, 2, 3, 4)):
    X = np.matrix(np.random.uniform(x[0], x[1], (count, len(coefs))))
    Y = np.zeros((count, 1))
    for i in range(len(coefs)):
        Y += coefs[i] * X[:, i]
    Y += np.matrix(np.random.uniform(shift[0], shift[1], (count, 1)))
    return X, Y


if __name__ == '__main__':
    X, Y = generate_linear_regression_nd(20, coefs=(10, 2, 3, 5))
    b = np.matrix([0., 10., 10., 10.]).transpose()
    b = LinearRegression(X, Y, b, 5).stochastic_grad_down(runs=1000)
    print(b)

