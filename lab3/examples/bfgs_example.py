import numpy as np

from lab3.BFGS import bfgs


def bfgs_example1():
    # f(x) =x1^2 + x2^2 + x3^2 + ...
    f = lambda x: sum([i ** 2 for i in x])
    grad_f = lambda x: np.array([2 * i for i in x])

    n = 3
    x0 = np.array([1 for i in range(n)]).T
    print(bfgs(f, grad_f, x0))


def bfgs_example2():
    def f2(x):
        '''
        FUNCTION TO BE OPTIMISED
        '''
        d = len(x)
        return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(d - 1))

    # grad with math rules (by definition)
    def grad(f, x):
        '''
        CENTRAL FINITE DIFFERENCE CALCULATION
        '''
        h = np.cbrt(np.finfo(float).eps)
        d = len(x)
        nabla = np.zeros(d)
        for i in range(d):
            x_for = np.copy(x)
            x_back = np.copy(x)
            x_for[i] += h
            x_back[i] -= h
            nabla[i] = (f(x_for) - f(x_back)) / (2 * h)
        return nabla

    grad_f2 = lambda x: grad(f2, x)

    x0 = np.array([-1.2, 1]).T
    print(bfgs(f2, grad_f2, x0))
