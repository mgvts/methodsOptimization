from pprint import pprint

import numpy as np
import sympy as sp

from lab3.BFGS import bfgs
from lab1.grad import grad_down, grad_down_wolfe


def grad(f, x):
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


'''
func from here:
https://github.com/maxim092001/Itmo-University/blob/master/math-optimization/readme_images/lab4/lab4_2.png

calc for grad:
https://www.symbolab.com/solver/gradient-calculator
'''
# task1
f = lambda x: sum([i ** 2 for i in x])
grad_f = lambda x: np.array([2 * i for i in x])
# task2
f1 = lambda x: 100 * ((x[1] - x[0]) ** 2) + (1 - x[0]) ** 2
grad_f1 = lambda x: np.array([202 * x[0] - 200 * x[1] - 2, 200 * x[1] - 200 * x[0]])
# task3
f2 = lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
grad_f2 = lambda x: np.array([4 * (x[0] ** 3) + 4 * x[0] * x[1] - 42 * x[0] + 2 * (x[1] ** 2) - 14,
                              4 * (x[1] ** 3) + 4 * x[0] * x[1] - 26 * x[1] + 2 * (x[0] ** 2) - 22])


def bfgs_example1():
    n = 3
    x0 = np.array([1 for i in range(n)]).T
    print(bfgs(f, grad_f, x0).points[-1])


def bfgs_example2():
    def f2(x):
        d = len(x)
        return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(d - 1))

    # grad with math rules (by definition)

    grad_f2 = lambda x: grad(f2, x)

    x0 = np.array([-1.2, 1]).T
    print(bfgs(f2, grad_f2, x0).points[-1])


def show(points: [np.array]):
    '''
    вдохновлялся lab2/examples/linear_regression/create_image_5.py
    '''
    pprint(points)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use('TkAgg')
    plt.figure(figsize=(12, 5))
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-20, 20, 0.01)
    xgrid, ygrid = np.meshgrid(x, y)
    z = (xgrid) ** 2 + (ygrid) ** 2
    plt.contour(xgrid, ygrid, z, colors='black', linewidths=0.2, levels=100, alpha=0.6)
    b = points
    x = []
    y = []
    for i in range(0, len(b)):
        x.append(b[i][0])
        y.append(b[i][1])
    plt.plot(x, y, 'v-c', alpha=1, label="hui", lw=0.5, mec='r', mew=1, ms=1)
    plt.scatter(1, 1, 1, 'b')
    print("ans = " + str(points[-1]))

    plt.title(f"Gg")
    plt.legend()
    plt.show()


# show(bfgs(f=f,
#           grad_f=grad_f,
#           x0=np.array([10, -10])).points)

# seems good
i_good = bfgs(f=f2,
              grad_f=grad_f2,
              x0=np.array([10, -10])).points
# pprint(i_good)
print(len(i_good))
print(*[f"point:{i}, value:{f2(i)}" for i in i_good], sep="\n")

print("st")
i_bad = grad_down_wolfe(2,
                        "(x0^2 + x1 - 11)^2 + (x0 + x1^2 - 7)^2",
                        sp.Matrix([[10, -10]])).points

print(len(i_bad))
print(f"result value {f2(i_bad[-1])}")
print(*[f"point:[{i[0,0]},{i[0, 1]}], value:{f2(i)} {count=}" for i, count in zip(i_bad, range(len(i_bad)))], sep="\n")
# show(bfgs(f=f2,
#           grad_f=grad_f2,
#           x0=np.array([10, -10])).points)
