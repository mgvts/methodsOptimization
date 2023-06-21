'''
f(x) = 1/2 * x^T * Q * x + b^T x + c
'''
import random
from pprint import pprint

import numpy as np
from lab3.BFGS import bfgs
from lab1.tools import fast_generate_quadratic_func
from lab1.fast_grad import grad_down_wolfe
import pandas as pd


def qfunc(Q: np.matrix, b: np.array, c: float):
    def f(x: np.array) -> float:
        return (1 / 2) * x.T @ Q @ x + b.T @ x + c

    def grad(x: np.array) -> np.array:
        return Q @ x + b

    return f, grad


# q = np.matrix([[1, 0], [0, 1]])
# b = np.array([2, 0])
# c = 1
# x = np.array([1, 0])
# f, grad = qfunc(q, b, c)


df = pd.DataFrame(columns=['n', 'k', 'name', 'was_broken', 'iters'])

eps = 1e-4
maxIter = 100
for _ in range(5):
    for n in [2, 5, 10, 20]:
        for k in [1, 5, 10]:
            print(f"{n=} {k=}")
            x = np.array([random.randint(-10, 10) for _ in range(n)])
            Qfunc = fast_generate_quadratic_func(n, k)
            gradient_search_out = grad_down_wolfe(Qfunc, list(x), eps=eps, max_inter=maxIter)
            df = pd.concat([df, pd.DataFrame({'n': [n], 'k': [k], 'name': ['gradient_search'],
                                 'was_broken': [gradient_search_out.was_broken],
                                 'iters': [len(gradient_search_out.points)]})], ignore_index=True)
            f, grad = qfunc(Qfunc.A, np.array(Qfunc.b.flatten())[0], Qfunc.c)
            bfgs_out = bfgs(f, grad, x, eps=eps, max_iter=maxIter)
            df = pd.concat([df, pd.DataFrame({'n': [n], 'k': [k], 'name': ['bfgs'],
                                 'was_broken': [bfgs_out.was_broken],
                                 'iters': [len(bfgs_out.points)]})], ignore_index=True)

df.to_csv('stat.csv')
