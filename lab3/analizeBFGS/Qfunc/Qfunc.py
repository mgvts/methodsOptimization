'''
f(x) = 1/2 * x^T * Q * x + b^T x + c
'''
import random
from pprint import pprint

import numpy as np
from lab3.BFGS import bfgs
from lab1.tools import fast_generate_quadratic_func
from lab3.analizeBFGS.grad import const_grad_down, wolfe_grad_down

import pandas as pd


def qfunc(Q: np.matrix, b: np.array, c: float):
    def f(x: np.array) -> float:
        return (1 / 2) * x.T @ Q @ x + b.T @ x + c

    def grad(x: np.array) -> np.array:
        return Q @ x + b

    return f, grad


df = pd.DataFrame(columns=['n', 'k', 'name', 'alpha', 'was_broken', 'iters'])

if __name__ == '__main__':
    eps = 1e-4
    maxIter = 1000
    for _ in range(5):
        for n in [2, 5, 10, 20]:
            for k in [1, 5, 10]:
                print(f"{n=} {k=}")
                x = np.array([random.randint(-10, 10) for _ in range(n)])
                Qfunc = fast_generate_quadratic_func(n, k)
                f, grad = qfunc(Qfunc.A, np.array(Qfunc.b.flatten())[0], Qfunc.c)

                const_search_out = const_grad_down(f, grad, x, alpha=1e-4, eps=eps, max_iter=maxIter)
                df = pd.concat([df, pd.DataFrame({'n': [n], 'k': [k], 'name': ['const_search'],
                                                  'alpha': ['0.0001'],
                                                  'was_broken': [const_search_out.was_broken],
                                                  'iters': [len(const_search_out.points)]})], ignore_index=True)

                const_search_out = const_grad_down(f, grad, x, alpha=1e-2, eps=eps, max_iter=maxIter)
                df = pd.concat([df, pd.DataFrame({'n': [n], 'k': [k], 'name': ['const_search'],
                                                  'alpha': ['0.01'],
                                                  'was_broken': [const_search_out.was_broken],
                                                  'iters': [len(const_search_out.points)]})], ignore_index=True)

                wolfe_search_out = wolfe_grad_down(f, grad, x, eps=eps, max_iter=maxIter)
                df = pd.concat([df, pd.DataFrame({'n': [n], 'k': [k], 'name': ['wolfe_search'],
                                                  'alpha': [None],
                                                  'was_broken': [wolfe_search_out.was_broken],
                                                  'iters': [len(wolfe_search_out.points)]})], ignore_index=True)

                bfgs_out = bfgs(f, grad, x, eps=eps, max_iter=maxIter)
                df = pd.concat([df, pd.DataFrame({'n': [n], 'k': [k], 'name': ['bfgs'],
                                                  'alpha': [None],
                                                  'was_broken': [bfgs_out.was_broken],
                                                  'iters': [len(bfgs_out.points)]})], ignore_index=True)

    df.to_csv('stat.csv')
