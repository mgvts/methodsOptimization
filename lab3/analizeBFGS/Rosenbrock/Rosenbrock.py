'''
f(x,y)=(1-x)^{2}+100(y-x^{2})^{2}.

f(x) = sum_{i=1}^{N-1} [(1-x_i)^2 + 100(x_{i+1} - x_i^2)^2]
'''
import random

import numpy as np
import pandas as pd

from lab3.analizeBFGS.grad import const_grad_down, wolfe_grad_down
from lab3.BFGS import bfgs


def Rosenbrock():
    def f(x: np.array):
        n = len(x)
        return sum([((1 - x[i]) ** 2 + 100 * ((x[i + 1] - x[i]) ** 2)) for i in range(n - 1)])

    def grad(x: np.array):
        from lab3.examples.bfgs_example import grad
        return grad(f, x)

    return f, grad


if __name__ == "__main__":
    df = pd.DataFrame(columns=['n', 'name', 'alpha', 'was_broken', 'iters'])

    eps = 1e-4
    maxIter = 1000
    for _ in range(5):
        for n in [2, 5, 10, 20]:
            print(f"{n=}")
            x = np.array([(random.random() - 0.5) * 20 for _ in range(n)])

            f, grad = Rosenbrock()

            const_search_out = const_grad_down(f, grad, x, alpha=1e-4, eps=eps, max_iter=maxIter)
            df = pd.concat([df, pd.DataFrame({'n': [n],  'name': ['const_search'],
                                              'alpha': ['0.0001'],
                                              'was_broken': [const_search_out.was_broken],
                                              'iters': [len(const_search_out.points)]})], ignore_index=True)

            const_search_out = const_grad_down(f, grad, x, alpha=1e-2, eps=eps, max_iter=maxIter)
            df = pd.concat([df, pd.DataFrame({'n': [n],  'name': ['const_search'],
                                              'alpha': ['0.01'],
                                              'was_broken': [const_search_out.was_broken],
                                              'iters': [len(const_search_out.points)]})], ignore_index=True)

            wolfe_search_out = wolfe_grad_down(f, grad, x, eps=eps, max_iter=maxIter)
            df = pd.concat([df, pd.DataFrame({'n': [n],  'name': ['wolfe_search'],
                                              'alpha': [None],
                                              'was_broken': [wolfe_search_out.was_broken],
                                              'iters': [len(wolfe_search_out.points)]})], ignore_index=True)

            bfgs_out = bfgs(f, grad, x, eps=eps, max_iter=maxIter)
            df = pd.concat([df, pd.DataFrame({'n': [n],  'name': ['bfgs'],
                                              'alpha': [None],
                                              'was_broken': [bfgs_out.was_broken],
                                              'iters': [len(bfgs_out.points)]})], ignore_index=True)

    df.to_csv('stat.csv')

    # print(const_grad_down(Rosenbrock()[0], Rosenbrock()[1], np.array([1, -1], )).points)


