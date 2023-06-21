'''
f(x,y)=(1-x)^{2}+100(y-x^{2})^{2}.

f(x) = sum_{i=1}^{N-1} [(1-x_i)^2 + 100(x_{i+1} - x_i^2)^2]
'''
import random

import numpy as np
import pandas as pd

from lab3.analizeBFGS.grad import const_grad_down
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
    df = pd.DataFrame(columns=['n', 'k', 'name', 'was_broken', 'iters', 'ans'])

    eps = 1e-4
    maxIter = 100
    for _ in range(5):
        for n in [2, 5, 10, 20]:
            for k in [1, 5, 10]:
                print(f"{n=} {k=}")
                x = np.array([(random.random() - 0.5) * 20 for _ in range(n)])

                f, grad = Rosenbrock()

                gradient_search_out = const_grad_down(f, grad, list(x), eps=eps, max_iter=maxIter)
                df = pd.concat([df, pd.DataFrame({'n': [n], 'k': [k], 'name': ['gradient_search'],
                                                  'was_broken': [gradient_search_out.was_broken],
                                                  'iters': [len(gradient_search_out.points)],
                                                  'ans': [gradient_search_out.points[-1]]})], ignore_index=True)
                bfgs_out = bfgs(f, grad, x, eps=eps, max_iter=maxIter)
                df = pd.concat([df, pd.DataFrame({'n': [n], 'k': [k], 'name': ['bfgs'],
                                                  'was_broken': [bfgs_out.was_broken],
                                                  'iters': [len(bfgs_out.points)],
                                                  'ans': [bfgs_out.points[-1]]
                                                  })], ignore_index=True)

    print('end')
    df.to_csv('stat.csv')
