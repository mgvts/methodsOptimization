import random

import numpy as np
import pandas as pd

from lab3.analizeBFGS.grad import const_grad_down, wolfe_grad_down
from lab3.BFGS import bfgs


def strangeFunc():
    def f(x: np.array):
        n = len(x)
        if n != 2:
            raise AssertionError
        return np.sin((x[0] ** 2) / 2 - (x[1] ** 2) / 4 + 3) * np.cos(2 * x[0] + 1 - np.exp(x[1]))

    def grad(x: np.array):
        inside_sqw = (x[0]**2)/2 - (x[1]**2)/4 + 3
        exp_comp = 2*x[0] + 1 - np.exp(x[1])
        return np.array([-2*np.sin(exp_comp)*np.sin(inside_sqw) +
                            x[0]*np.cos(exp_comp)*np.cos(inside_sqw),
                        np.exp(x[1])*np.sin(exp_comp)*np.sin(inside_sqw) -
                         (x[1]/2)*np.cos(inside_sqw)*np.cos(exp_comp)])

    return f, grad


if __name__ == "__main__":
    df = pd.DataFrame(columns=['n', 'name', 'was_broken', 'iters', 'start', 'res'])


    eps = 1e-4
    maxIter = 1000
    for _ in range(60):
        n = 2
        print(f"{_=}")
        x = np.array([(random.random() - 0.5) * 20 for _ in range(n)])

        f, grad = strangeFunc()

        const_search_out = const_grad_down(f, grad, x, alpha=1e-4, eps=eps, max_iter=maxIter)
        df = pd.concat([df, pd.DataFrame({'n': [n], 'name': ['const_search_0.0001'],
                                          'was_broken': [const_search_out.was_broken],
                                          'iters': [len(const_search_out.points)],
                                          'start': [x],
                                          'res': [f(const_search_out.points[-1])]
                                          })], ignore_index=True)

        const_search_out = const_grad_down(f, grad, x, alpha=1e-2, eps=eps, max_iter=maxIter)
        df = pd.concat([df, pd.DataFrame({'n': [n], 'name': ['const_search_0.01'],
                                          'was_broken': [const_search_out.was_broken],
                                          'iters': [len(const_search_out.points)],
                                          'start': [x],
                                          'res': [f(const_search_out.points[-1])]
                                          })], ignore_index=True)

        wolfe_search_out = wolfe_grad_down(f, grad, x, eps=eps, max_iter=maxIter)
        df = pd.concat([df, pd.DataFrame({'n': [n], 'name': ['wolfe_search'],
                                          'was_broken': [wolfe_search_out.was_broken],
                                          'iters': [len(wolfe_search_out.points)],
                                          'start': [x],
                                          'res': [f(wolfe_search_out.points[-1])]
                                          })], ignore_index=True)

        bfgs_out = bfgs(f, grad, x, eps=eps, max_iter=maxIter)
        df = pd.concat([df, pd.DataFrame({'n': [n], 'name': ['bfgs'],
                                          'was_broken': [bfgs_out.was_broken],
                                          'iters': [len(bfgs_out.points)],
                                          'start': [x],
                                          'res': [f(bfgs_out.points[-1])]
                                          })], ignore_index=True)

    df.to_csv('stat.csv')
