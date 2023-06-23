import random

import numpy as np
from lab3.BFGS import bfgs, lbfgs
from lab1.tools import fast_generate_quadratic_func
from lab3.analizeBFGS.grad import const_grad_down, wolfe_grad_down

import pandas as pd

from lab3.analizeBFGS.Qfunc.Qfunc import qfunc

df = pd.DataFrame(columns=['n', 'name', 'was_broken', 'iters'])

b = 10


def func(A):
    f = lambda x: -np.sum(np.log(1 - A.dot(x))) - np.sum(np.log(1 - x*x))
    grad = lambda x: np.sum(A.T / (1 - A.dot(x)), axis=1) + 2 * x / (1 - np.power(x, 2))
    return f, grad

n = 3000
m = 100
x = np.zeros(n)
max_iter = 100
tol = 1e-5
for i in range(10):
    A = np.random.rand(m, n) * 10
    m = 10
    f, grad = func(A)

    try:
        bfgs_out = bfgs(f, grad, x)
        l_bfgs_out = lbfgs(f, grad, x, m=m)
        df = pd.concat([df, pd.DataFrame({'n': [n], 'name': ['bfgs'],
                                          'alpha': [None],
                                          'was_broken': [bfgs_out.was_broken],
                                          'iters': [len(bfgs_out.points)]})], ignore_index=True)
        df = pd.concat([df, pd.DataFrame({'n': [n], 'name': ['l_bfgs'],
                                          'alpha': [None],
                                          'was_broken': [l_bfgs_out.was_broken],
                                          'iters': [len(l_bfgs_out.points)]})], ignore_index=True)
        print(len(bfgs_out.points), bfgs_out.was_broken)
        print(len(l_bfgs_out.points), l_bfgs_out.was_broken)

    except:
        pass
df.to_csv('stat.csv')
