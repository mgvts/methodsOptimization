from random import uniform

import numpy as np

from lab1.fast_grad import grad_down_dichotomy, grad_down
from lab1.tools import FastQFunc

f = FastQFunc(2, np.matrix([
    [4, 0],
    [0, 2]
]), np.matrix([[2, -9]]), 0)

f = FastQFunc(2, np.matrix([
    [20, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)

ALPH = 0.0
EPS = 0.0001

x = grad_down(f, [10, 10], alpha=ALPH, eps=EPS, max_inter=100000)
# x = grad_down_dichotomy(f, [10, 10], max_inter=100000)

print('Не сошелся!!!!!!!!' if x.was_broken else 'сошелся')
print(len(x.points))
print(x.points[-1])
