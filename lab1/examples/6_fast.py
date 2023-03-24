import numpy as np

from lab1.tools import fast_generate_quadratic_func, FastQFunc

from lab1.fast_grad import grad_down, grad_down_dichotomy, grad_down_wolfe

from random import uniform, randint

# f = fast_generate_quadratic_func(2, 10)
k = 100
f = fast_generate_quadratic_func(2, k)
x = grad_down_dichotomy(f, [uniform(-10, 10) for i in range(2)])

print(len(x.points), x.was_broken)
print(x.points[-1])