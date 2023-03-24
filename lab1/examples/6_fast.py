import numpy as np

from lab1.tools import fast_generate_quadratic_func, FastQFunc

from lab1.fast_grad import grad_down, grad_down_dichotomy, grad_down_wolfe

# f = fast_generate_quadratic_func(2, 10)
f = FastQFunc(2, np.matrix([[1, 0],
                            [0, 1]]), np.matrix([2, 0]), 0)
x = grad_down_dichotomy(f, [10.0, 10.0])

print(x.points[-1])
