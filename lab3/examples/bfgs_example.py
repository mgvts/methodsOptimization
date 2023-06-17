import numpy as np

from lab3.BFGS import bfgs


f = lambda x: sum([i ** 2 for i in x])
grad_f = lambda x: np.array([2 * i for i in x])

n = 3
x0 = np.array([1 for i in range(n)]).T
print(bfgs(f, grad_f, x0))
