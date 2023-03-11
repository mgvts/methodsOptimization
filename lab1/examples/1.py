import random

from lab1.grad import grad_down_metric_between_difference, grad_down_metric
from lab1.tools import Func, to_args
from time import sleep
import sympy as sp

n = 2
stringFunc = "x0^2 + x1^2 - 10"

# если x = Matrix([[-10, 0]]) , то бесконечный цикл
print(x := grad_down_metric_between_difference(n, stringFunc))
print(Func(n, stringFunc).eval(to_args(x, n)))
print(x := grad_down_metric(n, stringFunc))
print(Func(n, stringFunc).eval(to_args(x, n)))
# -9.99937549173455
# Matrix([[-1, 2]])
# Matrix([[-2.23493731242445e-5, 4.46987462484890e-5]])
# -9.99999999750253
