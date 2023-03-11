import random

from lab1.grad import grad_down_metric
from lab1.tools import Func, to_args
from time import sleep
import sympy as sp

n = 2
stringFunc = "x0^2 * x1^2 - 10"

# если x = Matrix([[-10, 0]]) , то бесконечный цикл
print(x := grad_down_metric(n, stringFunc))
print(Func(n, stringFunc).eval(to_args(x, n)))
