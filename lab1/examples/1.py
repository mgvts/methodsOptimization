from random import randint

from lab1.grad import grad_down_between_difference, grad_down
from lab1.tools import Func, to_args
import sympy as sp

n = 2
stringFunc = "x0^2 + x1^2 - 10"

start_point = sp.Matrix([[randint(-10, 10) for i in range(n)]])
print(x := grad_down_between_difference(n, stringFunc, start_point, alpha=0.001))
# print(x := grad_down(n, stringFunc, start_point, alpha=0.001).points[-1])
print(Func(n, stringFunc).eval(to_args(x, n)))
