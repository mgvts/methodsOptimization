import sympy as sp
from random import randint
from lab1.grad import grad_down_wolfe
from lab1.tools import Func, to_args

n = 2
stringFunc = "x0^2 + x1^2 - 10"


start_point = sp.Matrix([[randint(-10, 10) for i in range(n)]])
print(x := grad_down_wolfe(n, stringFunc, start_point).points[-1])
print(Func(n, stringFunc).eval(to_args(x, n)))
