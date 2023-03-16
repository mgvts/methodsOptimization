from random import randint

from lab1.grad import grad_down_dichotomy, dichotomy
from lab1.tools import Func, to_args
import sympy as sp

n = 2
stringFunc = "x0^2 + x1^2 - 10"

start_point = sp.Matrix([[randint(-10, 10) for i in range(n)]])
print(x := grad_down_dichotomy(n, stringFunc, start_point))
print('Найденная точка:', x.points[-1])

n = 1
stringFunc = "x0^2 + 2*x0 + 1"
print(dichotomy(lambda x: Func(n, stringFunc).f.subs("x0", x), a=-10000, b=10000))
