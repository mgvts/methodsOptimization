from random import randint

from lab1.grad import grad_down_dichotomy
from lab1.tools import Func, to_args
import sympy as sp

n = 2
stringFunc = "x0^2 + x1^2 - 10"

# если x = Matrix([[-10, 0]]) , то бесконечный цикл
start_point = sp.Matrix([[randint(-10, 10) for i in range(n)]])
