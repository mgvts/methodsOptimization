import random

from lab1.grad import grad_down
from lab1.tools import Func
from time import sleep
import sympy as sp

n = 2
stringFunc = "x0^2 * x1^2 - 10"

# если x = Matrix([[-10, 0]]) , то бесконечный цикл
print(grad_down(n, stringFunc))
