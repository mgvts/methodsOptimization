import random

from lab1.tools import Func
from time import sleep
import sympy as sp

n = 2
stringFunc = "x0^2 + x1^2 + 10"
f = Func(2, stringFunc)
eps = 0.01
alpha = 0.1
# именно столько скобок [[]]
x = sp.Matrix([[random.randint(0, 10) for i in range(n)]])
print(x)


def to_args(t):
    return [(f"x{i}", t[i]) for i in range(n)]


while True:
    # ||∇f(x)|| < ε
    if f.metric_of_gradient_in_point(to_args(x)) < eps:
        break

    y = x - alpha * f.grad(to_args(x))
    if f.eval(to_args(y)) < f.eval(to_args(x)):
        x = y

        # alpha = const in 1st task
        # alpha = alpha / 2
print(x)
print(f.eval(to_args(x)))
