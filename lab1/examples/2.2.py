# conspects page 31
import random
from lab1.tools import Func
import sympy as sp
import math


# lambda_k=min_{\lambda} (\,f(x^{[k]}-\lambda\nabla f(x^{[k]}))
# findMin ищет лямбду на которую надо умножить градиент
def findMin():
    def calc_min_iterations():
        return sp.log((b - a - delta) / (2 * eps - delta), 2)

    a = 0
    b = 1
    eps = 0.001
    delta = 0.0015
    N = math.ceil(calc_min_iterations())
    x1 = (a + b - delta) / 2
    x2 = (a + b + delta) / 2
    for i in range(N):
        print(f"{x1 = } {x2 = } {f.eval([('x0', x1)]) = } {f.eval([('x0', x2)]) = }")
        # 1 step
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2

        # 2 step
        if f.eval([('x0', x1)]) <= f.eval([('x0', x2)]):
            b = x2
        else:
            a = x1

        # 3 step
        eps_i = (b - a) / 2
        if eps_i <= eps:
            break
    return (a + b) / 2


def to_args(t):
    return [(f"x{i}", t[i]) for i in range(n)]


n = 2
# todo fall
stringFunc = "x0^2 + 2*x1 + 1"
f = Func(2, stringFunc)
eps = 0.01
alpha = 0.1
x = sp.Matrix([[random.randint(0, 10) for _ in range(n)]])

while True:
    # ||∇f(x)|| < ε
    if f.metric_of_gradient_in_point(to_args(x)) < eps:
        break
    while True:
        y = x - alpha * f.grad(to_args(x))
        if f.eval(to_args(y)) < f.eval(to_args(x)):
            x = y
            break
        alpha = findMin()

    print(f"{x = }")
    print(f"{alpha = }")

print(x)
