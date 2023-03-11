# conspects page 31
import math
import random

import sympy as sp

from lab1.grad import grad_down_metric
from lab1.tools import Func, to_args


# lambda_k=min_{\lambda} (\,f(x^{[k]}-\lambda\nabla f(x^{[k]}))
# findMin ищет лямбду на которую надо умножить градиент
def dichotomy(f, eps=0.001, delta=0.00015):
    def calc_min_iterations():
        return sp.log((b - a - delta) / (2 * eps - delta), 2)

    a = 0
    b = 1
    N = math.ceil(calc_min_iterations())
    x1 = (a + b - delta) / 2
    x2 = (a + b + delta) / 2
    for i in range(N):
        # print(f"{x1 = } {x2 = } {f.eval([('x0', x1)]) = } {f.eval([('x0', x2)]) = }")
        # 1 step
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2

        # 2 step
        if f(x1) <= f(x2):
            b = x2
        else:
            a = x1

        # 3 step
        eps_i = (b - a) / 2
        if eps_i <= eps:
            break
    return (a + b) / 2


n = 2
string_func = "x0^2 + x1^2 - 10"
f = Func(n, string_func)
x = sp.Matrix([[random.randint(-10, 10) for i in range(n)]])
x = sp.Matrix([[-10, 0]])
a = dichotomy(lambda a: f.eval(to_args(x - a * f.grad(to_args(x, n)), n)))
print(a)
# eps = 0.0001
# alpha = 0.001
# x = sp.Matrix([[random.randint(0, 10) for _ in range(n)]])
#
# while True:
#     # ||∇f(x)|| < ε
#     if f.metric_of_gradient_in_point(to_args(x)) < eps:
#         break
#     while True:
#         y = x - alpha * f.grad(to_args(x))
#         if f.eval(to_args(y)) < f.eval(to_args(x)):
#             x = y
#             break
#         alpha = findMin()
#
#     print(f"{x = }")
#     print(f"{alpha = }")
#
# print(x)



#
# # если x = Matrix([[-10, 0]]) , то бесконечный цикл
# start_point = sp.Matrix([[random.randint(-10, 10) for i in range(n)]])
# print(x := grad_down_metric(n, stringFunc,
#                             start_point, findMin))
# print(Func(n, stringFunc).eval(to_args(x, n)))
