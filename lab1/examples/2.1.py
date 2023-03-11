# одномерный поиск
# conspects page 9-10
from lab1.tools import Func
import sympy as sp
import math


def calc_min_iterations():
    return sp.log((b - a - delta) / (2 * eps - delta), 2)


# a b esp delta то параметры
# тут просто одноммерный поиск (тернарный поиск)
stringFunc = "x0^2 + 2*x0^2 + 1"
f = Func(1, stringFunc)
a = -10
b = 10
eps = 0.001
# must be from (0, 2eps)
delta = 0.0015
N = math.ceil(calc_min_iterations())
print(f"{N =}")

x1 = (a + b - delta) / 2
x2 = (a + b + delta) / 2
t1 = (b - x1) / (b - a)
t2 = (x2 - a) / (b - a)
# t1 and t2 must be lim = 1/2
print(f"{t1 = } {t2 = }")
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

# 4 step
X = (a + b) / 2
print(X)
# X is root of this function
