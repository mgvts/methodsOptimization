from random import randint

import sympy as sp

from lab1.grad import grad_down, grad_down_dichotomy
from lab1.tools import Func, to_args

n = 2
stringFunc = "x0^2 + x1^2 + 5*x0"
# stringFunc = "x0^2 + x1^2"
# stringFunc = "10*x0^2 + x1^2"


start_point = sp.Matrix([[10, 10]])

x1 = grad_down(n, stringFunc, start_point, alpha=0.02, eps=0.0000001, max_inter=100000)
x2 = grad_down_dichotomy(n, stringFunc, start_point)

print(x2.dichotomy_count, x2.points)

print('Кол-во итераций/Точки grad_down', len(x2.points))
print('Сошелся: ', not x2.was_broken)
print('Кол-во: ', sum(x2.dichotomy_count) + len(x2.points) * 3)

# print('Кол-во итераций/Точки grad_down_dichotomy', len(x2.points), x2.points)
# print('Альфа grad_down_dichotomy', len(x2.alpha), x2.alpha)


print('Найденная точка grad_down', x1.points[-1])
# print('Найденная точка grad_down_dichotomy', x2.points[-1])

print()
print('Данные от запуска grad_down')
print('Eps: ', x1.eps)
print('Metrics: ', x1.metrics[-1])

# print('String Func: ', x1.string_func)
# print('String N: ', x1.n)

