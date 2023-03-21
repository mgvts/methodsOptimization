from random import randint

import sympy as sp

from lab1.grad import grad_down, grad_down_dichotomy
from lab1.tools import Func, to_args

n = 2
stringFunc = "2 * x0^2 + (x1-3)^2 + 2*x0 -3*x1 - 10"

# start_point = sp.Matrix([[randint(-10, 10) for i in range(n)]])
start_point = sp.Matrix([[10, 10]])
# max_inter прервёт выполнение алгоритма, после стольких итераций. was_broken -> True

x1 = grad_down(n, stringFunc, start_point, alpha=0.02, eps=0.0000001, max_inter=100000)
# print(x2 := grad_down_dichotomy(n, stringFunc, start_point))

print('Кол-во итераций/Точки grad_down', len(x1.points))
print('Сошелся: ', not x1.was_broken)

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

