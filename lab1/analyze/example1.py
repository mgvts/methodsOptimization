from random import randint

import sympy as sp

from lab1.grad import grad_down, grad_down_dichotomy
from lab1.tools import Func, to_args

n = 2
stringFunc = "x0^2 + x1^2 - 10"

start_point = sp.Matrix([[randint(-10, 10) for i in range(n)]])
# max_inter прервёт выполнение алгоритма, после стольких итераций. was_broken -> True

print(x1 := grad_down(n, stringFunc, start_point, alpha=0.01, max_inter=20))
print(x2 := grad_down_dichotomy(n, stringFunc, start_point))

print('Точки grad_down', len(x1.points), x1.points)
print('Точки grad_down_dichotomy', len(x2.points), x2.points)
print('Альфа grad_down_dichotomy', len(x2.alpha), x2.alpha)


print('Найденная точка grad_down', x1.points[-1])
print('Найденная точка grad_down_dichotomy', x2.points[-1])

print()
print('Данные от запуска grad_down')
print('Eps: ', x1.eps)
print('Metrics: ', x1.metrics)
print('String Func: ', x1.string_func)
print('String N: ', x1.n)

