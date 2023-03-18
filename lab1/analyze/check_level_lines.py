import sympy as sp

from lab1.grad import grad_down, grad_down_dichotomy

n = 2
stringFunc = "10*x0^2 + x1^2"

level = 30

step = 0.1
x_min = -(level / 10) ** 0.5
x_max = (level / 10) ** 0.5

a = x_max
b = level ** 0.5

print(x_min, x_max)
meth_g = []
meth_d = []

for i in range(0, int((x_max - x_min) / step)):
    x0 = x_min + step * i
    x11 = (level - 10 * x0 ** 2) ** 0.5
    x12 = -((level - 10 * x0 ** 2) ** 0.5)

    # x**2/a**2 + y**2/b**2 = 1 свойство эллипса
    print(x0, x11, x0 ** 2 / a ** 2 + x11 ** 2 / b ** 2)
    print(x0, x12, x0 ** 2 / a ** 2 + x12 ** 2 / b ** 2)

    start_point = sp.Matrix([[x0, x11]])
    meth_g.append(len(grad_down(n, stringFunc, start_point, alpha=0.01, max_inter=1000).points))
    meth_d.append(len(grad_down_dichotomy(n, stringFunc, start_point, max_inter=1000).points))

    start_point = sp.Matrix([[x0, x12]])
    meth_g.append(len(grad_down(n, stringFunc, start_point, alpha=0.01, max_inter=1000).points))
    meth_d.append(len(grad_down_dichotomy(n, stringFunc, start_point, max_inter=1000).points))

    print(set(meth_d), set(meth_g))

# print(x1 := grad_down(n, stringFunc, start_point, alpha=0.01, max_inter=1000))
# print(x2 := grad_down_dichotomy(n, stringFunc, start_point))
#
# print('Кол-во итераций/Точки grad_down', len(x1.points), x1.points)
# print('Кол-во итераций/Точки grad_down_dichotomy', len(x2.points), x2.points)
# print('Альфа grad_down_dichotomy', len(x2.alpha), x2.alpha)
#
#
# print('Найденная точка grad_down', x1.points[-1])
# print('Найденная точка grad_down_dichotomy', x2.points[-1])
#
# print()
# print('Данные от запуска grad_down')
# print('Eps: ', x1.eps)
# print('Metrics: ', x1.metrics)
# print('String Func: ', x1.string_func)
# print('String N: ', x1.n)
