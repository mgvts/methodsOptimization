import sympy as sp

from lab1.grad import grad_down, grad_down_dichotomy

n = 2
c = 1
stringFunc = f"{c}*x0^2 + x1^2"

level = 10000

step = 12
x_min = -(level / c) ** 0.5
x_max = (level / c) ** 0.5

a = x_max
b = level ** 0.5

print(x_min, x_max)
meth_g = []
meth_d = []

for i in range(1, int((x_max - x_min) / step)):
    x0 = x_min + step * i
    x11 = (level - c * x0 ** 2) ** 0.5
    x12 = -((level - c * x0 ** 2) ** 0.5)

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

# RESULTS :
# circle
# c = 1, step = 0.1, level = 40 | {3}, {389}
# c = 1, step = 2, level = 100 | {3}, {412}
# c = 1, step = 4, level = 900 | {3}, {466}
# c = 1, step = 8, level = 1900 | {3}, {485}
# c = 1, step = 8, level = 1900 | {3}, {485}
# c = 1, step = 12, level = 10000 | {4}, {526}
