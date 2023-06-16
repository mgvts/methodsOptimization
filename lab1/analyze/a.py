import threading
import time

import matplotlib as mpl
from sympy import plot_implicit, symbols, Matrix

from lab1.grad import grad_down, grad_down_dichotomy
from lab1.tools import Func

"""
Желтый цвет - сошлось с максимальным значением
Красный цвет - сошлось с минимальным значением

Синий цвет - не сошлось

"""


max_inter = 1000
stringFunc = "10*x0^2 + (x1)^2"
n = 2
func = Func(n, stringFunc)
f = func.f

x_range = [-20, 20]
y_range = [-20, 20]
step = 5

x0, x1 = symbols('x0 x1')

# init
p = plot_implicit(f, x_var=(x0, *x_range), y_var=(x1, *y_range), show=False, line_color='red')


def rgba_to_hex(rgb):
    return '%02x%02x%02x' % rgb


# grad field
p.rectangles = []
x_len = x_range[1] - x_range[0]
y_len = y_range[1] - y_range[0]

# find min/max
mm = 9999999
mx = 0

matr = []
for i in range(round(x_len / step)):
    matr.append([0] * round(y_len / step))

print('PROCESS start points')

count = 0
final_count = round(x_len / step) * round(y_len / step)
for i in range(0, round(x_len / step)):
    for j in range(0, round(y_len / step)):
        _x = (x_range[0] + i * step, x_range[0] + (i + 1) * step)
        _y = (y_range[0] + j * step, y_range[0] + (j + 1) * step)
        grad = grad_down(2, stringFunc, Matrix([[_x[0] + step / 2, _y[0] + step / 2]]), alpha=0.01, max_inter=max_inter)
        matr[i][j] = len(grad.points)
        count += 1
        print(f'{count} / {final_count} {len(grad.points)} {grad.points[-1]}')


mm = 9999999
mx = 0
for i in range(0, round(x_len / step)):
    for j in range(0, round(y_len / step)):
        mm = min(mm, matr[i][j])
        mx = max(mx, matr[i][j])

if mx == mm:
    mm = 0

for i in range(0, round(x_len / step)):
    for j in range(0, round(y_len / step)):
        _x = (x_range[0] + i * step, x_range[0] + (i + 1) * step)
        _y = (y_range[0] + j * step, y_range[0] + (j + 1) * step)

        z = int(255 * (matr[i][j] - mm) / (mx - mm))
        c = rgba_to_hex((255, z, 0)) + '82'
        if matr[i][j] > max_inter:
            c = '0000ff82'

        p.rectangles.append(
            {
                'xy': (_x[0], _y[0]),
                'width': step,
                'height': step,
                'color': '#' + c,
                'ec': None
            }
        )

# p.extend(p_graph)

# PlotGrid(1, 2, p)
p.show()
