from random import randint

import matplotlib as mpl
import numpy as np
import sympy as sp
from sympy import plot_implicit, symbols
from sympy.plotting.plot import MatplotlibBackend, Plot

from lab1.grad import grad_down_dichotomy, grad_down, grad_down_wolfe
from lab1.tools import Func, get_metric2

mpl.use('TkAgg')


def analize(start_point, out_from_const, out_from_dichotomy):
    print("start_point", end="\t")
    print(start_point)
    print("grad_down")
    print(out_from_const)
    print("grad_down_dichotomy")
    print(out_from_dichotomy)
    print(f"last points: статичный{out_from_const.points[-1].values()} "
          f"дихотомия{out_from_dichotomy.points[-1].values()} "
          f"equals: {out_from_const.points[-1].values() == out_from_dichotomy.points[-1].values()}", sep="\n")
    print()
    print("разница в значениях на итерациях")
    print("i       static              dichotomy")
    for i in range(len(out_from_const.points)):
        print(f"{i = }")
        print(f"точки {out_from_const.points[i].values()} {out_from_dichotomy.points[i].values()}")
        print(f"alpha {out_from_const.alpha[i]} {out_from_dichotomy.alpha[i]}")
        print(f"метрика 2 от стартовой до данной точки")
        print(get_metric2(start_point - out_from_const.points[i]), get_metric2(start_point - out_from_dichotomy.points[i]))
        print(f"метрика 2 от коннечной до данной точки")
        print(get_metric2(out_from_const.points[-1] - out_from_const.points[i]), get_metric2(out_from_dichotomy.points[-1] - out_from_dichotomy.points[i]))
        print()
        print()


def get_sympy_subplots(plot: Plot):
    backend = MatplotlibBackend(plot)
    backend.process_series()
    backend.plt.tight_layout()
    return backend.plt


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size
                       )


# stringFunc = "2*x0^2 + (x1-3)^2 + 2*x0 - 3*x1 - 10"
# start_point = sp.Matrix([[-10 for i in range(n)]])

stringFunc = "2*x0^2 + (x1-3)^2 + 2*x0 - 3*x1"
start_point = sp.Matrix([[10, 10]])
start_points = [
    (10, 10, 'green'),
                (-10, -10, 'red'),
                (10, -10, 'blue'),
                (-10, 10, 'yellow'),
                (10, 0, 'pink'),
                (-10, 0, 'gray'),
                (0, 10, 'orange'),
                (0, -10, 'black')
]


n = 2
func = Func(n, stringFunc)
f = func.f
func2 = Func(n, stringFunc)
f2 = func2.f
x, y = symbols('x0, x1')
func3 = Func(n,stringFunc)
f3 = func3.f


x_range = [-20, 20]
y_range = [-20, 20]
def get_p():
    p = plot_implicit(f, show=False, points=300, x_var=(x, *x_range), y_var=(y, *y_range))
    p.extend(plot_implicit(Func(n, "2*x0^2 + (x1-3)^2 + 2*x0 - 3*x1 - 10").f, show=False, points=300, x_var=(x, *x_range), y_var=(y, *y_range)))
    p.extend(plot_implicit(Func(n, "2*x0^2 + (x1-3)^2 + 2*x0 - 3*x1 - 20").f, show=False, points=300, x_var=(x, *x_range), y_var=(y, *y_range)))
    p.extend(plot_implicit(Func(n, "2*x0^2 + (x1-3)^2 + 2*x0 - 3*x1 - 50").f, show=False, points=300, x_var=(x, *x_range), y_var=(y, *y_range)))
    p.extend(plot_implicit(Func(n, "2*x0^2 + (x1-3)^2 + 2*x0 - 3*x1 - 100").f, show=False, points=300, x_var=(x, *x_range), y_var=(y, *y_range)))
    p.extend(plot_implicit(Func(n, "2*x0^2 + (x1-3)^2 + 2*x0 - 3*x1 - 200").f, show=False, points=300, x_var=(x, *x_range), y_var=(y, *y_range)))
    p.extend(plot_implicit(Func(n, "2*x0^2 + (x1-3)^2 + 2*x0 - 3*x1 - 300").f, show=False, points=300, x_var=(x, *x_range), y_var=(y, *y_range)))
    # p.extend(plot_implicit(Func(n, "x0^2 + x1 ^ 2 - 500").f, show=False, points=300, x_var=(x, *x_range), y_var=(y, *y_range)))
    # p.extend(plot_implicit(Func(n, "x0^2 + x1 ^ 2 - 1000").f, show=False, points=300, x_var=(x, *x_range), y_var=(y, *y_range)))

    return p

plt1 = get_sympy_subplots(get_p())

for p in start_points:
    start_point = sp.Matrix([[p[0], p[1]]])
    x2 = grad_down_dichotomy(n, stringFunc, start_point, max_inter=100)
    out_from_const = x2
    for i in range(1, len(x2.points)):
        x1, y1 = [x2.points[i - 1][0], x2.points[i][0]], [x2.points[i - 1][1], x2.points[i][1]]
        line = plt1.gca().plot(x1, y1, marker='o', ms=1, color=p[2])
        try:
            add_arrow(line[0], size=12)
        except IndexError:
            # small dist -> not line, point
            pass

plt1.title('Dichotomy, Func: 2*x0^2 + (x1-3)^2 + 2*x0 - 3*x1')

plt2 = get_sympy_subplots(get_p())
for p in start_points:
    start_point = sp.Matrix([[p[0], p[1]]])
    x2 = grad_down(n, stringFunc, start_point, max_inter=1000, alpha=0.05)
    out_from_dichotomy = x2
    for i in range(1, len(x2.points)):
        x1, y1 = [x2.points[i - 1][0], x2.points[i][0]], [x2.points[i - 1][1], x2.points[i][1]]
        line2 = plt2.gca().plot(x1, y1, marker='o', ms=1, color=p[2])
        try:
            add_arrow(line2[0], size=12)
        except IndexError:
            # small dist -> not line, point
            pass

plt2.title('Const, Func: 2*x0^2 + (x1-3)^2 + 2*x0 - 3*x1, alpha: 0.05')

plt3 = get_sympy_subplots(get_p())

for p in start_points:
    start_point = sp.Matrix([[p[0], p[1]]])
    x3 = grad_down_wolfe(n, stringFunc, start_point, max_inter=100)
    out_from_const = x3
    for i in range(1, len(x3.points)):
        x1, y1 = [x3.points[i - 1][0], x3.points[i][0]], [x3.points[i - 1][1], x3.points[i][1]]
        line3 = plt3.gca().plot(x1, y1, marker='o', ms=1, color=p[2])
        try:
            add_arrow(line3[0], size=12)
        except IndexError:
            # small dist -> not line, point
            pass

plt3.title('Wolfe, Func: 2*x0^2 + (x1-3)^2 + 2*x0 - 3*x1')

plt1.show()
# plt2.show()
# plt3.show()
