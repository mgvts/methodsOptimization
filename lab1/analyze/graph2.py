import matplotlib as mpl
from sympy import plot_implicit, symbols
from sympy.plotting import PlotGrid

from lab1.tools import Func

mpl.use('TkAgg')

stringFunc = "2*x0^2 + (x1-3)^2 + 2*x0 - 3*x1 - 10"
n = 2
func = Func(n, stringFunc)
f = func.f

x_range = [-10, 10]
y_range = [-10, 10]
step = 0.5

x0, x1 = symbols('x0 x1')

# init
p = plot_implicit(f, x_var=(x0, *x_range), y_var=(x1, *y_range), show=False, line_color='red')


# grad field
p.rectangles = []
x_len = x_range[1] - x_range[0]
y_len = y_range[1] - y_range[0]
for i in range(0, round(x_len / step)):
    for j in range(0, round(y_len / step)):
        _x = (x_range[0] + i * step, x_range[0] + (i + 1) * step)
        _y = (y_range[0] + j * step, y_range[0] + (j + 1) * step)
        p.rectangles.append(
            {
                'xy': (_x[0], _y[0]),
                'width': step,
                'height': step,
                'color': '#43ff642b',
                'ec': None
            }
        )

# p.extend(p_graph)

PlotGrid(1, 2, p, p)
