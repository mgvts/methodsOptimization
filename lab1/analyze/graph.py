from random import randint

import sympy as sp
from sympy import plot_implicit, symbols
from sympy.plotting.plot import MatplotlibBackend, Plot

from lab1.grad import grad_down_dichotomy
from lab1.tools import Func
import matplotlib as mpl
mpl.use('TkAgg')

def get_sympy_subplots(plot: Plot):
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.plt


stringFunc = "x0^2 + x1 - 2"
f = Func(2, stringFunc).f
x, y = symbols('x, y')
# todo range
p = plot_implicit(f, show=False, points=10)

plt = get_sympy_subplots(p)

n = 2
stringFunc = "x0^2 + x1^2 - 10"

start_point = sp.Matrix([[randint(-4, 4) for i in range(n)]])

x2 = grad_down_dichotomy(n, stringFunc, start_point)
print(x2.points[-1])
for i in range(1, len(x2.points)):
    x1, y1 = [x2.points[i - 1][0], x2.points[i - 1][1]], [x2.points[i][0], x2.points[i][1]]
    plt.plot(x1, y1, marker='o', ms=4)

plt.show()
