import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')


x = sp.symbols("x")
string_func = "x^2"
f = sp.sympify(string_func)
xpoints = np.arange(-10, 10)
f_np = sp.lambdify(x, f, "numpy")
ypoints = f_np(xpoints)
ypoints = np.array([f.subs(x, i) for i in xpoints])

plt.plot(xpoints, ypoints)
plt.show()
