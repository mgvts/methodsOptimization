import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from lab1.tools import Func

mpl.use('TkAgg')


def make_data():
    stringFunc = "10*x0^2 + x1 ^ 2"
    n = 2
    func = Func(n, stringFunc)
    x = np.arange(-100, 100, 0.5)
    y = np.arange(-100, 100, 0.5)
    xgrid, ygrid = np.meshgrid(x, y)

    # wow watafuck
    z = sp.lambdify(func.sp_variables, func.f, 'numpy')

    return xgrid, ygrid, z


if __name__ == '__main__':
    x, y, z = make_data()

    # !!!
    cs = plt.contour(x, y, z, levels=20)
    plt.clabel(cs)
    plt.show()
