import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from lab3.BFGS import bfgs
from lab3.analizeBFGS.grad import const_grad_down, wolfe_grad_down
from sincos import strangeFunc
mpl.use('TkAgg')

def make_data():
    f, grad = strangeFunc()

    x = np.arange(-1, 1, 0.05)
    y = np.arange(-1, 1, 0.05)
    xgrid, ygrid = np.meshgrid(x, y)

    z = f([xgrid, ygrid])
    return xgrid, ygrid, z


if __name__ == '__main__':
    x, y, z = make_data()

    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')

    axes.plot_surface(x, y, z, edgecolor='k', linewidth=0.3)
    plt.show()

