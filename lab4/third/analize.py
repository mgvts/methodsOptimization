import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.optimize import rosen
import matplotlib.pyplot as plt
from pprint import pprint

# mpl.use('TkAgg')

x_bounds = (-2.0, 2.0)
y_bounds = (-1.0, 3.0)


def plot_results(data, x0, bounds=None):
    plt.figure(figsize=(12, 5))

    x = np.arange(-15, 30, 0.1)
    y = np.arange(-15, 30, 0.1)
    xgrid, ygrid = np.meshgrid(x, y)

    title = ''
    x = []
    y = []

    for j in data['bound_points']:
        x.append(j[0])
        y.append(j[1])

    plt.plot(x, y, 'o-g', alpha=1, label='bound_points', lw=0.5, mec='g', mew=1, ms=1)
    x = []
    y = []
    for j in data['no_bound_points']:
        x.append(j[0])
        y.append(j[1])

    plt.plot(x, y, 'o-m', alpha=1, label='no_bound_points', lw=0.5, mec='m', mew=1, ms=1)

    if bounds:
        lin_bnds_x, lin_bnds_y = bounds
        print(len(lin_bnds_x))
        print(len(lin_bnds_y))

        plt.fill(lin_bnds_x, lin_bnds_y, linewidth=1, alpha=0.2)

    plt.xlim(-24, 24)
    plt.ylim(-10, 10)

    plt.title(
        f"Start point=({x0[0]}, {x0[1]}) bound_points {len(data['bound_points'])} no_bound_points {len(data['no_bound_points'])}")
    plt.legend()
    plt.show()


