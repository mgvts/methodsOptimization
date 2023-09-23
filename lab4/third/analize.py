import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import rosen
import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.use('TkAgg')

x_bounds = (-2.0, 2.0)
y_bounds = (-1.0, 3.0)

def plot_results(data):
    plt.figure(figsize=(12, 5))

    x = np.arange(0, 12, 0.01)
    y = np.arange(-20, 20, 0.01)
    xgrid, ygrid = np.meshgrid(x, y)

    # z = 2 * (xgrid - 2) ** 2 + (ygrid - 1) ** 2
    # plt.contour(xgrid, ygrid, z, colors='black', linewidths=0.2, levels=100, alpha=0.6)

    title = ''

    for j in data['bound_points']:
        x = []
        y = []
        x.append(j[0])
        y.append(j[1])

    plt.plot(x, y, 'o-g', alpha=1, label='bound_points', lw=0.5, mec='g', mew=1, ms=1)

    for j in data['no_bound_points']:
        x = []
        y = []
        x.append(j[0])
        y.append(j[1])

    plt.plot(x, y, 'o-m', alpha=1, label='no_bound_points', lw=0.5, mec='m', mew=1, ms=1)

    plt.scatter(2, 1, 1, 'b')

    plt.title(f"Start point=(0, 0) bound_points {len(data['bound_points'])} no_bound_points {len(data['no_bound_points'])}")
    plt.legend()
    plt.show()

def minimize_with_linear_constraints():
    A = np.array([[1, 2], [-1, 1]])
    b = np.array([4, 1])
    linear_constraint = LinearConstraint(A, -np.inf, b)

    x0 = np.array([0.0, 0.0])

    path_storage_bounds = []
    res = minimize(rosen, x0, constraints=[linear_constraint], callback=lambda x: path_storage_bounds.append(x))

    path_storage_no_bounds = []
    res = minimize(rosen, x0, callback=lambda x: path_storage_no_bounds.append(x))

    data = {
        'bound_points' : path_storage_bounds,
        'no_bound_points' : path_storage_no_bounds
    }
    plot_results(data)


minimize_with_linear_constraints()