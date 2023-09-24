from pprint import pprint

import numpy as np
from scipy.optimize import rosen, LinearConstraint, minimize

from lab4.third.analize import plot_results


def minimize_with_linear_constraints():
    def f(st):
        stor = st
        def ff(x):
            stor.append(x)
            return rosen(x)

        return ff


    A = np.array([[1, 1], [-1, 11], [1, -1], [1, 9]])
    lb = np.array([-np.inf, 10, -np.inf, 10])
    ub = np.array([10, np.inf, 10, np.inf])
    linear_constraint = LinearConstraint(A, lb=lb, ub=ub)

    x0 = np.array([0, 7.5])

    path_storage_bounds = []
    print(minimize(f(path_storage_bounds), x0, constraints=[linear_constraint]))

    path_storage_no_bounds = []
    print(minimize(f(path_storage_no_bounds), x0))

    data = {
        'bound_points': path_storage_bounds,
        'no_bound_points': path_storage_no_bounds
    }

    bounds = [[-8, 0, 8.3, 1], [2, 10, 1.6, 1]]
    plot_results(data, x0, bounds)


minimize_with_linear_constraints()