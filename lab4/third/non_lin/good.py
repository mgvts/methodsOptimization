from pprint import pprint

import numpy as np
from scipy.optimize import rosen, NonlinearConstraint, minimize

from lab4.third.analize import plot_results


def minimize_with_nonlinear_constraints():
    def f(st):
        stor = st

        def ff(x):
            stor.append(x)
            return rosen(x)

        return ff

    def func(x):
        return np.sqrt(x[0] ** 2 + x[1] ** 2)

    nonlinear_constraint = NonlinearConstraint(func, lb=-np.inf, ub=10)

    x0 = np.array([20, 0])
    path_storage_bounds = [x0]
    print(minimize(rosen, x0, method="trust-constr", constraints=[nonlinear_constraint],
                   callback=lambda x, ans_i: path_storage_bounds.append(x)
                   ))

    pprint(path_storage_bounds)
    print()
    path_storage_no_bounds = [x0]
    print(minimize(rosen, x0, callback=lambda x: path_storage_no_bounds.append(x)))
    pprint(path_storage_no_bounds)

    data = {
        'bound_points': path_storage_bounds,
        'no_bound_points': path_storage_no_bounds
    }

    x_bonds = np.linspace(-10, 10, num=100)
    y_bounds = [np.sqrt(100 - i * i ) for i in x_bonds]
    bounds = (list(x_bonds) + list(x_bonds), y_bounds + [-i for i in y_bounds])
    plot_results(data, x0, bounds)


minimize_with_nonlinear_constraints()
