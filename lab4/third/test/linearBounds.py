import numpy as np
import scipy
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import rosen, rosen_der
from matplotlib import pyplot as plt

def f(x):
    print(x)
    return rosen(x)

x0 = np.array([ 10,  -10])
# min -> 0 at (1 for i in range(len(x)))

A = np.array([[1, 1], [1, 1], [1, -1], [1,-1]])
ub = np.array([10, np.inf, 10, np.inf])
lb = np.array([-np.inf, -10, -np.inf, -10])

bounds = LinearConstraint(A=A, lb=lb, ub=ub)

res = minimize(f, x0, tol=1e-6, constraints=bounds, options={'disp': True})
print(res)
