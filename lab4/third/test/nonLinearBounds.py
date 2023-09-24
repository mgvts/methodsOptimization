import numpy as np
import scipy
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from scipy.optimize import rosen, rosen_der
from matplotlib import pyplot as plt

from lab2.examples.linear_regression import twod_line
from lab2.linear_regression import LinearRegression
from lab4.linear.pytorch_linear_regression import PyTorchLinearRegression

count = 100
X, Y = twod_line.generate_linear_regression_2d(count, shift=(2, 2))
b = np.matrix([100., 100.])

# regression = LinearRegression(X, Y, b, None)
# torch_regression = PyTorchLinearRegression(X, Y, b, count / 2)

# reg = regression.nesterov_stochastic_grad_down_points(y=0.9, alpha=0.0001, runs=1000)
# torch = torch_regression.nesterov_stochastic_grad_down_points(y=0.9, alpha=0.0001, runs=1000)
f = lambda x: sum([i ** 2 for i in x])
x0 = np.array([ 10,  -10])
# min -> 0 at (1 for i in range(len(x)))

f_bounds = lambda x: x[1] **2 +x[2]**2

ub = np.array([10, np.inf, 10, np.inf])
lb = np.array([-np.inf, -10, -np.inf, -10])

bounds = NonlinearConstraint(f_bounds, lb=lb, ub=ub)

res = minimize(rosen, x0, tol=1e-6, constraints=bounds)
print(res)
