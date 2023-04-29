import random

import numpy as np


class LinearRegression:
    """
    :param X [x0, x1 ... xn] xi - [xi0, xi1 ... xik]
    :param Y [y0, y1 ... yn]
    :param batch set None for don't use batch
    """

    def __init__(self, X, Y, start_point, batch):
        self.X = X
        self.Y = Y
        self.start_point = start_point
        self.batch = batch
        self.math_iters = 0

    def get_error_in_point(self, b):
        return (self.X * b - self.Y).transpose() * (self.X * b - self.Y)

    def get_grad_in_point(self, b, use_batch=True):
        if use_batch and self.batch is not None:
            select = []
            while len(select) < self.batch:
                t = random.randint(0, len(self.X) - 1)
                if t not in select:
                    select.append(t)
                _X = self.X[select, :]
            _Y = self.Y[select, :]
        else:
            _X = self.X
            _Y = self.Y
        self.math_iters += 4
        return _X.transpose() * _X * b - _X.transpose() * _Y

    def grad_down(self, alpha=0.001, runs=1000):
        """ example from first lab """
        b = self.start_point.copy()
        for i in range(runs):
            b -= alpha * self.get_grad_in_point(b, use_batch=False)
        return b

    def stochastic_grad_down(self, alpha=0.001, runs=1000):
        """ 1. """
        b = self.start_point.copy()
        for i in range(runs):
            b -= alpha * self.get_grad_in_point(b)
        return b

    def stochastic_grad_down_with_math_iter(self, alpha=0.001, runs=1000, eps=0.0001):
        """ 1. """
        self.math_iters = 0
        b = self.start_point.copy()
        points = []
        points.append(b.copy())
        for i in range(runs):
            b -= alpha * self.get_grad_in_point(b)
            points.append(b.copy())
            self.math_iters += 2
            if self.get_error_in_point(b) < eps:
                break
        return points, self.math_iters

    def stochastic_grad_down_points(self, alpha=0.001, runs=1000, eps=0.0001):
        """ for git """
        self.math_iters = 0
        points = []
        b = self.start_point.copy()
        points.append(b.copy())
        for i in range(runs):
            b -= alpha * self.get_grad_in_point(b)
            points.append(b.copy())
            if self.get_error_in_point(b) < eps:
                break
        return points

    def stochastic_grad_down_exponential(self, alpha=0.001, c=0.01, runs=1000):
        """ 2. """
        b = self.start_point.copy()
        for i in range(runs):
            b -= alpha * self.get_grad_in_point(b, use_batch=False)
            alpha *= np.e ** c
        return b

    def momentum_stochastic_grad_down(self, y=0.9, alpha=0.001, runs=1000):
        """ 3. """
        u_p = np.zeros((len(self.start_point), 1))
        b = self.start_point.copy()
        for i in range(runs):
            u_p = y * u_p + alpha * self.get_grad_in_point(b)
            b = b - u_p
        return b

    def momentum_stochastic_grad_down_with_math_iters(self, y=0.8, alpha=0.001, runs=1000, eps=0.0001):
        """ 3. """
        self.math_iters = 0
        u_p = np.zeros((len(self.start_point), 1))
        b = self.start_point.copy()
        points = []
        points.append(b.copy())
        for i in range(runs):
            u_p = y * u_p + alpha * self.get_grad_in_point(b)
            b = b - u_p
            self.math_iters += 4
            points.append(b.copy())
            if self.get_error_in_point(b) < eps:
                break
        return points, self.math_iters

    def nesterov_stochastic_grad_down(self, y=0.9, alpha=0.001, runs=1000):
        """ 3. """
        u_p = np.zeros((len(self.start_point), 1))
        b = self.start_point.copy()
        for i in range(runs):
            u_p = y * u_p + alpha * self.get_grad_in_point(b - y * u_p)
            b = b - u_p
        return b

    def nesterov_stochastic_grad_down_with_math_iters(self, y=0.8, alpha=0.0001, runs=1000, eps=0.0001):
        """ 3. """
        self.math_iters = 0
        points = []
        u_p = np.zeros((len(self.start_point), 1))
        b = self.start_point.copy()
        points.append(b.copy())
        for i in range(runs):
            u_p = y * u_p + alpha * self.get_grad_in_point(b - y * u_p)
            b = b - u_p
            self.math_iters += 6
            points.append(b.copy())
            if self.get_error_in_point(b) < eps:
                break
        return points, self.math_iters

    def adagrad_stochastic_grad_down(self, alpha=0.7, runs=1000):
        """ 3. """
        G = np.zeros((len(self.start_point), 1))
        b = self.start_point.copy()
        for i in range(runs):
            l = self.get_grad_in_point(b)
            G += np.square(l)
            a = alpha / (np.power(G, 1 / 2) + 0.1 ** 8)
            a = np.diag(a.transpose()[0])
            b -= a * l
        return b

    def adagrad_stochastic_grad_down_with_math_iters(self, alpha=0.7, runs=1000, eps=0.0001):
        """ 3. """
        self.math_iters = 0
        points = []
        G = np.zeros((len(self.start_point), 1))
        b = self.start_point.copy()
        points.append(b.copy())
        for i in range(runs):
            l = self.get_grad_in_point(b)
            G += np.square(l)
            a = alpha / (np.power(G, 1 / 2) + 0.1 ** 8)
            a = np.diag(a.transpose()[0])
            b -= a * l
            self.math_iters += len(self.start_point) + 6
            points.append(b.copy())
            if self.get_error_in_point(b) < eps:
                break
        return points, self.math_iters

    def rms_stochastic_grad_down(self, W=4, alpha=0.7, runs=1000):
        """ 3. """
        last_g = []
        G = np.zeros((len(self.start_point), 1))
        b = self.start_point.copy()
        for i in range(runs):
            l = self.get_grad_in_point(b)
            last_g.append(l)
            if len(last_g) > W:
                _l = last_g.pop(0)
                G -= np.square(_l)
            G += np.square(l)
            g = alpha / (np.power(1 / W * G, 1 / 2) + 0.1 ** 8)
            g = np.diag(g.transpose()[0])
            b -= g * l
        return b

    def rms_stochastic_grad_down_with_math_iters(self, W=4, alpha=0.7, runs=1000, eps=0.0001):
        """ 3. """
        self.math_iters = 0
        points = []
        last_g = []
        G = np.zeros((len(self.start_point), 1))
        b = self.start_point.copy()
        points.append(b.copy())
        for i in range(runs):
            l = self.get_grad_in_point(b)
            last_g.append(l)
            if len(last_g) > W:
                _l = last_g.pop(0)
                G -= np.square(_l)
                self.math_iters += len(_l)
            G += np.square(l)
            g = alpha / (np.power(1 / W * G, 1 / 2) + 0.1 ** 8)
            g = np.diag(g.transpose()[0])
            b -= g * l
            points.append(b.copy())
            self.math_iters += len(l) + 8
            if self.get_error_in_point(b) < eps:
                break
        return points, self.math_iters

    def adam_stochastic_grad_down(self, b1=0.9, b2=0.9, alpha=0.01, runs=1000):
        """ 3. """
        m = 0
        u = 0
        b = self.start_point.copy()
        for i in range(runs):
            l = self.get_grad_in_point(b)
            m = b1 * m + (1 - b1) * l
            u = b2 * u + (1 - b2) * np.square(l)
            # _m = m / (1 - b1 ** (i + 1))
            # _u = u / (1 - b2 ** (i + 1))
            b -= alpha * m / (np.sqrt(u) + 0.1 ** 8)
        return b

    def adam_stochastic_grad_down_with_math_iters(self, b1=0.9, b2=0.9, alpha=0.01, runs=1000, eps=0.0001):
        """ 3. """
        self.math_iters = 0
        m = 0
        u = 0
        b = self.start_point.copy()
        points = []
        points.append(b.copy())
        for i in range(runs):
            l = self.get_grad_in_point(b)
            m = b1 * m + (1 - b1) * l
            u = b2 * u + (1 - b2) * np.square(l)
            # _m = m / (1 - b1 ** (i + 1))
            # _u = u / (1 - b2 ** (i + 1))
            b -= alpha * m / (np.sqrt(u) + 0.1 ** 8)
            self.math_iters += len(u) + len(l) + 13
            points.append(b.copy())
            if self.get_error_in_point(b) < eps:
                break
        return points, self.math_iters


def get_batch_name(count, batch):
    if batch == 1:
        return "SGD"
    if batch == count:
        return "GD"
    if count > batch > 1:
        return "Minibatch_GD"
    raise AssertionError("bad values")


def isError(lr, val, run):
    r = float(lr.get_error_in_point(val[-1]))
    if r > 0.0001 or len(val) > run:
        return None
    else:
        return r
