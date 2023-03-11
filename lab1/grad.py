import random

import sympy as sp

from lab1.tools import *

eps = 0.01
alpha = 0.001


def grad_down_metric(n, string_func):
    f = Func(2, string_func)
    # именно столько скобок [[]]
    # todo move x in args
    x = sp.Matrix([[random.randint(-10, 10) for i in range(n)]])
    print(x)
    # x = sp.Matrix([[-10, 0]])

    def to_args(t):
        return [(f"x{i}", t[i]) for i in range(n)]

    while True:
        # ||∇f(x)|| < ε
        if f.metric_of_gradient_in_point(to_args(x)) < eps:
            break

        y = x - alpha * f.grad(to_args(x))
        if f.eval(to_args(y)) < f.eval(to_args(x)):
            x = y
            # alpha = const in 1st task
            # alpha = alpha / 2
        # print(f.metric_of_gradient_in_point(to_args(x)), x, y)
    return x

def grad_down_metric_between_difference(n, string_func):
    f = Func(2, string_func)
    # именно столько скобок [[]]
    # todo move x in args
    x = sp.Matrix([[random.randint(-10, 10) for i in range(n)]])
    print(x)
    # x = sp.Matrix([[-10, 0]])

    def to_args(t):
        return [(f"x{i}", t[i]) for i in range(n)]

    while True:
        # ||∇f(x)|| < ε
        y = x - alpha * f.grad(to_args(x))

        if get_metric2(f.grad(y) - f.grad(x)) < eps:
            break

        if f.eval(to_args(y)) < f.eval(to_args(x)):
            x = y
            # alpha = const in 1st task
            # alpha = alpha / 2
        # print(f.metric_of_gradient_in_point(to_args(x)), x, y)
    return x