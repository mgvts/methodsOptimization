from lab1.tools import Func, get_metric2, to_args

from random import randint
import sympy as sp

eps_CONST = 0.0001
alpha_CONST = 0.001


def grad_down_metric(n: int, string_func: str,
                     start_point: sp.Matrix,
                     function_alpha,
                     eps=eps_CONST,
                     alpha=alpha_CONST) -> sp.Matrix:
    """
    :param n: how many variables in function
    :param string_func: string of Function
    :param start_point: Start point like sp.Matrix([[1,1]])
    :param function_alpha: function to changed  alpha
    :param alpha: alpha or lambda in gradient method
    :param eps: eps in gradient method
    :return: point where function go min
    """
    f = Func(n, string_func)
    # именно столько скобок [[]]
    # todo move x in args

    x = start_point
    # print(x)

    while True:
        # ||∇f(x)|| < ε
        if f.metric_of_gradient_in_point(to_args(x, n)) < eps:
            break

        y = x - alpha * f.grad(to_args(x, n))
        if f.eval(to_args(y, n)) < f.eval(to_args(x, n)):
            x = y
        alpha = function_alpha(f)
        print(f"{alpha = }")
        # print(f.metric_of_gradient_in_point(to_args(x)), x, y)
    return x


def grad_down_metric_between_difference(n, string_func,
                                        start_point: sp.Matrix,
                                        function_alpha,
                                        eps=eps_CONST,
                                        alpha=alpha_CONST):
    """
    :param n: how many variables in function
    :param string_func: string of Function
    :param start_point: Start point like sp.Matrix([[1,1]])
    :param alpha: alpha or lambda in gradient method
    :param eps: eps in gradient method
    :return: point where function go min
    """
    f = Func(n, string_func)
    # именно столько скобок [[]]
    # x = sp.Matrix([[randint(-10, 10) for i in range(n)]])
    x = start_point
    print(x)

    # x = sp.Matrix([[-10, 0]])

    while True:
        # ||∇f(x)|| < ε
        y = x - alpha * f.grad(to_args(x, n))
        print(x, y)
        if get_metric2(f.grad(to_args(y, n)) - f.grad(to_args(x, n))) < eps:
            break

        if f.eval(to_args(y, n)) < f.eval(to_args(x, n)):
            x = y
            # alpha = const in 1st task
            # alpha = alpha / 2
        alpha = function_alpha(f, alpha=alpha)

        # print(f.metric_of_gradient_in_point(to_args(x)), x, y)
    return x
