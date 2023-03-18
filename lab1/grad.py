import math
from dataclasses import dataclass

import sympy as sp

from lab1.tools import Func, get_metric2, to_args


@dataclass
class OutputDTO:
    points: list[
        sp.Matrix]
    points_with_floats: list[list[float]]
    alpha: list[float]
    eps: float
    metrics: list[float]
    string_func: str
    n: int
    was_broken: bool
    iter: int


eps_CONST = 0.0001
alpha_CONST = 0.1
max_INTER = 1000


# todo можно обьединить grad_down() и grad_down_dichotomy()

def grad_down(n: int, string_func: str,
              start_point: sp.Matrix,
              eps=eps_CONST,
              alpha=alpha_CONST,
              max_inter=max_INTER) -> OutputDTO:
    """
    :param max_inter: max interaction should be run
    :param n: how many variables in function
    :param string_func: string of Function
    :param start_point: Start point like sp.Matrix([[1,1]])
    :param alpha: alpha or lambda in gradient method
    :param eps: eps in gradient method
    :return: OutputDTO, dataclass for analyze
    """
    f = Func(n, string_func)
    x = start_point
    out = OutputDTO(
        points=[],
        points_with_floats=[],
        alpha=[],
        string_func=string_func,
        n=n,
        was_broken=False,
        eps=eps,
        metrics=[],
        iter=0
    )

    while True:
        y = x - alpha * f.grad(to_args(x, n))
        metr = get_metric2(f.grad(to_args(y, n)) - f.grad(to_args(x, n)))

        out.points.append(x)
        out.points_with_floats.append(x.values())
        out.alpha.append(alpha)
        out.metrics.append(metr)
        out.iter += 1

        # ||∇f(x)|| < ε
        if metr < eps:
            break

        if f.eval(to_args(y, n)) < f.eval(to_args(x, n)):
            x = y

        if len(out.points) > max_inter:
            out.was_broken = True
            return out
    return out


def dichotomy(f, eps=0.001, delta=0.00015, a=0, b=1):
    def calc_min_iterations():
        return sp.log((b - a - delta) / (2 * eps - delta), 2)

    N = math.ceil(calc_min_iterations())
    x1 = (a + b - delta) / 2
    x2 = (a + b + delta) / 2
    for i in range(N):
        # 1 step
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2

        # 2 step
        if f(x1) <= f(x2):
            b = x2
        else:
            a = x1

        # 3 step
        eps_i = (b - a) / 2
        if eps_i <= eps:
            break
    return (a + b) / 2


def grad_down_dichotomy(n: int, string_func: str,
                        start_point: sp.Matrix,
                        eps=eps_CONST,
                        max_inter=max_INTER) -> OutputDTO:
    """
    :param max_inter: max interaction should be run
    :param n: how many variables in function
    :param string_func: string of Function
    :param start_point: Start point like sp.Matrix([[1,1]])
    :param function_alpha: function to changed  alpha
    :param eps: eps in gradient method
    :return: OutputDTO, dataclass for analyze
    """
    f = Func(n, string_func)
    x = start_point

    out = OutputDTO(
        points=[],
        points_with_floats=[],
        alpha=[],
        string_func=string_func,
        n=n,
        was_broken=False,
        eps=eps,
        metrics=[],
        iter=0
    )

    while True:
        alpha = dichotomy(lambda a: f.eval(to_args(x - a * f.grad(to_args(x, n)).evalf(), n)))
        y = x - alpha * f.grad(to_args(x, n))
        metr = get_metric2(f.grad(to_args(y, n)) - f.grad(to_args(x, n)))

        out.points.append(x)
        out.points_with_floats.append(x.values())
        out.alpha.append(alpha)
        out.metrics.append(metr)
        out.iter += 1

        # ||∇f(x)|| < ε
        if metr < eps:
            break

        if f.eval(to_args(y, n)) < f.eval(to_args(x, n)):
            x = y
        if len(out.points) > max_inter:
            out.was_broken = True
            return out
    return out


# todo now used only in 1.py but actually need replace to grad_down
def grad_down_between_difference(n, string_func,
                                 start_point: sp.Matrix,
                                 eps=eps_CONST,
                                 alpha=alpha_CONST):
    """
    deprecated
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

    # x = sp.Matrix([[-10, 0]])

    while True:
        # ||∇f(x)|| < ε
        y = x - alpha * f.grad(to_args(x, n))
        if get_metric2(f.grad(to_args(y, n)) - f.grad(to_args(x, n))) < eps:
            break

        if f.eval(to_args(y, n)) < f.eval(to_args(x, n)):
            x = y
    return x


def parse_eval(f: Func):
    return lambda x: f.eval(to_args(x, f.get_n()))


def wolfe_conditions(f: Func, x: sp.Matrix, alpha: float, c1=0.01, c2=0.99):
    fx = parse_eval(f)(x)
    grad_fx = f.grad(to_args(x, f.get_n()))
    p = f.grad(to_args(x, f.get_n()))
    fx_alpha = parse_eval(f)(x - alpha * p)
    grad_fx_alpha = f.grad(to_args(x - alpha * p, f.get_n()))
    cond1 = fx_alpha <= fx + c1 * alpha * (grad_fx * p.transpose())[0]
    cond2 = (grad_fx_alpha * p.transpose())[0] >= c2 * (grad_fx_alpha * p.transpose())[0]

    return cond1 and cond2


# todo
def find_alpha_with_wolfe(f: Func,
                           start_point: sp.Matrix,
                           c1=0.001, c2=0.99):
    alpha = 1
    while not wolfe_conditions(f, start_point, alpha, c1=c1, c2=c2):
        alpha *= 0.5
    return alpha

def grad_down_wolfe(n: int, string_func: str,
                        start_point: sp.Matrix,
                        eps=eps_CONST,
                        max_inter=max_INTER) -> OutputDTO:
    """
    :param max_inter: max interaction should be run
    :param n: how many variables in function
    :param string_func: string of Function
    :param start_point: Start point like sp.Matrix([[1,1]])
    :param function_alpha: function to changed  alpha
    :param eps: eps in gradient method
    :return: OutputDTO, dataclass for analyze
    """
    f = Func(n, string_func)
    x = start_point

    out = OutputDTO(
        points=[],
        points_with_floats=[],
        alpha=[],
        string_func=string_func,
        n=n,
        was_broken=False,
        eps=eps,
        metrics=[],
        iter=0
    )

    while True:
        alpha = find_alpha_with_wolfe(f, x)
        # alpha = dichotomy(lambda a: f.eval(to_args(x - a * f.grad(to_args(x, n)).evalf(), n)))
        y = x - alpha * f.grad(to_args(x, n))
        metr = get_metric2(f.grad(to_args(y, n)) - f.grad(to_args(x, n)))

        out.points.append(x)
        out.points_with_floats.append(x.values())
        out.alpha.append(alpha)
        out.metrics.append(metr)
        out.iter += 1

        # ||∇f(x)|| < ε
        if metr < eps:
            break

        if f.eval(to_args(y, n)) < f.eval(to_args(x, n)):
            x = y
        if len(out.points) > max_inter:
            out.was_broken = True
            return out
    return out