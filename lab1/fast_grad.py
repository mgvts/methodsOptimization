import math
from dataclasses import dataclass

from lab1.tools import Func, get_metric3, to_args, FastQFunc
import numpy as np

@dataclass
class OutputDTO:
    points: list[float]
    points_with_floats: list[list[float]]
    alpha: list[float]
    eps: float
    metrics: list[float]
    string_func: str
    n: int
    was_broken: bool
    iter: int
    dichotomy_count: list[int]


eps_CONST = 0.0001
alpha_CONST = 0.1
max_INTER = 1000


# todo можно обьединить grad_down() и grad_down_dichotomy()

def grad_down(func: FastQFunc,
              start_point: [float],
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
    x = np.matrix(start_point).transpose()
    out = OutputDTO(
        points=[],
        points_with_floats=[],
        alpha=[],
        string_func="none",
        n=func.n,
        was_broken=False,
        eps=eps,
        metrics=[],
        iter=0,
        dichotomy_count=[]
    )

    while True:
        y = x - alpha * func.grad(x)

        metr = get_metric3(func.grad(y) - func.grad(x)) # todo change metric

        out.points.append(x)
        out.points_with_floats.append(x.values())
        out.alpha.append(alpha)
        out.metrics.append(metr)
        out.iter += 1

        # ||∇f(x)|| < ε
        if metr < eps:
            break

        if func.eval(y) < func.eval(x):
            x = y

        if len(out.points) > max_inter:
            out.was_broken = True
            return out

        if len(out.points) > 10:
            if out.points[-1] == out.points[-2]:
                out.was_broken = True
                return out

    return out


def dichotomy(f, eps=0.001, delta=0.00015, a=0, b=1):
    def calc_min_iterations():
        return math.log((b - a - delta) / (2 * eps - delta), 2)

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
    return (a + b) / 2, N


def grad_down_dichotomy(func: FastQFunc,
                        start_point: [float],
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
    x = np.matrix(start_point)

    out = OutputDTO(
        points=[],
        points_with_floats=[],
        alpha=[],
        string_func="none",
        n=func.n,
        was_broken=False,
        eps=eps,
        metrics=[],
        iter=0,
        dichotomy_count=[]
    )

    while True:
        alpha, count = dichotomy(lambda a: func.eval(x - a * func.grad(x)))
        y = x - alpha * func.grad(x)
        metr = get_metric3(func.grad(y) - func.grad(x))

        out.dichotomy_count.append(count)
        out.points.append(x)
        out.points_with_floats.append(x.values())
        out.alpha.append(alpha)
        out.metrics.append(metr)
        out.iter += 1

        # ||∇f(x)|| < ε
        if metr < eps:
            break

        if func.eval(y) < func.eval(x):
            x = y
        if len(out.points) > max_inter:
            out.was_broken = True
            return out
    return out
