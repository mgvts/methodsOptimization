import math
from dataclasses import dataclass

from lab1.tools import get_metric3, to_args, FastQFunc
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
alpha_CONST = 0.02
max_INTER = 10000


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

        out.points.append(x.transpose().tolist()[0])
        out.points_with_floats.append(x)
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


def dichotomy(f, eps=0.0001, delta=0.00015, a=0, b=1):
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
        alpha, count = dichotomy(lambda a: func.eval(x - a * func.grad(x)))
        y = x - alpha * func.grad(x)
        metr = get_metric3(func.grad(y) - func.grad(x))

        out.dichotomy_count.append(count)
        out.points.append(x.transpose().tolist()[0])
        out.points_with_floats.append(x)
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



def parse_eval(f: FastQFunc):
    return lambda x: f.eval(to_args(x, f.get_n()))


def wolfe_conditions(func: FastQFunc, x: np.matrix, alpha: float, c1=0.01, c2=0.99):
    grad_fx = func.grad(x)
    p = func.grad(x)
    fx_alpha = func.eval(x - alpha * p)
    grad_fx_alpha = func.grad(x - alpha * p)
    cond1 = fx_alpha <= func.eval(x) + c1 * alpha * (grad_fx.transpose() * p)
    cond2 = (grad_fx_alpha.transpose() * p) >= c2 * (grad_fx_alpha.transpose() * p)
    return cond1 and cond2


# todo
def find_alpha_with_wolfe(f: FastQFunc,
                          start_point: np.matrix,
                          c1=0.001, c2=0.99):
    alpha = 1
    while not wolfe_conditions(f, start_point, alpha, c1=c1, c2=c2):
        alpha *= 0.5
    return alpha

def grad_down_wolfe(func: FastQFunc,
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
        dichotomy_count = []
    )

    while True:
        alpha = find_alpha_with_wolfe(func, x)
        # alpha = dichotomy(lambda a: f.eval(to_args(x - a * f.grad(to_args(x, n)).evalf(), n)))
        y = x - alpha * func.grad(x)
        metr = get_metric3(func.grad(y) - func.grad(x))

        out.points.append(x.transpose().tolist()[0])
        out.points_with_floats.append(x)
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