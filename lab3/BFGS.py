from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class outputDTO:
    points: list[np.array]
    was_broken: bool
    alphas: [float]


# Callable[[Arg1Type, Arg2Type], ReturnType]

def bfgs(f: Callable[[np.array], float],
         grad_f: Callable[[np.array], np.array],
         x0: np.array,
         eps=1e-6,
         max_iter=10) -> outputDTO:
    """
    Алгоритм BFGS который позволяет искать минимум функции

    :param f: функция которую требуется исследовать
    :param grad_f: функция, которая является градиентом данной
    :param x0: начальная точка
    :param eps: необходимая точность алгоритма
    :param max_iter: максимальное число итераций, если нужная точность не достигнута
    :return: такая точка(вектор) x, что достигается минимум функции
    :raises AssertionError: если в процессе alpha будет нулём (метод не сошёлся)
    """
    out = outputDTO(points=[x0], was_broken=False,
                    alphas=[])
    n = len(x0)
    # this is casual way, but instead I may be any 'good' matrix
    H = np.eye(n)
    x = x0
    for i in range(max_iter):
        try:
            grad = grad_f(x)

            if np.linalg.norm(grad) < eps:
                break
            # changing p = - H_k *∇ f_k
            # same as p = -np.dot(H, grad)
            p = - H @ grad
            # alpha finding with well known line search with Wolfe conditions

            alpha = line_search(f, grad_f, x, p)
            out.alphas.append(alpha)
            # немного хитро, но присмотритесь, это то что надо
            # (x_new = x + alpha*p; s = x_new - x)
            s = alpha * p
            x_new = x + s

            y = grad_f(x_new) - grad
            rho = 1 / (y @ s)
            # в отчёте будут красивые картинки, просто поверьте, что это работает
            # np.outer(s, y) = s @ (y.T)
            A = np.eye(n) - rho * np.outer(s, y)
            B = np.eye(n) - rho * np.outer(y, s)
            H = A @ H @ B + rho * np.outer(s, s)
            x = x_new
            out.points.append(x)
        except Exception as e:
            # print(e)
            out.was_broken = True
            break
    return out


def line_search(f: Callable[[np.array], float],
                grad_f: Callable[[np.array], np.array],
                x: np.array,
                p: np.array) -> float:
    alpha = 4
    c1 = 0.01
    c2 = 0.9
    eps = 1e-10

    while not (get_cond1(f, grad_f, x, alpha, p, c1=c1) and
               get_cond2(grad_f, x, alpha, p, c2=c2)):
        alpha *= 0.5
        if alpha <= eps:
            raise AssertionError("in line search alpha equals zero")
    return alpha


def get_cond1(f: Callable[[np.array], float],
              grad_f: Callable[[np.array], np.array],
              x: np.array,
              alpha: float,
              p: np.array, c1: float) -> bool:
    return f(x + alpha * p) <= f(x) + c1 * alpha * (grad_f(x).T @ p)


# also H@p same as np.dot(H, p)
def get_cond2(grad_f: Callable[[np.array], np.array],
              x: np.array, alpha: float, p: np.array, c2: float) -> bool:
    return grad_f(x + alpha * p).T @ p >= c2 * (grad_f(x).T @ p)


def lbfgs(f: Callable[[np.array], float],
          grad_f: Callable[[np.array], np.array],
          x0: np.array,
          eps=1e-6,
          max_iter=10,
          m=10) -> np.array:
    out = outputDTO(points=[x0], was_broken=False,
                    alphas=[])
    n = len(x0)
    H = np.eye(n)
    x = x0
    s_list = []
    y_list = []
    rho_list = []
    g = grad_f(x)
    d = -g
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < eps:
            break
        p = - H @ grad
        try:
            alpha = line_search(f, grad_f, x, p)
        except AssertionError as e:
            out.was_broken = True
            break
        s = alpha * p
        x_new = x + s

        y = grad_f(x_new) - grad
        rho = 1 / (y @ s)
        if np.isinf(rho):
            out.was_broken = True
            break

        # добавляем векторы s и y в очередь
        if len(s_list) == m:
            s_list.pop(0)
            y_list.pop(0)
            rho_list.pop(0)
        s_list.append(s)
        y_list.append(y)
        rho_list.append(rho)

        # вычисляем q и r
        q = grad.copy()
        alpha_list = []
        for j in range(len(s_list) - 1, -1, -1):
            alpha_j = rho_list[j] * s_list[j].dot(q)
            q -= alpha_j * y_list[j]
            alpha_list.append(alpha_j)
        r = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1]) * q
        for j in range(len(s_list)):
            beta_j = rho_list[j] * y_list[j].dot(r)
            r += (alpha_list.pop() - beta_j) * s_list[j]

        # H = np.outer(s, y) / (y @ y) + np.diag(r)

        x = x_new
        g = grad_f(x)
        d = -r
        out.points.append(x)
    return out
