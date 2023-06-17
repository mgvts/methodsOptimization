from collections import Callable

import numpy as np


# Callable[[Arg1Type, Arg2Type], ReturnType]

def bfgs(f: Callable[[np.array], float],
         grad_f: Callable[[np.array], np.array],
         x0: np.array,
         eps=1e-6,
         max_iter=10) -> np.array:
    n = len(x0)
    # but correct way is use H = B_0^-1 where B_0 is hessian of current function
    H = np.eye(n)
    x = x0
    for i in range(max_iter):
        grad = grad_f(x)
        # norm(grad) this 2d casual norm
        if np.linalg.norm(grad) < eps:
            break
        # changing p = - H_k *∇ f_k
        # same as p = -np.dot(H, grad)
        p = -H @ grad
        # alpha finding with well known Wolfe search
        alpha = line_search(f, grad_f, x, p)

        # немного хитро, но присмотритесь, это то что надо
        # (x_new = x + alpha*p; s = x_new - x)
        s = alpha * p
        x_new = x + s

        y = grad_f(x_new) - grad
        rho = 1 / np.dot(y, s)
        # в отчёте будут красивые картинки, просто поверьте, что это работает
        # np.outer(s, y) = s @ (y.T)
        A = np.eye(n) - rho * np.outer(s, y)
        B = np.eye(n) - rho * np.outer(y, s)
        H = A @ H @ B + rho * np.outer(s, s)
        x = x_new
    return x


def line_search(f: Callable[[np.array], float],
                grad_f: Callable[[np.array], np.array],
                x: np.array,
                p: np.array) -> float:
    alpha = 1
    c1 = 1e-4
    c2 = 0.9

    while not (get_cond1(f, grad_f, x, alpha, p, c1=c1) and
               get_cond2(grad_f, x, alpha, p, c2=c2)):
        alpha *= 0.5

    return alpha


# also H@p same as np.dot(H, p)
def get_cond1(f: Callable[[np.array], float],
              grad_f: Callable[[np.array], np.array],
              x: np.array, alpha: float, p: np.array, c1: float) -> bool:
    return f(x + alpha * p) <= f(x) + c1 * alpha * (grad_f(x).T @ p)


# also H@p same as np.dot(H, p)
def get_cond2(grad_f: Callable[[np.array], np.array],
              x: np.array, alpha: float, p: np.array, c2: float) -> bool:
    return grad_f(x - alpha * p).T @ p >= c2 * (grad_f(x).T @ p)
