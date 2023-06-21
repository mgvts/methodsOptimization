from pprint import pprint

import numpy as np
from lab3.BFGS import outputDTO, line_search


def const_grad_down(f, grad_f, x, eps=1e-4, max_iter=100):
    out = outputDTO(points=[x],
                    was_broken=False)
    alpha = 0.1
    while True:
        grad = grad_f(x)
        if np.linalg.norm(grad) < eps:
            break
        if len(out.points) >= max_iter:
            out.was_broken = True
            break
        x -= alpha * grad
        out.points.append(x)

    return out


def wolfe_grad_down(f, grad_f, x, eps=1e-4, max_iter=100):
    out = outputDTO(points=[x],
                    was_broken=False)
    alpha = 0.01
    iter_num = 0
    while True:
        grad = grad_f(x)
        alpha = line_search(f, grad_f, x, grad)
        if np.linalg.norm(grad) < eps:
            break
        if iter_num >= max_iter:
            out.was_broken = True
            break
        x -= alpha * grad
        iter_num += 1
        out.points.append(x)
    return out


pprint(const_grad_down(lambda x: x[0] ** 2 + x[1] ** 2, lambda x: np.array([2 * x[0], 2 * x[1]]),
                       np.array([1., -1.])).points)
