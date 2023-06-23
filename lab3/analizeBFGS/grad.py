from pprint import pprint

import numpy as np
from lab3.BFGS import outputDTO, line_search


def const_grad_down(f, grad_f, x, alpha=0.0001, eps=1e-4, max_iter=100, ):
    out = outputDTO(points=[x.copy()],
                    was_broken=False,
                    alphas=None)
    x = x.astype(np.float64)
    while True:
        grad = -grad_f(x)
        if np.linalg.norm(grad) < eps:
            break
        if len(out.points) >= max_iter:
            out.was_broken = True
            break
        x += alpha * grad.astype(np.float64)
        out.points.append(x.copy())
    return out


def wolfe_grad_down(f, grad_f, x, eps=1e-4, max_iter=100):
    out = outputDTO(points=[x.copy()],
                    was_broken=False,
                    alphas=[])
    x = x.astype(np.float64)
    while True:
        grad = -grad_f(x)
        try:
            alpha = line_search(f, grad_f, x, grad)
            out.alphas.append(alpha)
        except AssertionError:
            out.was_broken = True
            break

        if np.linalg.norm(grad) < eps:
            break
        if len(out.points) >= max_iter:
            out.was_broken = True
            break
        x += alpha * grad.astype(np.float64)
        out.points.append(x.copy())
    return out
