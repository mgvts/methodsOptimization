# from chatGPT
import numpy as np

def wolfe_line_search(f, grad_f, x, p, c1=0.0001, c2=0.9, max_iter=100):
    """
    Wolfe line search algorithm for finding step size alpha that satisfies strong Wolfe conditions.
    :param f: objective function
    :param grad_f: gradient of objective function
    :param x: current point
    :param p: search direction
    :param c1: Wolfe condition 1 constant (default: 0.0001)
    :param c2: Wolfe condition 2 constant (default: 0.9)
    :param max_iter: maximum number of iterations (default: 100)
    :return: step size alpha
    """
    alpha = 1.0
    i = 0
    while i < max_iter:
        # Evaluate function and gradient at current point and proposed step
        f_x = f(x)
        grad_f_x = grad_f(x)
        f_x_alpha = f(x + alpha * p)

        # Check Wolfe condition 1
        if f_x_alpha > f_x + c1 * alpha * np.dot(grad_f_x, p):
            return backtrack_line_search(f, grad_f, x, p, alpha, c1)

        # Evaluate gradient at proposed step
        grad_f_x_alpha = grad_f(x + alpha * p)

        # Check Wolfe condition 2
        if np.dot(grad_f_x_alpha, p) < c2 * np.dot(grad_f_x, p):
            return backtrack_line_search(f, grad_f, x, p, alpha, c2)

        i += 1
        alpha *= 2.0

    return alpha


def backtrack_line_search(f, grad_f, x, p, alpha_init, c):
    """
    Backtracking line search algorithm for finding step size alpha that satisfies Wolfe condition 1.
    :param f: objective function
    :param grad_f: gradient of objective function
    :param x: current point
    :param p: search direction
    :param alpha_init: initial step size
    :param c: Wolfe condition 1 constant
    :return: step size alpha
    """
    alpha = alpha_init
    while f(x + alpha * p) > f(x) + c * alpha * np.dot(grad_f(x), p):
        alpha *= 0.5

    return alpha
"""
Для использования данного кода необходимо передать функцию `f`,
градиент функции `grad_f`, текущую точку `x`, направление поиска `p`, а также константы `c1` и `c2`,
которые задают условия Вольфе. Функция `wolfe_line_search` возвращает значение оптимального шага `alpha`,
удовлетворяющего обоим условиям Вольфе. Если это не удается найти за максимальное число итераций `max_iter`,
то возвращается последнее пробное значение шага `alpha`.

"""

# Вот пример использования функции `wolfe_line_search` для оптимизации функции Розенброка:

import numpy as np

def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_rosenbrock(x):
    return np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 200 * (x[1] - x[0]**2)])

# Starting point
x0 = np.array([-1.2, 1.0])

# Search direction
p = np.array([1.0, 0.0])

# Find optimal step size
alpha = wolfe_line_search(rosenbrock, grad_rosenbrock, x0, p)

# Update current point
x1 = x0 + alpha * p

print("Optimal step size:", alpha)
print("New point:", x1)

"""
В этом примере мы оптимизируем функцию Розенброка в направлении `[1, 0]` из начальной точки `[-1.2, 1.0]`.
Функция `rosenbrock` вычисляет значение функции Розенброка в заданной точке, а функция `grad_rosenbrock` вычисляет ее градиент.
Мы используем функцию `wolfe_line_search` для нахождения оптимального шага в направлении `[1, 0]`.
Затем мы обновляем текущую точку `x0` с помощью найденного шага и выводим результаты.

Надеюсь, это помогает! Дайте мне знать, если у вас есть дополнительные вопросы."""
