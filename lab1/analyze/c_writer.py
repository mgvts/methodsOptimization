import json

from sympy import plot_implicit, symbols, Matrix

from lab1.grad import grad_down, grad_down_dichotomy
from lab1.tools import Func

base = [
    # (0.5, "x0^2 + (x1)^2", 1, '3a_count_iter_1_1_1.json', 'const'),
    # (0.8, "x0^2 + (x1)^2", 1, '3a_count_iter_1_1_2.json', 'const'),
    # (0.001, "x0^2 + (x1)^2", 1, '3a_count_iter_1_1_3.json', 'const'),
    #
    # (0.02, "10*x0^2 + (x1)^2", 1, '3a_count_iter_1_2_1.json', 'const'),
    # (0.001, "10*x0^2 + (x1)^2", 1, '3a_count_iter_1_2_2.json', 'const'),
    # (0.01, "10*x0^2 + (x1)^2", 1, '3a_count_iter_1_2_3.json', 'const'),

    # (0.02, "2 * x0^2 + (x1-3)^2 + 2*x0 -3*x1 - 10", 1, '3a_count_iter_1_3_1.json', 'const'),
    # (0.001, "2 * x0^2 + (x1-3)^2 + 2*x0 -3*x1 - 10", 1, '3a_count_iter_1_3_2.json', 'const'),
    # (0.04, "2 * x0^2 + (x1-3)^2 + 2*x0 -3*x1 - 10", 1, '3a_count_iter_1_3_3.json', 'const'),

    (-1, "x0^2 + (x1)^2", 1, '3a_count_iter_d_1_1.json', 'dichotomy'),
    (-1, "10*x0^2 + (x1)^2", 1, '3a_count_iter_d_2_1.json', 'dichotomy'),
    (-1, "2 * x0^2 + (x1-3)^2 + 2*x0 -3*x1 - 10", 1, '3a_count_iter_d_3_1.json', 'dichotomy'),
]

stringFunc = "x0^2 + (x1)^2"

alpha = 0.8
step = 5
for alpha, stringFunc, step, name, _type in base:
    max_inter = 10_000
    n = 2
    func = Func(n, stringFunc)
    f = func.f

    x_range = [-100, 100]
    y_range = [-100, 100]

    x0, x1 = symbols('x0 x1')

    # init
    p = plot_implicit(f, x_var=(x0, *x_range), y_var=(x1, *y_range), show=False, line_color='red')


    def rgba_to_hex(rgb):
        return '%02x%02x%02x' % rgb


    # grad field
    p.rectangles = []
    x_len = x_range[1] - x_range[0]
    y_len = y_range[1] - y_range[0]

    # find min/max
    mm = 9999999
    mx = 0

    matr = {}
    print('PROCESS start points')

    count = 0
    final_count = round(x_len / step) * round(y_len / step)
    for i in range(0, round(x_len / step)):
        for j in range(0, round(y_len / step)):
            _x = (x_range[0] + i * step, x_range[0] + (i + 1) * step)
            _y = (y_range[0] + j * step, y_range[0] + (j + 1) * step)
            x = _x[0] + step / 2
            y = _y[0] + step / 2
            if type == 'const':
                grad = grad_down(2, stringFunc, Matrix([[x, y]]), alpha=alpha, max_inter=max_inter)
            else:
                grad = grad_down_dichotomy(2, stringFunc, Matrix([[x, y]]), max_inter=max_inter)

            if not grad.was_broken or len(grad.points) == max_inter + 1:
                matr[f'{x}, {y}'] = len(grad.points)
            else:
                matr[f'{x}, {y}'] = -1
            count += 1
            print(f'{count} / {final_count} {len(grad.points)} {grad.points[-1]} {name}')

    with open('../data/' + name, 'w') as file:
        file.write(json.dumps({
            'x_range': x_range,
            'y_range': y_range,
            'alpha': alpha,
            'eps':  0.0001,
            'type': _type,
            'func': stringFunc,
            'matrix': matr
        }))
