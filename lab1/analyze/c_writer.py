import json

import numpy as np
from sympy import symbols

from lab1.fast_grad import grad_down, grad_down_dichotomy
from lab1.tools import FastQFunc

# x0^2 + (x1)^2
f1 = FastQFunc(2, np.matrix([
    [2, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)

# "10*x0^2 + (x1)^2"
f2 = FastQFunc(2, np.matrix([
    [20, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)

# "9*x0^2 + (x1)^2"
f21 = FastQFunc(2, np.matrix([
    [18, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)
# "7*x0^2 + (x1)^2"
f22 = FastQFunc(2, np.matrix([
    [14, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)
# "5*x0^2 + (x1)^2"
f23 = FastQFunc(2, np.matrix([
    [10, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)
# "3*x0^2 + (x1)^2"
f24 = FastQFunc(2, np.matrix([
    [6, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)
# "2*x0^2 + (x1)^2"
f25 = FastQFunc(2, np.matrix([
    [4, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)
# "1.9*x0^2 + (x1)^2"
f221 = FastQFunc(2, np.matrix([
    [3.8, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)
# "1.7*x0^2 + (x1)^2"
f222 = FastQFunc(2, np.matrix([
    [3.4, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)
# "1.5*x0^2 + (x1)^2"
f223 = FastQFunc(2, np.matrix([
    [3, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)
# "1.3*x0^2 + (x1)^2"
f224 = FastQFunc(2, np.matrix([
    [2.6, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)
# "1.1*x0^2 + (x1)^2"
f225 = FastQFunc(2, np.matrix([
    [2.2, 0],
    [0, 2]
]), np.matrix([[0, 0]]), 0)



# "2 * x0^2 + (x1-3)^2 + 2*x0 -3*x1 - 10"
f3 = FastQFunc(2, np.matrix([
    [4, 0],
    [0, 2]
]), np.matrix([[2, -9]]), 10)

base = [
    # (0.5, "x0^2 + (x1)^2", f1, 1, '3a_count_iter_1_1_1.json', 'const'),
    # (0.2, "x0^2 + (x1)^2", f1, 1, '3a_count_iter_1_1_2.json', 'const'),
    # (0.01, "x0^2 + (x1)^2", f1, 1, '3a_count_iter_1_1_3.json', 'const'),

    # (0.08, "10*x0^2 + (x1)^2", f2, 1, '3a_count_iter_1_2_1.json', 'const'),
    # (0.09, "10*x0^2 + (x1)^2", f2, 1, '3a_count_iter_1_2_2.json', 'const'),
    # (0.01, "10*x0^2 + (x1)^2", f2, 1, '3a_count_iter_1_2_3.json', 'const'),
    #
    # (0.2, "2 * x0^2 + (x1-3)^2 + 2*x0 -3*x1 - 10", f3, 1, '3a_count_iter_1_3_1.json', 'const'),
    # (0.09, "2 * x0^2 + (x1-3)^2 + 2*x0 -3*x1 - 10", f3, 1, '3a_count_iter_1_3_2.json', 'const'),
    # (0.05, "2 * x0^2 + (x1-3)^2 + 2*x0 -3*x1 - 10", f3, 1, '3a_count_iter_1_3_3.json', 'const'),
    #
    # (-1, "x0^2 + (x1)^2", f1, 1, '3a_count_iter_d_1_1.json', 'dichotomy'),
    # (-1, "10*x0^2 + (x1)^2", f2, 1, '3a_count_iter_d_2_1.json', 'dichotomy'),
    # (-1, "2 * x0^2 + (x1-3)^2 + 2*x0 -3*x1 - 10", f3, 1, '3a_count_iter_d_3_1.json', 'dichotomy'),

    (-1, "9*x0^2 + x1^2", f21, 1, '3a_count_iter_d_4_1.json', 'dichotomy'),
    (-1, "7*x0^2 + x1^2", f22, 1, '3a_count_iter_d_4_2.json', 'dichotomy'),
    (-1, "5*x0^2 + x1^2", f23, 1, '3a_count_iter_d_4_3.json', 'dichotomy'),
    (-1, "3*x0^2 + x1^2", f24, 1, '3a_count_iter_d_4_4.json', 'dichotomy'),
    (-1, "2*x0^2 + x1^2", f25, 1, '3a_count_iter_d_4_5.json', 'dichotomy'),
    (-1, "1.9*x0^2 + x1^2", f221, 1, '3a_count_iter_d_5_1.json', 'dichotomy'),
    (-1, "1.7*x0^2 + x1^2", f222, 1, '3a_count_iter_d_5_2.json', 'dichotomy'),
    (-1, "1.5*x0^2 + x1^2", f223, 1, '3a_count_iter_d_5_3.json', 'dichotomy'),
    (-1, "1.3*x0^2 + x1^2", f224, 1, '3a_count_iter_d_5_4.json', 'dichotomy'),
    (-1, "1.1*x0^2 + x1^2", f225, 1, '3a_count_iter_d_5_5.json', 'dichotomy'),
]

def writer(alpha, stringFunc, func, step, name, _type):
    max_inter = 10_000
    n = 2

    x_range = [-100, 100]
    y_range = [-100, 100]

    x0, x1 = symbols('x0 x1')


    # init

    def rgba_to_hex(rgb):
        return '%02x%02x%02x' % rgb


    # grad field
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
            x = _x[0]
            y = _y[0]
            if _type == 'const':
                grad = grad_down(func, np.matrix([[x, y]]), alpha=alpha, max_inter=max_inter)
            else:
                grad = grad_down_dichotomy(func, np.matrix([[x, y]]), max_inter=max_inter)

            if not grad.was_broken or len(grad.points) == max_inter + 1:
                matr[f'{x}, {y}'] = len(grad.points)
            else:
                matr[f'{x}, {y}'] = -1
            count += 1
            if count % 100 == 0:
                print(f'{count} / {final_count} {len(grad.points)} {grad.points[-1]} {name}')

    with open('../data/' + name, 'w') as file:
        file.write(json.dumps({
            'x_range': x_range,
            'y_range': y_range,
            'alpha': alpha,
            'eps': 0.0001,
            'type': _type,
            'func': stringFunc,
            'matrix': matr
        }))

import multiprocessing

if __name__ == '__main__':
    process = []
    for alpha, stringFunc, func, step, name, _type in base:
        proc = multiprocessing.Process(target=writer, args=(alpha, stringFunc, func, step, name, _type,))
        process.append(proc)

    for i in process:
        i.start()

    for i in process:
        i.join()


import multiprocessing

if __name__ == '__main__':
    process = []
    for alpha, stringFunc, func, step, name, _type in base:
        proc = multiprocessing.Process(target=writer, args=(alpha, stringFunc, func, step, name, _type,))
        process.append(proc)

    for i in process:
        i.start()

    for i in process:
        i.join()

