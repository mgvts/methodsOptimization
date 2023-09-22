import cProfile
import linecache
import os
import pstats
import time
import tracemalloc

import numpy as np

from lab2.examples.linear_regression import twod_line
from lab2.linear_regression import LinearRegression
from lab4.linear.pytorch_linear_regression import PyTorchLinearRegression

count = 100
X, Y = twod_line.generate_linear_regression_2d(count, shift=(2, 2))
b = np.matrix([100., 100.]).transpose()

regression = LinearRegression(X, Y, b, count / 2)
torch_regression = PyTorchLinearRegression(X, Y, b, count / 2)

data = [
    {
        'name': "Minibatch CD",
        'f': lambda: regression.stochastic_grad_down(alpha=0.001, runs=10000),
    },
    {
        'name': "Minibatch CD Pytorch",
        'f': lambda: torch_regression.stochastic_grad_down(alpha=0.001, runs=10000, eps=0.0000001),
    },
    {
        "name": "Momentum",
        'f': lambda: regression.momentum_stochastic_grad_down(y=0.8, alpha=0.001, runs=10000),
    },
    {
        "name": "Nesterov",
        'f': lambda: regression.nesterov_stochastic_grad_down(y=0.9, alpha=0.0001, runs=10000),
    },
    {
        "name": "Momentum Pytorch",
        'f': lambda: torch_regression.momentum_stochastic_grad_down(y=0.8, alpha=0.001, runs=10000, eps=0.0000001),
    },
    {
        "name": "Nesterov Pytorch",
        'f': lambda: torch_regression.nesterov_stochastic_grad_down(y=0.9, alpha=0.0001, runs=10000, eps=0.0000001),
    },
    {
        "name": "Adagrad",
        'f': lambda: regression.adagrad_stochastic_grad_down(alpha=10, runs=10000),
    },
    {
        'name': "Adagrad Pytorch",
        'f': lambda: torch_regression.adagrad_stochastic_grad_down(alpha=10, runs=10000, eps=0.0000001),
    },
    {
        "name": "RMS",
        'f': lambda: regression.rms_stochastic_grad_down(W=4, alpha=0.2, runs=10000),
    },
    {
        'name': "RMS Pytorch",
        'f': lambda: torch_regression.rms_stochastic_grad_down(W=0.99, alpha=0.2, runs=10000, eps=0.0000001),
    },
    {
        "name": "Adam",
        'f': lambda: regression.adam_stochastic_grad_down(b1=0.9, b2=0.9, alpha=0.11, runs=10000),
    },
    {
        'name': "ADAM Pytorch",
        'f': lambda: torch_regression.adam_stochastic_grad_down(b1=0.9, b2=0.9, alpha=0.11, runs=1000, eps=0.0001),
    },
]


def display_top(snapshot, key_type='lineno', limit=5):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


print("=" * 120)


def profile(_task):
    task = _task['f']
    for i in range(5):
        print(f"TIME. PROFILING: {_task['name']}. # {i}")
        print("=" * 120)
        profiler = cProfile.Profile()
        profiler.enable()
        task()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
        print("=" * 120)
        print(f"MEM. PROFILING: {_task['name']}. # {i}")
        print("=" * 120)

        tracemalloc.start()
        task()
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot)
        print("=" * 120)
        tracemalloc.stop()


for i in data:
    profile(i)

