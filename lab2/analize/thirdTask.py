import dataclasses
import json
import math
import time
from dataclasses import dataclass
from pprint import pprint
from random import randint

import numpy as np
import pandas as pd
from lab2.examples.linear_regression.twod_line import generate_linear_regression_2d

from lab2.linear_regression import LinearRegression, get_batch_name, isError


@dataclass
class OutputDTO:
    count: int | None
    alpha: float | None
    run: int | None
    name: str | None
    res_point: [float]
    first_err: float | None
    result_err: float | None
    delta_time: float | None
    math_iters: int | None


counts = [10 * i for i in range(1, 22, 3)]
start_points = [np.matrix([10., 10.]).transpose(),
                np.matrix([-10., 10.]).transpose(),
                np.matrix([10., -10.]).transpose(),
                np.matrix([-10., -10.]).transpose()]

runs = [1000]
names = ["adagrad",
         "adam",
         "momentum",
         "rms",
         "nesterov",
         "stochastic"]


def start(alphas, i):
    data = []
    for count in counts:
        print(f"{count=} {names[i]}")
        for batch in [count / 2]:
            X, Y = generate_linear_regression_2d(count)
            for start_point in start_points:
                lr = LinearRegression(X, Y, start_point, batch=batch)
                first_err = lr.get_error_in_point(start_point)
                method = [
                    lr.adagrad_stochastic_grad_down_with_math_iters,
                    lr.adam_stochastic_grad_down_with_math_iters,
                    lr.momentum_stochastic_grad_down_with_math_iters,
                    lr.rms_stochastic_grad_down_with_math_iters,
                    lr.nesterov_stochastic_grad_down_with_math_iters,
                    lr.stochastic_grad_down_with_math_iter
                ][i]
                name = names[i]
                for alpha in alphas:
                    for run in runs:
                        if batch > count:
                            continue
                        st = time.time()
                        points, math_iters = method(alpha=alpha, runs=run)
                        delta = time.time() - st
                        if math_iters is None or math_iters == 0:
                            raise AssertionError
                        res = points[-1]
                        out = OutputDTO(
                            count=count,
                            alpha=alpha,
                            first_err=None if
                            math.isnan(float(first_err.item(0))) or math.isinf(float(first_err.item(0)))
                            else float(first_err.item(0)),
                            run=len(points),
                            name=name,
                            res_point=None if
                            (math.isnan(float(res.item(0))) or math.isinf(float(res.item(0)))) or
                            (math.isnan(float(res.item(1))) or math.isinf(float(res.item(1))))
                            else [float(res.item(0)), float(res.item(1))],
                            result_err=None
                            if math.isnan(float(lr.get_error_in_point(res).item(0)))
                               or math.isinf(float(lr.get_error_in_point(res).item(0)))
                            else isError(lr, points, run),
                            delta_time=delta,
                            math_iters=math_iters,
                        )
                        data.append(out)
    print("writing...")
    with open("../data/3analize_" + name + ".json", "w") as file:
        file.write(json.dumps([dataclasses.asdict(i) for i in data], indent=4))
    print("check")
    return data


start(sorted([0.0001, 0.0005, 0.00075, 0.001, 0.00125, 0.0025, 0.005,
              0.02]), 2)

# start(sorted([0.01, 0.025, 0.05, 0.05, 0.075, 0.1, 0.200, 1, 0.50, 5.00]), 1)

# start([0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 5, 20], 3)

# start([1, 2, 50, 100, 1000000], 0)


# 0.0001 , 0.00025, 0.0003 , 0.0005 , 0.0007 , 0.001
# start(sorted([0.00040, 0.0001, 0.00025, 0.0003, 0.0005, 0.0007, 0.001]), 4)

# start([i*0.0001 for i in range(1, 1000, 10)], 5)
