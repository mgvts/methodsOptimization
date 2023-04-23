import json
from dataclasses import dataclass
from pprint import pprint
from random import randint

import numpy as np
import pandas as pd
from lab2.examples.linear_regression.twod_line import generate_linear_regression_2d

from lab2.linear_regression import LinearRegression


@dataclass
class OutputDTO:
    count: int
    alpha: float
    batch: int
    run: int
    name: str
    res_point: np.matrix
    first_err: np.matrix
    result_err: np.matrix
    batch_name: str


counts = [10 ** i for i in range(1, 4)]
print(f"{counts=}")
alpha = 0.001
run = 1000

data = []


def get_batch_name(count, batch):
    if batch == 1:
        return "SGD"
    if batch == count:
        return "GD"
    if count > batch > 1:
        return "Minibatch_GD"
    raise AssertionError("bad values")


for count in counts:
    for batch in [i for i in range(1, count + 1, int(count // 10))] + [count]:
        X, Y = generate_linear_regression_2d(count, x=(-100, 100), shift=(-100, 100))
        start_point = np.matrix([float(randint(-10, 10)) for _ in range(2)]).transpose()
        lr = LinearRegression(X, Y, start_point, batch=batch)
        first_err = lr.get_error_in_point(start_point)
        methods = [lr.adagrad_stochastic_grad_down,
                   lr.adam_stochastic_grad_down,
                   lr.momentum_stochastic_grad_down,
                   lr.rms_stochastic_grad_down,
                   lr.nesterov_stochastic_grad_down]
        first_err = lr.get_error_in_point(start_point)
        if (batch > count):
            continue
        for method, name in zip(methods, ["adagrad",
                                          "adam",
                                          "momentum",
                                          "rms",
                                          "nesterov"]):
            print(f"{count=} {batch=} {name=}")
            res = method(alpha=alpha, runs=run)
            out = OutputDTO(
                count=count,
                alpha=alpha,
                batch=batch,
                batch_name=get_batch_name(count, batch),
                first_err=first_err,
                run=run,
                name=name,
                res_point=res,
                result_err=lr.get_error_in_point(res),
            )
            data.append(out)

print("writing...")
with open("../data/1analize.json", "w") as file:
    for entry in data:
        file.write(json.dumps({
            "count": entry.count,
            "alpha": alpha,
            "batch": entry.batch,
            "batch_name": entry.batch_name,
            "first_err": entry.first_err.item(0),
            "run": entry.run,
            "name": entry.name,
            "res_point": entry.res_point.item(0),
            "result_err": entry.result_err.item(0),
        }))

print("check")
