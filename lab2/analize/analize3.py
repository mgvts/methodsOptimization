import dataclasses
import json
import math
from dataclasses import dataclass
from pprint import pprint
from random import randint

import numpy as np
import pandas as pd
from twod_line import generate_linear_regression_2d

from lab2.linear_regression import LinearRegression, get_batch_name


@dataclass
class OutputDTO:
    count: int
    alpha: float
    batch: int
    run: int
    name: str
    res_point: float
    first_err: float
    result_err: float
    batch_name: str


counts = [10 ** i for i in range(1, 4)]
alphas = [0.1, 0.01, 0.001, 0.3, 0.5, 0.7, 0.9]
alphas.sort()
print(f"{alphas=}")
runs = [10 ** i for i in range(1, 4)]
print(f"{runs=}")
shift = (-100, 100)
data = []

for count in counts:
    for batch in [i for i in range(1, count + 1, int(count // 10))] + [count]:
        print(f"{count=} {batch=}")
        X, Y = generate_linear_regression_2d(count, x=(-100, 100), shift=(-100, 100))
        start_point = np.matrix([float(randint(-10, 10)) for _ in range(2)]).transpose()
        lr = LinearRegression(X, Y, start_point, batch=batch)
        first_err = lr.get_error_in_point(start_point)
        methods = [lr.adagrad_stochastic_grad_down,
                   lr.adam_stochastic_grad_down,
                   lr.momentum_stochastic_grad_down,
                   lr.rms_stochastic_grad_down,
                   lr.nesterov_stochastic_grad_down]
        for alpha in alphas:
            for run in runs:
                if batch > count:
                    continue
                for method, name in zip(methods, ["adagrad",
                                                  "adam",
                                                  "momentum",
                                                  "rms",
                                                  "nesterov"]):
                    res = method(alpha=alpha, runs=run)
                    out = OutputDTO(
                        count=count,
                        alpha=alpha,
                        batch=batch,
                        batch_name=get_batch_name(count, batch),
                        first_err=None if
                        math.isnan(float(first_err.item(0))) or math.isinf(float(first_err.item(0)))
                        else float(first_err.item(0)),
                        run=run,
                        name=name,
                        res_point=None if
                        math.isnan(float(res.item(0))) or math.isinf(float(res.item(0)))
                        else float(res.item(0)),
                        result_err=None
                        if math.isnan(float(lr.get_error_in_point(res).item(0)))
                           or math.isinf(float(lr.get_error_in_point(res).item(0)))
                        else float(lr.get_error_in_point(res).item(0)),
                    )
                    data.append(out)


print("writing...")
with open("../data/1analize.json", "w") as file:
    file.write(json.dumps([dataclasses.asdict(i) for i in data], indent=4))
print("check")
