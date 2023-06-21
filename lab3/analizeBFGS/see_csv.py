import numpy as np
import pandas as pd
from lab3.analizeBFGS.Rosenbrock.Rosenbrock import Rosenbrock
import re


def from_str_to_array(x: str) -> np.array:
    return np.array([float(i) for i in re.split("[, ]", x[1:-1]) if i != ''])


def read(path):
    df = pd.read_csv(path)
    print(df)
    df_grad = df[df['name'] == 'gradient_search']

    df_bfgs = df[df['name'] == 'bfgs']

    print(df_grad.head())
    print()
    print(df_grad['iters'].aggregate([np.mean]))
    print()
    print(df_bfgs.head())
    print()
    print(df_bfgs['iters'].aggregate([np.mean]))
    f, grad = Rosenbrock()
    df_bfgs['norm'] = df_bfgs['ans'].map(lambda x: np.linalg.norm(f(from_str_to_array(x))))
    print(df_bfgs.head())
    df_grad['norm'] = df_grad['ans'].map(lambda x: np.linalg.norm(f(from_str_to_array(x))))
    print(df_grad.head())


# read('D:\\3sem\\methodsOptimization\\lab3\\analizeBFGS\\Qfunc\\stat.csv')
read('D:\\3sem\\methodsOptimization\\lab3\\analizeBFGS\\Rosenbrock\\stat.csv')
