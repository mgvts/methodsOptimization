from decimal import Decimal
from random import randint, uniform

import numpy as np
import sympy as sp

INF = 1_000_000


class Func:
    """
        variables is [xi for i in range(number_of_variables)]
    """

    def __init__(self, number_of_variables: int, function_string, *args):
        sp.init_printing(use_unicode=True)
        self.f = sp.sympify(function_string)
        if len(self.f.free_symbols) != number_of_variables:
            raise AssertionError("n must be equals to amount of different variables")
        self.n = number_of_variables
        self.sp_variables = sp.symbols("x:" + str(number_of_variables))
        self.string_variables = ["x" + str(i) for i in range(number_of_variables)]
        self.v = list(sp.ordered(self.f.free_symbols))
        self.g = self._grad()

    def diff(self, variable):
        return self.f.diff(self.sp_variables[self.string_variables.index(variable)])

    def get_n(self):
        return self.n

    """
        [("x0", 1), ("x1", 2)] ->  [(x0, 1), (x1, 2)]
        where x1 x2 is sp.symbols
    """

    def _parse_arguments(self, l):
        res = [(self.sp_variables[self.string_variables.index(variable)], value) for variable, value in l]
        return res

    # unused
    # def _parse_function_string(self, s:str):
    #     import re
    #     delimiters = "^", "+", "(c)", " ", "\n", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "**", "sin", "cos", "(", ")", "e"
    #     regex_pattern = '|'.join(map(re.escape, delimiters))
    #     return [i for i in re.split(regex_pattern, s) if i != ""]

    def eval(self, variable_value):
        if type(variable_value) == sp.Matrix:
            self.f.subs(self._parse_arguments(to_args(variable_value, self.n)))
        return self.f.subs(self._parse_arguments(variable_value))

    def grad(self, variable_value) -> sp.Matrix:
        """
        :param variable_value: point where we calculating function
        # [("x0", 1), ("x1", 2)]
        :return: vector of meaning grad in point
        """
        return self.g.subs(self._parse_arguments(variable_value))

    def _grad(self):
        gradient = lambda ff, v: sp.Matrix([ff]).jacobian(v)
        g = gradient(self.f, self.v)
        return g

    def metric_of_gradient_in_point(self, variable_value) -> float:
        """
        :param variable_value: point where we calculating function
        # [("x0", 1), ("x1", 2)]
        :return: ||∇f(x)||
        """
        return get_metric2(self.grad(variable_value))

    def __str__(self):
        return str(self.f)


class QFunc:
    """
        Quadratic function
    """

    def __init__(self, n, A, b, c):
        sp.init_printing(use_unicode=True)
        if A is None:
            self.A = sp.eye(n)
            b = sp.Matrix([0 for i in range(n)])
            c = 0
        else:
            self.A = A
        self.list_variables = sp.symbols("x:" + str(n))
        self.v = sp.Matrix(list(sp.ordered(self.list_variables)))
        self._createFunc(n, self.A, b, c)
        self.numpy_f = sp.lambdify(self.f.free_symbols, self.f, 'numpy')

    def _createFunc(self, n, A, b, c):
        self.f = sp.sympify(0)
        for i in range(n):
            for j in range(n):
                self.f += A[i, j] * self.v[i] * self.v[j]
        for i in range(n):
            self.f += b[i] * self.v[i]
        self.f += c

    def __str__(self):
        return str(self.f)

    def get_A(self):
        return self.A

    def cond(self):
        lambdas = self.get_lamdas()
        # print(f"{lambdas = }")
        L = max(lambdas)
        l = min(lambdas)
        # print(f"{L = } {l = }")
        return L / l

    """
        :return list of tuples [(lambda, gamma(lambda)]
        gamma(lambda) its power in det(A - tE)
    """

    def get_lamdas(self):
        return list(dict(self.A.eigenvals()).keys())


class FastQFunc:
    """
        Quadratic function with numpy
    """

    def __init__(self, n, A: np.matrix, b: np.Matrix, c):
        sp.init_printing(use_unicode=True)
        self.A = A
        self.b = b
        self.c = c
        # self.f = A*x*x + bx + c
        # self.grad = Ax + b

    def grad_in_point(self, point: np.Matrix):
        return np.dot(self.A, point) + self.b

    def eval(self, point: np.Matrix):
        return np.dot(np.matmul(self.A, point), point) + np.dot(self.b, point) + self.c

    def cond(self):
        lambdas = self.get_lamdas()
        L = max(lambdas)
        l = min(lambdas)
        return L / l

    """
        :return list of tuples [(lambda, gamma(lambda)]
        gamma(lambda) its power in det(A - tE)
    """

    def get_lamdas(self):
        return list(dict(self.A.eigenvals()).keys())


def generate_quadratic_func(n: int, k: float) -> QFunc:
    """
    :param n: Count of vars
    :param k: Number of cond
    :return: Random QFunc
    """

    if k < 1:
        raise AssertionError("k must be >= 1")
    # если n == 1, то mi = ma, и k всегда = 1
    if n == 1:
        raise AssertionError("n must be > 1")

    # 1. генерируем диагональную матрицу, diag(a_min ... a_max)
    # генерируем a_min, a_max
    if int(k) == 1:
        a_min = Decimal(1)
    else:
        a_min = Decimal(randint(1, int(k) - 1))
    a_max = Decimal(Decimal(k) * a_min)
    a_max, a_min = max(a_max, a_min), min(a_max, a_min)
    v = [float(a_min)] + [uniform(float(a_min), float(a_max)) for _ in range(n - 2)] + [float(a_max)]
    A = sp.diag(*v)

    # A - диагональная матрица с числом обусловленности k
    # A - уже квадратичная форма. в каноническом виде
    # 2. любая квадратичная форма приводится к каноническому виду, с помощью ортоганального преобразования
    #   Q^(T) * B * Q = A
    #   Q - ортоганальная матрица -> Q^(-1) = Q^(T)
    #   тогда B = Q * A * Q^(T)
    #   нужно сгенерировтаь ортоганальную матрицу
    # Note важно тк монла получиться матрица из 0 и это хуйня
    C = sp.Matrix(np.random.randint(1, INF, (n, n)))
    Q, R = C.QRdecomposition()

    B = Q * A * Q.transpose()

    return QFunc(n, B, sp.Matrix([0 for _ in range(n)]), 5)


def fast_generate_quadratic_func(n: int, k: float) -> QFunc:
    """
    :param n: Count of vars
    :param k: Number of cond
    :return: Random QFunc
    """

    if k < 1:
        raise AssertionError("k must be >= 1")
    # если n == 1, то mi = ma, и k всегда = 1
    if n == 1:
        raise AssertionError("n must be > 1")

    # 1. генерируем диагональную матрицу, diag(a_min ... a_max)
    # генерируем a_min, a_max
    if int(k) == 1:
        a_min = Decimal(1)
    else:
        a_min = Decimal(randint(1, int(k) - 1))
    a_max = Decimal(Decimal(k) * a_min)
    a_max, a_min = max(a_max, a_min), min(a_max, a_min)
    v = [float(a_min)] + [uniform(float(a_min), float(a_max)) for _ in range(n - 2)] + [float(a_max)]
    A = np.diag(v)

    # A - диагональная матрица с числом обусловленности k
    # A - уже квадратичная форма. в каноническом виде
    # 2. любая квадратичная форма приводится к каноническому виду, с помощью ортоганального преобразования
    #   Q^(T) * B * Q = A
    #   Q - ортоганальная матрица -> Q^(-1) = Q^(T)
    #   тогда B = Q * A * Q^(T)
    #   нужно сгенерировтаь ортоганальную матрицу
    # Note важно тк монла получиться матрица из 0 и это хуйня
    C = np.matrix(np.random.randint(1, INF, (n, n)))
    Q, R = np.linalg.qr(C)
    # Q, R = C.QRdecomposition()

    B = np.matmul(np.matmul(Q, A), Q.transpose())

    return QFunc(n, B, np.matrix([0 for _ in range(n)]), 5)


def to_args(t, n):
    return [(f"x{i}", t[i]) for i in range(n)]


def get_metric2(x: sp.Matrix):
    res = 0
    for i in x:
        res += i * i
    return sp.sqrt(res).evalf()

def get_metric3(x: np.matrix):
    res = 0
    for i in x:
        res += i * i
    return np.sqrt(res)
