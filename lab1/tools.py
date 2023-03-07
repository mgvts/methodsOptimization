from decimal import Decimal
from random import randint, random, uniform

import numpy as np
import sympy as sp


class Func:
    """
        variables is [xi for i in range(number_of_variables)]
    """

    def __init__(self, number_of_variables: int, function_string, *args):
        sp.init_printing(use_unicode=True)
        self.sp_variables = sp.symbols("x:" + str(number_of_variables))
        self.string_variables = ["x" + str(i) for i in range(number_of_variables)]
        self.f = sp.sympify(function_string)

    def diff(self, variable):
        return self.f.diff(self.sp_variables[self.string_variables.index(variable)])

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

    """
        variable_value = [("x0", 1), ("x1", 2)]
        variable_value = value
    """

    def eval(self, variable_value):
        return self.f.subs(self._parse_arguments(variable_value))

    """
        without vectors like grad("0.5x^2 + bx + c") = ax + b
        without vectors like grad("x^2 + y^2") = 2x + 2y
    """

    def grad(self):
        result = sp.sympify("0")
        for xi in self.sp_variables:
            result += self.f.diff(xi)
        return Func(len(self.sp_variables), str(result))

    """
        variable_value = [("x0", 1), ("x1", 2)]
    """

    def metric(self, variable_value):
        v = list(sp.ordered(self.f.free_symbols))
        gradient = lambda ff, v: sp.Matrix([ff]).jacobian(v)
        g = gradient(self.f, v).subs(self._parse_arguments(variable_value))
        res = 0
        for i in g:
            res += i * i
        return sp.sqrt(res).evalf()

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

    def cond(self):
        lambdas = self.get_lamdas()
        print(f"{lambdas = }")
        L = max([i for i in lambdas])
        l = min([i for i in lambdas])
        print(f"{L = } {l = }")
        return L / l

    """
        :return list of tuples [(lambda, gamma(lambda)]
        gamma(lambda) its power in det(A - tE)
    """

    def get_lamdas(self):
        return list(dict(self.A.eigenvals()).keys())


# fall if mi == ma
# 3 3.(3), вроде пофиксил
def create_random_quadratic_func(n: int, k: float):
    if k < 1:
        raise AssertionError("k must be >= 1")

    def get_vector(mi, ma, size):
        print(f"{mi = } {ma = }")
        if mi == ma:
            return [mi for _ in range(size)]
        if int(ma) - int(mi) == 1:
            res = [random() + mi for _ in range(size)]
        else:
            res = [uniform(float(mi), float(ma)) for _ in range(size)]
        res[0] = mi
        res[-1] = ma
        return res

    def change_basis(diag_a):
        # 50 - constВ
        b = sp.Matrix(np.random.randint(0, 50, (n, n)))
        q, r = b.QRdecomposition()
        # q - ортоганальная
        a = q * diag_a * q.transpose()
        a: sp.Matrix
        print('After change basis, mi =', min(a.eigenvals().keys()), 'ma =', max(a.eigenvals().keys()))
        print('Current cond: ', a.condition_number(), 'Reference cond', k)
        return a

    # b and c must be anyone
    b = sp.Matrix([0 for _ in range(n)])
    c = 5
    if int(k) == 1:
        a_min = 1
    else:
        a_min = Decimal(randint(1, int(k) - 1))
    a_max = Decimal(k * a_min)
    a_max, a_min = max(a_max, a_min), min(a_max, a_min)

    return QFunc(n, change_basis(sp.diag(*get_vector(a_min, a_max, n))), b, c)


a = create_random_quadratic_func(5, 10)
print(a.f)
