import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sympy as sp
import math
import re


class Func:
    """
        variables is [xi for i in range(number_of_variables)]
    """
    def __init__(self, number_of_variables: int, function_string, *args):
        sp.init_printing(use_unicode=True)
        self.sp_variables = sp.symbols("x:" + str(number_of_variables))
        self.string_variables = ["x" + str(i) for i in range(number_of_variables)]
        print(f"{self.sp_variables = }")
        print(f"{self.string_variables = }")
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

    def __str__(self):
        return str(self.f)

