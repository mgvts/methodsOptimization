import unittest

from lab1.grad import grad_down
from lab1.tools import Func

# func, point, ans
tests_grad_in_point = [
    (3, 'x0 + ln(x2^2 + x1^2)', [2, 1, 1], [1, 1, 1]),
    # https://xn--24-6kcaa2awqnc8dd.xn--p1ai/gradient-funkcii-v-tochke.html
    (3, 'x0^2 + 2*x0*x1 + x1^2 + x2^2', [1, 1, 1], [4, 4, 2]),  # https://math.semestr.ru/math/grad.php
    (3, 'x0^2 + 2*x0*x1 + x1^2 + x2^2', [2, -1, 0], [2, 1, 0])
    # a очно праивльный тест? из калькулятора  grad(u)=(2·x+2·y)·i + (2·x+2·y)·j + 2·z·k
]

tests_s = [
    # (1, 'x0^2', [0]),
    # (1, 'x0^2 + 10', [0]),
    # (1, 'x0^4 + 10', [0]),
    # (1, 'x0^6 - 10', [0]),
    # (2, 'x0^4 + x1^2 - 10', [0, 0]),
    # (2, 'x0^6 + (x1 - 1)^2 - 10', [0, 1]),
    # (2, '(x0 + 8)^6 + (x1 - 1)^2 - 10', [-8, 1]),
    # (2, '(x0 + 8)^6 + (x1 - 1)^2 - 10', [-8, 1]),
    (2, 'x0^2 * x1^2 - 10', [0, 0]),
    (2, 'x0^2 * (x1+2)^2 - 10', [0, -2])
]


def to_args(t, n):
    return [(f"x{i}", t[i]) for i in range(n)]


class TestRandomGradient(unittest.TestCase):
    def test_readable(self):
        for n, s, m, a in tests_grad_in_point:
            f = Func(n, s)
            self.assertEqual(list(f.grad(to_args(m, n))), a)

    # долго
    def test_readable_grad(self):
        for n, s, a in tests_s:
            y = list(grad_down(n, s))
            print(s, y)
            for i in range(n):
                self.assertLess(abs(a[i] - y[i]), 0.5)
