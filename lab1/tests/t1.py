import unittest


# func, point, ans
tests_grad_in_point = [
    ('x0 + ln(x2^2 + x1^2)', [2, 1, 1], [1, 1, 1]), # https://xn--24-6kcaa2awqnc8dd.xn--p1ai/gradient-funkcii-v-tochke.html
    ('x0^2 + 2*x1*x2 + x2^2 + x3^2', [1, 1, 1], [4, 4, 2]), # https://math.semestr.ru/math/grad.php
    ('x0^2 + 2*x1*x2 + x2^2 + x3^2', [2, -1, 0], [2, 1, 0])
]

tests = [
    ('x1^2', [0]),
    ('x1^2 + 10', [0]),
    ('x1^4 + 10', [0]),
    ('x1^6 - 10', [0]),
    ('x1^6 + x2^2 - 10', [0, 0]),
    ('x1^6 + (x2 - 1)^2 - 10', [0, 1]),
    ('(x1 + 8)^6 + (x2 - 1)^2 - 10', [-8, 1]),
    ('(x1 + 8)^6 + (x2 - 1)^2 - 10', [-8, 1]),
    ('x1^2 * x2^2 - 10', [0, 0]),
    ('x1^2 * (x2+2)^2 - 10', [0, -2])
]

class TestRandomGradient(unittest.TestCase):
    def test_readable(self):
        pass
