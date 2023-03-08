from lab1.tools import generate_quadratic_func

import unittest
import random


class TestRandomQFunc(unittest.TestCase):
    def test_n(self):
        for i in range(10):
            n = random.randint(2, 10)
            a = generate_quadratic_func(n, 5)
            self.assertEqual(len(a.A), n ** 2)

    def test_k(self):
        for i in range(10):
            n = random.randint(2, 10)
            k = random.randint(1, 10)
            a = generate_quadratic_func(n, k)
            self.assertEqual(round(k, 2), round(a.cond(), 2))

    def test_k_float(self):
        for i in range(40):
            n = random.randint(2, 10)
            k = random.uniform(1, 10)
            a = generate_quadratic_func(n, k)
            self.assertLess(abs(a.cond() - k), 0.001)
