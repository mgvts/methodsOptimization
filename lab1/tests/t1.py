import unittest


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
