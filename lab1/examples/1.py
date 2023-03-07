import numpy as np

from lab1.tools import Func

n = 2
stringFunc = "x0^2 + x1^2 + 10"
f = Func(2, stringFunc)
eps = 0.01
alpha = 0.1
x = np.random.randint(0, 10, (1, n))


def to_args(t):
    return [(f"x{i}", t[0][i]) for i in range(n)]


while True:
    # ||∇f(x)|| < ε
    if f.grad().eval(to_args(x)) < eps:
        break

    y = x - alpha * f.grad().eval(to_args(x))
    if f.eval(to_args(y)) < f.eval(to_args(x)):
        x = y

        # alpha = const in 1st task
        # alpha = alpha / 2
    print(x)
print(x)
print(f.eval(to_args(x)))
