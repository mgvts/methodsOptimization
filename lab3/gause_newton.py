import numpy as np

from lab3.generate_no_linear_regression import generate_first_case

X, Y = generate_first_case()

R = np.append(X, Y, axis=1)
B = np.matrix([10, 5]).T
ITERS = 3000

cnt = 0
for i in range(ITERS):
    J = []
    b1 = B[0, 0]
    b2 = B[1, 0]
    RB = []
    eps = 0
    for u in R:
        x = u[0, 0]
        J.append(
            [
                1 / (b1 ** 2 + 2 * np.exp(x) * b1 * b2 + b2 ** 2 * np.exp(2 * x)),
                np.exp(x) / (np.exp(2 * x) * b2 ** 2 + 2 * np.exp(x) * b1 * b2 + b1 ** 2)
            ]
        )
        s = u[0, 1] - 1 / (b1 + b2 * np.exp(x))
        RB.append(s)
        eps += (s) ** 2
    if eps < 0.01:
        break
    print(eps)
    J = np.matrix(J)
    RB = np.matrix(RB).T
    B = B - 0.002 * np.linalg.inv(J.T * J) * J.T * RB
    cnt += 1
print(B)
print(cnt)
