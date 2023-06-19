import numpy as np

R = np.matrix(
    [
        [0.535, 8.5416],
        [0.7225, 8.2922],
        [1.0948, 7.699],
        [2.6901, 4.0431],
        [3.132, 3.0378],
        [4.6131, 0.9025],
        [4.9355, 0.6705],
        [6.2199, 0.1951],
        [7.2687, 0.0692],
        [8.2892, 0.0251],
        [8.7415, 0.016],
        [9.6237, 0.0066]
    ]
)

B = np.matrix([0.1, 0]).T

for i in range(10):
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
    J = np.matrix(J)
    RB = np.matrix(RB).T

    bgn = np.linalg.inv(J.T * J) * J.T * RB
    bsd = J.T * RB
    t = np.linalg.norm(bsd) ** 2 / np.linalg.norm(J * bsd) ** 2

    a = (bgn + t * bsd) / 2
    B -= a

print(B)
