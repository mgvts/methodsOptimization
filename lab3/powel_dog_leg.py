import numpy as np
from lab3.util import derivative

R = np.matrix(
    [
        [0.038, 0.050],
        [0.194, 0.127],
        [0.425, 0.094],
        [0.626, 0.2122],
        [1.253, 0.2729],
        [2.500, 0.2665],
        [3.740, 0.3317],
    ]
)

B = np.matrix([0.9, 0.2]).T

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
                -x / (b2 + x),
                (b1 * x) / (b2 + x) ** 2
            ]
        )
        s = u[0, 1] - (b1 * x) / (b2 + x)
        RB.append(s)
        eps += (u[0, 1] - s) ** 2

    print(eps)

    J = np.matrix(J)
    RB = np.matrix(RB).T
    r = np.linalg.matrix_rank(J)
    if r < J.shape[1]:
        print('failed: rg(J) =', r)
        break

    bgn = np.linalg.inv(J.T * J) * J.T * RB
    bsd = J.T * RB
    t = np.linalg.norm(bsd) ** 2 / np.linalg.norm(J * bsd) ** 2

    a = (bgn + t * bsd) / 2
    B -= a

print(B)
