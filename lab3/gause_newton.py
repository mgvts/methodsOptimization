import numpy as np
from lab3.util import derivative

R = np.matrix(
    [
        [1.1345, 0.725, 3.734],
        [1.488, 57.547, 1.512],
        [2.3457, 0.74, 1.321],
        [1.3754, 0.457, 21.123]
    ]
)

B = np.matrix([1.999, 1.0, 5.1223]).T
# print(R * B)

for i in range(10):
    J = []
    for u in R:
        f = lambda i: (u * i)[0, 0]

        J.append([derivative(f, p[0, 0]) for p in B])
    J = np.matrix(J)

    r = np.linalg.matrix_rank(J)
    if r < J.shape[1]:
        print('failed: rg(J) =', r)
        break

    print(J)
    B = B - np.linalg.inv(J.T * J) * J.T * (R * B)
    print(B)
