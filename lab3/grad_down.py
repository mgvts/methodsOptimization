import numpy as np
from matplotlib import pyplot as plt

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


def make_data():
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1)
    xgrid, ygrid = np.meshgrid(x, y)

    g_1 = lambda b2, b1, i: 1 / (b1 ** 2 + 2 * np.exp(i[0, 0]) * b1 * b2 + b2 ** 2 * np.exp(2 * i[0, 0]))
    g_2 = lambda b2, b1, i: np.exp(i[0, 0]) / (np.exp(2 * i[0, 0]) * b2 ** 2 + 2 * np.exp(i[0, 0]) * b1 * b2 + b1 ** 2)

    z = 0

    for i in R:
        z += abs(g_1(xgrid, ygrid, i) + g_2(xgrid, ygrid, i)) ** 2

    return xgrid, ygrid, z


# x, y, z = make_data()
# plt.contour(x, y, z)
# plt.show()

# y = 1/(b1 + b2 * e ^ x)
# y(b1 + b2 * e ^ x) = 1
# yb1 + b2 * e ^ x = 1

X = []
Y = []
for i in R:
    x = i[0, 0]
    y = i[0, 1]
    X.append([y, y * np.exp(x)])
    Y.append(1)

print(X)
print(Y)