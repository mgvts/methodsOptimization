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


def run(B):
    eps_prev = 10000000
    for i in range(50):
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
        if eps < 0.001:
            break
        # if eps_prev < eps:
        #     raise ValueError
        eps_prev = eps
        J = np.matrix(J)
        RB = np.matrix(RB).T
        B = B - np.linalg.inv(J.T * J) * J.T * RB
    return i


x = np.arange(-10, 10, 0.1)
y = np.arange(-10, 10, 0.1)
X, Y = np.meshgrid(x, y)
Z = 0 * X * Y

cnt_x = 0
for i in x:
    cnt_y = 0
    for j in y:
        try:
            it = run(np.matrix([i, j]).T)
            if it == 49:
                it = -1
            else:
                if j < -0.025 and i < 0.2:
                    print(i, j)
        except:
            it = -1
        Z[cnt_y, cnt_x] = it
        cnt_y += 1
    cnt_x += 1

fig, ax = plt.subplots()
c = plt.pcolormesh(X, Y, Z, cmap='viridis', vmin=0., linewidths=1, snap=True)
ax.axis([X.min(), X.max(), Y.min(), Y.max()])
fig.colorbar(c, ax=ax)

plt.show()
