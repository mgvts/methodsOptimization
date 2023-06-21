import numpy as np
from matplotlib import pyplot as plt

R = np.matrix(
    [[0.3538, 8.7165], [0.6032, 8.4363], [1.0404, 7.6312], [2.7325, 4.1316], [2.9829, 3.0954], [4.7056, 1.0344], [4.8328, 0.7546], [6.3445, 0.0874], [7.23, 0.0856], [8.4027, -0.047], [8.6061, -0.0838], [9.7713, -0.162]]
)


def run(B):
    eps_prev = 10000000
    OUT = [B]
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
        OUT.append(B)
    return OUT



def run2(B):
    eps_prev = 10000000
    OUT = [[B.copy(), B.copy(), B.copy()]]
    for i in range(100):
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
        J = np.matrix(J)
        RB = np.matrix(RB).T

        c = 0.1
        bgn = c * np.linalg.inv(J.T * J) * J.T * RB
        bsd = J.T * RB
        t = c * np.linalg.norm(bsd) ** 2 / np.linalg.norm(J * bsd) ** 2

        _B = (B - bgn + B - t * bsd) / 2
        OUT.append([_B.copy(), B - bgn, B - t * bsd])
        B = _B
    return OUT


fig, ax = plt.subplots()
ax.plot(0.1, 0.01, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")

B = run2(np.matrix([0.3, 0]).T)
x = []
y = []
prev = None
for i in B:
    print(i[0])
    x.append(i[0][0, 0])
    y.append(i[0][1, 0])
    if prev:
        ax.plot([prev[0], i[1][0, 0]], [prev[1], i[1][1, 0]], 'o-m', alpha=0.5, label='0,0', lw=0.5, mec='c', mew=1, ms=1)
        ax.plot([prev[0], i[2][0, 0]], [prev[1], i[2][1, 0]], 'o-g', alpha=0.5, label='0,0', lw=0.5, mec='c', mew=1, ms=1)
        ax.plot([i[1][0, 0], i[2][0, 0]], [i[1][1, 0], i[2][1, 0]], 'o-b', alpha=0.5, label='0,0', lw=0.5, mec='c', mew=1, ms=1)

    prev = [i[0][0, 0], i[0][1, 0]]
ax.plot(x, y, 'v-b', alpha=1, label='0,0', lw=0.5, mec='c', mew=1, ms=1)


# ax.set_ylim(-0.1, 0.1)
# ax.set_xlim(-0.1, 1)
plt.show()
