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
    OUT = [B.copy()]
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
        J = np.matrix(J)
        RB = np.matrix(RB).T

        bgn = np.linalg.inv(J.T * J) * J.T * RB
        bsd = J.T * RB
        t = np.linalg.norm(bsd) ** 2 / np.linalg.norm(J * bsd) ** 2

        a = (bgn + t * bsd) / 2
        B -= a
        OUT.append(B.copy())
    return OUT


fig, ax = plt.subplots()
ax.plot(0.1, 0.01, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")

B = run(np.matrix([0.1, 0]).T)
x = []
y = []
for i in B:
    x.append(i[0, 0])
    y.append(i[1, 0])
ax.plot(x, y, 'v-m', alpha=1, label='0,0', lw=0.5, mec='c', mew=1, ms=1)


B = run2(np.matrix([0.1, 0]).T)
print(B)
x = []
y = []
for i in B:
    x.append(i[0, 0])
    y.append(i[1, 0])
ax.plot(x, y, 'v-b', alpha=1, label='0,0', lw=0.5, mec='c', mew=1, ms=1)

B = run(np.matrix([0.2, 0.01]).T)
x = []
y = []
for i in B:
    x.append(i[0, 0])
    y.append(i[1, 0])
ax.plot(x, y, 'v-m', alpha=1, label='0,0', lw=0.5, mec='c', mew=1, ms=1)

B = run2(np.matrix([0.2, 0.01]).T)
print(B)
x = []
y = []
for i in B:
    x.append(i[0, 0])
    y.append(i[1, 0])
ax.plot(x, y, 'v-b', alpha=1, label='0,0', lw=0.5, mec='c', mew=1, ms=1)


# ax.set_ylim(-0.1, 0.1)
# ax.set_xlim(-0.1, 1)
plt.show()
