import numpy as np
from matplotlib import pyplot as plt

from lab3.generate_no_linear_regression import generate_first_case

X, Y = generate_first_case()

R = np.append(X, Y, axis=1)



def run(B):
    eps_prev = -1
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
        if eps < 0.01:
            break
        # if eps_prev < eps and eps_prev != -1:
        #     raise ValueError
        J = np.matrix(J)
        RB = np.matrix(RB).T

        bgn = np.linalg.inv(J.T * J) * J.T * RB
        bsd = J.T * RB
        t = np.linalg.norm(bsd) ** 2 / np.linalg.norm(J * bsd) ** 2

        a = (bgn + t * bsd) / 2
        B -= a
    return i


x = np.arange(-0.1, 1, 0.01)
y = np.arange(-0.1, 0.1, 0.01)
#
# x = np.arange(-0.1, 1, 0.03)
# y = np.arange(-0.1, 0.1, 0.01)
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
