import numpy as np
from matplotlib import pyplot as plt

from lab2.linear_regression import LinearRegression
import matplotlib as mpl

mpl.use('TkAgg')

count = 10

X = np.matrix(
    [[8.5416, 14.584339906038576], [8.2922, 17.078413778150818], [7.699, 23.00911519619328],
     [4.0431, 59.56759537565088], [3.0378, 69.62568726997998], [0.9025, 90.96851077788254],
     [0.6705, 93.29517799680318], [0.1951, 98.06759331746257], [0.0692, 99.28014323584033],
     [0.0251, 99.91427488174314], [0.016, 100.1163934115714], [0.0066, 99.78465061643551]]
)

Y = np.matrix(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
).T

R = np.matrix(
    [[0.3538, 8.7165], [0.6032, 8.4363], [1.0404, 7.6312], [2.7325, 4.1316], [2.9829, 3.0954], [4.7056, 1.0344], [4.8328, 0.7546], [6.3445, 0.0874], [7.23, 0.0856], [8.4027, -0.047], [8.6061, -0.0838], [9.7713, -0.162]]
)

count = len(Y)
print(count)

b = np.matrix([4., -1.]).transpose()

regression = LinearRegression(X, Y, b.copy(), count)

data = [
    # {
    #     'name': "Minibatch CD",
    #     'points': regression.stochastic_grad_down_points(alpha=0.001, runs=1000, eps=0.0001),
    #     'c': 'o-g',
    #     'mec': 'g'
    # },
    {
        "name": "Momentum",
        'points': regression.momentum_stochastic_grad_down_points(y=0.8, alpha=0.00001, runs=1000),
        'c': 'o-m',
        'mec': 'm'
    },
    # {
    #     "name": "Nesterov",
    #     'points': regression.nesterov_stochastic_grad_down_points(y=0.9, alpha=0.000001, runs=1000),
    #     'c': 'o-b',
    #     'mec': 'b'
    # },
    {
        "name": "Adagrad",
        'points': regression.adagrad_stochastic_grad_down_points(alpha=1, runs=1000),
        'c': 'v-c',
        'mec': 'c'
    },
    # {
    #     "name": "RMS",
    #     'points': regression.rms_stochastic_grad_down_points(W=10, alpha=0.1, runs=1000),
    #     'c': 'v-r',
    #     'mec': 'r'
    # },
    # {
    #     "name": "Adam",
    #     'points': regression.adam_stochastic_grad_down_points(b1=0.9, b2=0.9, alpha=0.01, runs=1000),
    #     'c': 'v-b',
    #     'mec': 'b'
    # }
]

plt.figure(figsize=(5, 5))

x = np.arange(-1, 5, 0.1)
y = np.arange(-3, 5, 0.1)
xgrid, ygrid = np.meshgrid(x, y)

z = 0 * xgrid * ygrid
r = LinearRegression(X, Y, b, count)

x_cnt = 0
for i in x:
    y_cnt = 0
    for j in y:
        m = r.get_grad_in_point(np.matrix([i, j]).T)
        z[y_cnt, x_cnt] = abs(m[0, 0] + m[1, 0])
        y_cnt += 1
    x_cnt += 1

plt.contour(xgrid, ygrid, z, colors='black', linewidths=0.2, levels=100)
for j in data:
    x = []
    y = []
    _b = j['points']
    for i in range(0, len(_b)):
        x.append(_b[i][0, 0])
        y.append(_b[i][1, 0])
    plt.plot(x, y, j['c'], alpha=1, label=j['name'], lw=0.5, mec=j['mec'], mew=1, ms=1)

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
        if eps < 0.1:
            break
        # if eps_prev < eps:
        #     raise ValueError
        eps_prev = eps
        J = np.matrix(J)
        RB = np.matrix(RB).T
        B = B - 0.01 *np.linalg.inv(J.T * J) * J.T * RB
        OUT.append(B)
    return OUT



def run2(B):
    eps_prev = 10000000
    OUT = [[B.copy(), B.copy(), B.copy()]]
    for i in range(200):
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
        if eps < 0.1:
            break
        # if eps_prev < eps:
        #     raise ValueError
        J = np.matrix(J)
        RB = np.matrix(RB).T

        c = 0.01
        bgn = c * np.linalg.inv(J.T * J) * J.T * RB
        bsd = J.T * RB
        t = c * np.linalg.norm(bsd) ** 2 / np.linalg.norm(J * bsd) ** 2

        _B = (B - bgn + B - t * bsd) / 2
        OUT.append([_B.copy(), B - bgn, B - t * bsd])
        B = _B
    return OUT

print(b)
B = run2(b.copy())
x = []
y = []
prev = None
for i in B:
    print(i[0])
    x.append(i[0][0, 0])
    y.append(i[0][1, 0])
    if prev:
        plt.plot([prev[0], i[1][0, 0]], [prev[1], i[1][1, 0]], 'o-m', alpha=0.5, label='0,0', lw=0.5, mec='c', mew=1, ms=1)
        plt.plot([prev[0], i[2][0, 0]], [prev[1], i[2][1, 0]], 'o-g', alpha=0.5, label='0,0', lw=0.5, mec='c', mew=1, ms=1)
        plt.plot([i[1][0, 0], i[2][0, 0]], [i[1][1, 0], i[2][1, 0]], 'o-b', alpha=0.5, label='0,0', lw=0.5, mec='c', mew=1, ms=1)

    prev = [i[0][0, 0], i[0][1, 0]]
plt.plot(x, y, 'v-b', alpha=1, label='0,0', lw=0.5, mec='c', mew=1, ms=1)


B = run(b)
x = []
y = []
for i in B:
    x.append(i[0, 0])
    y.append(i[1, 0])
plt.plot(x, y, 'v-m', alpha=1, label='0,0', lw=0.5, mec='c', mew=1, ms=1)

# plt.legend()

plt.show()
