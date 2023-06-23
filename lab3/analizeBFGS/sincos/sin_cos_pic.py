# создание двумерного массива значений функции
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from lab3.BFGS import bfgs
from lab3.analizeBFGS.grad import const_grad_down, wolfe_grad_down
from sincos import strangeFunc

mpl.use('TkAgg')

f, grad = strangeFunc()

maxIter = 10 ** 3
eps = 1e-4

start = np.array([-0.4, 0.3])
const_search_out = const_grad_down(f, grad, start, alpha=1e-3, eps=eps, max_iter=maxIter).points
print("const", const_search_out[-1], f(const_search_out[-1]))

wolfe_search_out = wolfe_grad_down(f, grad, start, eps=eps, max_iter=maxIter).points
print("wolf", wolfe_search_out[-1], f(wolfe_search_out[-1]))
bfgs_out = bfgs(f, grad, start, eps=eps, max_iter=maxIter).points
print("bfgs", bfgs_out[-1], f(bfgs_out[-1]))

compared_array_x = [i[0] for i in const_search_out] + [i[0] for i in wolfe_search_out] + [i[0] for i in bfgs_out]
compared_array_y = [i[1] for i in const_search_out] + [i[1] for i in wolfe_search_out] + [i[1] for i in bfgs_out]

print(f"{max(compared_array_x) = } {min(compared_array_x)} x")
print(f"{max(compared_array_y) = } {min(compared_array_y)} y")
print(f'{max([f(np.array([i, j])) for i, j in zip(compared_array_x, compared_array_y)])} \n'
      f'{min([f(np.array([i, j])) for i, j in zip(compared_array_x, compared_array_y)])}')

x = np.arange(-1, 0, 0.005)
y = np.arange(-0.6, 0.4, 0.005)
xgrid, ygrid = np.meshgrid(x, y)

z = f([xgrid, ygrid])
plt.contour(x, y, z)


# отображение точек и линий как в предыдущем примере
def concat_points(points, color, name):
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    plt.plot(x, y, marker='o', label=name, color=color,
             lw=0.5, mew=1, ms=1)


concat_points(const_search_out, 'red', 'const')
concat_points(wolfe_search_out, 'green', 'wolfe')
concat_points(bfgs_out, 'blue', 'bfgs')

plt.legend()

plt.title(f"start point{start}")
# отображение графика
print('end')
plt.show()
