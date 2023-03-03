# Квадратичная функция:
# f(x) = 1/2 * x^T * A * X
# или f(x) = 1/2 * x^T * A * X - b ^ T * x
# число обусловленности k = a_max(A) / a_min(A) >= 1 a - собственное число
# A^T = A
# Всегда есть собственные числа?

import random
import math
from decimal import Decimal

k = 10
n = 10
# k = Decimal(input())
# n = int(input())

assert n >= 2

a_max = Decimal(random.randint(math.ceil(k / 2), k))
a_min = Decimal(k / a_max)

print(a_max, a_min)

# Сгенерировали A_diag
# B * A_diag * B^(-1) = C
# B * A_diag * B^(-1) = C^T
# C - квадратичная матрица
# Нужно сгенерировать B, C
# Думаю, что надо сгенерировать B, с условиями
# 1. B*B^(-1) = E
# 2. B * A_diag * B^(-1) = (B * A_diag * B^(-1))^T
#                          (B * A_diag * B^(-1))^T = B^(-1)^T * A_diag * B^T
