## 1.
### Постановка задачи:
#### Реализуйте градиентный спуск с постоянным шагом (learning rate).

#### Описание решения:
Градиентный спуск с постоянным шагом - это метод оптимизации функции, который использует производную (градиент) функции для поиска минимума.
Шаг, с которым мы двигаемся в направлении градиента, остается постоянным на протяжении всего процесса оптимизации.

В коде, который приведен ниже, мы начинаем с некоторой начальной точки x.
Затем мы вычисляем градиент функции f в этой точке, используя метод grad(), и перемещаемся на некоторое расстояние в направлении антиградиента.
Расстояние, на которое мы перемещаемся, определяется постоянным шагом alpha.

Мы затем проверяем, достигнут ли критерий остановки, который устанавливает, что норма градиента должна быть меньше некоторого значения eps.
Если это условие не выполняется, мы повторяем процесс, начиная с новой точки y.

Градиентный спуск с постоянным шагом может быть эффективным методом оптимизации в тех случаях,
когда функция имеет единственный минимум и этот минимум достигается в пределах области определения.
Однако, если функция имеет множество локальных минимумов, градиентный спуск может "застрять" в одном из них.

```python
while True:
    y = x - alpha * f.grad(x)
    metr = get_metric2(f.grad(y) - f.grad(x))

    # ||∇f(x)|| < ε
    if metr < eps:
        break

    x = y
```

## 2.

### Постановка задачи:
#### Реализуйте метод одномерного поиска (<ins>метод дихотомии</ins>, метод Фибоначчи,метод золотого сечения) и градиентный спуск на его основе.

Для решения данной задачи мы выбрали метод дихотомии

Для реализации метода дихотомии была написана функция dichotomy, которая принимает на вход функцию f,
точность eps, дельту delta и начальные значения a и b.
Функция calc_min_iterations расчитывает необходимое количество итераций для достижения заданной точности.
Затем в цикле происходит сам процесс поиска оптимального значения шага.
Метод дихотомии представляет собой реализацию тернарного поиска

Для градиентного спуска была написана основная функция, в которой используется метод дихотомии для определения оптимального шага,
на который стоит сместиться при поиске точки.
Основной цикл повторяется до тех пор, пока норма градиента не станет меньше заданной точности eps.

```python
def dichotomy(f, eps=0.001, delta=0.00015, a=0, b=1):
    def calc_min_iterations():
        return sp.log((b - a - delta) / (2 * eps - delta), 2)

    N = math.ceil(calc_min_iterations())
    x1 = (a + b - delta) / 2
    x2 = (a + b + delta) / 2
    for i in range(N):
        # 1 step
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2

        # 2 step
        if f(x1) <= f(x2):
            b = x2
        else:
            a = x1

        # 3 step
        eps_i = (b - a) / 2
        if eps_i <= eps:
            break
    return (a + b) / 2

while True:
    alpha = dichotomy(lambda a: f.eval(x - a * f.grad(x),))

    y = x - alpha * f.grad(x)
    metr = get_metric2(f.grad(y) - f.grad(x))

    # ||∇f(x)|| < ε
    if metr < eps:
        break

    x = y
```

## 4.

Выбранные функции:

1. $x^2 + y^2 = 0$
2. $10x^2 + y^2 = 0$
3. $2x^2 + (y-3)^2 + 2x - 3y - 10 = 0$

### a) Исследование сходимости градиентного спуска с постоянным шагом

#### 1. Исследуем влияние коэффициента `alpha` на сходимость

Зафиксируем точку (x, y) и посмотрим для каких `alpha` алгоритм будет сходиться

Уравнение $20x^2 + y^2 = 0$, точка (20, 20)

| Значение `alpha` | Значение `eps` | Колличество итераций |
|------------------|:--------------:|---------------------:|
| 0.1              |     0.0001     |           Не сошелся |
| 0.2              |     0.0001     |           Не сошелся |
| 0.01             |     0.001      |                  332 |
| 0.01             |     0.0001     |                  446 |
| 0.02             |     0.0001     |                  239 |
| 0.03             |     0.0001     |                  165 |
| 0.04             |     0.0001     |                  126 |
| 0.05             |     0.0001     |           Не сошелся |
| 0.09             |     0.0001     |           Не сошелся |
| 0.001            |     0.0001     |                 3340 |
| 0.001            |    0.00001     |                 4491 |
| 0.04             |    0.00001     |                  154 |
| 0.04             |  0.0000000001  |                  292 |
| 0.001            |  0.0000000001  |                10241 |

Вывод 1: Чем меньше значение `alpha` тем больше нужно кол-во итераций для поска минимума точки

Вывод 2: Существует верхний порог `alpha` после которого метод перестает рассходиться

Вывод 3: Приближение к этому порогу даёт увеличение кол-ва итераций, в таблице значение `alpha=0.0499`

Вывод 4: Значение `eps` не влияет на сходимость метода

Вывод 5: Меньшее значение `eps` увеличивает кол-во итераций

Исследование влияния начальной точки на сходимость в `4 (c)`

#### Результаты для выбранных функций:

1. Уравнение $x^2 + y^2 = 0$, точка (10, 10)

| Значение `alpha` | Значение `eps` | Колличество итераций |
|------------------|:--------------:|---------------------:|
| 0.9999999        |     0.0001     |              >100000 |
| 0.9              |     0.0001     |                   60 |
| 0.8              |     0.0001     |                   27 |
| 0.5              |     0.0001     |                    2 |
| 0.001            |     0.0001     |                 3167 |
| 0.001            |   0.00000001   |                 7768 |
| 0.5              |   0.00000001   |                    2 |

2. Уравнение $10x^2 + y^2 = 0$, точка (10, 10)

| Значение `alpha` | Значение `eps` | Колличество итераций |
|------------------|:--------------:|---------------------:|
| 0.9              |     0.0001     |           Не сошелся |
| 0.1              |     0.0001     |           Не сошелся |
| 0.099999         |     0.0001     |              >100000 |
| 0.01             |     0.0001     |                  412 |
| 0.02             |     0.0001     |                  222 |
| 0.001            |     0.0001     |                 2994 |
| 0.001            |   0.0000001    |                 6445 |
| 0.02             |   0.0000001    |                  391 |

Вывод 6: Чем больше обусловленность функции, тем меньшее значение `alpha` нужно

3. Уравнение $2x^2 + (y-3)^2 + 2x - 3y - 10 = 0$, точка (10, 10)

| Значение `alpha` | Значение `eps` | Колличество итераций |
|------------------|:--------------:|---------------------:|
| 0.9              |     0.0001     |           Не сошелся |
| 0.1              |     0.0001     |                   46 |
| 0.2              |     0.0001     |                   22 |
| 0.4              |     0.0001     |                   28 |
| 0.5              |     0.0001     |           Не сошелся |
| 0.49             |     0.0001     |                 3404 |
| 0.001            |     0.0001     |                 2994 |
| 0.001            |   0.0000001    |                 6146 |
| 0.02             |   0.0000001    |                  376 |

### b) Исследование эффективности градиентного спуска с одномерным поиском

Кол-во вычислений градиента для постоянного шага = кол-во итераций * 3

Кол-во вычислений градиента с одномерным поиском = кол-во итераций * 3 + кол-во при вычислении в дихтоми

| Функция                                | `alpha` / Кол-во вычислений с постоянным шагом | Кол-во вычислений с дихтоми |
|----------------------------------------|:----------------------------------------------:|----------------------------:|
| $x^2 + y^2 = 0$                        |                  0.01 / 1338                   |                          39 |
| $x^2 + y^2 = 0$                        |                   0.04 / 338                   |                          39 |
| $10x^2 + y^2 = 0$                      |                   0.02 / 666                   |                         156 |
| $10x^2 + y^2 = 0$                      |                  0.001 / 8982                  |                         156 |
| $2x^2 + (y-3)^2 + 2x - 3y - 10 = 0$    |                  0.001 / 8982                  |                         117 |
| $2x^2 + (y-3)^2 + 2x - 3y - 10 = 0$    |                  0.02 / 1128                   |                         117 |

Вывод 7: Алгоритм с одномерным поиском работает быстрее и требует меньше вычислений

Вывод 8: Алгоритм с одномерным поиском находит лучшие alpha на каждом шагу

Вывод 9: Алгоритм с одномерным поиском требует `sp.log((b - a - delta) / (2 * eps - delta), 2)` дополнительных операций
вычисления градиента
в отличие от обычного алгоритма со спуском

### c) Исследование работы методов в зависимости от начальной точки

### d) Исследование нормализации на сходимость и примере масштабирования осей

## 5.

Алгоритм генерации:

```python3

def generate_quadratic_func(n: int, k: float) -> QFunc:
    """
    :param n: Count of vars
    :param k: Number of cond
    :return: Random QFunc
    """

    if k < 1:
        raise AssertionError("k must be >= 1")
    # если n == 1, то mi = ma, и k всегда = 1
    if n == 1:
        raise AssertionError("n must be > 1")

    # 1. генерируем диагональную матрицу, diag(a_min ... a_max)
    # генерируем a_min, a_max
    if int(k) == 1:
        a_min = Decimal(1)
    else:
        a_min = Decimal(randint(1, int(k) - 1))
    a_max = Decimal(Decimal(k) * a_min)
    a_max, a_min = max(a_max, a_min), min(a_max, a_min)
    v = [float(a_min)] + [uniform(float(a_min), float(a_max)) for _ in range(n - 2)] + [float(a_max)]
    A = sp.diag(*v)

    # A - диагональная матрица с числом обусловленности k
    # A - уже квадратичная форма. в каноническом виде
    # 2. любая квадратичная форма приводится к каноническому виду, с помощью ортоганального преобразования
    #   Q^(T) * B * Q = A
    #   Q - ортоганальная матрица -> Q^(-1) = Q^(T)
    #   тогда B = Q * A * Q^(T)
    #   нужно сгенерировтаь ортоганальную матрицу
    # Note важно тк монла получиться матрица из 0 и это хуйня
    C = sp.Matrix(np.random.randint(1, INF, (n, n)))
    Q, R = C.QRdecomposition()

    B = Q * A * Q.transpose()

    return QFunc(n, B, sp.Matrix([0 for _ in range(n)]), 5)
```

1. Генерируем диагональную матрицу `A` с собственными числами. Такими что `a_max / a_min = k`
2. Генерируем рандомную матрицу `B`
3. С помощью `QR` разложения получаем из сгенерированной матрицы `B`, рандомную ортоганальную митрицу Q
4. Меняем базис матрицы `A` с помощью ортоганального преобразования `Q * A * Q^T`


##8. (Доп задание)

### Постановка задачи:
#### Реализуйте градиентный спуск с постоянным шагом (learning rate).