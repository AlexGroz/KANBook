"""Минимальная KAN 1 -> 1 без внешних зависимостей.

Демонстрирует ровно то, что описано в главе 4:
  * равномерные B-сплайны степени k на сетке из G интервалов;
  * функцию ребра phi(x) = w_b * SiLU(x) + sum(c_k * B_k(x)) + b;
  * обучение коэффициентов обычным градиентным спуском (Adam)
    по аналитическим градиентам MSE.

Запуск:  python3 kan_demo_pure.py
Время:   ~1 секунда на любом ноутбуке (чистый Python, без numpy).
"""

import math
import random

# --------------------------------------------------------------------------
# 1. B-сплайны
# --------------------------------------------------------------------------

G = 5          # число интервалов сетки
K = 3          # степень сплайна
LO, HI = -1.0, 1.0


def make_knots(grid_size=G, degree=K, lo=LO, hi=HI):
    """Равномерные «зажатые» узлы: degree+1 копия на концах."""
    knots = [lo] * (degree + 1)
    knots += [lo + (hi - lo) * i / grid_size for i in range(1, grid_size)]
    knots += [hi] * (degree + 1)
    return knots


def basis_row(knots, degree, x):
    """Значения всех базисных функций степени degree в точке x."""
    m = len(knots)
    # степень 0: индикаторы интервалов
    B = [1.0 if knots[i] <= x < knots[i + 1] else 0.0 for i in range(m - 1)]
    # правый конец включаем в последний ненулевой интервал
    if abs(x - knots[-1]) < 1e-12:
        B[m - degree - 2] = 1.0

    # рекурсия Кокса–де Бура
    for d in range(1, degree + 1):
        count = m - d - 1
        nxt = [0.0] * count
        for j in range(count):
            den1 = knots[j + d] - knots[j]
            den2 = knots[j + d + 1] - knots[j + 1]
            term1 = ((x - knots[j]) / den1) * B[j] if den1 > 1e-12 else 0.0
            term2 = ((knots[j + d + 1] - x) / den2) * B[j + 1] if den2 > 1e-12 else 0.0
            nxt[j] = term1 + term2
        B = nxt
    return B


def silu(x):
    if x < -700.0:
        return 0.0
    return x / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------
# 2. Данные: f(x) = sin(pi * x) на [-1, 1]
# --------------------------------------------------------------------------

N = 101
XS = [LO + (HI - LO) * i / (N - 1) for i in range(N)]
YS = [math.sin(math.pi * x) for x in XS]

KNOTS = make_knots()
N_BASIS = G + K
BASIS = [basis_row(KNOTS, K, x) for x in XS]

# проверка разбиения единицы — полезный санитарный тест
for row in BASIS:
    assert abs(sum(row) - 1.0) < 1e-9, "базисные функции не образуют разбиение единицы"


# --------------------------------------------------------------------------
# 3. Модель: один слой KAN 1 -> 1
# --------------------------------------------------------------------------

random.seed(0)
coef = [random.gauss(0.0, 0.1) for _ in range(N_BASIS)]  # коэффициенты сплайна
w_base = [1.0]                                           # вес SiLU (список — чтобы обновлять)
bias = [0.0]


def forward(x, row):
    spline = sum(c * b for c, b in zip(coef, row))
    return w_base[0] * silu(x) + spline + bias[0]


def mse():
    total = 0.0
    for x, target, row in zip(XS, YS, BASIS):
        diff = forward(x, row) - target
        total += diff * diff
    return total / N


# --------------------------------------------------------------------------
# 4. Adam по аналитическим градиентам
# --------------------------------------------------------------------------

LR = 0.05
STEPS = 4000
BETA1, BETA2 = 0.9, 0.999
EPS = 1e-8

# параметры и их моменты
params = [coef, w_base, bias]
grads = [[0.0] * N_BASIS, [0.0], [0.0]]
m = [[0.0] * N_BASIS, [0.0], [0.0]]
v = [[0.0] * N_BASIS, [0.0], [0.0]]

start_mse = mse()

for step in range(1, STEPS + 1):
    # обнуляем градиенты
    for g in grads:
        for i in range(len(g)):
            g[i] = 0.0

    # накапливаем dL/dtheta для MSE = (1/N) * sum (pred - y)^2
    for x, target, row in zip(XS, YS, BASIS):
        diff = forward(x, row) - target
        scale = 2.0 * diff / N
        base_val = silu(x)
        grads[1][0] += scale * base_val
        grads[2][0] += scale
        for k in range(N_BASIS):
            grads[0][k] += scale * row[k]

    # шаг Adam
    for group, grad_group, m_group, v_group in zip(params, grads, m, v):
        for i in range(len(group)):
            g = grad_group[i]
            m_group[i] = BETA1 * m_group[i] + (1 - BETA1) * g
            v_group[i] = BETA2 * v_group[i] + (1 - BETA2) * g * g
            m_hat = m_group[i] / (1 - BETA1 ** step)
            v_hat = v_group[i] / (1 - BETA2 ** step)
            group[i] -= LR * m_hat / (math.sqrt(v_hat) + EPS)


# --------------------------------------------------------------------------
# 5. Результат
# --------------------------------------------------------------------------

print("KAN 1 -> 1, без зависимостей")
print(f"сетка G = {G}, степень k = {K}, базисных функций = {N_BASIS}")
print(f"параметров: {N_BASIS} коэффициентов + 1 базовый вес + 1 смещение = {N_BASIS + 2}")
print(f"MSE до обучения: {start_mse:.6f}")
print(f"MSE после {STEPS} шагов: {mse():.8f}")
print()
print("  x      цель      предсказание")
for x in (-1.0, -0.5, 0.0, 0.5, 1.0):
    row = basis_row(KNOTS, K, x)
    print(f"{x:+.2f}   {math.sin(math.pi * x):+.5f}   {forward(x, row):+.5f}")
print()
print("коэффициенты сплайна:")
print("  " + ", ".join(f"{c:+.4f}" for c in coef))
