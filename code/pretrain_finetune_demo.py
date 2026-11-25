"""Мини-эксперимент: pretrain на зашумлённых данных -> fine-tune.

Идея: модель сначала учится на нескольких зашумлённых версиях целевой
функции (это и есть «доменный претрейн»), затем дообучается на чистой цели.
Сравниваем с обучением с нуля за то же число шагов.

Запуск: python3 code/pretrain_finetune_demo.py
Время:  ~0.5 секунды на CPU.
"""

import math
import random

G, K = 8, 3
LO, HI = -1.0, 1.0
N = 81
XS = [LO + (HI - LO) * i / (N - 1) for i in range(N)]


def make_knots():
    t = [LO] * (K + 1)
    t += [LO + (HI - LO) * i / G for i in range(1, G)]
    t += [HI] * (K + 1)
    return t


def basis_row(knots, x):
    m = len(knots)
    B = [1.0 if knots[i] <= x < knots[i + 1] else 0.0 for i in range(m - 1)]
    if abs(x - knots[-1]) < 1e-12:
        B[m - K - 2] = 1.0
    for d in range(1, K + 1):
        count = m - d - 1
        nxt = [0.0] * count
        for j in range(count):
            den1 = knots[j + d] - knots[j]
            den2 = knots[j + d + 1] - knots[j + 1]
            t1 = ((x - knots[j]) / den1) * B[j] if den1 > 1e-12 else 0.0
            t2 = ((knots[j + d + 1] - x) / den2) * B[j + 1] if den2 > 1e-12 else 0.0
            nxt[j] = t1 + t2
        B = nxt
    return B


def silu(x):
    return x / (1.0 + math.exp(-x)) if x > -700 else 0.0


T = make_knots()
NB = G + K
BASIS = [basis_row(T, x) for x in XS]
BASE = [silu(x) for x in XS]


def target(x):
    return math.sin(math.pi * x)


class Model:
    """KAN 1 -> 1: phi(x) = w * SiLU(x) + spline(x) + b."""

    def __init__(self, seed):
        rng = random.Random(seed)
        self.coef = [rng.gauss(0.0, 0.1) for _ in range(NB)]
        self.w = 1.0
        self.b = 0.0

    def pred(self, i):
        return self.w * BASE[i] + self.b + sum(
            c * b for c, b in zip(self.coef, BASIS[i])
        )

    def mse(self, ys):
        return sum((self.pred(i) - ys[i]) ** 2 for i in range(N)) / N

    def step(self, ys, lr):
        grad_c = [0.0] * NB
        grad_w = 0.0
        grad_b = 0.0
        for i in range(N):
            d = 2.0 * (self.pred(i) - ys[i]) / N
            grad_w += d * BASE[i]
            grad_b += d
            for k in range(NB):
                grad_c[k] += d * BASIS[i][k]
        for k in range(NB):
            self.coef[k] -= lr * grad_c[k]
        self.w -= lr * grad_w
        self.b -= lr * grad_b


def main():
    rng = random.Random(0)
    clean = [target(x) for x in XS]
    noisy = [[target(x) + rng.gauss(0.0, 0.25) for x in XS] for _ in range(6)]

    # этап 1: доменный pretrain на зашумлённых версиях
    pre = Model(seed=1)
    for step in range(3000):
        pre.step(noisy[step % len(noisy)], lr=0.08)
    pre_start = pre.mse(clean)

    # этап 2: fine-tune на чистой цели
    for _ in range(200):
        pre.step(clean, lr=0.05)
    pre_after = pre.mse(clean)

    # контроль: обучение с нуля за те же 200 шагов
    fresh = Model(seed=2)
    fresh_start = fresh.mse(clean)
    for _ in range(200):
        fresh.step(clean, lr=0.05)
    fresh_after = fresh.mse(clean)

    print("clean target MSE")
    print(f"pretrained at start    : {pre_start:.8f}")
    print(f"fresh at start         : {fresh_start:.8f}")
    print(f"pretrained after 200 ft: {pre_after:.8f}")
    print(f"fresh after 200 ft     : {fresh_after:.8f}")
    print(f"improvement ratio      : {fresh_after / pre_after:.1f}x")


if __name__ == "__main__":
    main()
