"""Полная минимальная реализация KAN на PyTorch (CPU-friendly).

Примеры:
    python kan_torch.py --task 1d --steps 1500
    python kan_torch.py --task 2d --steps 2000
    python kan_torch.py --task 1d --extend

Зависимости: torch >= 2.0 (CPU достаточно).
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# B-сплайны
# --------------------------------------------------------------------------

class BSplineBasis(nn.Module):
    """Равномерные «зажатые» B-сплайны степени degree на [lo, hi].

    На сетке из grid_size интервалов получается grid_size + degree
    базисных функций — ровно столько коэффициентов нужно каждому ребру.
    """

    def __init__(self, grid_size=5, degree=3, lo=-1.0, hi=1.0):
        super().__init__()
        self.grid_size = grid_size
        self.degree = degree
        self.lo = lo
        self.hi = hi
        self.register_buffer("knots", self._make_knots())

    def _make_knots(self):
        k, G = self.degree, self.grid_size
        knots = [self.lo] * (k + 1)
        knots += [self.lo + (self.hi - self.lo) * i / G for i in range(1, G)]
        knots += [self.hi] * (k + 1)
        return torch.tensor(knots, dtype=torch.float32)

    @property
    def n_basis(self):
        return self.grid_size + self.degree

    def forward(self, x):
        """x: (..., n_in) -> (..., n_in, n_basis)."""
        t = self.knots
        k = self.degree
        m = t.numel()
        shape = x.shape
        xf = x.reshape(-1)

        # степень 0
        B = ((xf[:, None] >= t[None, :-1]) & (xf[:, None] < t[None, 1:])).to(x.dtype)

        # правый конец включаем в последний ненулевой интервал
        at_right = torch.isclose(xf, t[-1])
        if at_right.any():
            B[at_right, m - k - 2] = 1.0

        # рекурсия Кокса–де Бура, векторизованная по точкам
        for d in range(1, k + 1):
            den1 = t[d:m - 1] - t[:m - d - 1]
            den2 = t[d + 1:m] - t[1:m - d]
            c1 = (xf[:, None] - t[None, :m - d - 1]) / den1.clamp_min(1e-12)
            c2 = (t[None, d + 1:m] - xf[:, None]) / den2.clamp_min(1e-12)
            c1 = torch.where(den1 > 1e-12, c1, torch.zeros_like(c1))
            c2 = torch.where(den2 > 1e-12, c2, torch.zeros_like(c2))
            B = c1 * B[:, :m - d - 1] + c2 * B[:, 1:m - d]

        return B.clamp_min(0.0).reshape(*shape, self.n_basis)


# --------------------------------------------------------------------------
# Слой и модель
# --------------------------------------------------------------------------

class KANLayer(nn.Module):
    def __init__(self, n_in, n_out, grid_size=5, degree=3, lo=-1.0, hi=1.0):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.basis = BSplineBasis(grid_size, degree, lo, hi)

        self.coef = nn.Parameter(torch.empty(n_out, n_in, self.basis.n_basis))
        nn.init.normal_(self.coef, mean=0.0, std=0.1)
        self.base_weight = nn.Parameter(torch.ones(n_out, n_in))
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.scale = 1.0 / math.sqrt(n_in)

    def forward(self, x):
        # x: (B, n_in)
        basis = self.basis(x)                              # (B, n_in, G+k)
        spline = torch.einsum("bik,oik->bo", basis, self.coef)
        base = F.silu(x) @ self.base_weight.T               # (B, n_out)
        return (base + spline) * self.scale + self.bias


class KAN(nn.Module):
    def __init__(self, widths, grid_size=5, degree=3, lo=-1.0, hi=1.0):
        super().__init__()
        self.widths = list(widths)
        self.layers = nn.ModuleList(
            KANLayer(widths[i], widths[i + 1], grid_size, degree, lo, hi)
            for i in range(len(widths) - 1)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def extend_grid(self, new_grid_size):
        """Перестроить сплайны на более тонкой сетке, сохранив функцию."""
        for layer in self.layers:
            old = layer.basis
            device = layer.coef.device
            xs = torch.linspace(old.lo, old.hi, 256, device=device)

            old_basis = old(xs)                                    # (S, n_in, nb_old)
            values = torch.einsum("sik,oik->soi", old_basis, layer.coef)  # (S, n_out, n_in)

            new_basis = BSplineBasis(
                new_grid_size, old.degree, old.lo, old.hi
            ).to(device)
            new_B = new_basis(xs)                                  # (S, n_in, nb_new)

            # Для каждого ребра решаем МНК: new_B @ c_new ≈ values
            A = new_B.permute(1, 0, 2)                             # (n_in, S, nb_new)
            Y = values.permute(2, 0, 1)                            # (n_in, S, n_out)
            solution = torch.linalg.lstsq(A, Y).solution            # (n_in, nb_new, n_out)

            layer.coef = nn.Parameter(solution.permute(2, 0, 1).contiguous())
            layer.basis = new_basis


# --------------------------------------------------------------------------
# Данные и обучение
# --------------------------------------------------------------------------

def make_data_1d(n=256):
    x = torch.linspace(-1.0, 1.0, n).unsqueeze(1)
    y = torch.sin(math.pi * x)
    return x, y


def make_data_2d(n=600, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=gen) * 2.0 - 1.0
    y = torch.exp(torch.sin(math.pi * x[:, :1]) + x[:, 1:2] ** 2)
    return x, y


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train(model, x, y, steps=1500, lr=1e-2, l1=0.0, log_every=300):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        model.train()
        pred = model(x)
        loss = F.mse_loss(pred, y)

        if l1 > 0.0:
            reg = sum(p.abs().mean() for p in model.parameters() if p.ndim == 3)
            loss = loss + l1 * reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0:
            with torch.no_grad():
                mse = F.mse_loss(model(x), y).item()
            print(f"  шаг {step:5d}   MSE = {mse:.7f}")

    return model


def symbolic_fit_1d(model, lo=-1.0, hi=1.0, points=200):
    """Подобрать простую формулу для выученного ребра 1 -> 1."""
    x = torch.linspace(lo, hi, points).unsqueeze(1)
    with torch.no_grad():
        y = model(x).squeeze(1)

    funcs = {
        "x": lambda z: z,
        "x^2": lambda z: z * z,
        "x^3": lambda z: z ** 3,
        "sin": torch.sin,
        "cos": torch.cos,
        "tanh": torch.tanh,
        "exp": torch.exp,
        "abs": torch.abs,
    }

    best = None
    b_values = torch.linspace(0.5, 4.0, 36)
    for name, fn in funcs.items():
        for b in b_values:
            z = fn(b * x.squeeze(1))
            A = torch.stack([z, torch.ones_like(z)], dim=1)
            sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution
            pred = (A @ sol).squeeze(1)
            mse = torch.mean((pred - y) ** 2).item()
            if best is None or mse < best[0]:
                best = (mse, name, float(b), float(sol[0, 0]), float(sol[1, 0]))
    return best


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Минимальная KAN на PyTorch")
    parser.add_argument("--task", choices=["1d", "2d"], default="1d")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--extend", action="store_true", help="grid extension: 3 -> 10")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.task == "1d":
        x, y = make_data_1d()
        widths = [1, 1]
    else:
        x, y = make_data_2d(seed=args.seed)
        widths = [2, 3, 1]

    print(f"задача: {args.task}, архитектура: {widths}, сетка G={args.grid}, k={args.degree}")

    if args.extend:
        model = KAN(widths, grid_size=3, degree=args.degree)
        print(f"параметров до расширения: {count_parameters(model)}")
        print("этап 1: грубая сетка G=3")
        train(model, x, y, steps=args.steps // 2, lr=args.lr, l1=args.l1)
        model.extend_grid(10)
        print(f"параметров после расширения: {count_parameters(model)}")
        print("этап 2: сетка G=10")
        train(model, x, y, steps=args.steps - args.steps // 2, lr=args.lr, l1=args.l1)
    else:
        model = KAN(widths, grid_size=args.grid, degree=args.degree)
        print(f"параметров: {count_parameters(model)}")
        train(model, x, y, steps=args.steps, lr=args.lr, l1=args.l1)

    with torch.no_grad():
        final_mse = F.mse_loss(model(x), y).item()
    print(f"итоговый MSE: {final_mse:.8f}")

    if args.task == "1d":
        mse, name, b, a, d = symbolic_fit_1d(model)
        print(f"символическая проверка: {a:+.3f} * {name}({b:.3f} * x) {d:+.3f}   (MSE {mse:.2e})")


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"время: {time.time() - started:.2f} c")

