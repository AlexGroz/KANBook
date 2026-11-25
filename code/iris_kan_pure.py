"""KAN-классификатор Iris на чистом Python.

Обучает один слой KAN 4 -> 3 (B-сплайны + SiLU), оценивает на отложенной
выборке и в 5-фолдовой кросс-валидации, экспортирует веса в JSON и JS.

Запуск:
    python3 code/iris_kan_pure.py --cv
    python3 code/iris_kan_pure.py --export

Зависимости: только стандартная библиотека Python.
"""

import argparse
import csv
import json
import math
import os
import random
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "iris.csv")
ASSETS_JS = os.path.join(HERE, "..", "assets", "iris_model.js")
MODEL_JSON = os.path.join(HERE, "iris_model.json")

G, K = 5, 3
LO, HI = -1.0, 1.0
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


# --------------------------------------------------------------------------
# B-сплайны (те же формулы, что в книге)
# --------------------------------------------------------------------------

def make_knots(grid_size=G, degree=K, lo=LO, hi=HI):
    knots = [lo] * (degree + 1)
    knots += [lo + (hi - lo) * i / grid_size for i in range(1, grid_size)]
    knots += [hi] * (degree + 1)
    return knots


def basis_row(knots, degree, x):
    m = len(knots)
    B = [1.0 if knots[i] <= x < knots[i + 1] else 0.0 for i in range(m - 1)]
    if abs(x - knots[-1]) < 1e-12:
        B[m - degree - 2] = 1.0
    for d in range(1, degree + 1):
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
    if x < -700.0:
        return 0.0
    return x / (1.0 + math.exp(-x))


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


# --------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------

def load_rows(path=DATA):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            x = [float(r[name]) for name in FEATURES]
            rows.append((x, r["species"]))
    return rows


def stratified_split(rows, test_ratio=0.2, seed=0):
    rng = random.Random(seed)
    train, test = [], []
    for label in sorted({r[1] for r in rows}):
        part = [r for r in rows if r[1] == label]
        rng.shuffle(part)
        cut = int(round(len(part) * (1.0 - test_ratio)))
        train += part[:cut]
        test += part[cut:]
    return train, test


def fit_normalizer(rows):
    mins = [min(r[0][i] for r in rows) for i in range(len(FEATURES))]
    maxs = [max(r[0][i] for r in rows) for i in range(len(FEATURES))]
    return mins, maxs


def normalize(x, mins, maxs):
    return [LO + (HI - LO) * (x[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(x))]


def prepare(rows, mins, maxs, knots, classes):
    idx = {c: i for i, c in enumerate(classes)}
    S, Bv, Y = [], [], []
    for x, label in rows:
        xn = normalize(x, mins, maxs)
        Bv.append([silu(v) for v in xn])
        S.append([basis_row(knots, K, v) for v in xn])
        Y.append(idx[label])
    return S, Bv, Y
class KanClassifier:
    def __init__(self, n_in=4, n_out=3, nbasis=G + K, seed=1):
        rng = random.Random(seed)
        self.n_in, self.n_out, self.nb = n_in, n_out, nbasis
        self.coef = [[[rng.gauss(0.0, 0.1) for _ in range(nbasis)]
                      for _ in range(n_in)] for _ in range(n_out)]
        self.base = [[1.0] * n_in for _ in range(n_out)]
        self.bias = [0.0] * n_out

    def logits(self, basis, base_vals):
        out = []
        for k in range(self.n_out):
            z = self.bias[k]
            for i in range(self.n_in):
                z += self.base[k][i] * base_vals[i]
                z += sum(self.coef[k][i][b] * basis[i][b] for b in range(self.nb))
            out.append(z)
        return out

    def predict_proba(self, basis, base_vals):
        return softmax(self.logits(basis, base_vals))

    def train(self, S, Bv, Y, epochs=1500, lr=0.02, seed=1):
        n = len(Y)
        mc = [[[0.0] * self.nb for _ in range(self.n_in)] for _ in range(self.n_out)]
        vc = [[[0.0] * self.nb for _ in range(self.n_in)] for _ in range(self.n_out)]
        mb = [[0.0] * self.n_in for _ in range(self.n_out)]
        vb = [[0.0] * self.n_in for _ in range(self.n_out)]
        mB = [0.0] * self.n_out
        vB = [0.0] * self.n_out
        b1, b2, eps = 0.9, 0.999, 1e-8

        for ep in range(1, epochs + 1):
            gc = [[[0.0] * self.nb for _ in range(self.n_in)] for _ in range(self.n_out)]
            gb = [[0.0] * self.n_in for _ in range(self.n_out)]
            gB = [0.0] * self.n_out

            for s, base_vals, y in zip(S, Bv, Y):
                p = self.predict_proba(s, base_vals)
                for k in range(self.n_out):
                    d = (p[k] - (1.0 if k == y else 0.0)) / n
                    gB[k] += d
                    for i in range(self.n_in):
                        gb[k][i] += d * base_vals[i]
                        for b in range(self.nb):
                            gc[k][i][b] += d * s[i][b]

            for k in range(self.n_out):
                mB[k] = b1 * mB[k] + (1 - b1) * gB[k]
                vB[k] = b2 * vB[k] + (1 - b2) * gB[k] ** 2
                self.bias[k] -= lr * (mB[k] / (1 - b1 ** ep)) / (
                    math.sqrt(vB[k] / (1 - b2 ** ep)) + eps)
                for i in range(self.n_in):
                    mb[k][i] = b1 * mb[k][i] + (1 - b1) * gb[k][i]
                    vb[k][i] = b2 * vb[k][i] + (1 - b2) * gb[k][i] ** 2
                    self.base[k][i] -= lr * (mb[k][i] / (1 - b1 ** ep)) / (
                        math.sqrt(vb[k][i] / (1 - b2 ** ep)) + eps)
                    for b in range(self.nb):
                        mc[k][i][b] = b1 * mc[k][i][b] + (1 - b1) * gc[k][i][b]
                        vc[k][i][b] = b2 * vc[k][i][b] + (1 - b2) * gc[k][i][b] ** 2
                        self.coef[k][i][b] -= lr * (mc[k][i][b] / (1 - b1 ** ep)) / (
                            math.sqrt(vc[k][i][b] / (1 - b2 ** ep)) + eps)

    def evaluate(self, S, Bv, Y):
        correct = 0
        ce = 0.0
        conf = [[0] * self.n_out for _ in range(self.n_out)]
        for s, base_vals, y in zip(S, Bv, Y):
            p = self.predict_proba(s, base_vals)
            ce -= math.log(max(p[y], 1e-12))
            pred = max(range(self.n_out), key=lambda k: p[k])
            correct += pred == y
            conf[y][pred] += 1
        return {
            "accuracy": correct / len(Y),
            "ce": ce / len(Y),
            "confusion": conf,
            "n": len(Y),
        }
# --------------------------------------------------------------------------
# Кросс-валидация и экспорт
# --------------------------------------------------------------------------

def cross_validate(rows, folds=5, epochs=1500, seed=0):
    classes = sorted({r[1] for r in rows})
    by_class = {c: [r for r in rows if r[1] == c] for c in classes}
    rng = random.Random(seed)
    for c in classes:
        rng.shuffle(by_class[c])

    accs = []
    for fold in range(folds):
        train, test = [], []
        for c in classes:
            part = by_class[c]
            start = round(len(part) * fold / folds)
            end = round(len(part) * (fold + 1) / folds)
            test += part[start:end]
            train += part[:start] + part[end:]

        mins, maxs = fit_normalizer(train)
        knots = make_knots()
        S, Bv, Y = prepare(train, mins, maxs, knots, classes)
        St, Bt, Yt = prepare(test, mins, maxs, knots, classes)
        model = KanClassifier(len(FEATURES), len(classes), G + K, seed=seed + fold)
        model.train(S, Bv, Y, epochs=epochs)
        res = model.evaluate(St, Bt, Yt)
        accs.append(res["accuracy"])
        print(f"  фолд {fold + 1}: accuracy = {res['accuracy']:.4f} (n={res['n']})")

    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / len(accs)
    print(f"  CV accuracy: {mean:.4f} ± {math.sqrt(var):.4f}")
    return mean, math.sqrt(var), accs


def export_model(rows, train_rows, test_rows, epochs=3000, seed=0):
    classes = sorted({r[1] for r in rows})
    mins, maxs = fit_normalizer(train_rows)
    knots = make_knots()
    S, Bv, Y = prepare(train_rows, mins, maxs, knots, classes)
    St, Bt, Yt = prepare(test_rows, mins, maxs, knots, classes)

    model = KanClassifier(len(FEATURES), len(classes), G + K, seed=seed)
    model.train(S, Bv, Y, epochs=epochs)
    train_metrics = model.evaluate(S, Bv, Y)
    test_metrics = model.evaluate(St, Bt, Yt)
    print(f"  train accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  test  accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  confusion (test): {test_metrics['confusion']}")

    payload = {
        "name": "Iris KAN 4 -> 3",
        "version": "1.0",
        "features": FEATURES,
        "classes": classes,
        "min": mins,
        "max": maxs,
        "grid": G,
        "degree": K,
        "knots": knots,
        "coef": model.coef,
        "base": model.base,
        "bias": model.bias,
        "metrics": {
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_ce": test_metrics["ce"],
            "test_n": test_metrics["n"],
            "confusion": test_metrics["confusion"],
        },
        "notes": "Учебная модель. Не является медицинской рекомендацией.",
    }

    with open(MODEL_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    js = "window.IRIS_KAN_MODEL = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n"
    with open(ASSETS_JS, "w", encoding="utf-8") as f:
        f.write(js)

    print(f"  сохранено: {os.path.relpath(MODEL_JSON)}")
    print(f"  сохранено: {os.path.relpath(ASSETS_JS)}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="KAN на Iris (чистый Python)")
    parser.add_argument("--cv", action="store_true", help="5-фолдовая кросс-валидация")
    parser.add_argument("--export", action="store_true", help="обучить и сохранить модель")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--cv-epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows()
    print(f"данные: {len(rows)} примеров, {len(FEATURES)} признака, "
          f"{len({r[1] for r in rows})} класса")

    if args.cv:
        print("5-фолдовая кросс-валидация:")
        cross_validate(rows, folds=5, epochs=args.cv_epochs, seed=args.seed)

    if args.export:
        train, test = stratified_split(rows, 0.2, seed=args.seed)
        print(f"обучение: {len(train)} примеров, отложено: {len(test)}")
        export_model(rows, train, test, epochs=args.epochs, seed=args.seed)

    if not args.cv and not args.export:
        print("Укажите --cv и/или --export (см. --help).")


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"время: {time.time() - started:.2f} c")


