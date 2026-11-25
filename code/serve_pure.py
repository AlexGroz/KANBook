"""Локальный HTTP-сервис для обученной Iris KAN. Без зависимостей.

Запуск:
    python3 code/serve_pure.py
    # http://127.0.0.1:8000/demo.html
    # POST http://127.0.0.1:8000/api/predict

Сервис отдаёт статические файлы книги и JSON API предсказания.
"""

import argparse
import json
import math
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from iris_kan_pure import basis_row, silu, softmax, FEATURES  # noqa: E402

MODEL_PATH = os.path.join(HERE, "iris_model.json")
with open(MODEL_PATH, encoding="utf-8") as f:
    MODEL = json.load(f)

MAX_BODY = 4096


def predict(values):
    """values: список из 4 чисел в сантиметрах."""
    if len(values) != len(FEATURES):
        raise ValueError(f"нужно {len(FEATURES)} признака")

    for i, value in enumerate(values):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("все признаки должны быть числами")
        lo, hi = MODEL["min"][i], MODEL["max"][i]
        if value < lo or value > hi:
            raise ValueError(
                f"{FEATURES[i]} = {value} вне диапазона [{lo}, {hi}]"
            )

    xn = [
        -1.0 + 2.0 * (values[i] - MODEL["min"][i]) / (MODEL["max"][i] - MODEL["min"][i])
        for i in range(len(values))
    ]
    basis = [basis_row(MODEL["knots"], MODEL["degree"], x) for x in xn]

    logits = []
    for k, _ in enumerate(MODEL["classes"]):
        z = MODEL["bias"][k]
        for i in range(len(FEATURES)):
            z += MODEL["base"][k][i] * silu(xn[i])
            for b, coef in enumerate(MODEL["coef"][k][i]):
                z += coef * basis[i][b]
        logits.append(z)

    probs = softmax(logits)
    best = max(range(len(probs)), key=lambda k: probs[k])
    return {
        "class": MODEL["classes"][best],
        "confidence": probs[best],
        "probabilities": dict(zip(MODEL["classes"], probs)),
        "normalized": dict(zip(FEATURES, xn)),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/model":
            meta = {
                "name": MODEL["name"],
                "version": MODEL["version"],
                "features": MODEL["features"],
                "classes": MODEL["classes"],
                "metrics": MODEL["metrics"],
                "notes": MODEL["notes"],
            }
            self._send_json(200, meta)
            return
        if path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/predict":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY:
            self._send_json(413, {"error": "слишком большой или пустой запрос"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            features = payload.get("features")
            if not isinstance(features, list):
                raise ValueError("ожидается поле features: [4 числа]")
            result = predict([float(v) for v in features])
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return

        self._send_json(200, result)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    parser = argparse.ArgumentParser(description="HTTP-сервис Iris KAN")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"сервис: http://{args.host}:{args.port}/demo.html")
    print(f"API:    POST http://{args.host}:{args.port}/api/predict")
    print("остановка: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлено")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
