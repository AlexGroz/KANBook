# Код к KANBook

Все примеры рассчитаны на слабый CPU. Основной путь не требует numpy и PyTorch.

## Файлы

| Файл | Что делает | Зависимости |
| --- | --- | --- |
| `kan_demo_pure.py` | KAN 1 → 1: базис, forward, Adam, аналитические градиенты | нет |
| `kan_torch.py` | KAN на PyTorch: слои, grid extension, L1, символическая подгонка | torch >= 2.0 |
| `iris_kan_pure.py` | KAN 4 → 3 на Iris: CV, обучение, экспорт JSON/JS | нет |
| `pretrain_finetune_demo.py` | Warm start: pretrain на шуме → fine-tune | нет |
| `serve_pure.py` | HTTP-сервис: статика + `/api/predict` | нет |
| `test_predict.py` | 5 автотестов инференса | нет |
| `MODEL_CARD.md` | Модель-карточка: данные, метрики, ограничения | — |
| `Dockerfile` | Контейнер для сервиса | Docker |
| `data/iris.csv` | Датасет Iris (150 строк) | — |
| `iris_model.json` | Обученные веса и метрики | — |

## Быстрый старт

```bash
# математика KAN
python3 kan_demo_pure.py

# Iris: честная оценка и экспорт
python3 iris_kan_pure.py --cv --cv-epochs 1200
python3 iris_kan_pure.py --export --epochs 3000

# претрейн
python3 pretrain_finetune_demo.py

# тесты и сервис
python3 test_predict.py
python3 serve_pure.py
```

Сервис: <http://127.0.0.1:8000/demo.html>, API:
`POST /api/predict` с телом `{"features":[5.9,2.8,4.3,1.3]}`.

## Ожидаемые результаты

```
kan_demo_pure.py:        MSE 0.270003 -> 0.00000707, ~1.1 c
iris_kan_pure.py --cv:   CV accuracy 0.9533 ± 0.0267, ~12 c
iris_kan_pure.py --export: test accuracy 1.0000, ~6 c
pretrain_finetune_demo:  0.000290 vs 0.014658 после 200 шагов, ~0.4 c
test_predict.py:         5 tests OK
```

Числа зависят от seed и версии Python. Сравнивайте конфигурации при одинаковых
условиях.

## PyTorch

```bash
pip install -r requirements.txt
python3 kan_torch.py --task 1d --steps 1500
python3 kan_torch.py --task 2d --steps 2500
python3 kan_torch.py --task 1d --steps 3000 --extend
```

## Docker

```bash
docker build -f code/Dockerfile -t kan-iris .
docker run --rm -p 8000:8000 kan-iris
```

## Ограничения

- Iris — учебный набор; модель не является медицинской рекомендацией.
- Входы вне `[min, max]` отклоняются: KAN не умеет надёжно экстраполировать.
- Не загружайте pickle/torch-файлы из недоверенных источников.
