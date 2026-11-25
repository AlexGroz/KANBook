"""Автотесты модели и API-логики (без запуска сервера).

Запуск: python3 code/test_predict.py
"""

import unittest

from serve_pure import predict


class PredictTest(unittest.TestCase):
    def test_versicolor(self):
        result = predict([5.9, 2.8, 4.3, 1.3])
        self.assertEqual(result["class"], "versicolor")
        self.assertGreater(result["confidence"], 0.99)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=6)

    def test_setosa(self):
        self.assertEqual(predict([5.0, 3.4, 1.5, 0.2])["class"], "setosa")

    def test_virginica(self):
        self.assertEqual(predict([6.5, 3.0, 5.5, 1.8])["class"], "virginica")

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            predict([99.0, 2.8, 4.3, 1.3])

    def test_wrong_length(self):
        with self.assertRaises(ValueError):
            predict([5.9, 2.8, 4.3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
