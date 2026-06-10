import unittest

import numpy as np

from mlp.network import MLP
from mlp.optimizers import adam


class AdamOptimizerTests(unittest.TestCase):
    def test_adam_updates_parameters(self):
        weights = [np.array([[0.2, -0.1]])]
        biases = [np.array([[0.0, 0.0]])]
        dW = [np.array([[0.1, -0.2]])]
        db = [np.array([[0.05, -0.03]])]

        state = None
        t = 0
        state, t = adam(weights, biases, dW, db, learning_rate=0.01, t=t, optimizer_state=state)

        self.assertIsNotNone(state)
        self.assertGreater(t, 0)
        self.assertTrue(np.all(np.isfinite(weights[0])))
        self.assertTrue(np.all(np.isfinite(biases[0])))

    def test_train_accepts_optimizer_argument(self):
        model = MLP([2, 2, 1])
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.eye(1, 1, k=0).repeat(2, axis=0)

        history = model.train(X, y, epochs=1, batch_size=2, learning_rate=0.01, optimizer='adam')

        self.assertIn('loss', history)
        self.assertIn('accuracy', history)


if __name__ == '__main__':
    unittest.main()
