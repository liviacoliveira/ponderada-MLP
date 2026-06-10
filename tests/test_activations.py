import unittest

import numpy as np

from mlp.activations import relu, relu_derivative, softmax


class ActivationTests(unittest.TestCase):
    def test_relu_returns_zero_for_negative_values(self):
        x = np.array([-3.0, -1.0, 0.0, 2.0, 4.5])

        result = relu(x)

        expected = np.array([0.0, 0.0, 0.0, 2.0, 4.5])
        np.testing.assert_allclose(result, expected)

    def test_relu_derivative_is_one_for_positive_values(self):
        x = np.array([-2.0, 0.0, 0.1, 3.0])

        result = relu_derivative(x)

        expected = np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_softmax_outputs_probabilities_that_sum_to_one(self):
        x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])

        result = softmax(x)

        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.all(result >= 0))
        np.testing.assert_allclose(np.sum(result, axis=1), np.ones(result.shape[0]), atol=1e-7)

    def test_softmax_is_stable_for_large_logits(self):
        x = np.array([[1000.0, 1000.0, 1000.0]])

        result = softmax(x)

        expected = np.full((1, 3), 1.0 / 3.0)
        np.testing.assert_allclose(result, expected, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
