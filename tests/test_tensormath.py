import numpy as np
import tensorflow as tf
import alphai_cromulon_oracle.tensormaths as tm
np.random.seed(42)


class TestTensormath(tf.test.TestCase):

    def test_selu(self):
        parameters = ([10, 2.0, 1.0, 0.1, 0.0, -0.1, -1.0],
                      [10.50700987, 2.10140197, 1.05070099, 0.1050701, 0., - 0.16730527, - 1.11133074])

        with self.test_session():
            actual_result = tm.selu(np.asarray(parameters[0])).eval()
            self.assertArrayNear(actual_result, parameters[1], err=1e-5)

    def test_inv_selu(self):
        parameters = ([10, 2.0, 1.0, 0.1, 0.0, -0.1, -1.0],
                      [10.50700987, 2.10140197, 1.05070099, 0.1050701, 0., - 0.16730527, - 1.11133074])

        with self.test_session():
            actual_result = tm.inv_selu(np.asarray(parameters[1])).eval()
            self.assertArrayNear(actual_result, parameters[0], err=1e-5)

    def test_kelu(self):
        parameters = ([10, 2.0, 1.0, 0.1, 0.0, -0.1, -1.0],
                      [30., 6., 3., 0.3, 0., -0.03333333, -0.33333333])

        with self.test_session():
            actual_result = tm.kelu(np.asarray(parameters[0])).eval()
            self.assertArrayNear(actual_result, parameters[1], err=1e-5)

    def test_inv_kelu(self):
        parameters = ([10, 2.0, 1.0, 0.1, 0.0, -0.1, -1.0],
                      [30., 6., 3., 0.3, 0., -0.03333333, -0.33333333])

        with self.test_session():
            actual_result = tm.inv_kelu(np.asarray(parameters[1])).eval()
            self.assertArrayNear(actual_result, parameters[0], err=1e-5)

    def test_centred_gaussian(self):
        shapes = [(3,), (2, 2)]
        results = [np.array([-0.28077507, -0.1377521, -0.67632961]),
                   np.array([[-0.28077507, - 0.1377521], [-0.67632961, 0.02458041]])]

        with self.test_session():
            for shape, expected_result in zip(shapes, results):
                actual_result = tm.centred_gaussian(shape, sigma=1., seed=42).eval()
                self.assertArrayNear(actual_result.flatten(), expected_result.flatten(), err=1e-5)

    def test_perfect_centred_guassian(self):
        shapes = [(3,), (2, 2)]
        results = [
            np.array([1.22474492e+00, 2.26148963e-17, -1.22474492e+00]),
            np.array([[1.36289537, -1.36289537], [0.37751323, -0.37751323]])
        ]

        for shape, expected_result in zip(shapes, results):
            actual_result = tm.perfect_centred_gaussian(shape, sigma=1.)
            self.assertArrayNear(actual_result.flatten(), expected_result.flatten(), err=1e-5)

    def test_log_gaussian(self):
        xs = [-1.0, 0.2, 2.0]
        mus = [0.1, 0.2, 1.0]
        sigma = [0.1, 0.2, 1.0]
        results = [-59.1164, 0.690499, -1.41894]

        with self.test_session():
            for x, mu, sigma, expected_result in zip(xs, mus, sigma, results):
                actual_result = tm.log_gaussian(x, mu, sigma).eval()
                self.assertAlmostEqual(actual_result, expected_result, places=4)

    def test_log_gaussian_logsigma(self):
        xs = [-1.0, 0.2, 2.0]
        mus = [0.1, 0.2, 1.0]
        sigma = [0.1, 0.2, 10.0]
        results = [-1.51427, -1.11894, -10.9189]

        with self.test_session():
            for x, mu, sigma, expected_result in zip(xs, mus, sigma, results):
                actual_result = tm.log_gaussian_logsigma(x, mu, sigma).eval()
                self.assertAlmostEqual(actual_result, expected_result, places=4)

    def test_unit_gaussian(self):
        xs = [1.0, 1.2, 0.0]
        results = [0.241971, 0.194186, 0.398922]

        with self.test_session():
            for x, expected_result in zip(xs, results):
                actual_result = tm.unit_gaussian(x).eval()
                self.assertAlmostEqual(actual_result, expected_result, places=4)

    def test_sinh_shift(self):
        xs = [-1., 0.0, 1., 0.1]
        cs = [-1., 0.0, 1., 0.1]
        results = [-1.51935, 0.0, 1.51935, 0.198854]

        with self.test_session():
            for x, c, expected_result in zip(xs, cs, results):
                actual_result = tm.sinh_shift(x, c).eval()
                self.assertAlmostEqual(actual_result, expected_result, places=4)
