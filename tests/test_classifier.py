import numpy as np
import unittest

import alphai_crocubot_oracle.classifier as cl

EPS = 1e-10
N_BINS = 10
N_EDGES = N_BINS + 1
N_DATA = 100
MIN_EDGE = 0
MAX_EDGE = 10
TEST_EDGES = np.linspace(MIN_EDGE, MAX_EDGE, num=N_EDGES)
TEST_BIN_CENTRES = np.linspace(0.5, 9.5, num=N_BINS)
TEST_ARRAY = np.linspace(MIN_EDGE + EPS, MAX_EDGE - EPS, num=N_DATA)
TEST_DATA = np.stack((TEST_ARRAY, TEST_ARRAY, TEST_ARRAY))

RTOL = 1e-5
ATOL = 1e-8


class TestClassifier(unittest.TestCase):

    def test_make_template_distribution(self):

        dist = cl.make_template_distribution(TEST_DATA, N_BINS)

        print('dist edges:', TEST_EDGES, dist["bin_edges"])
        self.assertTrue(np.allclose(dist["bin_edges"], TEST_EDGES, rtol=RTOL, atol=ATOL))

    def test_compute_bin_centres(self):

        bin_centres = cl.compute_bin_centres(TEST_EDGES)
        true_centre = 1.5

        self.assertTrue(np.allclose(bin_centres[1], true_centre, rtol=RTOL, atol=ATOL))

    def test_compute_bin_widths(self):

        bin_widths = cl.compute_bin_widths(TEST_EDGES)
        true_widths = np.ones(shape=(N_BINS))

        self.assertTrue(np.allclose(bin_widths, true_widths, rtol=RTOL, atol=ATOL))

    def test_compute_balanced_bin_edges(self):

        balanced_edges = cl.compute_balanced_bin_edges(TEST_DATA.flatten(), N_BINS)

        self.assertTrue(np.allclose(balanced_edges, TEST_EDGES, rtol=RTOL, atol=ATOL))

    def test_classify_labels(self):

        true_classification = np.zeros(N_BINS)
        true_classification[5] = 1
        label = np.array([5.01])

        binned_label = cl.classify_labels(TEST_EDGES, label)
        self.assertTrue(np.allclose(binned_label, true_classification, rtol=RTOL, atol=ATOL))

    def test_declassify_labels(self):
        # Check the mean and variance of a simple pdf [00001000]

        test_classification = np.zeros(N_BINS)
        test_classification[5] = 1
        bin_width = 1

        dist = cl.make_template_distribution(TEST_DATA, N_BINS)
        pdf_arrays = dist["pdf"]

        mean, variance = cl.declassify_labels(dist, pdf_arrays)
        true_mean = np.mean(TEST_DATA)

        true_variance = bin_width ** 2 / 12

        self.assertAlmostEquals(mean, true_mean)
        self.assertAlmostEquals(variance, true_variance)

    def test_extract_point_estimates(self):
        # Set up a mock of two pdfs
        pdf_array = np.zeros(shape=(2, N_BINS))
        index_a = 2
        index_b = 5
        pdf_array[0, index_a] = 1
        pdf_array[1, index_b] = 1

        estimated_points = cl.extract_point_estimates(TEST_BIN_CENTRES, pdf_array)
        point_a = TEST_BIN_CENTRES[index_a]
        point_b = TEST_BIN_CENTRES[index_b]
        points = [point_a, point_b]

        self.assertTrue(np.allclose(estimated_points, points, rtol=RTOL, atol=ATOL))

    def test_calc_sheppards_correction(self):

        bin_width = 10
        est_correction = cl.calc_sheppards_correction(bin_width)
        correction = bin_width ** 2 / 12

        self.assertAlmostEquals(est_correction, correction)

    def test_calc_mean_bin_width(self):

        mean = cl.calc_mean_bin_width(TEST_EDGES)
        true_mean = (MAX_EDGE - MIN_EDGE) / N_BINS

        self.assertAlmostEquals(mean, true_mean)
