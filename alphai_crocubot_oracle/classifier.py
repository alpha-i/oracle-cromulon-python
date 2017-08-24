# These functions act as interface between point estimates and binned estimates
# This will probably be moved to a data transformation library at a later point
# This module is only used by oracle.py and iotools.py


import numpy as np


def make_template_distribution(training_labels, n_bins):
    """ Returns the best-fit characteristics of the continuous probability distribution fit to the input data"""

    # We'll treat the data as homogeneous for now. If features/labels are normalised/whitened this should be ok.
    # The bins will also have better noise properties this way.
    training_labels = training_labels.flatten()

    pdf_type = find_best_fit_pdf_type(training_labels)

    dist = {"type": pdf_type, "mean": np.mean(training_labels), "median": np.median(training_labels),
            "sigma": np.std(training_labels)}

    bin_edges = compute_balanced_bin_edges(training_labels, n_bins)
    pdf, edges = np.histogram(training_labels, bin_edges, density=False)
    pdf = pdf / training_labels.shape[0]

    dist["n_bins"] = n_bins
    dist["bin_centres"] = compute_bin_centres(bin_edges)
    dist["bin_edges"] = bin_edges
    dist["bin_widths"] = compute_bin_widths(dist["bin_edges"])
    dist["mean_bin_width"] = calc_mean_bin_width(dist["bin_edges"])
    dist['shep_correction'] = calc_sheppards_correction(dist["bin_widths"])
    dist["pdf"] = pdf

    return dist


def find_best_fit_pdf_type(x):
    return 'Gaussian'  # TODO: enable tests for t-distributed data; lognormal, etc


def compute_bin_centres(bin_edges):
    """ Finds the bin centres """
    return 0.5 * (bin_edges[1:] + bin_edges[:-1])


def compute_bin_widths(bin_edges):
    """ Finds the bin widths """
    return bin_edges[1:] - bin_edges[:-1]


def compute_balanced_bin_edges(x, n_bins):
    """ Finds the bins needed such that they equally divied the data.
    """

    n_xvals = len(x)
    xrange = np.linspace(0, n_xvals, n_bins + 1)
    n_array = np.arange(n_xvals)
    print("Assigning", str(x.shape), "to", n_bins, "bins")
    return np.interp(xrange, n_array, np.sort(x))


def classify_labels(bin_edges, labels):
    """  Takes numerical values and returns their binned values in one-hot format

    :param bin_edges: One dimensional array
    :param labels:  One or two dimensional array (e.g. could be [batch_size, n_series])
    :return: return dimensions  [labels.shape, n_bins]
    """

    n_label_dimensions = labels.ndim
    label_shape = labels.shape
    labels = labels.flatten()

    n_labels = len(labels)
    n_bins = len(bin_edges) - 1
    binned_labels = np.zeros((n_labels, n_bins))

    for i in range(n_labels):
        binned_labels[i, :], _ = np.histogram(labels[i], bin_edges, density=False)

    if n_label_dimensions == 2:
        binned_labels = binned_labels.reshape(label_shape[0], label_shape[1], n_bins)
    elif n_label_dimensions == 3:
        binned_labels = binned_labels.reshape(label_shape[0], label_shape[1], label_shape[2], n_bins)
    elif n_label_dimensions > 3:
        raise NotImplementedError("Label dimension too high:", n_label_dimensions)

    return binned_labels


def declassify_labels(dist, pdf_arrays):
    """ Takes binned pdfs interprets as means and standard deviations
    pdf_arrays has shape [n_samples, n_bins]"""

    point_estimates = extract_point_estimates(dist["bin_centres"], pdf_arrays)

    mean = np.mean(point_estimates)
    variance = np.var(point_estimates) - dist['shep_correction']

    variance = np.maximum(variance, dist['shep_correction'])  # Prevent variance becoming too small

    return mean, variance


def extract_point_estimates(bin_centres, pdf_array):
    """ Convert classification into a point estimate of mean and variance.
    bin_centres has shape [ n_bins]
    pdf_array  has shape [n_samples, n_bins]
    """

    if pdf_array.ndim == 1:
        pdf_array = np.expand_dims(pdf_array, axis=0)

    n_points = pdf_array.shape[0]
    points = np.zeros((n_points))

    normalisation_offset = np.sum(pdf_array[0, :]) - 1.0
    assert np.abs(normalisation_offset) < 1e-3, 'Probability mass function not normalised'

    for i in range(n_points):
        pdf = pdf_array[i, :]
        points[i] = np.sum(bin_centres * pdf)

    return points


def calc_sheppards_correction(bin_widths):
    """Computes the extend to which the variance is overestimated when using binned data. """

    return np.mean(bin_widths ** 2) / 12


def calc_mean_bin_width(bin_edges):

    n_bins = len(bin_edges) - 1
    full_gap = np.abs(bin_edges[-1] - bin_edges[0])

    return full_gap / n_bins
