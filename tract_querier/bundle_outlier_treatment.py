import numpy as np

from tract_querier.tensor_covariance import fiberSegments

def hausdorff_distance(fiber1, fiber2, **kwargs):
    return np.sqrt(((fiber1[None,...] - fiber2[:, None, ...]) ** 2).sum(-1).max())

def wassermann_inner_product(fiber1, fiber2, sigma1=4, sigma2=4, tapering=np.inf):
    sqr_distance = ((fiber1[None,...] - fiber2[:, None, ...]) ** 2).sum(-1)

    s1_p_s2 = sigma1 + sigma2
    s1s2 = sigma1 * sigma2

    #Tapering the matrix at 3 times the minimum sigma
    exp_term = np.exp( - .5 *  sqr_distance / s1_p_s2) * (sqr_distance < tapering * min(sigma1, sigma2))

    sim = (s1s2 / np.sqrt(s1_p_s2) * exp_term).sum()

    return np.sqrt(2) * sim

def inner_product_matrix(fibers, inner_product_function=wassermann_inner_product, **kwargs):
    distance_matrix = np.zeros((len(fibers), len(fibers)))
    for i in xrange(len(fibers)):
        for j in xrange(i + 1):
            distance_matrix[i, j] = inner_product_function(fibers[i], fibers[j], **kwargs)
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix


def bundle_distances(fibers, inner_product=wassermann_inner_product, **kwargs):
    distance_matrix = inner_product_matrix(inner_product, kwargs, fibers)

    squared_norms = np.diag(distance_matrix)

    distance_matrix *= -2
    distance_matrix += squared_norms[:, None]
    distance_matrix += squared_norms[None, :]

    return distance_matrix ** .5


def bundle_normed_distances(fibers, inner_product=wassermann_inner_product, **kwargs):
    distance_matrix = inner_product_matrix(inner_product, kwargs, fibers)

    norms = np.sqrt(np.diag(distance_matrix))

    distance_matrix /= norms[:, None]
    distance_matrix /= norms[None, :]

    distance_matrix *= -2
    distance_matrix += 2

    return distance_matrix.round(6) ** .5

def z_score_outlier_rejection(fibers, z_score_reject=4, **kwargs):
    #inner_products_whole_bundle = inner_product_matrix(fibers, **kwargs)

    #inner_products_to_whole_bundle = inner_products_whole_bundle.mean(1)

    #whole_bundle_squared_norm = inner_products_to_whole_bundle.sum()

    #distances_to_whole_bundle = whole_bundle_squared_norm + inner_products_whole_bundle.diagonal() - 2 * inner_products_to_whole_bundle

    import ipdb; ipdb.set_trace()

    fiber_gps = [ fiberSegments.FiberSegmentGaussianProcess(f, thickness=4) for f in fibers ]
    whole_bundle_gp = fiberSegments.FiberBundleSegmentGaussianProcess(fiber_gps)

    inner_products_whole_bundle = np.array([whole_bundle_gp * f for f in fiber_gps])
    squared_norms = np.array([f * f for f in fiber_gps])

    #prototype_tract_index = inner_products_whole_bundle.argmax()
    #prototype_tract_squared_norm = squared_norms[prototype_tract_index]
    whole_bundle_squared_norm = whole_bundle_gp * whole_bundle_gp

    distances_to_whole_bundle = whole_bundle_squared_norm + squared_norms - 2 * inner_products_whole_bundle
    #angles = inner_products_whole_bundle / ((whole_bundle_squared_norm * squared_norms) ** .5)
    #inner_products_prototype_tract = np.array([fiber_gps[prototype_tract_index] * f for f in fiber_gps])

    #distances_prototype_tract = np.sqrt(squared_norms + prototype_tract_squared_norm - 2 * inner_products_prototype_tract)

    measure = distances_to_whole_bundle + 1e-10
    #Estimation of the gamma function parameters
    s = np.log(measure.mean()) - np.log(measure).mean()
    k = (3 - s + np.sqrt((s-3) ** 2 + 24 * s) ) / (12 * s)
    theta = measure.mean() / k

    mean = k * theta
    std = np.sqrt(k) * theta

    centered_measure = measure - mean
    z_score = centered_measure / std

    return np.where(z_score < z_score_reject)[0], z_score


