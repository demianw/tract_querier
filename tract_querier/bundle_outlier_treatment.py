import numpy as np
from scipy import ndimage

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
    inner_products = inner_product_matrix(fibers, **kwargs)

    inner_products_to_whole_bundle = inner_products.mean(1)

    whole_bundle_squared_norm = inner_products_to_whole_bundle.sum()

    distances_to_whole_bundle = whole_bundle_squared_norm + inner_products.diagonal() - 2 * inner_products_to_whole_bundle

    #Estimation of the gamma function parameters
    s = np.log(distances_to_whole_bundle.mean()) - np.log(distances_to_whole_bundle).mean()
    k = (3 - s + np.sqrt((s-3) ** 2 + 24 * s) ) / (12 * s)
    theta = distances_to_whole_bundle.mean() / k

    mean = k * theta
    std = np.sqrt(k) * theta

    z_score = (distances_to_whole_bundle - mean) / std

    return np.where(np.abs(z_score) < z_score_reject)[0], np.abs(z_score)


def mean_density_map(fibers, sigma=4, resolution=4):
    points = np.vstack(fibers)
    sigma_ = sigma / resolution
    points_max = np.ceil(points.max(0) + sigma)
    points_min = np.floor(points.min(0) - sigma)
    box_dimensions = np.ceil((np.ceil(points_max) - np.floor(points_min)) / resolution).astype(int)
    im = np.empty(box_dimensions)
    mean_im = np.empty(box_dimensions)

    for i,f in enumerate(fibers):
        im[:] = 0
        fiber_ijk = ((f - points_min)/resolution).round().astype(int)
        im[ tuple(fiber_ijk.T)] = 1
        ndimage.gaussian_filter(im, sigma_, output=im)
        mean_im += im

    return mean_im

def bundle_z_scores(fibers, sigma=4, resolution=4):
    points = np.vstack(fibers)
    sigma_ = sigma / resolution
    points_max = np.ceil(points.max(0) + sigma)
    points_min = np.floor(points.min(0) - sigma)
    box_dimensions = np.ceil((np.ceil(points_max) - np.floor(points_min)) / resolution).astype(int)
    im = np.empty(box_dimensions)
    mean_im = np.empty(box_dimensions)

    fibers_ijk = []
    for i,f in enumerate(fibers):
        im[:] = 0
        fibers_ijk.append(((f - points_min)/resolution).round().astype(int))
        im[ tuple(fibers_ijk[-1].T)] = 1
        ndimage.gaussian_filter(im, sigma_, output=im)
        mean_im += im

    z_scores = np.empty(len(fibers))
    mean_im_norm = mean_im.sum()
    for i,f in enumerate(fibers_ijk):
        im[:] = 0
        im[ tuple(f.T)] = 1
        ndimage.gaussian_filter(im, sigma_, output=im)
        z_scores[i] = (mean_im * im).sum() / np.sqrt(mean_im_norm * im.sum())

    z_scores = (z_scores - z_scores.mean()) / z_scores.std()

    return z_scores

def bundle_without_outliers(fibers, sigma=4, resolution=4, z_score=3):
    z_scores = bundle_z_scores(fibers, sigma=sigma, resolution=resolution)
    indices_to_keep = (z_scores > -z_score).nonzero()[0]

    return indices_to_keep, [fibers[i] for i in indices_to_keep]


