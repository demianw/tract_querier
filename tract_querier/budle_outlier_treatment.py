import numpy as np
from scipy import ndimage

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


