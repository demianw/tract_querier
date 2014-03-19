from .decorator import tract_math_operation
from warnings import warn


try:
    from collections import OrderedDict
except ImportError:  # Python 2.6 fix
    from ordereddict import OrderedDict

import numpy

import nibabel
from nibabel.spatialimages import SpatialImage

from ..tractography import Tractography, tractography_to_file, tractography_from_files


@tract_math_operation(': counts the number of tracts', needs_one_tract=False)
def count(tractographies):
    results = {'tract file #': [], 'number of tracts': []}
    for i, tractography in enumerate(tractographies):
        results['tract file #'].append(i)
        results['number of tracts'].append(len(tractography.tracts()))
    return results


@tract_math_operation(': print the names of scalar data associated with each tract')
def scalars(tractography):
    return {
        'scalar attributes':
        tractography.tracts_data().keys()
    }


@tract_math_operation(': calculates mean and std of tract length')
def length_mean_std(tractography):
    lengths = numpy.empty(len(tractography.tracts()))
    for i, tract in enumerate(tractography.tracts()):
        lengths[i] = tract_length(tract)

    mean = lengths.mean()
    std = lengths.std()

    return OrderedDict((
        ('length mean (mm)', mean),
        ('length std (mm^2)', std)
    ))


def tract_length(tract):
    d2 = numpy.sqrt((numpy.diff(tract, axis=0) ** 2).sum(1))
    return d2.sum()


@tract_math_operation('<volume unit>: calculates the volume of a tract based on voxel occupancy of a certain voxel volume', needs_one_tract=False)
def tract_volume(tractographies, resolution):
    results = OrderedDict()
    results['tract name'] = []
    results['tract volume'] = []
    resolution = float(resolution)
    neighbors = numpy.array([
        [0, 1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, 0, -1]
    ])

    for tractography in tractographies:
        voxels = voxelized_tract(tractography, resolution)

        dilated_voxels = set()
        dilated_voxels.update(voxels)
        eroded_voxels = set()
        for voxel in voxels:
            neighbors_list = zip(*(neighbors + voxel).T)
            dilated_voxels.update(neighbors_list)
            if len(voxels.intersection(neighbors_list)) == len(neighbors):
                eroded_voxels.add(voxel)

        # print len(dilated_voxels), len(voxels), len(eroded_voxels)
        approx_voxels = (len(dilated_voxels) - len(eroded_voxels)) / 2.
        results['tract name'].append(tractography.filename)
        results['tract volume'].append(approx_voxels * (resolution ** 3))

    return results

@tract_math_operation('<scalar>: calculates mean and std of a scalar quantity for each tract')
def scalar_tract_mean_std(tractography, scalar):
    try:
        tracts = tractography.original_tracts_data()[scalar]
        result = OrderedDict((
            ('tract file', []),
            ('mean %s' % scalar, []),
            ('std %s' % scalar, [])
        ))
        for i, t in enumerate(tracts):
            result['tract file'].append('Tract %04d' % i)
            result['mean %s' % scalar].append(t.mean())
            result['std %s' % scalar].append(t.std())

        return result

    except KeyError:
        raise ValueError("Tractography does not contain this scalar data")


@tract_math_operation('<scalar>: calculates median of a scalar quantity for each tract')
def scalar_tract_median(tractography, scalar):
    try:
        tracts = tractography.original_tracts_data()[scalar]
        result = OrderedDict((
            ('tract file', []),
            ('median %s' % scalar, []),
        ))
        for i, t in enumerate(tracts):
            result['tract file'].append('Tract %04d' % i)
            result['median %s' % scalar].append(float(numpy.median(t)))

        return result
    except KeyError:
        raise ValueError("Tractography does not contain this scalar data")


@tract_math_operation('<scalar>: calculates mean and std of a scalar quantity over tracts')
def scalar_mean_std(tractography, scalar):
    try:
        scalars = tractography.tracts_data()[scalar]
        all_scalars = numpy.vstack(scalars)
        mean = all_scalars.mean(0)
        std = all_scalars.std(0)
        return OrderedDict((
            ('mean %s' % scalar, float(mean)),
            ('std %s' % scalar, float(std))
        ))

    except KeyError:
        raise ValueError("Tractography does not contain this scalar data")


@tract_math_operation('<scalar>: calculates median of a scalar quantity over tracts')
def scalar_median(tractography, scalar):
    try:
        scalars = tractography.tracts_data()[scalar]
        all_scalars = numpy.vstack(scalars)
        median = numpy.median(all_scalars)

        return OrderedDict((
            ('median %s' % scalar, float(median)),
        ))

    except KeyError:
        raise ValueError("Tractography does not contain this scalar data")


@tract_math_operation(': Minimum and maximum distance between two consecutive points')
def tract_point_distance_min_max(tractography):
    dist_min = numpy.empty(len(tractography.tracts()))
    dist_max = numpy.empty(len(tractography.tracts()))
    for i, tract in enumerate(tractography.tracts()):
        dist = tract_length(tract)
        dist_min[i] = dist.min()
        dist_max[i] = dist.max()
    print dist_min.min(), dist_max.max()


@tract_math_operation('<points per tract> <tractography_file_output>: subsamples tracts to a maximum number of points')
def tract_subsample(tractography, points_per_tract, file_output=None):
    tractography.subsample_tracts(int(points_per_tract))

    return Tractography(
        tractography.tracts(),  tractography.tracts_data()
    )


@tract_math_operation('<points per tract> <tractography_file_output>: resamples tracts to a fixed number of points')
def tract_resubsample(tractography, points_per_tract, file_output=None):
    tractography.subsample_tracts(int(points_per_tract), True)

    return Tractography(
        tractography.tracts(),  tractography.tracts_data()
    )


@tract_math_operation('<mm per tract> <tractography_file_output>: subsamples tracts to a maximum number of points')
def tract_remove_short_tracts(tractography, min_tract_length, file_output=None):

    min_tract_length = float(min_tract_length)

    tracts = tractography.tracts()
    data = tractography.tracts_data()

    tract_ix_to_keep = [
        i for i, tract in enumerate(tractography.tracts())
        if tract_length(tract) > min_tract_length
    ]

    selected_tracts = [tracts[i] for i in tract_ix_to_keep]

    selected_data = dict()
    for key, item in data.items():
        if len(item) == len(tracts):
            selected_data_items = [item[i] for i in tract_ix_to_keep]
            selected_data[key] = selected_data_items
        else:
            selected_data[key] = item

    return Tractography(selected_tracts, selected_data)


@tract_math_operation('<image> <quantity_name> <tractography_file_output>: maps the values of an image to the tract points')
def tract_map_image(tractography, image, quantity_name, file_output=None):
    from scipy import ndimage

    image = nibabel.load(image)

    ijk_points = tract_in_ijk(image, tractography)
    image_data = image.get_data().squeeze()

    if image_data.ndim > 3:
        for i in xrange(image_data.shape[-1]):
            image = image_data[..., i]
            new_scalar_data = ndimage.map_coordinates(
                image.T, ijk_points.T
            )[:, None]
            tractography.original_tracts_data()[
                quantity_name + '_%04d' % i
            ] = new_scalar_data
        return tractography
    else:
        new_scalar_data_flat = ndimage.map_coordinates(
            image_data.T, ijk_points.T
        )[:, None]
        start = 0
        new_scalar_data = []
        for tract in tractography.original_tracts():
            new_scalar_data.append(
                new_scalar_data_flat[start: start + len(tract)])
            start += len(tract)
        tractography.original_tracts_data()[quantity_name] = new_scalar_data

        return Tractography(
            tractography.original_tracts(),  tractography.original_tracts_data()
        )


@tract_math_operation('<bins> <qty> <output>')
def tract_tract_confidence(tractography, bins, qty, file_output=None):
    bins = int(bins)
    lengths = numpy.empty(len(tractography.tracts()))
    tracts = tractography.tracts()
    tracts_prob_data = []
    tracts_length_bin = []
    for i, tract in enumerate(tracts):
        lengths[i] = tract_length(tract)
        tracts_prob_data.append(numpy.zeros(len(tract)))
        tracts_length_bin.append(numpy.zeros(len(tract)))

    length_histogram_counts, length_histogram_bins = numpy.histogram(lengths, normed=True, bins=bins)

    for i in xrange(1, bins):
        tract_log_prob = []
        indices_bin = ((length_histogram_bins[i - 1] < lengths) * (lengths < length_histogram_bins[i])).nonzero()[0]
        if len(indices_bin) == 0:
            continue

        for j in indices_bin:
            tract_log_prob.append(numpy.log(tractography.tracts_data()[qty][j]).sum())
        tract_log_prob = numpy.array(tract_log_prob)
        tract_log_prob = numpy.nan_to_num(tract_log_prob)
        lp_a0 = tract_log_prob[tract_log_prob < 0].max()
        tract_log_prob_total = numpy.log(numpy.exp(tract_log_prob - lp_a0).sum()) + lp_a0
        tract_prob = numpy.exp(tract_log_prob - tract_log_prob_total)

        for tract_number, tract_prob in zip(indices_bin, tract_prob):
            tracts_prob_data[tract_number][:] = tract_prob
            tracts_length_bin[tract_number][:] = length_histogram_bins[i - 1]

    tractography.tracts_data()['tprob'] = tracts_prob_data
    tractography.tracts_data()['tprob_bin'] = tracts_length_bin

    return tractography

@tract_math_operation('<image> <mask_out>: calculates the mask image from a tract on the space of the given image')
def tract_generate_mask(tractography, image, file_output=None):
    image = nibabel.load(image)
    mask = tract_mask(image, tractography)

    return SpatialImage(mask, image.get_affine())


@tract_math_operation('<image> [smoothing] <image_out>: calculates the probabilistic tract image for these tracts', needs_one_tract=False)
def tract_generate_population_probability_map(tractographies, image, smoothing=0, file_output=None):
    from scipy import ndimage
    image = nibabel.load(image)
    smoothing = float(smoothing)

    if isinstance(tractographies, Tractography):
        tractographies = [tractographies]

    prob_map = tract_mask(image, tractographies[0]).astype(float)
    if smoothing > 0:
        prob_map = ndimage.gaussian_filter(prob_map, smoothing)

    for tract in tractographies[1:]:
        aux_map = tract_mask(image, tract)
        if smoothing > 0:
            aux_map = ndimage.gaussian_filter(aux_map, smoothing)
        prob_map += aux_map

    prob_map /= len(tractographies)

    return SpatialImage(prob_map, image.get_affine())


@tract_math_operation('<image> <image_out>: calculates the probabilistic tract image for these tracts', needs_one_tract=False)
def tract_generate_probability_map(tractographies, image, file_output=None):
    image = nibabel.load(image)

    prob_map = tract_probability_map(image, tractographies[0]).astype(float)

    for tract in tractographies[1:]:
        if len(tract.tracts()) == 0:
            continue
        new_prob_map = tract_mask(image, tract)
        prob_map = prob_map + new_prob_map - (prob_map * new_prob_map)

    return SpatialImage(prob_map, image.get_affine())


@tract_math_operation('<tractography_out>: strips the data from the tracts', needs_one_tract=True)
def tract_strip(tractography, file_output=None):
    tractography_out = Tractography(tractography.tracts())

    return tractography_out


@tract_math_operation('<tractography_out>: takes the union of all tractographies', needs_one_tract=False)
def tract_merge(tractographies, file_output=None):
    all_tracts = []
    all_data = {}
    keys = [set(t.tracts_data().keys()) for t in tractographies]
    common_keys = keys[0].intersection(*keys[1:])
    for tract in tractographies:
        tracts = tract.tracts()
        all_tracts += tract.tracts()
        data = tract.tracts_data()
        for k in common_keys:
            if len(data[k]) == len(tracts):
                if k not in all_data:
                    all_data[k] = []
                all_data[k] += data[k]
            else:
                all_data[k] = data[k]

    return Tractography(all_tracts, all_data)



@tract_math_operation('r ICC')
def tract_icc(tractography, resolution, *other_tracts):
    import rpy2.robjects as robjects
    from rpy2.robjects.numpy2ri import numpy2ri
    r = robjects.r
    r.library('irr')

    result = OrderedDict((
        ('tract file', []),
        ('kappa value', []),
        ('totals', []),
    ))

    resolution = float(resolution)
    voxels = voxelized_tract(tractography, resolution)
    for other_tract_fname in other_tracts:
        other_tract = tractography_from_files(other_tract_fname)
        other_voxels = voxelized_tract(other_tract, resolution)

        all_voxels = voxels.union(other_voxels)

        for _ in xrange(1):
            all_voxels = dilate_voxels(all_voxels)

        #all_voxels = dilated_voxels
        all_voxels = list(all_voxels)
        mask = numpy.zeros((len(all_voxels), 2), dtype=int)

        for i, vx in enumerate(all_voxels):
            if vx in voxels:
                mask[i, 0] = 1
            if vx in other_voxels:
                mask[i, 1] = 1

        r_mask = numpy2ri(mask)
        res = r.icc(r_mask, model='twoway', type='agreement')
        #res = r.icc(r_mask, model='twoway')
        result['totals'].append((mask.all(1).sum() + (~mask.any(1)).sum()) * 1. / len(all_voxels))
        #res = r.kappa2(r_mask, "equal")
        result['tract file'].append(other_tract_fname)
        for t in res.iteritems():
            if t[0] == 'value':
                result['kappa value'].append(numpy.round(t[1][0],2))

    return result

def dilate_voxels(voxels):
    neighbors = numpy.array([
        [0, 1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, 0, -1]
    ])

    dilated_voxels = set()
    dilated_voxels.update(voxels)
    for voxel in voxels:
        neighbors_list = zip(*(neighbors + voxel).T)
        dilated_voxels.update(neighbors_list)
    return dilated_voxels


@tract_math_operation('<volume unit> <tract1.vtk> ... <tractN.vtk>: calculates the kappa value of the first tract with the rest in the space of the reference image')
def tract_kappa(tractography, resolution, *other_tracts):
    resolution = float(resolution)

    voxels = voxelized_tract(tractography, resolution)

    result = OrderedDict((
        ('tract file', []),
        ('kappa value', [])
    ))

    for tract in other_tracts:
        voxels1 = voxelized_tract(
            tractography_from_files(tract),
            resolution
        )

        all_voxels = numpy.array(list(voxels.union(voxels1)))
        N = (all_voxels.max(0) - all_voxels.min(0)).prod()
        pp = len(voxels.intersection(voxels1)) * 1.
        pn = len(voxels.difference(voxels1)) * 1.
        np = len(voxels1.difference(voxels)) * 1.
        nn = N - pp - pn - np
        observed_agreement = (pp + nn) / N
        chance_agreement = (
            (pp + pn) * (pp + np) + (nn + np) * (nn + pn)) / (N * N)

        k = (observed_agreement - chance_agreement) / (1 - chance_agreement)

        result['tract file'].append(tract)
        result['kappa value'].append(k)

    return result


@tract_math_operation('<volume> <threshold> <tract1.vtk> ... <tractN.vtk>: calculates the kappa value of the first tract with the rest in the space of the reference image')
def tract_kappa_volume(tractography, volume, threshold, resolution, *other_tracts):
    resolution = float(resolution)

    volume = nibabel.load(volume)
    mask = (volume.get_data() > threshold).astype(int)
    voxels = tract_mask(mask, tractography)

    result = OrderedDict((
        ('tract file', []),
        ('kappa value', [])
    ))

    for tract in other_tracts:
        voxels1 = voxelized_tract(
            tractography_from_files(tract), resolution)

        all_voxels = numpy.array(list(voxels.union(voxels1)))
        N = (all_voxels.max(0) - all_voxels.min(0)).prod()
        pp = len(voxels.intersection(voxels1)) * 1.
        pn = len(voxels.difference(voxels1)) * 1.
        np = len(voxels1.difference(voxels)) * 1.
        nn = N - pp - pn - np
        observed_agreement = (pp + nn) / N
        chance_agreement = (
            (pp + pn) * (pp + np) + (nn + np) * (nn + pn)) / (N * N)

        k = (observed_agreement - chance_agreement) / (1 - chance_agreement)

        result['tract file'].append(tract)
        result['kappa value'].append(k)

    return result


@tract_math_operation('<volume unit> <tract1.vtk> ... <tractN.vtk>: calculates the dice coefficient of the first tract with the rest in the space of the reference image')
def tract_dice(tractography, resolution, *other_tracts):
    resolution = float(resolution)

    voxels = voxelized_tract(tractography, resolution)

    result = OrderedDict((
        ('tract file', []),
        ('dice coefficient', [])
    ))

    for tract in other_tracts:
        voxels1 = voxelized_tract(
            tractography_from_files(tract),
            resolution
        )
        result['tract file'].append(tract)
        result['dice coefficient'].append(
            2 * len(voxels.intersection(voxels1)) * 1. /
            (len(voxels) + len(voxels1))
        )

    return result


def voxelized_tract(tractography, resolution):
    from itertools import izip
    all_points = numpy.vstack(tractography.tracts())
    all_points /= resolution
    all_points = all_points.round(0).astype(int)
    return set(izip(*(all_points.T)))


@tract_math_operation('<var> <tract_out>: smoothes the tract by convolving with a sliding window')
def tract_smooth(tractography, var, file_output=None):
    from sklearn.neighbors import BallTree

    var = float(var)
    std = var ** 2

    points = tractography.original_tracts()

    all_points = numpy.vstack(points)
    bt = BallTree(all_points)
    N = len(all_points) / 3
    I = numpy.eye(3)[None, ...]
    for i, tract in enumerate(tractography.original_tracts()):
        # all_points = numpy.vstack(points[:i] + points[i + 1:])
        # bt = BallTree(all_points)

        diff = numpy.diff(tract, axis=0)
        diff = numpy.vstack((diff, diff[-1]))
        lengths = numpy.sqrt((diff ** 2).sum(1))
        # cum_lengths = numpy.cumsum(lengths)

        diff_norm = diff / lengths[:, None]
        tangent_lines = diff_norm[:, None, :] * diff_norm[:, :, None]
        normal_planes = I - tangent_lines
#        weight_matrices = normal_planes + 1e10 * tangent_lines

        N = max(len(d) for d in bt.query_radius(tract, var * 3))

        close_point_distances, close_point_indices = bt.query(
            tract, N
        )

        close_points = all_points[close_point_indices]
        difference_vectors = close_points - tract[:, None, :]
        projected_vectors = (
            normal_planes[:, None, :] *
            difference_vectors[..., None]
        ).sum(-2)
        projected_points = projected_vectors + tract[:, None, :]
        # projected_distances2 = (projected_vectors**2).sum(-1)
        # projected_weights = numpy.exp(- .5 * projected_distances2 / std)
        # projected_weights /= projected_weights.sum(-1)[:, None]

        weights = numpy.exp(
            -.5 * close_point_distances ** 2 / std
        )[..., None]
        weights /= weights.sum(-2)[..., None]

        # tract += (weights * projected_vectors).sum(-2)

#        weighted_distances = (
#            weight_matrices[:, None, :] *
#            difference_vectors[..., None]
#        ).sum(-2)
#        weighted_distances *= difference_vectors
#        weighted_distances = weighted_distances.sum(-1) ** .5
        # weighted_points = (projected_points * weights).sum(1)

        weighted_points = (projected_points * weights).sum(1)

        tract[:] = weighted_points
        # tract /= norm_term

    return Tractography(
        tractography.original_tracts(),
        tractography.original_tracts_data()
    )


def tract_mask(image, tractography):
    image_data = image.get_data()
    mask = numpy.zeros_like(image_data, dtype=float)
    if len(tractography.tracts()) == 0:
        return mask

    ijk_points = tract_in_ijk(image, tractography)

    ijk_clipped = ijk_points.clip(
        (0, 0, 0), numpy.array(image_data.shape) - 1
    ).astype(int)

    mask[tuple(ijk_clipped.T)] = 1
    return mask


def tract_probability_map(image, tractography):
    ijk_tracts = each_tract_in_ijk(image, tractography)
    image_data = image.get_data()

    probability_map = numpy.zeros_like(image_data, dtype=float)
    for ijk_points in ijk_tracts:
        ijk_clipped = ijk_points.clip(
            (0, 0, 0), numpy.array(image_data.shape) - 1
        ).astype(int)

        probability_map[tuple(ijk_clipped.T)] += 1

    probability_map /= len(ijk_tracts)
    return probability_map


def each_tract_in_ijk(image, tractography):
    ijk_tracts = []
    for tract in tractography.tracts():
        ijk_tracts.append(numpy.dot(numpy.linalg.inv(image.get_affine()), numpy.hstack((
            tract,
            numpy.ones((len(tract), 1))
        )).T).T[:, :-1])
    return ijk_tracts


def tract_in_ijk(image, tractography):
    ras_points = numpy.vstack(tractography.tracts())
    ijk_points = numpy.dot(numpy.linalg.inv(image.get_affine()), numpy.hstack((
        ras_points,
        numpy.ones((len(ras_points), 1))
    )).T).T[:, :-1]
    return ijk_points


@tract_math_operation('<tract_out>: compute the protoype tract')
def tract_prototype_median(tractography, file_output=None):
    from .tract_obb import prototype_tract

    tracts = tractography.tracts()
    data = tractography.tracts_data()
    prototype_ix = prototype_tract(tracts)

    selected_tracts = [tracts[prototype_ix]]
    selected_data = dict()
    for key, item in data.items():
        if len(item) == len(tracts):
            selected_data_items = [item[prototype_ix]]
            selected_data[key] = selected_data_items
        else:
            selected_data[key] = item

    return Tractography(selected_tracts, selected_data)


@tract_math_operation('<smooth order> <tract_out>: compute the protoype tract')
def tract_prototype_mean(tractography, smooth_order, file_output=None):
    from .tract_obb import prototype_tract

    tracts = tractography.tracts()
    prototype_ix, leave_centers = prototype_tract(tracts, return_leave_centers=True)

    median_tract = tracts[prototype_ix]

    mean_tract = numpy.empty_like(median_tract)
    centers_used = set()
    for point in median_tract:
        closest_leave_center_ix = (
            ((leave_centers - point[None, :]) ** 2).sum(1)
        ).argmin()

        if closest_leave_center_ix in centers_used:
            continue

        mean_tract[len(centers_used)] = leave_centers[closest_leave_center_ix]
        centers_used.add(closest_leave_center_ix)

    mean_tract = mean_tract[:len(centers_used)]

    if smooth_order > 0:
        try:
            from scipy import interpolate

            tck, u = interpolate.splprep(mean_tract.T)
            mean_tract = numpy.transpose(interpolate.splev(u, tck))
        except ImportError:
            warn("A smooth order larger than 0 needs scipy installed")

    return Tractography([mean_tract], {})


@tract_math_operation('<volume unit> <tract1.vtk> ... <tractN.vtk>: calculates the Bhattacharyya coefficient of the first tract with the rest in the space of the reference image')
def tract_bhattacharyya_coefficient(tractography, resolution, *other_tracts):
    resolution = float(resolution)
    coord = ('X', 'Y', 'Z')
    result = OrderedDict(
        [('tract file', [])]
        + [
            ('bhattacharyya %s value' % coord[i], [])
            for i in xrange(3)
        ]
    )

    tractography_points = numpy.vstack(tractography.tracts())

    other_tracts_tractographies = [tractography_from_files(t_)
        for t_ in other_tracts
    ]

    other_tracts_points = [
        numpy.vstack(t_.tracts())
        for t_ in other_tracts_tractographies
    ]

    mn_ = tractography_points.min(0)
    mx_ = tractography_points.max(0)

    for pts in other_tracts_points:
        mn_ = numpy.minimum(mn_, pts.min(0))
        mx_ = numpy.maximum(mn_, pts.max(0))

    bins = numpy.ceil((mx_ - mn_) * 1. / resolution)
    hists_tract = [
        numpy.histogram(tractography_points[:, i], bins=bins[i], density=True, range=(mn_[i], mx_[i]))[0]
        for i in xrange(3)
    ]

    for tract, tract_points in zip(other_tracts, other_tracts_points):
        hists_other_tract = [
            numpy.histogram(tract_points[:, i], bins=bins[i], density=True, range=(mn_[i], mx_[i]))[0]
            for i in xrange(3)
        ]

        distances = [
            numpy.sqrt(
                hists_other_tract[i] * hists_tract[i] /
                (hists_other_tract[i].sum() * hists_tract[i].sum())
            ).sum()
            for i in xrange(3)
        ]
        for i in xrange(3):
            result['tract file'].append(tract)
            result['bhattacharyya %s value' % coord[i]].append(numpy.nan_to_num(distances[i]))

    return result

