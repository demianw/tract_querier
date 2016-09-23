from .decorator import tract_math_operation, set_dictionary_from_use_filenames_as_index

from warnings import warn

import numpy

import nibabel
from nibabel.spatialimages import SpatialImage

from ..tractography import (
    Tractography, tractography_to_file, tractography_from_files
)

import sys
import traceback
from . import tensor_operations
from . import tract_operations


try:
    from collections import OrderedDict
except ImportError:  # Python 2.6 fix
    from ordereddict import OrderedDict


@tract_math_operation(': print the names of scalar data associated with each tract')
def scalars(optional_flags, tractography):
    return {
        'scalar attributes':
        tractography.tracts_data().keys()
    }


@tract_math_operation(': counts the number of tracts', needs_one_tract=False)
def count(optional_flags, tractographies):
    results = OrderedDict()
    for default_tractography_name, (tract_name, tract) in enumerate(tractographies):
        measurement_dict = tensor_operations.compute_all_measures(tract, ['number of tracts'])
        results = set_dictionary_from_use_filenames_as_index(optional_flags,
                                                               tract_name, default_tractography_name,
                                                               results, measurement_dict)
    return results


@tract_math_operation(': calculates mean and std of tract length')
def length_mean_std(optional_flags, tractography):
    return tensor_operations.compute_all_measures(tractography, ['length mean (mm)', 'length std (mm^2)'])


@tract_math_operation('<volume unit>: calculates the volume of a tract based on voxel occupancy of a certain voxel volume')
def tract_volume(optional_flags, tractography, resolution):
    return tensor_operations.compute_all_measures(tractography, ['tract volume'], resolution=resolution)


@tract_math_operation('<scalar>: calculates mean and std of a scalar quantity that has been averaged along each tract', needs_one_tract=False)
def scalar_per_tract_mean_std(optional_flags, tractographies, scalar):
    results = OrderedDict()
    try:
        for default_tract_name, (tract_name, tract) in enumerate(tractographies):

            measurement_dict = tensor_operations.compute_all_measures(tract,
                                                    ['per tract distance weighted mean %s',
                                                     'per tract distance weighted std %s'],
                                                    scalars=[scalar])
            results = set_dictionary_from_use_filenames_as_index(optional_flags,
                                                                   tract_name, default_tract_name,
                                                                   results, measurement_dict)
    except KeyError:
        traceback.print_exc(file=sys.stdout)
        raise ValueError("Tractography does not contain this scalar data")

    return results


@tract_math_operation('<scalar>: calculates many DTI measurements along each tract if there are two tensor data attributes: "tensor1" and "tensor2"', needs_one_tract=False)
def scalar_compute_most(optional_flags, tractographies, scalar):
    if scalar == 'all':
        get_reference_tract = tractographies[0][1]
        scalars = [
            s for s in get_reference_tract.tracts_data().keys() if not s.startswith("tensor")]
    else:
        scalars = [scalar]
    results = OrderedDict()
    try:
        for default_tract_name, (tract_name, tract) in enumerate(tractographies):
            # First_decorate_tract
            if 'tensor1' in tract.tracts_data().keys():
                tract = tensor_operations.decorate_tract_with_measures(tract, 'tensor1')
                scalars.extend(
                    ['FA_tensor1', 'MD_tensor1', 'AX_tensor1', 'RD_tensor1', 'GA_tensor1'])
            if 'tensor2' in tract.tracts_data().keys():
                tract = tensor_operations.decorate_tract_with_measures(tract, 'tensor2')
                scalars.extend(
                    ['FA_tensor2', 'MD_tensor2', 'AX_tensor2', 'RD_tensor2', 'GA_tensor2'])

            measurement_dict = tensor_operations.compute_all_measures(tract,
                                                    ['per tract distance weighted mean %s',
                                                     'per tract distance weighted std %s',
                                                     'tract volume',
                                                     'length mean (mm)', 'length std (mm^2)',
                                                     'number of tracts'
                                                     ],
                                                    scalars=scalars, resolution=1.)
            results = set_dictionary_from_use_filenames_as_index(optional_flags,
                                                                   tract_name, default_tract_name,
                                                                   results, measurement_dict)
    except KeyError:
        traceback.print_exc(file=sys.stdout)
        raise ValueError("Tractography does not contain this tensor data")

    return results


@tract_math_operation('<scalar>: calculates mean and std of a scalar quantity for each tract')
def scalar_tract_mean_std(optional_flags, tractography, scalar):

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
def scalar_tract_median(optional_flags, tractography, scalar):
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
def scalar_mean_std(optional_flags, tractography, scalar):
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
def scalar_median(optional_flags, tractography, scalar):
    try:
        scalars = tractography.tracts_data()[scalar]
        all_scalars = numpy.vstack(scalars)
        median = numpy.median(all_scalars)

        return OrderedDict((
            ('median %s' % scalar, float(median)),
        ))

    except KeyError:
        raise ValueError("Tractography does not contain this scalar data")


@tract_math_operation(': Dumps all the data in the tractography', needs_one_tract=True)
def tract_dump(optional_flags, tractography):
    res = OrderedDict()
    tract_number = 'tract #'
    res[tract_number] = []
    res['x'] = []
    res['y'] = []
    res['z'] = []

    data = tractography.tracts_data()

    for k in data.keys():
        res[k] = []

    for i, tract in enumerate(tractography.tracts()):
        res[tract_number] += [i] * len(tract)
        res['x'] += list(tract[:, 0])
        res['y'] += list(tract[:, 1])
        res['z'] += list(tract[:, 2])

        for k in data.keys():
            res[k] += list(numpy.asarray(data[k][i]).squeeze())

    return res


@tract_math_operation(': Dumps tract endpoints', needs_one_tract=True)
def tract_dump_endpoints(optional_flags, tractography):
    res = OrderedDict()
    tract_number = 'tract #'
    res[tract_number] = []
    res['x'] = []
    res['y'] = []
    res['z'] = []

    for i, tract in enumerate(tractography.tracts()):
        res[tract_number] += [i] * 2
        res['x'] += list(tract[(0, -1), 0])
        res['y'] += list(tract[(0, -1), 1])
        res['z'] += list(tract[(0, -1), 2])

    return res


@tract_math_operation(': Minimum and maximum distance between two consecutive points')
def tract_point_distance_min_max(optional_flags, tractography):
    dist_min = numpy.empty(len(tractography.tracts()))
    dist_max = numpy.empty(len(tractography.tracts()))
    for i, tract in enumerate(tractography.tracts()):
        dist = tract_operations.tract_length(tract)
        dist_min[i] = dist.min()
        dist_max[i] = dist.max()
    print dist_min.min(), dist_max.max()


@tract_math_operation('<points per tract> <tractography_file_output>: subsamples tracts to a maximum number of points')
def tract_subsample(optional_flags, tractography, points_per_tract, file_output):
    tractography.subsample_tracts(int(points_per_tract))

    return Tractography(
        tractography.tracts(),  tractography.tracts_data(),
        **tractography.extra_args
    )


@tract_math_operation('<mm per tract> <tractography_file_output>: subsamples tracts to a maximum number of points')
def tract_remove_short_tracts(optional_flags, tractography, min_tract_length, file_output):

    min_tract_length = float(min_tract_length)

    tracts = tractography.tracts()
    data = tractography.tracts_data()

    tract_ix_to_keep = [
        i for i, tract in enumerate(tractography.tracts())
        if tract_operations.tract_length(tract) > min_tract_length
    ]

    selected_tracts = [tracts[i] for i in tract_ix_to_keep]

    selected_data = dict()
    for key, item in data.items():
        if len(item) == len(tracts):
            selected_data_items = [item[i] for i in tract_ix_to_keep]
            selected_data[key] = selected_data_items
        else:
            selected_data[key] = item

    return Tractography(
        selected_tracts, selected_data,
        **tractography.extra_args
    )


@tract_math_operation('<image> <quantity_name> <tractography_file_output>: maps the values of an image to the tract points')
def tract_map_image(optional_flags, tractography, image, quantity_name, file_output):
    from os import path
    from scipy import ndimage

    image = nibabel.load(image)

    ijk_points = tract_operations.tract_in_ijk(image, tractography)
    image_data = image.get_data()

    if image_data.ndim > 3:
        output_name, ext = path.splitext(file_output)
        output_name = output_name + '_%04d' + ext
        for i, image in enumerate(image_data):
            new_scalar_data = ndimage.map_coordinates(
                image, ijk_points.T
            )[:, None]
            tractography.original_tracts_data()[
                quantity_name] = new_scalar_data
            tractography_to_file(output_name % i, Tractography(
                tractography.original_tracts(),  tractography.original_tracts_data()))
    else:
        new_scalar_data_flat = ndimage.map_coordinates(
            image_data, ijk_points.T
        )[:, None]
        start = 0
        new_scalar_data = []
        for tract in tractography.original_tracts():
            new_scalar_data.append(
                new_scalar_data_flat[start: start + len(tract)].copy()
            )
            start += len(tract)
        tractography.original_tracts_data()[quantity_name] = new_scalar_data

        return Tractography(
            tractography.original_tracts(
            ),  tractography.original_tracts_data(),
            **tractography.extra_args
        )


@tract_math_operation(
    '<deformation> <tractography_file_output>: apply a '
    'non-linear deformation to a tractography'
)
def tract_deform(optional_flags, tractography, image, file_output=None):
    from scipy import ndimage
    import numpy as numpy

    image = nibabel.load(image)
    coord_adjustment = numpy.sign(numpy.diag(image.get_affine())[:-1])
    ijk_points = tract_operations.tract_in_ijk(image, tractography)
    image_data = image.get_data().squeeze()

    if image_data.ndim != 4 and image_data.shape[-1] != 3:
        raise ValueError('Image is not a deformation field')

    new_points = numpy.vstack(tractography.tracts())  # ijk_points.copy()
    for i in (0, 1, 2):
        image_ = image_data[..., i]
        deformation = ndimage.map_coordinates(
            image_, ijk_points.T
        ).squeeze()
        new_points[:, i] -= coord_adjustment[i] * deformation

    new_ras_points = new_points  # tract_in_ras(image, new_points)
    start = 0
    new_tracts = []
    for tract in tractography.original_tracts():
        new_tracts.append(
            new_ras_points[start: start + len(tract)].copy()
        )
        start += len(tract)

    return Tractography(
        new_tracts,  tractography.original_tracts_data(),
        **tractography.extra_args
    )


@tract_math_operation(
    '<transform> [invert] <tractography_file_output>: apply a '
    'affine transform to a tractography. '
    'transform is assumed to be in RAS format like Nifti.'
)
def tract_affine_transform(optional_flags,
                           tractography, transform_file, ref_image,
                           invert=False, file_output=None
                           ):
    import nibabel
    import numpy as numpy
    ref_image = nibabel.load(ref_image)
    ref_affine = ref_image.get_affine()
    transform = numpy.loadtxt(transform_file)
    invert = bool(invert)
    if invert:
        print "Inverting transform"
        transform = numpy.linalg.inv(transform)
    orig_points = numpy.vstack(tractography.tracts())
    new_points = nibabel.affines.apply_affine(transform, orig_points)
    start = 0
    new_tracts = []
    for tract in tractography.original_tracts():
        new_tracts.append(
            new_points[start: start + len(tract)].copy()
        )
        start += len(tract)

    extra_args = {
        'affine': ref_affine,
        'image_dims': ref_image.shape
    }

    # if tractography.extra_args is not None:
    #    tractography.extra_args.update(extra_args)
    #    extra_args = tractography.extra_args

    return Tractography(
        new_tracts,  tractography.original_tracts_data(),
        **extra_args
    )


@tract_math_operation('<bins> <qty> <output>')
def tract_tract_confidence(optional_flags, tractography, bins, qty, file_output=None):
    bins = int(bins)
    lengths = numpy.empty(len(tractography.tracts()))
    tracts = tractography.tracts()
    tracts_prob_data = []
    tracts_length_bin = []
    for i, tract in enumerate(tracts):
        lengths[i] = tract_operations.tract_length(tract)
        tracts_prob_data.append(numpy.zeros(len(tract)))
        tracts_length_bin.append(numpy.zeros(len(tract)))

    length_histogram_counts, length_histogram_bins = numpy.histogram(
        lengths, normed=True, bins=bins)

    for i in xrange(1, bins):
        tract_log_prob = []
        indices_bin = ((length_histogram_bins[
                       i - 1] < lengths) * (lengths < length_histogram_bins[i])).nonzero()[0]
        if len(indices_bin) == 0:
            continue

        for j in indices_bin:
            tract_log_prob.append(
                numpy.log(tractography.tracts_data()[qty][j]).sum())
        tract_log_prob = numpy.array(tract_log_prob)
        tract_log_prob = numpy.nan_to_num(tract_log_prob)
        lp_a0 = tract_log_prob[tract_log_prob < 0].max()
        tract_log_prob_total = numpy.log(
            numpy.exp(tract_log_prob - lp_a0).sum()) + lp_a0
        tract_prob = numpy.exp(tract_log_prob - tract_log_prob_total)

        for tract_number, tract_prob in zip(indices_bin, tract_prob):
            tracts_prob_data[tract_number][:] = tract_prob
            tracts_length_bin[tract_number][:] = length_histogram_bins[i - 1]

    tractography.tracts_data()['tprob'] = tracts_prob_data
    tractography.tracts_data()['tprob_bin'] = tracts_length_bin

    return tractography


@tract_math_operation('<image> <mask_out>: calculates the mask image from a tract on the space of the given image')
def tract_generate_mask(optional_flags, tractography, image, file_output):
    image = nibabel.load(image)
    mask = tract_operations.tract_mask(image, tractography)

    return SpatialImage(mask, image.get_affine())


@tract_math_operation('<image> [smoothing] <image_out>: calculates the probabilistic tract image for these tracts', needs_one_tract=False)
def tract_generate_population_probability_map(optional_flags, tractographies, image, smoothing=0, file_output=None):
    from scipy import ndimage
    image = nibabel.load(image)
    smoothing = float(smoothing)

    # tractographies includes tuples of (tractography filename, tractography
    # instance)
    if isinstance(tractographies[1], Tractography):
        tractographies = [tractographies]

    prob_map = tract_operations.tract_mask(image, tractographies[0][1]).astype(float)
    if smoothing > 0:
        prob_map = ndimage.gaussian_filter(prob_map, smoothing)

    for tract in tractographies[1:]:
        aux_map = tract_operations.tract_mask(image, tract[1])
        if smoothing > 0:
            aux_map = ndimage.gaussian_filter(aux_map, smoothing)
        prob_map += aux_map

    prob_map /= len(tractographies)

    return SpatialImage(prob_map, image.get_affine()),


@tract_math_operation('<image> <image_out>: calculates the probabilistic tract image for these tracts', needs_one_tract=False)
def tract_generate_probability_map(optional_flags, tractographies, image, file_output):
    image = nibabel.load(image)

    prob_map = tract_operations.tract_probability_map(image, tractographies[0][1]).astype(float)

    for tract in tractographies[1:]:
        if len(tract[1].tracts()) == 0:
            continue
        new_prob_map = tract_operations.tract_mask(image, tract[1])
        prob_map = prob_map + new_prob_map - (prob_map * new_prob_map)

    return SpatialImage(prob_map, image.get_affine())


@tract_math_operation('<tractography_out>: strips the data from the tracts', needs_one_tract=True)
def tract_strip(optional_flags, tractography, file_output):
    tractography_out = Tractography(tractography.tracts())

    return tractography_out


@tract_math_operation('<tractography_out>: takes the union of all tractographies', needs_one_tract=False)
def tract_merge(optional_flags, tractographies, file_output):
    all_tracts = []
    all_data = {}
    keys = [set(t[1].tracts_data().keys()) for t in tractographies]
    common_keys = keys[0].intersection(*keys[1:])

    affine = tractographies[0][1].extra_args.get('affine', None)
    image_dims = tractographies[0][1].extra_args.get('image_dims', None)

    for tract in tractographies:
        tracts = tract[1].tracts()
        if affine is not None and 'affine' in tract[1].extra_args:
            if (tract[1].affine != affine).any():
                affine = None
        if image_dims is not None and 'image_dims' in tract[1].extra_args:
            if (tract[1].image_dims != image_dims).any():
                image_dims = None
        all_tracts += tract[1].tracts()
        data = tract[1].tracts_data()
        for k in common_keys:
            if len(data[k]) == len(tracts):
                if k not in all_data:
                    all_data[k] = []
                all_data[k] += data[k]
            else:
                all_data[k] = data[k]

    return Tractography(
        all_tracts, all_data,
        affine=affine, image_dims=image_dims
    )


@tract_math_operation('<volume unit> <tract1.vtk> ... <tractN.vtk>: calculates the kappa value of the first tract with the rest in the space of the reference image')
def tract_kappa(optional_flags, tractography, resolution, *other_tracts):
    resolution = float(resolution)

    voxels = tract_operations.voxelized_tract(tractography, resolution)

    result = OrderedDict((
        ('tract file', []),
        ('kappa value', [])
    ))

    for tract in other_tracts:
        voxels1 = tract_operations.voxelized_tract(
            tractography_from_files(tract),
            resolution
        )

        all_voxels = numpy.array(list(voxels.union(voxels1)))
        N = (all_voxels.max(0) - all_voxels.min(0)).prod()
        pp = len(voxels.intersection(voxels1)) * 1.
        pn = len(voxels.difference(voxels1)) * 1.
        numpy = len(voxels1.difference(voxels)) * 1.
        nn = N - pp - pn - numpy
        observed_agreement = (pp + nn) / N
        chance_agreement = (
            (pp + pn) * (pp + numpy) + (nn + numpy) * (nn + pn)) / (N * N)

        k = (observed_agreement - chance_agreement) / (1 - chance_agreement)

        result['tract file'].append(tract)
        result['kappa value'].append(k)

    return result


@tract_math_operation('<volume> <threshold> <tract1.vtk> ... <tractN.vtk>: calculates the kappa value of the first tract with the rest in the space of the reference image')
def tract_kappa_volume(optional_flags, tractography, volume, threshold, resolution, *other_tracts):
    resolution = float(resolution)

    volume = nibabel.load(volume)
    mask = (volume.get_data() > threshold).astype(int)
    voxels = tract_operations.tract_mask(mask, tractography)

    result = OrderedDict((
        ('tract file', []),
        ('kappa value', [])
    ))

    for tract in other_tracts:
        voxels1 = tract_operations.voxelized_tract(
            tractography_from_files(tract), resolution)

        all_voxels = numpy.array(list(voxels.union(voxels1)))
        N = (all_voxels.max(0) - all_voxels.min(0)).prod()
        pp = len(voxels.intersection(voxels1)) * 1.
        pn = len(voxels.difference(voxels1)) * 1.
        numpy = len(voxels1.difference(voxels)) * 1.
        nn = N - pp - pn - numpy
        observed_agreement = (pp + nn) / N
        chance_agreement = (
            (pp + pn) * (pp + numpy) + (nn + numpy) * (nn + pn)) / (N * N)

        k = (observed_agreement - chance_agreement) / (1 - chance_agreement)

        result['tract file'].append(tract)
        result['kappa value'].append(k)

    return result


@tract_math_operation('<volume unit> <tract1.vtk> ... <tractN.vtk>: calculates the dice coefficient of the first tract with the rest in the space of the reference image')
def tract_dice(optional_flags, tractography, resolution, *other_tracts):
    resolution = float(resolution)

    voxels = tract_operations.voxelized_tract(tractography, resolution)

    result = OrderedDict((
        ('tract file', []),
        ('dice coefficient', [])
    ))

    for tract in other_tracts:
        voxels1 = tract_operations.voxelized_tract(
            tractography_from_files(tract),
            resolution
        )
        result['tract file'].append(tract)
        result['dice coefficient'].append(
            2 * len(voxels.intersection(voxels1)) * 1. /
            (len(voxels) + len(voxels1))
        )

    return result


@tract_math_operation('<var> <tract_out>: smoothes the tract by convolving with a sliding window')
def tract_smooth(optional_flags, tractography, var, file_output):
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
        tangent_lines = diff_norm[:, None, :] * diff_norm[:,:, None]
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
        tractography.original_tracts_data(),
        **tractography.extra_args
    )


@tract_math_operation('<tract_out>: compute the protoype tract')
def tract_prototype_median(optional_flags, tractography, file_output=None):
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

    return Tractography(selected_tracts, selected_data, **tractography.extra_args)


@tract_math_operation('<smooth order> <tract_out>: compute the protoype tract')
def tract_prototype_mean(optional_flags, tractography, smooth_order, file_output=None):
    from .tract_obb import prototype_tract

    tracts = tractography.tracts()
    prototype_ix, leave_centers = prototype_tract(
        tracts, return_leave_centers=True)

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
            mean_tract = numpy.ascontiguousarray(numpy.transpose(interpolate.splev(u, tck)))
        except ImportError:
            warn("A smooth order larger than 0 needs scipy installed")

    return Tractography([mean_tract], {}, **tractography.extra_args)


@tract_math_operation('<volume unit> <tract1.vtk> ... <tractN.vtk>: calculates the Bhattacharyya coefficient of the first tract with the rest in the space of the reference image')
def tract_bhattacharyya_coefficient(optional_flags, tractography, resolution, *other_tracts):
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
        numpy.histogram(tractography_points[:, i], bins=bins[
                        i], density=True, range=(mn_[i], mx_[i]))[0]
        for i in xrange(3)
    ]

    for tract, tract_points in zip(other_tracts, other_tracts_points):
        hists_other_tract = [
            numpy.histogram(
                tract_points[:, i], bins=bins[i], density=True, range=(mn_[i], mx_[i]))[0]
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
            result['bhattacharyya %s value' % coord[i]].append(
                numpy.nan_to_num(distances[i]))

    return result


@tract_math_operation(
    '<image> <label>: Flips tracts such that the first endpoint is '
    'in the given label',
    needs_one_tract=True
)
def tract_flip_endpoints_in_label(
    tractography, image, label, file_output=None
):
    image = nibabel.load(image)
    tracts_ijk = tract_operations.each_tract_in_ijk(image, tractography)
    image_data = image.get_data()
    label = int(label)
    print image_data.sum()
    needs_flip = []
    for ix, tract in enumerate(tracts_ijk):
        i, j, k = numpy.round(tract[0]).astype(int)
        l, m, n = numpy.round(tract[-1]).astype(int)

        e1 = image_data[i, j, k] == label
        e2 = image_data[l, m, n] == label
        if e2 and not e1:
            needs_flip.append(ix)
        elif e1 and e2:
            warn("At least one tract has both endpoints in the label")
        elif not(e1 or e2):
            warn("At least one tract none of its endpoints in the label")

    tracts = list(tractography.tracts())
    tracts_data = tractography.tracts_data()
    print "Flipped %d tracts" % len(needs_flip)
    for i in needs_flip:
        tracts[i] = tracts[i][::-1]
        for data_key, data_points in tracts_data:
            data_points[i] = data_points[i][::-1]

    return Tractography(
        tracts,  tracts_data,
        **tractography.extra_args
    )
