import numpy
from ..tractography import Tractography
from . import tract_operations
from ..tensor import scalar_measures

try:
    from collections import OrderedDict
except ImportError:  # Python 2.6 fix
    from ordereddict import OrderedDict



def compute_all_measures(tractography, desired_keys_list, scalars=None, resolution=None):

    unordered_results = dict()

    if ('number of tracts' in desired_keys_list):
        unordered_results['number of tracts'] = tract_operations.tract_count(
            tractography.tracts())

    if ('length mean (mm)' in desired_keys_list) or ('length std (mm^2)' in desired_keys_list):
        lengths = numpy.empty(len(tractography.tracts()))
        for i, one_tract in enumerate(tractography.tracts()):
            lengths[i] = tract_operations.tract_length(one_tract)
        unordered_results['length mean (mm)'] = lengths.mean()
        unordered_results['length std (mm^2)'] = lengths.std()

    if ('tract volume' in desired_keys_list) and (resolution is not None):
        resolution = float(resolution)
        voxels = tract_operations.voxelized_tract(tractography, resolution)

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
        eroded_voxels = set()
        for voxel in voxels:
            neighbors_list = zip(*(neighbors + voxel).T)
            dilated_voxels.update(neighbors_list)
            if len(voxels.intersection(neighbors_list)) == len(neighbors):
                eroded_voxels.add(voxel)
        # print len(dilated_voxels), len(voxels), len(eroded_voxels)
        approx_voxels = (len(dilated_voxels) - len(eroded_voxels)) / 2.
        approx_volume = approx_voxels * (resolution ** 3)
        unordered_results['tract volume'] = approx_volume

    if ('per tract distance weighted mean %s' in desired_keys_list ) or \
            ('per tract distance weighted std %s' in desired_keys_list):
        mean_keys_list = list()
        std_keys_list = list()
        for scalar in scalars:
            mean_key = 'per tract distance weighted mean %s' % scalar
            std_key = 'per tract distance weighted std %s' % scalar
            mean_keys_list.append(mean_key)
            std_keys_list.append(std_key)
            scalars = tractography.tracts_data()[scalar]
            weighted_scalars = numpy.empty((len(tractography.tracts()), 2))
            for line_index, t_data in enumerate(tractography.tracts()):
                tdiff = numpy.sqrt((numpy.diff(t_data, axis=0) ** 2).sum(-1))
                length = tdiff.sum()
                values = scalars[line_index][1:].squeeze()
                average = numpy.average(values, weights=tdiff)
                weighted_scalars[line_index, 0] = average
                weighted_scalars[line_index, 1] = length
            mean = numpy.average(
                weighted_scalars[:, 0], weights=weighted_scalars[:, 1])
            std = numpy.average(
                (weighted_scalars[:, 0] - mean) ** 2, weights=weighted_scalars[:, 1])
            unordered_results[mean_key] = mean
            unordered_results[std_key] = std
        mii = desired_keys_list.index('per tract distance weighted mean %s')
        desired_keys_list[mii:mii + 1] = mean_keys_list
        sii = desired_keys_list.index('per tract distance weighted std %s')
        desired_keys_list[sii:sii + 1] = std_keys_list
    # Make Ordered Dictionary
    ordered_dict = OrderedDict()
    for key in desired_keys_list:
        ordered_dict[key] = unordered_results[key]
    return ordered_dict


def tract_expand_tensor_metrics(tractography):
    from os import path
    from scipy import ndimage
    from numpy import linalg

    quantity_name = "tensor1_FA"
    start = 0
    new_scalar_data = []
    for tract in tractography.original_tracts():
        new_scalar_data.append(
            new_scalar_data_flat[start: start + len(tract)].copy()
        )
        start += len(tract)
    tractography.original_tracts_data()[quantity_name] = new_scalar_data

    return Tractography(
        tractography.original_tracts(),  tractography.original_tracts_data(),
        **tractography.extra_args
    )


def decorate_tract_with_measures(tractography, tensor_name):
    ot = tractography.original_tracts_data()
    all_tensors = ot[tensor_name]
    fa_fiber_list = list()
    md_fiber_list = list()
    ax_fiber_list = list()
    rd_fiber_list = list()
    ga_fiber_list = list()

    for one_fiber in all_tensors:
        fa_by_point = numpy.ndarray((len(one_fiber), 1), dtype=numpy.float32)
        md_by_point = numpy.ndarray((len(one_fiber), 1), dtype=numpy.float32)
        ax_by_point = numpy.ndarray((len(one_fiber), 1), dtype=numpy.float32)
        rd_by_point = numpy.ndarray((len(one_fiber), 1), dtype=numpy.float32)
        ga_by_point = numpy.ndarray((len(one_fiber), 1), dtype=numpy.float32)

        index = 0
        for one_tensor_values in one_fiber:
            one_tensor = numpy.reshape(one_tensor_values, (3, 3))
            _, eigenvals, _ = numpy.linalg.svd(one_tensor)
            fa_by_point[index] = scalar_measures.fractional_anisotropy_from_eigenvalues(eigenvals)
            md_by_point[index] = scalar_measures.mean_diffusivity(eigenvals)
            ax_by_point[index] = scalar_measures.axial_diffusivity(eigenvals)
            rd_by_point[index] = scalar_measures.radial_diffusivity(eigenvals)
            ga_by_point[index] = scalar_measures.geodesic_anisotropy(eigenvals)
            index = index + 1
        fa_fiber_list.append(fa_by_point)
        md_fiber_list.append(md_by_point)
        ax_fiber_list.append(ax_by_point)
        rd_fiber_list.append(rd_by_point)
        ga_fiber_list.append(ga_by_point)

    tractography.original_tracts_data()['FA_' + tensor_name] = fa_fiber_list
    tractography.original_tracts_data()['MD_' + tensor_name] = md_fiber_list
    tractography.original_tracts_data()['AX_' + tensor_name] = ax_fiber_list
    tractography.original_tracts_data()['RD_' + tensor_name] = rd_fiber_list
    tractography.original_tracts_data()['GA_' + tensor_name] = ga_fiber_list

    return Tractography(
        tractography.original_tracts(),  tractography.original_tracts_data(),
        **tractography.extra_args)
