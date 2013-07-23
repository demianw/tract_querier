from .decorator import tract_math_operation, TractMathWrongArgumentsError
from warnings import warn


try:
    from collections import OrderedDict
except ImportError:  # Python 2.6 fix
    from ordereddict import OrderedDict

import numpy

import nibabel
from nibabel.spatialimages import SpatialImage

from ..tractography import Tractography, tractography_to_file, tractography_from_files
from ..tensor import scalar_measures


@tract_math_operation('<prototype> <output_scalar_name> [distance threshold=8.] <file_output>: projects a scalar value of the tract to the prototype', needs_one_tract=False)
def tract_scalar_to_prototype(tractographies, prototype_tract, scalar_name, dist_threshold=8., file_output=None):
    divide_by_zero_err = numpy.geterr()
    numpy.seterr(divide='ignore', invalid='ignore')

    prototype = tractography_from_files([prototype_tract])
    if len(prototype.tracts()) != 1:
        raise TractMathWrongArgumentsError('Prototype file %s must have a single tract' % prototype_tract)
    for i, tractography in enumerate(tractographies):
        if scalar_name not in tractography.tracts_data():
            raise TractMathWrongArgumentsError(
                'Input tractography %d does not have '
                'the required scalar property' % i
            )

    prototype_tract = prototype.tracts()[0]
    tangents = numpy.r_[(prototype_tract[:-1] - prototype_tract[1:]), (prototype_tract[-2] - prototype_tract[-1])[None, :]]
    prototype_arc_length = numpy.r_[0, numpy.cumsum(numpy.sqrt((tangents ** 2).sum(1))[:-1])]
    tangent_tensor = tangents[..., None] * tangents[:, None, ...]

    output_scalar_name = scalar_name + '_%04d'
    output_data = prototype.tracts_data()
    for i, tractography in enumerate(tractographies):
        output_scalar_name_ = output_scalar_name % i
        if output_scalar_name_ in output_data:
            warn('Warning, scalar data %s exists in the prototype tractography' % output_scalar_name_)

        prototype_value_count, prototype_value_sqr_buffer, prototype_value_buffer = project_tractography_to_prototype(
            tractography, scalar_name,
            prototype_tract, tangent_tensor,
            dist_threshold=dist_threshold
        )

        mean_value = numpy.nan_to_num(prototype_value_buffer / prototype_value_count)
        mean_value[prototype_value_count == 0] = numpy.nan
        std_value = numpy.sqrt(numpy.nan_to_num(
            prototype_value_sqr_buffer / prototype_value_count -
            mean_value ** 2
        ))
        std_value[prototype_value_count == 0] = numpy.nan

        output_data[output_scalar_name_ + '_mean'] = [mean_value]
        output_data[output_scalar_name_ + '_std'] = [std_value]

    output_data['arc_length'] = [prototype_arc_length[:, None]]
    output_tractography = Tractography(prototype.tracts(), output_data)

    numpy.seterr(**divide_by_zero_err)
    return output_tractography


@tract_math_operation('<prototype> <output_scalar_name> [distance threshold=8.] <file_output>: projects the prototype arc-length to the tractography', needs_one_tract=True)
def tract_prototype_arclength(tractography, prototype_tract, output_scalar_name, dist_threshold=8., file_output=None):
    if isinstance(dist_threshold, str):
        dist_threshold = float(dist_threshold)
    prototype = tractography_from_files([prototype_tract])
    if len(prototype.tracts()) != 1:
        raise TractMathWrongArgumentsError('Prototype file %s must have a single tract' % prototype_tract)

    prototype_tract = prototype.tracts()[0]
    tangents = numpy.r_[(prototype_tract[:-1] - prototype_tract[1:]), (prototype_tract[-2] - prototype_tract[-1])[None, :]]
    prototype_arc_length = (numpy.r_[0, numpy.cumsum(numpy.sqrt((tangents ** 2).sum(1))[:-1])])[:, None]
    tangent_tensor = tangents[..., None] * tangents[:, None, ...]

    scalar_data = project_prototype_to_tractography(
        tractography, prototype_tract, tangent_tensor,
        prototype_arc_length, dist_threshold=dist_threshold
    )

    tracts_data = tractography.tracts_data()
    tracts_data[output_scalar_name] = scalar_data
    return Tractography(tractography.tracts(), tracts_data)


@tract_math_operation('<prototype> <scalar> <output_scalar_name> [distance threshold=8.] <output_tractography>: projects the prototype arc-length to the tractography', needs_one_tract=True)
def tract_prototype_scalar_to_tract(tractography, prototype_tract, scalar, output_scalar_name, dist_threshold=8., file_output=None):
    try:
        prototype = tractography_from_files([prototype_tract])
        if len(prototype.tracts()) != 1:
            raise TractMathWrongArgumentsError('Prototype file %s must have a single tract' % prototype_tract)

        prototype_tract = prototype.tracts()[0]
        tangents = numpy.r_[(prototype_tract[:-1] - prototype_tract[1:]), (prototype_tract[-2] - prototype_tract[-1])[None, :]]
        tangent_tensor = tangents[..., None] * tangents[:, None, ...]

        scalar_data = project_prototype_to_tractography(
            tractography, prototype_tract,
            tangent_tensor, prototype.tracts_data()[scalar][0],
            dist_threshold=dist_threshold
        )

        tracts_data = tractography.tracts_data()
        tracts_data[output_scalar_name] = scalar_data
        return Tractography(tractography.tracts(), tracts_data)
    except KeyError:
        raise TractMathWrongArgumentsError('Scalar value %s not found in prototype' % scalar)


def project_tractography_to_prototype(tractography, scalar_name, prototype_tract, tangent_tensor, dist_threshold=8.):
    dist_threshold2 = dist_threshold ** 2
    prototype_value_buffer = numpy.zeros((len(prototype_tract), 1))
    prototype_value_sqr_buffer = numpy.zeros((len(prototype_tract), 1))
    prototype_value_count = numpy.zeros((len(prototype_tract), 1), dtype=int)
    tracts = tractography.tracts()
    scalar_data = tractography.tracts_data()[scalar_name]
    for i, tract in enumerate(tracts):
        scalar_data_point = scalar_data[i]
        for j, point in enumerate(tract):
            ix = prototype_index_for_point(tangent_tensor, dist_threshold2, prototype_tract, point)
            if ix is None:
                continue
            value = scalar_data_point[j]
            prototype_value_buffer[ix, 0] += value
            prototype_value_sqr_buffer[ix, 0] += value ** 2
            prototype_value_count[ix, 0] += 1
    return prototype_value_count, prototype_value_sqr_buffer, prototype_value_buffer


def project_prototype_to_tractography(tractography, prototype_tract, tangent_tensor, prototype_data, dist_threshold=8.):
    dist_threshold2 = dist_threshold ** 2
    tracts = tractography.tracts()
    output_data = []
    nan_vector = numpy.repeat(numpy.nan, prototype_data.shape[1], axis=0)[None, :]
    for i, tract in enumerate(tracts):
        tract_data = numpy.repeat(nan_vector, len(tract), axis=0)
        for j, point in enumerate(tract):
            ix = prototype_index_for_point(tangent_tensor, dist_threshold2, prototype_tract, point)
            if ix is None:
                continue
            tract_data[j] = prototype_data[ix]
        output_data.append(tract_data)
    return output_data


def prototype_index_for_point(tangent_tensor, dist_threshold2, prototype_tract, point):
    differences = (prototype_tract - point[None, :])
    differences = (prototype_tract - point[None, :])
    distances2 = (differences ** 2).sum(1)
    sel_points = distances2 < dist_threshold2
    if not numpy.any(sel_points):
        return None

    differences_sel_points = differences[sel_points]

    mh_differences = (
        (tangent_tensor[sel_points] * differences_sel_points[:, None, :]).sum(2) *
        differences_sel_points
    ).sum(1)
    min_points_ix = (mh_differences == mh_differences.min()).nonzero()
    sel_points_ix = sel_points.nonzero()[0]
    if len(min_points_ix[0]) == 1:
        ix = sel_points_ix[min_points_ix[0][0]]
    else:
        ix = sel_points_ix[min_points_ix[0][((differences[min_points_ix, :] ** 2).sum(1)).argmin()]]
    return ix

@tract_math_operation('<measure> <output_scalar_name> <output_tractography>: calculates a tensor-derived measure', needs_one_tract=True)
def tract_tensor_measure(tractography, measure, scalar_name, file_output=None):
    try:
        tensors = tractography.tracts_data()['tensors']

        if measure == 'FA':
            measure_func = scalar_measures.fractional_anisotropy

        measure = []
        for td in tensors:
            t = td.reshape(len(td), 3, 3)
            measure.append(measure_func(t))

        tractography.tracts_data()[scalar_name] = measure

        return tractography

    except KeyError:
        raise TractMathWrongArgumentsError('Tensor data should be in the tractography and named "tensors"')

