from .decorator import tract_math_operation, TractMathWrongArgumentsError
from warnings import warn


try:
    from collections import OrderedDict
except ImportError:  # Python 2.6 fix
    from ordereddict import OrderedDict

import numpy


from ..tractography import Tractography, tractography_from_files


@tract_math_operation('<prototype> <output_scalar_name> [distance threshold=8.] <file_output>: projects a scalar value of the tract to the prototype', needs_one_tract=False)
def tract_scalar_to_prototype(tractographies, prototype_tract, scalar_name, dist_threshold=8., file_output=None):
    import re

    divide_by_zero_err = numpy.geterr()
    numpy.seterr(divide='ignore', invalid='ignore')

    prototype = tractography_from_files([prototype_tract])
    if len(prototype.tracts()) != 1:
        raise TractMathWrongArgumentsError('Prototype file %s must have a single tract' % prototype_tract)

    scalar_names = [
        scalar_name_ for scalar_name_ in tractographies[0].tracts_data()
        if re.match('^' + scalar_name + '$', scalar_name_)
    ]
    scalar_names_set = set(scalar_names)
    for i, tractography in enumerate(tractographies):
        if not set(tractography.tracts_data().keys()).issuperset(scalar_names_set):
            raise TractMathWrongArgumentsError(
                'Input tractography %d does not have '
                'the required scalar property' % i
            )

    prototype_tract = prototype.tracts()[0]
    tangents = numpy.r_[(prototype_tract[:-1] - prototype_tract[1:]), (prototype_tract[-2] - prototype_tract[-1])[None, :]]
    prototype_arc_length = numpy.r_[0, numpy.cumsum(numpy.sqrt((tangents ** 2).sum(1))[:-1])]
    tangent_tensor = tangents[..., None] * tangents[:, None, ...]

    for scalar_name in scalar_names:
        output_scalar_name = scalar_name
        output_data = prototype.tracts_data()
        for i, tractography in enumerate(tractographies):
            if len(tractographies) > 1:
                output_scalar_name_ = output_scalar_name + '_%04d' % i
            else:
                output_scalar_name_ = output_scalar_name

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


@tract_math_operation('<scalar name> <output_file>: Output a csv file per tract with the x,y,z coordinates, the arc-length and a set of scalar values')
def tract_export(tractography, scalar_name, file_output=None):
    import re

    output = OrderedDict([
        ('tract_number', []),
        ('x', []),
        ('y', []),
        ('z', []),
        ('arc_length', []),
    ])

    scalar_names = [
        scalar_name_ for scalar_name_ in tractography.tracts_data()
        if re.match('^' + scalar_name + '$', scalar_name_)
    ]

    if len(scalar_names) == 0:
        raise TractMathWrongArgumentsError('Scalar attribute %s not found' % scalar_name)

    for scalar_name_ in scalar_names:
        output[scalar_name_] = []

    for i, tract in enumerate(tractography.tracts()):
        x, y, z = tract.T

        tangents = numpy.r_[(tract[:-1] - tract[1:]), (tract[-2] - tract[-1])[None, :]]
        arc_length = numpy.r_[0, numpy.cumsum(numpy.sqrt((tangents ** 2).sum(1))[:-1])]

        output['x'] += x.tolist()
        output['y'] += y.tolist()
        output['z'] += z.tolist()
        output['arc_length'] += arc_length.tolist()
        output['tract_number'] += [i] * len(tract)

        for scalar_name_ in scalar_names:
            tract_data = tractography.tracts_data()[scalar_name_][i]
            output[scalar_name_] += tract_data.squeeze().tolist()

    return output


@tract_math_operation('<scalar name> <csv_input_file> <output_file>: Input a column of a csv file to the tractography')
def tract_import(tractography, scalar_name, csv_input, file_output=None):
    if len(tractography.tracts()) != 1:
        raise TractMathWrongArgumentsError('The input tractography for this operation can only have one tract')
    import re
    import csv

    reader = csv.DictReader(open(csv_input))

    scalar_names = [
        name for name in reader.fieldnames
        if re.match('^' + scalar_name + '$', name)
    ]

    tracts_data = tractography.tracts_data()
    tract = tractography.tracts()[0]
    for name in scalar_names:
        tracts_data[name] = [numpy.empty(len(tract))[:, None]]

    for i, row in enumerate(reader):
        for name in scalar_names:
            tracts_data[name][0][i] = float(row[name])

    if i < len(tract[0]):
        raise TractMathWrongArgumentsError('The input CSV needs to have %d rows' % len(tract))

    return tractography


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


