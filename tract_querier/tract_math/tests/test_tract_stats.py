from ... import tractography
from .. import tract_stats

import os
import tempfile

import numpy
from numpy.testing import assert_array_equal
from nose import tools


def test_tract_scalar_to_prototype_one_to_one(random=numpy.random.RandomState(0), length=50, bundle_size=50):
    prototype_tract = numpy.c_[
        numpy.zeros(length),
        numpy.zeros(length),
        numpy.linspace(0, 1, length)
    ]

    bundle_tracts = []
    data = {'point_id': []}
    for i in xrange(bundle_size):
        x, y = random.randn(2)
        bundle_tract = prototype_tract.copy()
        bundle_tract[:, 0] = x
        bundle_tract[:, 1] = y
        bundle_tracts.append(bundle_tract)
        data['point_id'].append(numpy.arange(length)[:, None])

    tractography_prototype = tractography.Tractography([prototype_tract])
    tractography_bundle = tractography.Tractography(bundle_tracts, data)

    fname_proto = tempfile.mkstemp(suffix='.vtk', prefix='proto')[1]

    tractography.tractography_to_file(
        fname_proto, tractography_prototype
    )

    projected_tractography = tract_stats.tract_scalar_to_prototype.original_function(
        [tractography_bundle], fname_proto, 'point_id'
    )

    os.remove(fname_proto)

    enumeration = numpy.arange(length)[:, None]

    tools.eq_(len(projected_tractography.tracts()), 1)
    assert_array_equal(prototype_tract, projected_tractography.tracts()[0])

    for k, v in projected_tractography.tracts_data().iteritems():
        if k.startswith('point_id') and k.endswith('mean'):
            assert_array_equal(enumeration, v[0])
        elif k.startswith('point_id') and k.endswith('std'):
            assert_array_equal(v[0], v[0] * 0)


def test_tract_scalar_to_prototype_one_to_many(random=numpy.random.RandomState(0), length=50, bundle_size=50):
    prototype_tract = numpy.c_[
        numpy.zeros(length),
        numpy.zeros(length),
        numpy.linspace(0, 1, length)
    ]

    bundle_tracts = []
    data = {'point_id': []}
    for i in xrange(bundle_size):
        x, y = random.randn(2)
        bundle_tract = prototype_tract[::2].copy()
        bundle_tract[:, 0] = x
        bundle_tract[:, 1] = y
        bundle_tracts.append(bundle_tract)
        data['point_id'].append(numpy.arange(length)[::2, None])

    tractography_prototype = tractography.Tractography([prototype_tract])
    tractography_bundle = tractography.Tractography(bundle_tracts, data)

    fname_proto = tempfile.mkstemp(suffix='.vtk', prefix='proto')[1]

    tractography.tractography_to_file(
        fname_proto, tractography_prototype
    )

    projected_tractography = tract_stats.tract_scalar_to_prototype.original_function(
        [tractography_bundle], fname_proto, 'point_id'
    )

    os.remove(fname_proto)

    enumeration = numpy.arange(length)[:, None]
    nanarray = numpy.repeat(numpy.nan, length)[:, None]

    tools.eq_(len(projected_tractography.tracts()), 1)
    assert_array_equal(prototype_tract, projected_tractography.tracts()[0])

    for k, v in projected_tractography.tracts_data().iteritems():
        v = v[0]
        if k.startswith('point_id') and k.endswith('mean'):
            assert_array_equal(v[0::2], enumeration[0::2])
            assert_array_equal(v[1::2], nanarray[1::2])
        elif k.startswith('point_id') and k.endswith('std'):
            assert_array_equal(v[0::2], v[0::2] * 0)
            assert_array_equal(v[1::2], nanarray[1::2])
