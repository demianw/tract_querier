from .. import Tractography

from nose.tools import with_setup
import copy
from itertools import izip, chain

from numpy.random import randint, randn

dimensions = None
tracts = None
tract_data = None
tractography = None
max_tract_length = 50
n_tracts = 50


def setup(*args, **kwargs):
    global dimensions
    global tracts
    global tract_data
    global tractography

    dimensions = [(randint(5, max_tract_length), 3) for _ in xrange(n_tracts)]
    tracts = [randn(*d) for d in dimensions]
    tract_data = {
        'a%d' % i: [
            randn(d[0], k)
            for d in dimensions
        ]
        for i, k in zip(xrange(4), randint(1, 5, 4))
    }
    tractography = Tractography(tracts, tract_data)


@with_setup(setup)
def test_creation():
    assert(tractography.tracts() is tracts and tractography.original_tracts() is tracts)
    assert(tractography.tracts_data() is tract_data and tractography.original_tracts_data() is tract_data)
    assert(not tractography.are_tracts_subsampled())
    assert(not tractography.are_tracts_filtered())


@with_setup(setup)
def test_subsample_tracts():
    tractography.subsample_tracts(5)

    assert(all(tract.shape[0] <= 5 for tract in tractography.tracts()))
    assert(
        all(
            all(values.shape[0] <= 5 for values in value)
            for value in tractography.tracts_data().values()
        )
    )
    assert(tractography.tracts() is not tracts and tractography.original_tracts() is tracts)
    assert(tractography.tracts_data() is not tract_data and tractography.original_tracts_data() is tract_data)
    assert(tractography.are_tracts_subsampled())

    tractography.unsubsample_tracts()
    assert(tractography.tracts() is tracts and tractography.original_tracts() is tracts)
    assert(tractography.tracts_data() is tract_data and tractography.original_tracts_data() is tract_data)
    assert(not tractography.are_tracts_subsampled())


@with_setup(setup)
def test_append():
    old_tracts = copy.deepcopy(tractography.tracts())
    new_data = {}
    for k, v in tract_data.iteritems():
        new_data[k] = v + v

    tractography.append(tracts, tract_data)
    for t1, t2 in izip(tractography.tracts(), chain(old_tracts, old_tracts)):
        assert((t1 == t2).all())

    for k in tractography.tracts_data():
        for t1, t2 in izip(tractography.tracts_data()[k], new_data[k]):
            assert((t1 == t2).all())
