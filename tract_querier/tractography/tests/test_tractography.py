from .. import Tractography

from nose.tools import with_setup
import copy
from itertools import izip, chain

from numpy import any, all
from numpy.random import randint, randn

dimensions = None
tracts = None
tracts_data = None
tractography = None
max_tract_length = 50
n_tracts = 50


def equal_tracts(a, b):
    for t1, t2 in izip(a, b):
        if not all(t1 == t2):
            return False

    return True


def equal_tracts_data(a, b):
    if set(a.keys()) != set(b.keys()):
        return False

    for k in a.keys():
        for t1, t2 in izip(a[k], b[k]):
            if not all(t1 == t2):
                return False
    return True


def equal_tractography(a, b):
    return (
        equal_tracts(a.tracts(), b.tracts()) and
        equal_tracts_data(a.tracts_data(), b.tracts_data())
    )


def setup(*args, **kwargs):
    global dimensions
    global tracts
    global tracts_data
    global tractography

    dimensions = [(randint(5, max_tract_length), 3) for _ in xrange(n_tracts)]
    tracts = [randn(*d) for d in dimensions]
    tracts_data = {
        'a%d' % i: [
            randn(d[0], k)
            for d in dimensions
        ]
        for i, k in zip(xrange(4), randint(1, 5, 4))
    }
    tractography = Tractography(tracts, tracts_data)


@with_setup(setup)
def test_creation():
    assert(equal_tracts(tractography.tracts(), tracts))
    assert(equal_tracts_data(tractography.tracts_data(), tracts_data))
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

    assert(not equal_tracts(tractography.tracts(), tractography.original_tracts()))
    assert(equal_tracts(tracts, tractography.original_tracts()))
    assert(not equal_tracts_data(tractography.tracts_data(), tractography.original_tracts_data()))
    assert(equal_tracts_data(tracts_data, tractography.original_tracts_data()))
    assert(tractography.are_tracts_subsampled())

    tractography.unsubsample_tracts()
    assert(equal_tracts(tractography.tracts(), tracts))
    assert(equal_tracts_data(tractography.tracts_data(), tracts_data))
    assert(not tractography.are_tracts_subsampled())
    assert(not tractography.are_tracts_filtered())


@with_setup(setup)
def test_append():
    old_tracts = copy.deepcopy(tractography.tracts())
    new_data = {}
    for k, v in tracts_data.iteritems():
        new_data[k] = v + v

    tractography.append(tracts, tracts_data)

    assert(equal_tracts(tractography.tracts(), chain(old_tracts, old_tracts)))
    assert(equal_tracts_data(tractography.tracts_data(), new_data))
