from .. import Tractography
from .. import (
    tractography_from_trackvis_file, tractography_to_trackvis_file,
    tractography_from_files, tractography_to_file
)

try:
    VTK = True
    from ..vtkInterface import (
        tractography_from_vtk_files, tractography_to_vtk_file,
    )
except ImportError:
    VTK = False

from nose.tools import with_setup
import copy
from itertools import chain

from numpy import all, eye, ones, allclose
from numpy.random import randint, randn
from numpy.testing import assert_array_equal

dimensions = None
tracts = None
tracts_data = None
tractography = None
max_tract_length = 50
n_tracts = 50


def equal_tracts(a, b):
    for t1, t2 in zip(a, b):
        if not (len(t1) == len(t2) and allclose(t1, t2)):
            return False

    return True


def equal_tracts_data(a, b):
    if set(a.keys()) != set(b.keys()):
        return False

    for k in a.keys():
        v1 = a[k]
        v2 = b[k]
        if isinstance(v1, str) and isinstance(v2, str) and v1 == v2:
            continue
        elif not isinstance(v1, str) and not isinstance(v2, str):
            for t1, t2 in zip(a[k], b[k]):
                if not (len(t1) == len(t2) and allclose(t1, t2)):
                    return False
        else:
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

    if 'test_active_data' in kwargs:
        test_active_data = kwargs['test_active_data']
    else:
        test_active_data = False

    dimensions = [(randint(5, max_tract_length), 3) for _ in range(n_tracts)]
    tracts = [randn(*d) for d in dimensions]
    tracts_data = {
        'a%d' % i: [
            randn(d[0], k)
            for d in dimensions
        ]
        for i, k in zip(range(4), randint(1, 3, 9))
    }

    if test_active_data:
        mask = 0
        for k, v in tracts_data.items():
            if mask & (1 + 2 + 4):
                break
            if v[0].shape[1] == 1 and mask & 1 == 0:
                tracts_data['ActiveScalars'] = k
                mask |= 1
            if v[0].shape[1] == 3 and mask & 2 == 0:
                tracts_data['ActiveVectors'] = k
                mask |= 2
            if v[0].shape[1] == 9 and mask & 4 == 0:
                tracts_data['ActiveTensors'] = k
                mask |= 4

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
    for k, v in tracts_data.items():
        new_data[k] = v + v

    tractography.append(tracts, tracts_data)

    assert(equal_tracts(tractography.tracts(), chain(old_tracts, old_tracts)))
    assert(equal_tracts_data(tractography.tracts_data(), new_data))


if VTK:
    @with_setup(setup)
    def test_saveload_vtk():
        import tempfile
        import os
        fname = tempfile.mkstemp('.vtk')[1]
        tractography_to_vtk_file(fname, tractography)

        new_tractography = tractography_from_vtk_files(fname)

        assert(equal_tracts(tractography.tracts(), new_tractography.tracts()))
        assert(equal_tracts_data(
            tractography.tracts_data(),
            new_tractography.tracts_data())
        )

        os.remove(fname)

    @with_setup(setup)
    def test_saveload_vtp():
        import tempfile
        import os
        fname = tempfile.mkstemp('.vtp')[1]
        tractography_to_vtk_file(fname, tractography)

        new_tractography = tractography_from_vtk_files(fname)

        assert(equal_tracts(tractography.tracts(), new_tractography.tracts()))
        assert(equal_tracts_data(tractography.tracts_data(), new_tractography.tracts_data()))

        os.remove(fname)


@with_setup(setup)
def test_saveload_trk():
    import tempfile
    import os
    fname = tempfile.mkstemp('.trk')[1]

    tract_data_new = {
        k: v
        for k, v in tractography.tracts_data().items()
        if (v[0].ndim == 1) or (v[0].ndim == 2 and v[0].shape[1] == 1)
    }

    tractography_ = Tractography(tractography.tracts(), tract_data_new)

    tractography_to_trackvis_file(
        fname, tractography_,
        affine=eye(4), image_dimensions=ones(3)
    )

    new_tractography = tractography_from_trackvis_file(fname)

    assert(equal_tracts(tractography_.tracts(), new_tractography.tracts()))
    assert(equal_tracts_data(tractography_.tracts_data(), new_tractography.tracts_data()))
    assert_array_equal(eye(4), new_tractography.affine)
    assert_array_equal(ones(3), new_tractography.image_dims)

    os.remove(fname)


@with_setup(setup)
def test_saveload():
    import tempfile
    import os

    extensions = ('.trk',)
    if VTK:
        extensions += ('.vtk', '.vtp')

    for ext in extensions:
        fname = tempfile.mkstemp(ext)[1]

        kwargs = {}

        if ext == '.trk':
            kwargs['affine'] = eye(4)
            kwargs['image_dimensions'] = ones(3)

            tract_data_new = {
                k: v
                for k, v in tractography.tracts_data().items()
                if (v[0].ndim == 1) or (v[0].ndim == 2 and v[0].shape[1] == 1)
            }

            tractography_ = Tractography(tractography.tracts(), tract_data_new)
        else:
            tractography_ = tractography

        tractography_to_file(
            fname, tractography_,
            **kwargs
        )

        new_tractography = tractography_from_files(fname)

        assert(equal_tracts(tractography_.tracts(), new_tractography.tracts()))
        assert(equal_tracts_data(
            tractography_.tracts_data(),
            new_tractography.tracts_data()
        ))

        os.remove(fname)
