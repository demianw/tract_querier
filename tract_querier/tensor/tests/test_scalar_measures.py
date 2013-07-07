from .. import scalar_measures

import numpy
from numpy.testing import assert_array_almost_equal


def test_fractional_anisotropy(N=10, random=numpy.random.RandomState(0)):
    tensors = random.randn(N, 3, 3)
    fa = numpy.empty(N)
    for i, t in enumerate(tensors):
        tt = numpy.dot(t, t.T)
        tensors[i] = tt
        ev = numpy.linalg.eigvalsh(tt)
        mn = ev.mean()
        fa[i] = numpy.sqrt(1.5 * ((ev - mn) ** 2).sum() / (ev ** 2).sum())

    assert_array_almost_equal(fa, scalar_measures.fractional_anisotropy(tensors))
