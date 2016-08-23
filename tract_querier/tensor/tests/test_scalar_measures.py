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


def test_volume_fraction(N=10, random=numpy.random.RandomState(0)):
    tensors = random.randn(N, 3, 3)
    vf = numpy.empty(N)
    for i, t in enumerate(tensors):
        tt = numpy.dot(t, t.T)
        tensors[i] = tt
        ev = numpy.linalg.eigvalsh(tt)
        mn = ev.mean()
        vf[i] = 1 - ev.prod() / (mn ** 3)

    assert_array_almost_equal(vf, scalar_measures.volume_fraction(tensors))


def test_tensor_determinant(N=10, random=numpy.random.RandomState(0)):
    tensors = random.randn(N, 3, 3)
    dt = numpy.empty(N)
    for i, t in enumerate(tensors):
        tt = numpy.dot(t, t.T)
        tensors[i] = tt
        dt[i] = numpy.linalg.det(tt)

    assert_array_almost_equal(dt, scalar_measures.tensor_det(tensors))


def test_tensor_traces(N=10, random=numpy.random.RandomState(0)):
    tensors = random.randn(N, 3, 3)
    res = numpy.empty(N)
    for i, t in enumerate(tensors):
        tt = numpy.dot(t, t.T)
        tensors[i] = tt
        res[i] = numpy.trace(tt)

    assert_array_almost_equal(res, scalar_measures.tensor_trace(tensors))


def test_tensor_contraction(N=10, random=numpy.random.RandomState(0)):
    tensors1 = random.randn(N, 3, 3)
    tensors2 = random.randn(N, 3, 3)

    res = numpy.empty(N)
    for i in range(N):
        t1 = tensors1[i]
        t2 = tensors2[i]
        tt1 = numpy.dot(t1, t1.T)
        tt2 = numpy.dot(t2, t2.T)
        tensors1[i] = tt1
        tensors2[i] = tt2
        res[i] = numpy.trace(numpy.dot(tt1, tt2.T))

    assert_array_almost_equal(res, scalar_measures.tensor_contraction(tensors1, tensors2))
