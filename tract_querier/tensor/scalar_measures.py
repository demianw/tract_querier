'''
Anisotropy measures from tensor operations
'''
import numpy


__all__ = [
    'fractional_anisotropy', 'volume_fraction',
    'eigenvalues', 'tensor_trace', 'tensor_contraction',
    'tensor_det'
]


def fractional_anisotropy(tensor_array):
    r'''
    Fractional Anisotropy (Basser et al.) measure computed as

    .. math::
        FA = \sqrt{\frac {3(mn \cdot I - T) : (mn \cdot I - T)} {2(T : T)}}

        mn = Tr(T) / 3
    '''
    mean_diffusion = tensor_trace(tensor_array) / 3.
    denominator = 2. * tensor_contraction(tensor_array, tensor_array)
    deviation = (tensor_array[:, 0, 0] - mean_diffusion) ** 2
    deviation += (tensor_array[:, 1, 1] - mean_diffusion) ** 2
    deviation += (tensor_array[:, 2, 2] - mean_diffusion) ** 2
    deviation += tensor_array[:, 0, 1] ** 2 * 2
    deviation += tensor_array[:, 0, 2] ** 2 * 2
    deviation += tensor_array[:, 2, 1] ** 2 * 2

    return numpy.sqrt(3. * deviation / denominator)


def volume_fraction(tensor_array):
    r'''
    Volume Fraction (Basser et al.) measure computed as:

    .. math::
        VF = 1 - \frac{|T|} {(Tr(T) / 3.)^3}
    '''
    mean_diffusion = tensor_trace(tensor_array) / 3.
    det_ = tensor_det(tensor_array)

    return 1. - det_ / (mean_diffusion ** 3)


def eigenvalues(tensor_array):
    r'''
    Eigenvalues of the tensors
    '''
    res = numpy.empty((len(tensor_array), 3), dtype=tensor_array.dtype)
    for i, t in enumerate(tensor_array):
        res[i] = numpy.linalg.eigvalsh(t)

    return res


def tensor_det(tensor_array):
    r'''
    Determinant of a tensor array
    '''
    return (
        tensor_array[:, 0, 0] * tensor_array[:, 1, 1] * tensor_array[:, 2, 2] +
        tensor_array[:, 0, 1] * tensor_array[:, 1, 2] * tensor_array[:, 2, 0] +
        tensor_array[:, 0, 2] * tensor_array[:, 1, 0] * tensor_array[:, 2, 1] -
        tensor_array[:, 0, 2] * tensor_array[:, 1, 1] * tensor_array[:, 2, 0] -
        tensor_array[:, 0, 1] * tensor_array[:, 1, 0] * tensor_array[:, 2, 2] -
        tensor_array[:, 0, 0] * tensor_array[:, 1, 2] * tensor_array[:, 2, 1]
    )


def tensor_trace(tensor_array):
    r'''
    Trace of each tensor
    '''
    diag = (0, 1, 2)
    return tensor_array[:, diag, diag].sum(-1)


def tensor_contraction(tensor_array_1, tensor_array_2):
    r'''
    Return the contraction of each tensor pair
    '''
    return (tensor_array_1 * tensor_array_2).sum(-1).sum(-1)
