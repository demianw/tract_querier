'''
Anisotropy measures from tensor operations
'''
import numpy


def fractional_anisotropy(tensor_array):
    r'''
    Fractional Anisotropy measure computed as

    .. math:
        FA = \sqrt{\frac {3(mn \cdot I - T) : (mn \cdot I - T)} {2(T : T)}}

        mn = Tr(T) / 3
    '''
    diag = (0, 1, 2)
    mean_diffusion = tensor_array[:, diag, diag].sum(-1) / 3.
    denominator = 2. * (tensor_array * tensor_array).sum(-1).sum(-1)
    deviation = tensor_array.copy()
    deviation[:, 0, 0] -= mean_diffusion
    deviation[:, 1, 1] -= mean_diffusion
    deviation[:, 2, 2] -= mean_diffusion
    deviation = (deviation * deviation).sum(-1).sum(-1)

    return numpy.sqrt(3. * deviation / denominator)
