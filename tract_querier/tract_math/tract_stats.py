from .decorator import tract_math_operation, TractMathWrongArgumentsError
from warnings import warn

import numpy


try:
    from collections import OrderedDict
except ImportError:  # Python 2.6 fix
    from ordereddict import OrderedDict


import nibabel

from tract_querier.tensor import scalar_measures
from tract_querier.tract_math.decorator import tract_math_operation, TractMathWrongArgumentsError


@tract_math_operation('<measure> [output_scalar_name] <output_tractography>: calculates a tensor-derived measure', needs_one_tract=True)
def tract_tensor_measure(tractography, measure, scalar_name=None, file_output=None):
    try:
        tensors = tractography.tracts_data()['tensors']
        measure = measure.upper()

        if scalar_name is None:
            scalar_name = measure

        if measure == 'FA':
            measure_func = scalar_measures.fractional_anisotropy
        elif measure == 'RD':
            measure_func = scalar_measures.radial_diffusivity
        elif measure == 'AD':
            measure_func = scalar_measures.axial_diffusivity
        elif measure == 'TR':
            measure_func = scalar_measures.tensor_trace
        elif measure == 'DET':
            measure_func = scalar_measures.tensor_det
        elif measure == 'VF':
            measure_func = scalar_measures.volume_fraction
        else:
            raise TractMathWrongArgumentsError(
                'Wrong anisotropy measure, avaliable measures are\n'
                '\tFA: Fractional anisotropy\n'
                '\tRD: Radial diffusivity\n'
                '\tAD: Axial diffusivity\n'
                '\tTR: Tract\n'
                '\tDET: Determinant\n'
                '\tVF: Volume fraction'
            )

        measure = []
        for td in tensors:
            t = td.reshape(len(td), 3, 3)
            measure.append(measure_func(t))

        tractography.tracts_data()[scalar_name] = measure

        return tractography

    except KeyError:
        raise TractMathWrongArgumentsError('Tensor data should be in the tractography and named "tensors"')


@tract_math_operation(
    '<scalars> <design_matrix> <output_scalar_prefix> <output_tractography>: fit a design matrix to a set of scalars on the tract',
    needs_one_tract=True
)
def tract_stat_ttest(tractography, scalars, design_matrix_fname, contrast_fname, output_scalar_prefix, file_output=None):
    import statsmodels.api as sm

    tract_data = tractography.tracts_data()

    if ' ' in scalars:
        scalars = scalars.split(' ')
    elif ',' in scalars:
        scalars = scalars.split(',')
    else:
        import re
        scalars = [
            scalar for scalar in tract_data.keys()
            if re.match('^' + scalars + '$', scalar)
        ]
        scalars.sort()

    try:
        design_matrix = numpy.atleast_2d(numpy.loadtxt(design_matrix_fname, comments='/'))
    except IOError:
        raise TractMathWrongArgumentsError('Error reading design matrix file %s' % design_matrix_fname)

    if design_matrix.shape[0] == 1:
        design_matrix = design_matrix.T

    if design_matrix.shape[0] != len(scalars):
        raise TractMathWrongArgumentsError('The number of rows in the matrix must be the same as scalar values')

    try:
        contrasts = numpy.atleast_2d(numpy.loadtxt(contrast_fname, comments='/'))
    except IOError:
        raise TractMathWrongArgumentsError('Error reading contrast file %s' % contrast_fname)

    if contrasts.shape[1] != design_matrix.shape[1]:
        raise TractMathWrongArgumentsError('The number of columns in the contrasts must be the same columns in the design matrix')

    tract_lengths = numpy.array([len(tract) for tract in tractography.tracts()])
    per_scalar_data = numpy.empty((tract_lengths.sum(), len(scalars)))

    for i, scalar in enumerate(scalars):
        try:
            value = tract_data[scalar]
            per_scalar_data[:, i] = numpy.hstack(value).squeeze()
        except KeyError:
            raise TractMathWrongArgumentsError('Scalar %s not found in the tractography' % scalar)

    for i, contrast in enumerate(contrasts):
        output_scalar = output_scalar_prefix + '_%04d' % i
        tract_data[output_scalar + '_t_score'] = []
        tract_data[output_scalar + '_t_pvalue'] = []
        tract_data[output_scalar + '_t_df'] = []
        tract_data[output_scalar + '_t_df'] = []
        tract_data[output_scalar + '_t_effect'] = []
        tract_data[output_scalar + '_t_sd'] = []
        point = 0
        for tract_length in tract_lengths:
            current_t = numpy.empty((tract_length, 1))
            current_p = numpy.empty((tract_length, 1))
            current_d = numpy.empty((tract_length, 1))
            current_e = numpy.empty((tract_length, 1))
            current_s = numpy.empty((tract_length, 1))

            for p in xrange(tract_length):
                glm = sm.GLM(per_scalar_data[point], design_matrix)
                t_test = glm.fit().t_test(contrast)
                current_t[p, :] = t_test.tvalue
                current_p[p, :] = t_test.pvalue
                current_d[p, :] = t_test.df_denom
                current_e[p, :] = t_test.effect
                current_s[p, :] = t_test.sd
                point += 1
            tract_data[output_scalar + '_t_score'].append(current_t)
            tract_data[output_scalar + '_t_pvalue'].append(current_p)
            tract_data[output_scalar + '_t_df'].append(current_d)
            tract_data[output_scalar + '_t_effect'].append(current_e)
            tract_data[output_scalar + '_t_sd'].append(current_s)

    return tractography


@tract_math_operation(
    '<scalar> <method> <significance level> <output tactography>: correct the p-values with FDR',
    needs_one_tract=True
)
def tract_stat_multiple_correction(tractography, method, scalar, threshold, file_output=None):
    method = method.strip()
    threshold = float(threshold)
    methods = (
        'bonferroni', 'sidak',
        'holm-sidak', 'holm',
        'simes-hochberg', 'hommel',
        'fdr_bh', 'fdr_by'
    )

    if method not in methods:
        methods_list = reduce(
            lambda x, y: x + '\t' + y + '\n',
            methods, ''
        )[:-1]
        raise TractMathWrongArgumentsError(
            'Method %s not in the list: \n' % scalar +
            methods_list
        )

    try:
        from statsmodels.stats.multitest import multipletests
        tracts_data = tractography.tracts_data()[scalar]
        new_tracts_data = []
        for tract_data in tracts_data:
            _, corr_pvalues, _, _ = multipletests(
                tract_data.squeeze(), alpha=threshold,
                method=method
            )
            new_tracts_data.append(corr_pvalues[:, None])
        tractography.tracts_data()['%s_corr_%s' % (scalar, method)] = new_tracts_data

        return tractography
    except KeyError:
        raise TractMathWrongArgumentsError('Scalar %s not in tractography' % scalar)
    except ImportError:
        raise TractMathWrongArgumentsError('statsmodule package not installed')
