# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The ants module provides basic functions for interfacing with WMQL."""

import os
import glob
from itertools import izip

from nipype.interfaces.base import (TraitedSpec, traits)
from nipype.interfaces.ants import utils


class AntsCommandInputSpec(utils.ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='%d', usedefault=True,
                             desc='image dimension (2 or 3)', position=1)
    reference_images = traits.List(traits.File, sep=',',
                           argstr='', desc='template file to warp to',
                           mandatory=True, copyfile=True, position=2)
    input_images = traits.List(traits.File, sep=',',
                       argstr='', desc='input image to warp to template',
                       mandatory=True, copyfile=False, position=3)
    max_iterations = traits.List(traits.Int, [30, 90, 20], argstr='-i %s',
                             sep='x', usedefault=True,
                             desc=('maximum number of iterations (must be '
                                   'list of integers in the form [J,K,L...]: '
                                   'J = coarsest resolution iterations, K = '
                                   'middle resolution interations, L = fine '
                                   'resolution iterations'), position=4)
    regularization_parameters = traits.List(traits.Float,
                                 argstr='-r Gauss[%s]', sep=',',
                                desc=('Parameters for the regularization'),
                                position=5)
    transformation_parameters = traits.List(traits.Float, argstr='-t SyN[%s]',
                            sep=',',
                            desc=('Parameters for the transformation'),
                            position=6)
    out_prefix = traits.Str('ants_', argstr='-o %s', usedefault=True,
                             desc=('Prefix that is prepended to all output '
                                   'files (default = ants_)'), position=7)

    use_histogram_matching = traits.Bool(True,
                                        argstr='--use-Histogram-Matching',
                                        usedefault=True, position=8)

    number_of_affine_iterations = traits.List(traits.Int, [10000, ] * 5,
                            argstr='--number-of-affine-iterations %s', sep='x',
                            usedefault=True, position=9)

    mi_option = traits.List(traits.Int, [32, 16000],
                            argstr='--MI-option %s', sep='x',
                            usedefault=True, position=10)


class AntsCommandOutputSpec(TraitedSpec):
    affine_transformation = traits.File(
        exists=True,
        desc='affine (prefix_Affine.txt)'
    )
    warp_field = traits.File(
        exists=True,
        desc='warp field (prefix_Warp.nii)'
    )
    inverse_warp_field = traits.File(exists=True,
                            desc='inverse warp field (prefix_InverseWarp.nii)')
    input_file = traits.File(
                            exists=True,
                            desc='input image (prefix_repaired.nii)')

    transforms = traits.List(traits.File, desc='transform list')

    inverse_transforms = traits.List(traits.File, desc='transform list with inversed warp (but not affine)')

class GenWarpFields(utils.ANTSCommand):
    """Uses ANTS to generate matrices to warp data from one space to another.

    Examples
    --------

    >>> warp = GenWarpFields()
    >>> warp.inputs.reference_images = ['Template_FA.nii', 'Template_MD.nii']
    >>> warp.inputs.input_images = ['fa.nii', 'md.nii']
    >>> warp.inputs.max_iterations = [30,90,20]
    >>> warp.inputs.regularization_parameters = [3, 0]
    >>> warp.inputs.transformation_parameters = [.25, 2, 0.05]
    >>> warp.cmdline
    'ANTS 3 -m CC[Template_FA.nii,fa.nii,1,4] -m CC[Template_MD.nii,md.nii,1,4]  -i 30x90x20 -r Gauss[3.0,0.0] -t SyN[0.25,2.0,0.05] -o ants_ --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000'
    """

    _cmd = 'ANTS'
    _internal = {}
    input_spec = AntsCommandInputSpec
    output_spec = AntsCommandOutputSpec

    def _format_arg(self, name, spec, value):
        if name.endswith('images'):
            if name == 'reference_images':
                self._internal[name] = value
            if name == 'input_images':
                self._internal[name] = value
            if (
                'reference_images' in self._internal and
                'input_images' in self._internal
            ):
                if len(self._internal['reference_images']) !=\
                    len(self._internal['input_images']):
                    raise ValueError(
                        'The number of reference and'
                        'input images must be the same'
                    )
                arg_string = ''
                for template, reference in izip(
                    self._internal['reference_images'],
                    self._internal['input_images']
                ):
                    arg_string += '-m CC[%s,%s,1,4] ' % (template, reference)
                return arg_string[:-1]
            else:
                return ''
        else:
            return super(GenWarpFields, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs['affine_transformation'] = os.path.join(os.getcwd(),
                            self.inputs.out_prefix + 'Affine.txt')

        outputs['warp_field'] = os.path.join(os.getcwd(),
                                             self.inputs.out_prefix +
                                             'Warp.nii.gz')
        outputs['inverse_warp_field'] = os.path.join(os.getcwd(),
                                                     self.inputs.out_prefix +
                                                     'InverseWarp.nii.gz')

        return outputs
