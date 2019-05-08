# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The ants module provides basic functions for interfacing with WMQL."""

import os
import glob


from nipype.interfaces.base import (TraitedSpec, traits)
from nipype.interfaces.ants import utils


class ANTSAffineCommandInputSpec(utils.ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='%d', usedefault=True,
                            desc='image dimension (2 or 3)', position=1)

    reference_image = traits.File(
        argstr='%s', desc='template file to warp to',
        mandatory=True, copyfile=True, position=2)

    input_image = traits.File(
        argstr='%s', desc='input image to warp to template',
        mandatory=True, copyfile=False, position=3)

    out_prefix = traits.Str('ants_', argstr='%s', usedefault=True,
                            desc=('Prefix that is prepended to all output '
                                  'files (default = ants_)'), position=4)

    rigid_affine = traits.Bool(False,
                               argstr='1',
                               usedefault=False, position=5
                               )


class ANTSAffineCommandOutputSpec(TraitedSpec):
    affine_transformation = traits.File(
        exists=True,
        desc='affine (prefix_Affine.txt)'
    )

    deformed_image = traits.File(
        exists=True,
        desc='deformed_image')


class GenAffine(utils.ANTSCommand):

    """Uses ANTS to generate matrices to warp data from one space to another.

    Examples
    --------

    >>> aff = GenAffine()
    >>> aff.inputs.reference_image = 'Template_FA.nii'
    >>> aff.inputs.input_image = 'fa.nii'
    >>> aff.inputs.rigid_affine = True
    >>> aff.cmdline
    'antsaffine.sh 3 Template_FA.nii fa.nii ants_ 1'
    """
    _cmd = 'antsaffine.sh'
    input_spec = ANTSAffineCommandInputSpec
    output_spec = ANTSAffineCommandOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs['affine_transformation'] = os.path.join(os.getcwd(),
                                                        self.inputs.out_prefix + 'Affine.txt')

        outputs['deformed_image'] = os.path.join(os.getcwd(),
                                                 self.inputs.out_prefix + 'deformed.nii.gz')

        return outputs


class ANTSCommandInputSpec(utils.ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='%d', usedefault=True,
                            desc='image dimension (2 or 3)', position=1)
    reference_images = traits.List(traits.File, sep=',',
                                   argstr='', desc='template file to warp to',
                                   mandatory=True, copyfile=True, position=2)
    input_images = traits.List(traits.File, sep=',',
                               argstr='', desc='input image to warp to template',
                               mandatory=True, copyfile=False, position=3)
    metric_weights = traits.List(traits.Float, [1], argstr='',
                                 desc='Different weights for metric combinations',
                                 usedefault=True
                                 )
    max_iterations = traits.List(traits.Int, [30, 90, 20], argstr='-i %s',
                                 sep='x', usedefault=True,
                                 desc=('maximum number of iterations (must be '
                                       'list of integers in the form [J,K,L...]: '
                                       'J = coarsest resolution iterations, K = '
                                       'middle resolution interations, L = fine '
                                       'resolution iterations'), position=4)
    regularization_parameters = traits.List(traits.Float, [3., 0.],
                                            argstr='-r Gauss[%s]', sep=',', usedefault=True,
                                            desc=('Parameters for the regularization'),
                                            position=5)
    transformation_parameters = traits.List(traits.Float, [.25, 2, 0.05], argstr='-t SyN[%s]',
                                            sep=',', usedefault=True,
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

    affine_metric = traits.Enum('MI', 'CC', 'MSQ', 'PR', argstr='--affine-metric-type %s',
                                desc=('Type of similartiy metric used for affine registration '
                                      '(CC = cross correlation, MI = mutual information, '
                                      'PR = probability mapping, MSQ = mean square difference)'),
                                usedefault=True, position=11)

    initial_affine = traits.File(exists=False,
                                 argstr='-a %s', desc='Initial affine transform',
                                 mandatory=False, copyfile=False, position=12)

    rigid_affine = traits.Bool(False,
                               argstr='--rigid-affine true',
                               usedefault=False, position=13
                               )

    not_continue_affine = traits.Bool(False,
                                      argstr='--continue-affine false',
                                      usedefault=False, position=14
                                      )


class ANTSCommandOutputSpec(TraitedSpec):
    affine_transformation = traits.File(
        exists=True,
        desc='affine (prefix_Affine.txt)'
    )
    warp_field = traits.File(
        exists=False,
        desc='warp field (prefix_Warp.nii)'
    )
    inverse_warp_field = traits.File(exists=False,
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
    >>> warp.inputs.metric_weights = [1., 1.25]
    >>> warp.inputs.max_iterations = [30,90,20]
    >>> warp.inputs.regularization_parameters = [3, 0]
    >>> warp.inputs.transformation_parameters = [.25, 2, 0.05]
    >>> warp.inputs.rigid_affine = True
    >>> warp.cmdline
    'ANTS 3 -m CC[Template_FA.nii,fa.nii,1.00,4] -m CC[Template_MD.nii,md.nii,1.25,4]  -i 30x90x20 -r Gauss[3.0,0.0] -t SyN[0.25,2.0,0.05] -o ants_ --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --affine-metric-type MI --rigid-affine true '

    >>> warp_aff = GenWarpFields()
    >>> warp_aff.inputs.reference_images = ['Template_FA.nii', 'Template_MD.nii']
    >>> warp_aff.inputs.input_images = ['fa.nii', 'md.nii']
    >>> warp_aff.inputs.max_iterations = [30,90,20]
    >>> warp_aff.inputs.regularization_parameters = [3, 0]
    >>> warp_aff.inputs.transformation_parameters = [.25, 2, 0.05]
    >>> warp_aff.inputs.initial_affine = 'initial_affine.txt'
    >>> warp_aff.inputs.not_continue_affine = True
    >>> warp_aff.cmdline
    'ANTS 3 -m CC[Template_FA.nii,fa.nii,1.00,4] -m CC[Template_MD.nii,md.nii,1.00,4]  -i 30x90x20 -r Gauss[3.0,0.0] -t SyN[0.25,2.0,0.05] -o ants_ --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --affine-metric-type MI -a initial_affine.txt --continue-affine false '
    """
    _cmd = 'ANTS'
    input_spec = ANTSCommandInputSpec
    output_spec = ANTSCommandOutputSpec

    def _format_arg(self, name, spec, value):
        if not hasattr(self, '_internal'):
            self._internal = {}
        if name in ('reference_images', 'input_images', 'metric_weights'):
            self._internal[name] = value
            if (
                'reference_images' in self._internal and
                'input_images' in self._internal and
                'metric_weights' in self._internal
            ):
                arg_string = ''
                N = len(self._internal['reference_images'])
                if len(self._internal['input_images']) != N:
                    raise ValueError(
                        'The number of reference and'
                        'input images must be the same'
                    )
                if len(self._internal['metric_weights']) == N:
                    pass
                elif len(self._internal['metric_weights']) == 1:
                    self._internal['metric_weights'] *= N
                else:
                    raise ValueError(
                        'The number of metric weights'
                        'must be 1 or the same as '
                        'input images'
                    )
                for template, reference, weight in zip(
                    self._internal['reference_images'],
                    self._internal['input_images'],
                    self._internal['metric_weights'],
                ):
                    arg_string += '-m CC[%s,%s,%0.2f,4] ' % (
                        template, reference, weight
                    )
                return arg_string[:-1]
            else:
                return ''
        else:
            return super(GenWarpFields, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs['affine_transformation'] = os.path.join(os.getcwd(),
                                                        self.inputs.out_prefix + 'Affine.txt')

        out_warp = os.path.join(os.getcwd(),
                                self.inputs.out_prefix +
                                'Warp.nii.gz')
        out_inv_warp = os.path.join(os.getcwd(),
                                    self.inputs.out_prefix +
                                    'InverseWarp.nii.gz')

        if os.path.exists(out_warp):
            outputs['warp_field'] = out_warp

        if os.path.exists(out_inv_warp):
            outputs['inverse_warp_field'] = out_inv_warp

        return outputs
