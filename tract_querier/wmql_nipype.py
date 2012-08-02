# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The ants module provides basic functions for interfacing with WMQL."""

import os
import glob

from nipype.interfaces.base import (CommandLine, CommandLineInputSpec, TraitedSpec, traits)

class TractQuerierInputSpec(CommandLineInputSpec):
    atlas_type = traits.Enum('Desikan', 'Mori', argstr='-q %s', usedefault=True,
                             desc='Atlas type for the queries')
    input_atlas = traits.File(desc = "Input Atlas volume", exists = True, mandatory = True, argstr="-a %s", copy_file=False)
    input_tractography = traits.File(desc = "Input Tractography", exists = True, mandatory = True, argstr="-t %s", copy_file=False)
    out_prefix = traits.Str('query', des = "prefix for the results", mandatory = False, argstr="-o %s", usedefault=True)
    queries = traits.List(desc = "Input queries", exists = True, mandatory = False, argstr="--query_selection %s")



class TractQuerierOutputSpec(TraitedSpec):
    output_queries = traits.List(exists=True, desc='resulting query files')

class TractQuerier(CommandLine):
    """Uses WMQL to generate white matter tracts

    Examples
    --------

    >>> from .wmql_nipype import TractQuerier
    >>> tract_querier = TractQuerier()
    >>> tract_querier.inputs.atlas_type = 'Desikan'
    >>> tract_querier.inputs.input_atlas = 'wmparc.nii.gz'
    >>> tract_querier.inputs.input_tractography = 'tracts.vtk'
    >>> tract_querier.inputs.out_prefix = 'query_'
    >>> tract_querier.inputs.queries = ['ilf.left' ,'ilf.right']
    >>> tract_querier.cmdline
    'tract_querier -q freesurfer_queries.qry -a wmparc.nii.gz -t tracts.vtk -o query_ --query_selection ilf.left,ilf.right'

    """

    _cmd = 'tract_querier'
    input_spec = TractQuerierInputSpec
    output_spec = TractQuerierOutputSpec


    def _format_arg(self, name, spec, value):
        if name == 'atlas_type':
            return spec.argstr%{"Mori":'mori_queries.qry', "Desikan":'freesurfer_queries.qry'}[value]
        elif name == 'queries':
            return spec.argstr%(''.join((q + ',' for q in value[:-1])) + value[-1])
        return super(TractQuerier, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_queries'] = glob.glob(os.path.join(os.getcwd(),
                                                        self.inputs.out_prefix +
                                                        '*.vtk')
                                             )
        return outputs


class MapImageToTractsInputSpec(CommandLineInputSpec):
    input_tractography = traits.File(desc = "Input Tractography", exists = False, mandatory = True, argstr="%s", copy_file=False, position=0)
    input_image = traits.File(desc = "Input Image", exists = False, mandatory = True, argstr="-i %s", copy_file=False)
    output_tractography = traits.File('out_', desc = "output tract name", mandatory = False, argstr="-o %s", usedefault=True)
    data_name = traits.String(desc = "Name of the property", mandatory = True, argstr="-n %s")



class MapImageToTractsOutputSpec(TraitedSpec):
    output_tractography = traits.File(exists=True, desc='output tractography')

class MapImageToTracts(CommandLine):
    """Uses WMQL to generate white matter tracts

    Examples
    --------

    >>> from .wmql_nipype import MapImageToTracts
    >>> tract_mapper = MapImageToTracts()
    >>> tract_mapper.inputs.output_tractography = 'tracts_w_fa.vtk'
    >>> tract_mapper.inputs.input_image = 'fa.nii.gz'
    >>> tract_mapper.inputs.data_name = 'FA'
    >>> tract_mapper.inputs.input_tractography = 'tracts.vtk'
    >>> tract_mapper.cmdline
    'tq_map_image_to_tracts tracts.vtk -n FA -i fa.nii.gz -o tracts_w_fa.vtk'

    """

    _cmd = 'tq_map_image_to_tracts'
    input_spec = MapImageToTractsInputSpec
    output_spec = MapImageToTractsOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'output_tractography':
            return spec.argstr%(os.path.join(
                os.getcwd(),
                value + os.path.basename(self.inputs.input_tractography)
            ))
        return super(MapImageToTracts, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_tractography'] = os.path.join(
            os.getcwd(),
            (
                self.inputs.output_tractography +
                os.path.basename(self.inputs.input_tractography)
            )
        )

        return outputs
        return outputs
