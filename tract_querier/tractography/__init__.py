from .tractography import Tractography
from .trackvis import tractography_from_trackvis_file, tractography_to_trackvis_file

from warnings import warn
import numpy

__all__ = [
    'Tractography',
    'tractography_from_trackvis_file', 'tractography_to_trackvis_file',
    'tractography_from_files',
    'tractography_from_file', 'tractography_to_file',
]

try:
    __all__ += [
        'tractography_from_vtk_files', 'tractography_to_vtk_file',
        'vtkPolyData_to_tracts', 'tracts_to_vtkPolyData'
    ]
    from .vtkInterface import (
        tractography_from_vtk_files, tractography_to_vtk_file,
        vtkPolyData_to_tracts, tracts_to_vtkPolyData
    )

except ImportError:
    warn(
        'VTK support not installed in this python distribution, '
        'VTK files will not be read or written'
    )


def tractography_from_files(filenames):
    if isinstance(filenames, str):
        filenames = [filenames]

    tracts = tractography_from_file(filenames[0])

    for filename in filenames[1:]:
        tracts_ = tractography_from_file(filename)
        tracts.append(tracts_.tracts(), tracts_.tracts_data())

    return tracts


def tractography_from_file(filename):
    if filename.endswith('trk'):
        return tractography_from_trackvis_file(filename)
    elif filename.endswith('vtk') or filename.endswith('vtp'):
        if 'tractography_from_vtk_files' in __all__:
            return tractography_from_vtk_files(filename)
        else:
            raise IOError("No VTK support installed, VTK files could not be read")
    else:
        raise IOError("File format not supported")


def tractography_to_file(filename, tractography, **kwargs):
    if filename.endswith('trk'):
        if 'affine' not in kwargs or kwargs['affine'] is None:
            if (
                    hasattr(tractography, 'affine') and 
                    tractography.affine is not None
            ):
                kwargs['affine'] = tractography.affine
            else:
                warn('Setting affine of trk file to the identity')
                kwargs['affine'] = numpy.eye(4)

        if (
                'image_dimensions' not in kwargs or
                kwargs['image_dimensions'] is None
        ):
            if (
                hasattr(tractography, 'image_dims') and
                tractography.image_dims is not None
            ):
                kwargs['image_dimensions'] = tractography.image_dims
            else:
                warn('Setting image_dimensions of trk file to: 1 1 1')
                kwargs['image_dimensions'] = numpy.ones(3)

        return tractography_to_trackvis_file(filename, tractography, **kwargs)

    elif filename.endswith('vtk') or filename.endswith('vtp'):
        if 'tractography_from_vtk_files' in __all__:
            return tractography_to_vtk_file(filename, tractography, **kwargs)
        else:
            raise IOError("No VTK support installed, VTK files could not be read")
    else:
        raise IOError("File format not supported")
