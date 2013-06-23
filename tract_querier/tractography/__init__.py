from .tractography import Tractography
from .trackvis import tractography_from_trackvis_file, tractography_to_trackvis_file


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
    pass


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
        return tractography_to_trackvis_file(filename, tractography, **kwargs)
    elif filename.endswith('vtk') or filename.endswith('vtp'):
        if 'tractography_from_vtk_files' in __all__:
            return tractography_to_vtk_file(filename, tractography, **kwargs)
        else:
            raise IOError("No VTK support installed, VTK files could not be read")
    else:
        raise IOError("File format not supported")
