from .tractography import Tractography
from .trackvis import tractography_from_trackvis_file, tractography_to_trackvis_file


__all__ = [
    'Tractography',
    'tractography_from_trackvis_file', 'tractography_to_trackvis_file',
    'tractography_from_file', 'tractography_to_file'
]

try:
    __all__ += ['tractography_from_vtk_files', 'tractography_to_vtk_file']
    from .vtkInterface import tractography_from_vtk_files, tractography_to_vtk_file

except ImportError:
    pass


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
