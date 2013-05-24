from . import vtkInterface
from .tractography import Tractography

__all__ = ['tractography_from_vtk_files', 'tractography_to_vtk_file', 'Tractography']


def tractography_from_vtk_files(vtk_file_names):
    tr = Tractography()

    if isinstance(vtk_file_names, str):
        vtk_file_names = [vtk_file_names]

    for file_name in vtk_file_names:
        tracts, tracts_data = vtkInterface.read_vtkPolyData(file_name)
        tr.append(tracts, tracts_data)

    return tr


def tractography_to_vtk_file(vtk_file_name, tractography):
    return vtkInterface.write_vtkPolyData(
        vtk_file_name,
        tractography.tracts(),
        tractography.tracts_data()
    )
