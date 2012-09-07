from . import vtkInterface
from .tractography import Tractography


def tractography_from_vtk_files(vtk_file_names):
    tr = Tractography()

    if len(vtk_file_names) == 1:
        vtk_file_names = vtk_file_names[0]

    if isinstance(vtk_file_names, str):
        pd_as_dictiononary = vtkInterface.read_vtkPolyData(vtk_file_names)
        tr.from_dictionary(pd_as_dictiononary)
    else:
        for file_name in vtk_file_names:
            pd_as_dictiononary = vtkInterface.read_vtkPolyData(file_name)
            tr.from_dictionary(pd_as_dictiononary, append=True)

    return tr


def tractography_to_vtkPolyData(tractography, vtkPolyData, selected_tracts=None):
    import vtk
    import numpy as np

    fibers = tractography.original_tracts()
    if selected_tracts is not None:
        fibers = [fibers[i] for i in selected_tracts]

    numberOfPoints = reduce(lambda x, y: x + y.shape[0], fibers, 0)
    numberOfCells = len(fibers)
    numberOfCellIndexes = numberOfPoints + numberOfCells

    linesUnsignedInt = vtk.vtkUnsignedIntArray()
    linesUnsignedInt.SetNumberOfComponents(1)
    linesUnsignedInt.SetNumberOfTuples(numberOfCellIndexes)
    lines = linesUnsignedInt.ToArray().squeeze()

    points_vtk = vtkPolyData.GetPoints()
    if points_vtk == []:
        points_vtk = vtk.vtkPoints()
        vtkPolyData.SetPoints(points_vtk)

    points_vtk.SetNumberOfPoints(numberOfPoints)
    points_vtk.SetDataTypeToFloat()
    points = points_vtk.GetData().ToArray()

    actual_line_index = 0
    actual_point_index = 0
    for i in xrange(numberOfCells):
        line_start = actual_line_index + 1
        line_end = actual_line_index + 1 + lines[actual_line_index]
        lines[actual_line_index] = fibers[i].shape[0]
        lines[line_start: line_end] = (
            np.arange(lines[actual_line_index]) +
            actual_point_index
        )
        points[lines[line_start: line_end]] = fibers[i]
        actual_point_index += line_start
        actual_line_index = line_end

    vtkPolyData.GetLines().GetData().DeepCopy(linesUnsignedInt)
