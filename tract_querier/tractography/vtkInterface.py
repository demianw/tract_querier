from itertools import izip
import vtk
from vtk.util import numpy_support as ns
import numpy as np


def read_vtkPolyData(filename):
    if filename.endswith('xml'):
        polydata_reader = vtk.vtkXMLPolyDataReader()
    else:
        polydata_reader = vtk.vtkPolyDataReader()

    polydata_reader.SetFileName(filename)
    polydata_reader.Update()

    polydata = polydata_reader.GetOutput()

    result = {}
    result['lines'] = ns.vtk_to_numpy(polydata.GetLines().GetData())
    result['points'] = ns.vtk_to_numpy(polydata.GetPoints().GetData())
    result['numberOfLines'] = polydata.GetNumberOfLines()

    data = {}
    if polydata.GetPointData().GetScalars():
        data['ActiveScalars'] =  polydata.GetPointData.GetScalars().GetName()
    if polydata.GetPointData().GetVectors():
        data['ActiveVectors'] =  polydata.GetPointData.GetVectors().GetName()
    if polydata.GetPointData().GetTensors():
        data['ActiveTensors'] =  polydata.GetPointData.GetTensors().GetName()

    for i in xrange(polydata.GetPointData().GetNumberOfArrays()):
        array = polydata.GetPointData().GetArray(i)
        data[polydata.GetPointData().GetArrayName(i)] = ns.vtk_to_numpy(array)

    result['pointData'] = data

    return result

def vtkPolyData_to_lines(polydata):
    lines_ids = ns.vtk_to_numpy(polydata.GetLines().GetData())
    points = ns.vtk_to_numpy(polydata.GetPoints().GetData())

    lines = []
    lines_indices = []
    actual_line_index = 0
    for i in xrange(polydata.GetNumberOfLines()):
        next_line_index =actual_line_index + lines_ids[actual_line_index] + 1

        lines_indices.append(lines_ids[actual_line_index + 1: next_line_index])
        lines.append(points[lines_indices[-1]])

        actual_line_index = next_line_index

    point_data = {}
    for i in xrange( polydata.GetPointData().GetNumberOfArrays() ):
        vtk_array = polydata.GetPointData().GetArray(i)
        array_data = ns.vtk_to_numpy(vtk_array)
        if array_data.ndim == 1:
            data = [
                ns.numpy.ascontiguousarray(array_data[line_indices][:, None])
                for line_indices in lines_indices
            ]
        else:
            data = [
                ns.numpy.ascontiguousarray(array_data[line_indices])
                for line_indices in lines_indices
            ]

        point_data[ vtk_array.GetName() ] = data

    scalars = polydata.GetPointData().GetScalars()
    if scalars is not None:
        point_data['ActiveScalars'] = scalars.GetName()

    vectors = polydata.GetPointData().GetVectors()
    if vectors is not None:
        point_data['ActiveVectors'] = vectors.GetName()

    tensors = polydata.GetPointData().GetTensors()
    if tensors is not None:
        point_data['ActiveTensors'] = tensors.GetName()

    return lines, lines_indices, point_data


def lines_to_vtkPolyData(lines, lines_indices=None, point_data={}):
    lengths = [len(p) for p in lines]
    line_starts = ns.numpy.r_[0, ns.numpy.cumsum(lengths)]
    if lines_indices is None:
        lines_indices = [
            ns.numpy.arange(length) + line_start
            for length, line_start in izip(lengths, line_starts)
        ]


    ids = ns.numpy.hstack([
        ns.numpy.r_[c[0], c[1]]
        for c in izip(lengths,lines_indices)
    ])
    vtk_ids = ns.numpy_to_vtkIdTypeArray(ids, deep=True)

    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(len(lines), vtk_ids)
    points = ns.numpy.vstack(lines).astype(
        ns.get_vtk_to_numpy_typemap()[vtk.VTK_DOUBLE]
    )
    points_array = ns.numpy_to_vtk(points, deep=True)

    poly_data = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(points_array)
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(cell_array)


    saved_keys = set()
    for key, value in point_data.items():
        if key in saved_keys:
            continue
        if key.startswith('Active'):
            saved_keys.add(value)
            name = value
            value = point_data[value]
        else:
            name = key

        if len(value) == len(lines):
            value_ = ns.numpy.vstack(value)
        elif len(value) == len(points):
            value_ = value
        else:
            raise ValueError("Data in %s does not have the correct number of items")

        vtk_value = ns.numpy_to_vtk(np.ascontiguousarray(value_), deep=True)
        vtk_value.SetName(name)
        if key == 'ActiveScalars':
            poly_data.GetPointData().SetScalars(vtk_value)
        if key == 'ActiveVectors':
            poly_data.GetPointData().SetVectors(vtk_value)
        if key == 'ActiveTensors':
            poly_data.GetPointData().SetTensors(vtk_value)
        else:
            poly_data.GetPointData().AddArray(vtk_value)

    poly_data.BuildCells()

    return poly_data


def writeLinesToVtkPolyData(filename, lines, point_data={}):
    poly_data = lines_to_vtkPolyData(lines, point_data=point_data)

    if filename.endswith('.xml'):
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetDataModeToBinary()
    else:
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToBinary()

    writer.SetFileName(filename)
    writer.SetInput(poly_data)
    writer.Write()


def writeLinesToVtkPolyData_pure_python( filename, lines, point_data={} ):
    file_ = open(filename, 'w')
    file_.write(__header__)
    file_.write(__polyDataType__)

    number_of_points = sum( [ len(line) for line in lines ] )

    file_.write(__points_header__(number_of_points))
    for line in lines:
        for point in line:
            file_.write(str(point).strip()[1:-1]+'\n')

    number_of_lines = len(lines)
    file_.write(__lines_header__(number_of_lines, number_of_points))
    points_for_line_saved = 0
    for line in lines:
        file_.write(
            "%d %s \n" % (
                len(line),
                reduce(
                    lambda x, y:x+' %d' % (y+points_for_line_saved),
                    xrange(len(line)), ''
                )
            ))
        points_for_line_saved += len(line)

    if point_data:
        file_.write(__point_data_header__(number_of_points))

        active_keys = write_active_components( file_, point_data )
        write_field_data( file_, number_of_points, active_keys, point_data )

    file_.flush()
    file_.close()

def get_number_of_components( data ):
    if hasattr(data[0], 'shape'):
	if len(data[0].shape)==0:
		return 1
        return data[0].shape[-1]
    if hasattr(data[0][0], '__len__'):
        return len(data[0][0])
    else:
        return 1

def write_active_components( file_, point_data ):
    active_keys = []
    for type_, fixed_number_of_components in [
        ('Scalars', None), ('Vectors', 3), ('Tensors', 9)
    ]:
        active_tag = 'Active'+type_
        if active_tag in point_data:
            name = point_data[active_tag]
            active_keys.append(name)
            data = point_data[name]
            number_of_components = get_number_of_components( data )

            if (
                (fixed_number_of_components is not None) and
                (fixed_number_of_components != number_of_components)
            ):
                raise ValueError(
                    "Active %s don't have %d components, it has %d" % (
                        type_, fixed_number_of_components, number_of_components)
                )
            if type_ == 'Scalars':
                file_.write(__point_data_attribute_header__(
                    type_.upper(), name, number_of_components)
                )
                file_.write('LOOKUP_TABLE default\n')
            else:
                file_.write(__point_data_attribute_header__(type_.upper(), name))

            write_line_data( file_, data )

    return active_keys

def write_field_data( file_, number_of_points, active_keys, point_data ):
    keys = (
        set(point_data.keys()) -
        set(active_keys) -
        set([key for key, data in point_data.items() if isinstance(data, str)])
    )
    if not keys:
        return

    file_.write( __field_data_header__( len(keys) ) )
    for key in keys:
        data = point_data[key]

        name = key
        number_of_components = get_number_of_components( data )

        if sum( len(d) for d in data ) != number_of_points:
            raise ValueError("Attribute %s does not have a tuple per point in the line"%key)

        file_.write(
            __field_data_attribute_header__(
                name, number_of_components, number_of_points
            )
        )

        write_line_data( file_, data )


def write_line_data( file_, data ):
    for line in data:
        for attribute in line:
            file_.write(
                str(attribute).replace('[', '')\
                .replace(']', '')\
                .replace(', ', ' ')\
                .strip()
                +'\n'
            )



def tractography_from_vtkPolyData(polydata):
    tractography = Tractography()

    tractography._originalFibers = []
    tractography._fiberData = {}
    tractography._originalLines = []
    tractography._originalData = {}
    tractography._fibers = []

    lines, lines_ids, point_data = vtkPolyData_to_lines(polydata)

    tractography._fibers = lines
    tractography._fiberData = point_data

    tractography._originalFibers = np.vstack(lines)
    tractography._originalLines = lines_ids
    tractography._originalData = dict((
        (key, np.vstack(value))
        for key, value in tractography._fiberData
    ))

__header__ = """
# vtk DataFile Version 3.0
vtk output
ASCII
"""

__polyDataType__ = "DATASET POLYDATA\n"

def __points_header__(number_of_points):
	return "POINTS %d float\n" % number_of_points

def __lines_header__(number_of_lines, number_of_points):
	return "LINES %d %d\n" % (number_of_lines, number_of_lines+number_of_points)

def __point_data_header__(number_of_points):
	return "POINT_DATA %d\n" % number_of_points

def __point_data_attribute_header__(type_, name, number_of_components = 0):
	return  "%s %s float %.d\n" % (type_, name, number_of_components)

def __field_data_header__(number_of_arrays):
	return "FIELD FieldData %d\n" % number_of_arrays

def __field_data_attribute_header__(
    array_name = '', number_of_components = 1,
    number_of_points = 1, data_type = 'float'
):
	return " % s %d %d %s\n" % (
        array_name, number_of_components,
        number_of_points, data_type
    )

