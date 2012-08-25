import numpy as np

try:
    import vtk
except:
    pass

try:
    from . import fiber_reparametrization as fr
except:
    pass

from . import vtkInterface

class tractography:

    _originalFibers = []
    _originalLines = []
    _originalData = {}
    _fiberData = {}
    _fibers = []
    _fibersMap = []
    _subsampledFibers = []
    _quantitiyOfPointsPerfiber = None
    _interpolated = False

    def __init__(self):
        self._originalFibers = []
        self._fiberData = {}
        self._originalLines = []
        self._originalData = {}
        self._fibers = []
        self._fibersMap = []
        self._quantitiyOfPointsPerfiber = None

        self._interpolated = False
        self._fiberKeepRatio = None

        self._subsampledFibers = []
        self._subsampledLines = []
        self._subsampledData = []

        return


    def from_vtkPolyData(self, vtkPolyData, ratio=1, append=False ):
        #Fibers and Lines are the same thing
        if not append:
            self._originalFibers = []
            self._fiberData = {}
            self._originalLines = []
            self._originalData = {}
            self._fibers = []
            self._fibersMap = []
            self._quantitiyOfPointsPerfiber = None

        self._interpolated = False
        self._fiberKeepRatio = ratio

        self._subsampledFibers = []
        self._subsampledLines = []
        self._subsampledData = []

        #Bug fix for the inhability to convert vtkId to a numeric type np array
        linesUnsignedInt = vtk.vtkUnsignedIntArray()
        linesUnsignedInt.DeepCopy( vtkPolyData.GetLines().GetData() )
        lines = linesUnsignedInt.ToArray().squeeze()
        points = vtkPolyData.GetPoints().GetData().ToArray()

        actualLineIndex = 0
        numberOfFibers = vtkPolyData.GetLines().GetNumberOfCells()
        for l in xrange( numberOfFibers ):
            #print '%.2f%%'%(l*1./numberOfFibers * 100.)
            self._fibers.append( points[ lines[actualLineIndex+1: actualLineIndex+lines[actualLineIndex]+1] ] )
            self._originalLines.append( np.array(lines[actualLineIndex+1: actualLineIndex+lines[actualLineIndex]+1],copy=True)  )
            actualLineIndex += lines[actualLineIndex]+1

        for i in xrange( vtkPolyData.GetPointData().GetNumberOfArrays() ):
            array = vtkPolyData.GetPointData().GetArray(i)
            array_data = array.ToArray()
            self._originalData[ array.GetName() ] = array_data
            self._fiberData[ array.GetName() ] =  map( lambda f: array_data[ f ], self._originalLines )

        if (vtkPolyData.GetPointData().GetScalars()!=[]):
            self._originalData['vtkScalars']=vtkPolyData.GetPointData().GetScalars().ToArray()
        if (vtkPolyData.GetPointData().GetTensors()!=[]):
            self._originalData['vtkTensors']=vtkPolyData.GetPointData().GetTensors().ToArray()
        if (vtkPolyData.GetPointData().GetVectors()!=[]):
            self._originalData['vtkVectors']=vtkPolyData.GetPointData().GetTensors().ToArray()

        self._originalFibers = list(self._fibers)
        if self._fiberKeepRatio!=1:
            self._fibers = self._originalFibers[::np.round(len(self._originalFibers)*self._fiberKeepRatio)]



    def from_dictionary(self, d, ratio=1, append=False ):
        #Fibers and Lines are the same thing
        if not append:
            self._originalFibers = []
            self._fiberData = {}
            self._originalLines = []
            self._originalData = {}
            self._fibers = []
            self._fibersMap = []
            self._quantitiyOfPointsPerfiber = None

        self._interpolated = False
        self._fiberKeepRatio = ratio

        self._subsampledFibers = []
        self._subsampledLines = []
        self._subsampledData = []



        #Bug fix for the inhability to convert vtkId to a numeric type np array
        lines = np.asarray(d['lines']).squeeze()
        points = d['points']

        actualLineIndex = 0
        numberOfFibers = d['numberOfLines']
        for l in xrange( numberOfFibers ):
            #print '%.2f%%'%(l*1./numberOfFibers * 100.)
            self._fibers.append( points[ lines[actualLineIndex+1: actualLineIndex+lines[actualLineIndex]+1] ] )
            self._originalLines.append( np.array(lines[actualLineIndex+1: actualLineIndex+lines[actualLineIndex]+1],copy=True)  )
            actualLineIndex += lines[actualLineIndex]+1


        pointDataKeys = [ it[0] for it in d['pointData'].items() if isinstance( it[1], np.ndarray ) ]
        if len(self._originalData.keys())>0 and (self._originalData.keys()!=pointDataKeys):
            raise ValueError('PointData not compatible')

        for k in pointDataKeys:
            array_data = d['pointData'][k]
            if not k in self._originalData:
                self._originalData[ k ] = array_data
                self._fiberData[ k ] =  map( lambda f: array_data[ f ], self._originalLines )
            else:
                np.vstack(self._originalData[k])
                self._fiberData[ k ].extend( map( lambda f: array_data[ f ], self._originalLines[-numberOfFibers:] ) )


#    if (vtkPolyData.GetPointData().GetScalars()!=[]):
#      self._originalData['vtkScalars']=vtkPolyData.GetPointData().GetScalars().ToArray()
#    if (vtkPolyData.GetPointData().GetTensors()!=[]):
#      self._originalData['vtkTensors']=vtkPolyData.GetPointData().GetTensors().ToArray()
#    if (vtkPolyData.GetPointData().GetVectors()!=[]):
#      self._originalData['vtkVectors']=vtkPolyData.GetPointData().GetTensors().ToArray()

        self._originalFibers = list(self._fibers)
        if self._fiberKeepRatio!=1:
            self._fibers = self._originalFibers[::np.round(len(self._originalFibers)*self._fiberKeepRatio)]


        if self._quantitiyOfPointsPerfiber!=None:
            self.subsampleFibers( self._quantitiyOfPointsPerfiber )

    def to_vtkPolyData(self,vtkPolyData, selectedFibers=None):

        fibers = self.getOriginalFibers()
        if selectedFibers!=None:
            fibers = [ fibers[i] for i in selectedFibers]

        numberOfPoints = reduce( lambda x,y: x+y.shape[0], fibers,0 )
        numberOfCells = len(fibers)
        numberOfCellIndexes = numberOfPoints+numberOfCells

        linesUnsignedInt = vtk.vtkUnsignedIntArray()
        linesUnsignedInt.SetNumberOfComponents(1)
        linesUnsignedInt.SetNumberOfTuples(numberOfCellIndexes)
        lines = linesUnsignedInt.ToArray().squeeze()
        #print lines.shape


        points_vtk = vtkPolyData.GetPoints()
        if points_vtk==[]:
            point_vtk = vtk.vtkPoints()
            vtkPolyData.SetPoints(points_vtk)
#      points_vtk.Delete()
        points_vtk.SetNumberOfPoints(numberOfPoints)
        points_vtk.SetDataTypeToFloat()
        points = points_vtk.GetData().ToArray()

        actualLineIndex = 0
        actualPointIndex = 0
        for i in xrange(numberOfCells):
            lines[actualLineIndex] = fibers[i].shape[0]
            lines[actualLineIndex+1: actualLineIndex+1+lines[actualLineIndex]] = np.arange( lines[actualLineIndex] )+actualPointIndex
            points[ lines[ actualLineIndex+1: actualLineIndex+1+lines[actualLineIndex] ] ] = fibers[i]
            actualPointIndex += lines[actualLineIndex]
            actualLineIndex = actualLineIndex + lines[actualLineIndex]+1


        vtkPolyData.GetLines().GetData().DeepCopy(linesUnsignedInt)

    def unsubsampleFibers(self):
        self._subsampledFibers = []

    def unfilterFibers(self):
        self._fibersMap = []

    def subsampleFibers(self, quantitiyOfPointsPerfiber):
        self._quantitiyOfPointsPerfiber = quantitiyOfPointsPerfiber
        self._subsampledFibers = []
        self._subsampledLines= []
        self._subsampledData = {}

        for k in self._fiberData:
            self._subsampledData[ k ] = []

        for i in xrange(len(self._fibers)):
            f = self._fibers[i]
            s = np.linspace( 0, f.shape[0]-1,min(f.shape[0],self._quantitiyOfPointsPerfiber) ).round().astype(int)
            self._subsampledFibers.append( f[s,:] )
            self._subsampledLines.append(s)
            for k in self._fiberData:
                self._subsampledData[ k ].append( self._fiberData[k][i][s] )


        self._interpolated = False

    def subsampleFibersEqualStep(self, step_ratio):
        self._step_ratio = step_ratio
        self._subsampledFibers = []
        self._subsampledLines= []
        self._subsampledData = {}

        for k in self._fiberData:
            self._subsampledData[ k ] = []

        for i in xrange(len(self._fibers)):
            f = self._fibers[i]
            s = np.arange(0, len(f), step_ratio)
            self._subsampledFibers.append( f[s, :] )
            self._subsampledLines.append(s)
            for k in self._fiberData:
                self._subsampledData[ k ].append( self._fiberData[k][i][s] )


        self._interpolated = False


    def subsampleInterpolatedFibers(self, step):

        self._quantitiyOfPointsPerfiber = step
        self._subsampledFibers = []
        for i, f in enumerate(self._fibers):
            print f
            reparametrized_fiber = fr.arc_length_fiber_parametrization(f, step=step)
            print reparametrized_fiber
            if f is not None:
                self._subsampledFibers.append(reparametrized_fiber[1:].T)
        self._interpolated = True

    def filterFibers(self, minimumNumberOfSamples ):

        if len(self._originalFibers)==0:
            self._originalFibers = self._fibers

        self._fibers = filter( lambda f: f.shape[0]>= minimumNumberOfSamples, self._originalFibers )
        self._fibersMap = filter( lambda i: self._originalFibers[i].shape[0]>= minimumNumberOfSamples, range(len(self._originalFibers)) )


        if self._quantitiyOfPointsPerfiber!=None:
            if self._interpolated:
                self.subsampleInterpolatedFibers( self._quantitiyOfPointsPerfiber )
            else:
                self.subsampleFibers( self._quantitiyOfPointsPerfiber )

    def areFibersFiltered(self):
        return self._fibersMap!=[]

    def areFibersSubsampled(self):
        return self._subsampledFibers!=[]

    def areSubsampledFibersInterpolated(self):
        return self._interpolated

    def getOriginalFibersData(self):
        return self._fiberData

    def getOriginalFibers(self):
        return self._originalFibers

    def getOriginalLines(self):
        return self._originalLines

    def getOriginalData(self):
        return self._originalData

    def getFilteredFibersMap(self):
        return self._fibersMap

    def getFibersToProcess(self):
        if self._subsampledFibers!=[]:
            return self._subsampledFibers
        elif self._fibers!=[]:
            return self._fibers

    def getFibersDataToProcess(self):
        if self._subsampledData!=[]:
            return self._subsampledData
        elif self._fiberData!=[]:
            return self._fiberData

def tractography_from_vtk_files(vtk_file_names):
    tr = tractography()

    if len(vtk_file_names) == 1:
        vtk_file_names = vtk_file_names[0]

    if isinstance(vtk_file_names, str):
        pd_as_dictiononary = vtkInterface.read_vtkPolyData(vtk_file_names)
        tr.from_dictionary(pd_as_dictiononary)
    else:
        for file_name in vtk_file_names:
            pd_as_dictiononary = vtkInterface.read_vtkPolyData(file_name)
            tr.from_dictionary(pd_as_dictiononary, append = True)

    return tr
