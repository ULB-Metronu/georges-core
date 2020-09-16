import os
import vtk as _vtk
import vtk.util.numpy_support as _vtk_np


def histogram3d_to_vtk(histogram3d,
                       filename='histogram.vti',
                       path='.',
                       origin_from_file=True
                       ):

    if origin_from_file is True:
        origin = histogram3d.scoring_mesh_translations
    else:
        origin = [0.0, 0.0, 0.0]

    imgdat = _vtk.vtkImageData()
    imgdat.GetPointData().SetScalars(
        _vtk_np.numpy_to_vtk(
            num_array=histogram3d.values.ravel(order='F'),
            deep=True,
            array_type=_vtk.VTK_FLOAT
        )
    )
    imgdat.SetDimensions(histogram3d.xnumbins, histogram3d.ynumbins, histogram3d.znumbins)
    imgdat.SetOrigin(origin[0] - histogram3d.coordinates_normalization * (histogram3d.edges[0][-1] - histogram3d.edges[0][0]) / 2,
                     origin[1] - histogram3d.coordinates_normalization * (histogram3d.edges[1][-1] - histogram3d.edges[1][0]) / 2,
                     origin[2] - histogram3d.coordinates_normalization * (histogram3d.edges[2][-1] - histogram3d.edges[2][0]) / 2
                     )
    imgdat.SetSpacing(
        histogram3d.coordinates_normalization * (histogram3d.edges[0][1] - histogram3d.edges[0][0]),
        histogram3d.coordinates_normalization * (histogram3d.edges[1][1] - histogram3d.edges[1][0]),
        histogram3d.coordinates_normalization * (histogram3d.edges[2][1] - histogram3d.edges[2][0])
    )
    writer = _vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(path, filename))
    writer.SetInputData(imgdat)
    writer.SetDataModeToBinary()
    writer.Write()

