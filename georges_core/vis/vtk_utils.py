import os
import vtk as _vtk
import vtk.util.numpy_support as _vtk_np

import pybdsim 

def to_vtk(Histogram3d,
           filename='histogram.vti',
           path='.',
           origin_from_file=True
           ):
    origin = None
    _pyroot_file = None

    if origin_from_file is True:
        _pyroot_file = pybdsim.Data.Load(os.path.join(Histogram3d.path, Histogram3d.filename))

    if _pyroot_file is not None:
        mt = _pyroot_file.GetModelTree()
        md = _pyroot_file.GetModel()
        mt.GetEntry(0)
        mesh_translation = [0.0, 0.0, 0.0]
        for name in md.model.scoringMeshName:
            if str(name) in Histogram3d.meshname:
                mesh_translation = md.model.scoringMeshTranslation[name]
        origin = [
            mesh_translation.x(),
            mesh_translation.y(),
            mesh_translation.z(),
        ]
    else:
        origin = [0.0, 0.0, 0.0] if origin is None else origin
    imgdat = _vtk.vtkImageData()
    imgdat.GetPointData().SetScalars(
        _vtk_np.numpy_to_vtk(
            num_array=Histogram3d.values.ravel(order='F'),
            deep=True,
            array_type=_vtk.VTK_FLOAT
        )
    )
    imgdat.SetDimensions(Histogram3d.xnumbins, Histogram3d.ynumbins, Histogram3d.znumbins)
    imgdat.SetOrigin(origin[0] - Histogram3d.coordinates_normalization * (Histogram3d.edges[0][-1] - Histogram3d.edges[0][0]) / 2,
                     origin[1] - Histogram3d.coordinates_normalization * (Histogram3d.edges[1][-1] - Histogram3d.edges[1][0]) / 2,
                     origin[2] - Histogram3d.coordinates_normalization * (Histogram3d.edges[2][-1] - Histogram3d.edges[2][0]) / 2
                     )
    imgdat.SetSpacing(
        Histogram3d.coordinates_normalization * (Histogram3d.edges[0][1] - Histogram3d.edges[0][0]),
        Histogram3d.coordinates_normalization * (Histogram3d.edges[1][1] - Histogram3d.edges[1][0]),
        Histogram3d.coordinates_normalization * (Histogram3d.edges[2][1] - Histogram3d.edges[2][0])
    )
    writer = _vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(path, filename))
    writer.SetInputData(imgdat)
    writer.SetDataModeToBinary()
    writer.Write()