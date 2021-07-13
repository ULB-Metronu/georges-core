import os
import vtk as _vtk
import vtk.util.numpy_support as _vtk_np
import numpy as np
import uproot


def expand_values_for_Paraview(histogram3d):

    nx = histogram3d.xnumbins
    ny = histogram3d.ynumbins
    nz = histogram3d.znumbins

    old_values = histogram3d.values
    new_values = np.ndarray(shape=(nx + 1, ny + 1, nz + 1))

    for x in range(nx+1):
        for y in range(ny+1):
            for z in range(nz+1):

                x_old = 0
                y_old = 0
                z_old = 0

                t1 = (x == 0 or (x == nx))
                t2 = (y == 0 or (y == ny))
                t3 = (z == 0 or (z == nz))

                if t1 and x != 0:
                    x_old = x - 1
                if t2 and y != 0:
                    y_old = y - 1
                if t3 and z != 0:
                    z_old = z - 1

                if t1 and t2 and t3:
                    new_values[x, y, z] = old_values[x_old, y_old, z_old]

                elif t1 and t2 and not t3:
                    new_values[x, y, z] = (old_values[x_old, y_old, z] + old_values[x_old, y_old, z - 1]) / 2

                elif t1 and not t2 and t3:
                    new_values[x, y, z] = (old_values[x_old, y, z_old] + old_values[x_old, y - 1, z_old]) / 2

                elif not t1 and t2 and t3:
                    new_values[x, y, z] = (old_values[x, y_old, z_old] + old_values[x - 1, y_old, z_old]) / 2

                elif t1 and not t2 and not t3:
                    new_values[x, y, z] = (old_values[x_old, y, z] + old_values[x_old, y, z - 1] + old_values[
                        x_old, y - 1, z] + old_values[x_old, y - 1, z - 1]) / 4

                elif not t1 and t2 and not t3:
                    new_values[x, y, z] = (old_values[x, y_old, z] + old_values[x, y_old, z - 1] + old_values[
                        x - 1, y_old, z] + old_values[x - 1, y_old, z - 1]) / 4

                elif not t1 and not t2 and t3:
                    new_values[x, y, z] = (old_values[x, y, z_old] + old_values[x, y - 1, z_old] + old_values[
                        x - 1, y, z_old] + old_values[x - 1, y - 1, z_old]) / 4

                elif not t1 and not t2 and not t3:
                    new_values[x, y, z] = (old_values[x, y, z] + old_values[x - 1, y, z] + old_values[x, y - 1, z] +
                                           old_values[x, y, z - 1] + old_values[x - 1, y - 1, z] + old_values[
                                               x - 1, y, z - 1] + old_values[x, y - 1, z - 1] + old_values[
                                               x - 1, y - 1, z - 1]) / 8

    return new_values

def histogram3d_to_vtk(histogram3d,
                       filename='histogram.vti',
                       path='.',
                       name='Flux',
                       origin_from_file=True,
                       origin=[0.0, 0.0, 0.0],
                       expand_for_Paraview=False
                       ):
    def copyAndNameArray(data, name):
        if data is not None:
            outdata = data.NewInstance()
            outdata.DeepCopy(data)
            outdata.SetName(name)
            return outdata
        else:
            return None

    values = histogram3d.values.ravel(order='F')
    dimensions = [histogram3d.xnumbins, histogram3d.ynumbins, histogram3d.znumbins]
    spacing = [histogram3d.coordinates_normalization * (histogram3d.edges[0][1] - histogram3d.edges[0][0]),
               histogram3d.coordinates_normalization * (histogram3d.edges[1][1] - histogram3d.edges[1][0]),
               histogram3d.coordinates_normalization * (histogram3d.edges[2][1] - histogram3d.edges[2][0])]

    if origin_from_file is True:
        origin = histogram3d.scoring_mesh_translations

    if expand_for_Paraview:

        values = expand_values_for_Paraview(histogram3d).ravel(order='F')
        dimensions = [histogram3d.xnumbins+1, histogram3d.ynumbins+1, histogram3d.znumbins+1]
        origin = np.array(origin) - np.array(spacing)/2

    imgdat = _vtk.vtkImageData()
    imgdat.GetPointData().SetScalars(
        copyAndNameArray(
            _vtk_np.numpy_to_vtk(
                num_array=values,
                deep=True,
                array_type=_vtk.VTK_FLOAT
            )
            , name
        )
    )
    imgdat.SetDimensions(dimensions[0], dimensions[1], dimensions[2])
    imgdat.SetOrigin(origin[0] - (
            histogram3d.coordinates_normalization * (histogram3d.edges[0][-1] - histogram3d.edges[0][0]) / 2) + (
                             histogram3d.coordinates_normalization * (
                             histogram3d.edges[0][1] - histogram3d.edges[0][0]) / 2),
                     origin[1] - (histogram3d.coordinates_normalization * (
                             histogram3d.edges[1][-1] - histogram3d.edges[1][0]) / 2) + (
                             histogram3d.coordinates_normalization * (
                             histogram3d.edges[1][1] - histogram3d.edges[1][0]) / 2),
                     origin[2] - (histogram3d.coordinates_normalization * (
                             histogram3d.edges[2][-1] - histogram3d.edges[2][0]) / 2) + (
                             histogram3d.coordinates_normalization * (
                             histogram3d.edges[2][1] - histogram3d.edges[2][0]) / 2)
                     )
    imgdat.SetSpacing(spacing)

    writer = _vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(path, filename))
    writer.SetInputData(imgdat)
    writer.SetDataModeToBinary()
    writer.Write()


def beam_to_vtk(filename, output='beam', option_ISO=False, option_NOT_ISO=False, option_primaries=False,
                option_secondaries=False):
    file = uproot.open(filename)
    evt = file.get('Event')

    partID_tracks = evt.arrays(['Trajectory.partID'], library='np')['Trajectory.partID']
    s_tracks = evt.arrays(['Trajectory.S'], library='np')['Trajectory.S']
    tracks = evt.arrays(['Trajectory.XYZ'], library='np')['Trajectory.XYZ']

    mb = _vtk.vtkMultiBlockDataSet()
    mb.SetNumberOfBlocks(len(tracks))

    colors = _vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    red = [255, 0, 0]
    green = [0, 255, 0]
    blue = [0, 0, 255]

    mbIndex = 0

    for i in range(len(tracks)):

        for j in range(len(tracks[i])):

            run = False

            if option_ISO and j == 0 and s_tracks[i].tolist()[j][-1] > 15.42:
                run = True
            if option_NOT_ISO and j == 0 and s_tracks[i].tolist()[j][-1] < 15.42:
                run = True
            if option_primaries and j == 0:
                run = True
            if option_secondaries and j != 0:
                run = True

            if partID_tracks[i][j] == 2212:
                color = blue
            elif partID_tracks[i][j] == 2112:
                color = green
            else:
                run = False

            if run:
                steps = tracks[i].tolist()[j]

                lines = _vtk.vtkCellArray()
                pts = _vtk.vtkPoints()

                line0 = _vtk.vtkLine()
                line1 = _vtk.vtkLine()

                pts.InsertNextPoint([steps[0].member('fX'), steps[0].member('fY'), steps[0].member('fZ')])
                line0.GetPointIds().SetId(0, 0)

                for k in range(1, len(steps) - 1):

                    if k % 2 == 0:
                        lines.InsertNextCell(line0)
                        colors.InsertNextTuple(color)
                        line0 = _vtk.vtkLine()
                    if k % 2 == 1 and k != 1:
                        lines.InsertNextCell(line1)
                        colors.InsertNextTuple(color)
                        line1 = _vtk.vtkLine()

                    pts.InsertNextPoint([steps[k].member('fX'), steps[k].member('fY'), steps[k].member('fZ')])
                    line0.GetPointIds().SetId(k % 2, k)
                    line1.GetPointIds().SetId((k - 1) % 2, k)

                # Create a polydata to store everything in
                linesPolyData = _vtk.vtkPolyData()

                # Add the points to the dataset
                linesPolyData.SetPoints(pts)

                # Add the lines to the dataset
                linesPolyData.SetLines(lines)

                # Color the lines
                # colors.InsertNextTuple(color)
                linesPolyData.GetCellData().SetScalars(colors)

                mb.SetBlock(mbIndex, linesPolyData)
                mbIndex += 1

        print(mbIndex)
        print(f"Progress: {i / len(tracks) * 100}%")

    writer = _vtk.vtkXMLMultiBlockDataWriter()
    writer.SetDataModeToAscii()
    writer.SetInputData(mb)
    print(f"Trying to write file {output}.vtm")
    writer.SetFileName(f"{output}.vtm")
    writer.Write()
