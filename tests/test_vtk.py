import os
import shutil

from georges_core.vis import vtk_utils


def test_write_beam_file():
    vtk_utils.beam_to_vtk(
        f"{os.path.join(os.getcwd(), 'externals/bdsim_output.root')}",
        option_primaries=True,
        option_secondaries=True,
    )
    os.remove("beam.vtm")
    shutil.rmtree("beam", ignore_errors=True)
