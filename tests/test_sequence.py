import os

import georges_core
import georges_core.codes_io
import georges_core.sequences
from georges_core.kinematics import Kinematics
from georges_core.units import ureg as _ureg


def test_madx_sequence():
    madx_line = georges_core.sequences.TwissSequence(
        path=f"{os.path.join(os.getcwd(), 'externals')}",
        filename="twiss_madx.tfs",
        with_units=True,
        with_beam=True,
        nparticles=100,
    )
    artist = georges_core.vis.MatplotlibArtist()
    artist.plot_cartouche(madx_line.df)
    artist.plot_beamline(madx_line.df)

    artist = georges_core.vis.PlotlyArtist()
    artist.plot_cartouche(madx_line.df, unsplit_bends=False)


def test_madng_sequence():
    kin = Kinematics(140 * _ureg.MeV)
    madng_line = georges_core.sequences.TwissSequence(
        path=f"{os.path.join(os.getcwd(), 'externals')}",
        filename="twiss_madng.tfs",
        with_units=True,
        lines=34,
        with_beam=False,
        kinematics=kin,
    )
    artist = georges_core.vis.MatplotlibArtist()
    artist.plot_cartouche(madng_line.df)
    artist.plot_beamline(madng_line.df)

    artist = georges_core.vis.PlotlyArtist()
    artist.plot_cartouche(madng_line.df)


def test_transport_sequence():
    transport_line = georges_core.sequences.TransportSequence(
        path=f"{os.path.join(os.getcwd(), 'externals')}",
        filename="calc.bml",
        flavor=georges_core.codes_io.transport.TransportInputIBAFlavor,
    )
    artist = georges_core.vis.MatplotlibArtist()
    artist.plot_cartouche(transport_line.df)
    artist.plot_beamline(transport_line.df)

    artist = georges_core.vis.PlotlyArtist()
    artist.plot_cartouche(transport_line.df)


def test_bdsim_sequence():
    # TODO Fix error when importing root: use conda env ?

    # bdsim_line = georges_core.sequences.BDSIMSequence(
    #     path=f"{os.path.join(os.getcwd(), 'externals')}",
    #     filename="bdsim_output.root",
    # )
    # artist = georges_core.vis.MatplotlibArtist()
    # artist.plot_cartouche(bdsim_line.df)
    # artist.plot_beamline(bdsim_line.df)
    #
    # artist = georges_core.vis.PlotlyArtist()
    # artist.plot_cartouche(bdsim_line.df)
    pass


def test_survey_sequence():
    kin = georges_core.Kinematics(140 * _ureg.MeV)
    survey_line = georges_core.sequences.SurveySequence(
        path=f"{os.path.join(os.getcwd(), 'externals')}",
        filename="survey.csv",
        kinematics=kin,
    )
    survey_line.expand()
    artist = georges_core.vis.MatplotlibArtist()
    artist.plot_cartouche(survey_line.df)
    artist.plot_beamline(survey_line.df)

    artist = georges_core.vis.PlotlyArtist()
    artist.plot_cartouche(survey_line.df)
