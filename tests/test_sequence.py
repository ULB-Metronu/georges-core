import georges_core
import georges_core.sequences
import georges_core.codes_io
from georges_core.units import ureg as _ureg
from georges_core.kinematics import Kinematics


def test_madx_sequence():
    madx_line = georges_core.sequences.TwissSequence(path='../examples/converter/MADX',
                                                     filename='twiss_madx.tfs',
                                                     with_units=True,
                                                     with_beam=True,
                                                     nparticles=100
                                                     )
    artist = georges_core.vis.MatplotlibArtist()
    artist.plot_cartouche(madx_line.df)
    artist.plot_beamline(madx_line.df)


def test_madng_sequence():
    kin = Kinematics(140 * _ureg.MeV)
    madng_line = georges_core.sequences.TwissSequence(path='../examples/converter/MADNG',
                                                      filename='twiss_madng.tfs',
                                                      with_units=True,
                                                      lines=34,
                                                      with_beam=False,
                                                      kinematics=kin)
    artist = georges_core.vis.MatplotlibArtist()
    artist.plot_cartouche(madng_line.df)
    artist.plot_beamline(madng_line.df)


def test_transport_sequence():
    transport_line = georges_core.sequences.TransportSequence(path='../examples/converter/TRANSPORT',
                                                              filename='calc.bml',
                                                              flavor=georges_core.codes_io.transport.TransportInputIBAFlavor)
    artist = georges_core.vis.MatplotlibArtist()
    artist.plot_cartouche(transport_line.df)
    artist.plot_beamline(transport_line.df)


def test_bdsim_sequence():
    bdsim_line = georges_core.sequences.BDSIMSequence(path='../examples/converter/BDSIM',
                                                      filename='output.root')
    artist = georges_core.vis.MatplotlibArtist()
    artist.plot_cartouche(bdsim_line.df)
    artist.plot_beamline(bdsim_line.df)


def test_survey_sequence():
    kin = georges_core.Kinematics(140 * _ureg.MeV)
    survey_line = georges_core.sequences.SurveySequence(path='../examples/converter/CSV',
                                                        filename="survey.csv",
                                                        kinematics=kin)
    survey_line.expand()
    artist = georges_core.vis.MatplotlibArtist()
    artist.plot_cartouche(survey_line.df)
    artist.plot_beamline(survey_line.df)
