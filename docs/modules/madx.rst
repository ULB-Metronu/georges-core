*****
MAD-X
*****
This module allows to run MAD-X through cpymad.
A sequence loaded with `georges_core.sequence` can be send to MAD-X
to compute the Twiss or perform a tracking simulation.

The example below creates a simple line with georges-core and perform a
Twiss with cpymad and the results are shown in a figure.

.. jupyter-execute::
    :hide-output:

    import georges_core
    import georges_core.madx
    from georges_core.units import ureg as _ureg
    from georges_core.sequences import Element
    from georges_core.sequences import PlacementSequence, SequenceMetadata
    import matplotlib.pyplot as plt

.. jupyter-execute::
    :hide-output:

    aper1 = 5 *_ureg.cm
    aper2 = 3.5 * _ureg.cm

    d1 = Element.Drift(NAME="D1",
                       L=0.3* _ureg.m,
                       APERTYPE="RECTANGULAR",
                       APERTURE=[aper1, aper2])

    qf = Element.Quadrupole(NAME="Q1",
                            L=0.3*_ureg.m,
                            K1=2*_ureg.m**-2,
                            APERTYPE="RECTANGULAR",
                            APERTURE=[aper1, aper2])

    d2 = Element.Drift(NAME="D2",
                       L=0.3*_ureg.m,
                       APERTYPE="CIRCULAR",
                       APERTURE=[aper1])

    b1 = Element.SBend(NAME="B1",
                       L=1*_ureg.m,
                       ANGLE=30*_ureg.degrees,
                       K1=0*_ureg.m**-2,
                       APERTYPE="CIRCULAR",
                       APERTURE=[aper1, aper1])

    d3 = Element.Drift(NAME="D3",
                       L=0.3*_ureg.m,
                       APERTYPE="CIRCULAR",
                       APERTURE=[aper1, aper1])

    qd = Element.Quadrupole(NAME="Q2",
                            L=0.3*_ureg.m,
                            K1=-2*_ureg.m**-2,
                            APERTYPE="RECTANGULAR",
                            APERTURE=[aper1, aper2])

    d4 = Element.Drift(NAME="D4",
                       L=0.3*_ureg.m,
                       APERTYPE="CIRCULAR",
                       APERTURE=[aper1, aper1])

    b2 = Element.SBend(NAME="B2",
                       L=1*_ureg.m,
                       ANGLE=-30*_ureg.degrees,
                       K1=0*_ureg.m**-2,
                       APERTYPE="RECTANGULAR",
                       APERTURE=[aper1, aper2])

    d5 = Element.Drift(NAME="D5",
                       L=0.3*_ureg.m,
                       APERTYPE="CIRCULAR",
                       APERTURE=[aper1, aper1])

    sequence = PlacementSequence(name="seq", metadata=SequenceMetadata(kinematics=georges_core.Kinematics(230 *_ureg.MeV)))

    sequence.place(d1,at_entry=0*_ureg.m)
    sequence.place_after_last(qf)
    sequence.place_after_last(d2)
    sequence.place_after_last(b1)
    sequence.place_after_last(d3)
    sequence.place_after_last(qd)
    sequence.place_after_last(d4)
    sequence.place_after_last(b2)
    sequence.place_after_last(d5)

.. jupyter-execute::

    mad_input = georges_core.madx.MadX(sequence=sequence)
    twiss = mad_input.twiss(sequence='seq');

    plt.plot(twiss.s, twiss.betx)
    plt.xlabel('S')
    plt.ylabel('BETX')
    plt.show()

.. note::

    This instance of cpymad can be easily converted into a BDSIM input
    using `pybdsim <http://www.pp.rhul.ac.uk/bdsim/pybdsim/convert.html>`_
