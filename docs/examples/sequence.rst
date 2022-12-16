*****************
Create a sequence
*****************

In this example, we will show how to create a simple sequence using the module `sequence` and how to display it using the
`vis` module.

Fist, we import the necessary module:

.. jupyter-execute::
   :hide-output:

    %matplotlib inline
    import matplotlib.pyplot as plt
    import georges_core
    from georges_core.units import ureg as _ureg
    from georges_core.sequences import Element
    from georges_core.sequences import PlacementSequence
    from georges_core.vis import MatplotlibArtist, PlotlyArtist

Then, we define the elements of our line:

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
                               APERTURE=[aper1, aper1])

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

Finally, we place the elements in a sequence.

.. jupyter-execute::
    :hide-output:

    sequence = PlacementSequence(name="Sequence")

    sequence.place(d1,at_entry=0*_ureg.m)
    sequence.place_after_last(qf)
    sequence.place_after_last(d2)
    sequence.place_after_last(b1)
    sequence.place_after_last(d3)
    sequence.place_after_last(qd)
    sequence.place_after_last(d4)
    sequence.place_after_last(b2)
    sequence.place_after_last(d5);

Of course, we can visualize the sequence with matplotlib or plotly.

.. jupyter-execute::

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15,8))

    artist = MatplotlibArtist(ax1)
    artist.plot_beamline(sequence.df, plane='X')
    artist.plot_cartouche(sequence.df)

    artist = MatplotlibArtist(ax2)
    artist.plot_beamline(sequence.df, plane='Y')
    artist.plot_cartouche(sequence.df)

.. jupyter-execute::

    artist = PlotlyArtist()
    artist.plot_cartouche(sequence.df)
    artist.render()
