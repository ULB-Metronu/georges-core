*****************
Beam distribution
*****************

Georges-core provides methods to load and analyze beam distributions.
These methods return a distribution object that can be used or analyzed.

.. jupyter-execute::
    :hide-output:

    import georges_core
    from georges_core.distribution import Distribution
    from georges_core.units import ureg as _ureg


Supported format for external files are *csv* and *parquet*::

    beam_distribution = Distribution.from_csv('beam.csv')
    beam_distribution = Distribution.from_parquet('beam.tar.gz')

A beam distribution can be defined from his parameters. Different methods are implemented, such as:

* *from_5d_sigma_matrix*

.. jupyter-execute::

    pass

* *from_5d_multigaussian_distribution*

.. jupyter-execute::
    :hide-output:
    :hide-code:

    x = 0.0 * _ureg.cm
    px = 0
    y = 0.0 * _ureg.cm
    py = 0
    dpp = 0
    xrms = 0.5 * _ureg.cm
    yrms = 0.5 * _ureg.cm
    pxrms = 0.5
    pyrms = 0.5
    dpprms = 0

.. jupyter-execute::

     beam_distribution = Distribution.from_5d_multigaussian_distribution(n=int(1e3),
                                                                        x=x,
                                                                        px=px,
                                                                        y=y,
                                                                        py=py,
                                                                        dpp=dpp,
                                                                        xrms=xrms,
                                                                        pxrms=pxrms,
                                                                        yrms=yrms,
                                                                        pyrms=pyrms,
                                                                        dpprms=dpprms)

* *from_twiss_parameters*

::

    beam_distribution = Distribution.from_twiss_parameters(n=int(1e3),
                                                               x=x,
                                                               px=px,
                                                               y=y,
                                                               py=py,
                                                               dpp=dpp,
                                                               betax=betax,
                                                               alphax=alphax,
                                                               betay=betay,
                                                               alphay=alphay,
                                                               emitx=emitx,
                                                               emity=emity,
                                                               dispx=dispx,
                                                               dispy=dispy,
                                                               dispxp=dispxp,
                                                               dispyp=dispyp,
                                                               dpprms=dpprms)

All these methods give a instance of a Class distribution.
This class has many properties to analyse the beam's distribution.

.. jupyter-execute::

    beam_distribution.mean

.. jupyter-execute::

    beam_distribution.std

.. jupyter-execute::

    beam_distribution.emit

.. jupyter-execute::

    beam_distribution.twiss
