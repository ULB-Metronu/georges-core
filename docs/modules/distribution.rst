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
    :hide-output:
    :hide-code:

    import numpy as np

    x = 0.0 * _ureg.cm
    px = 0
    y = 0.0 * _ureg.cm
    py = 0
    dpp = 0

    sigma = np.random.rand(5, 5)
    sigma = np.dot(sigma,sigma.T) # This ensure a positive covariance

    sigma_matrix = {
        "s11": sigma[0][0] * _ureg.m**2,
        "s12": sigma[0][1],
        "s13": sigma[0][2],
        "s14": sigma[0][3],
        "s15": sigma[0][4],
        "s22": sigma[1][1],
        "s23": sigma[1][2],
        "s24": sigma[1][3],
        "s25": sigma[1][4],
        "s33": sigma[2][2] * _ureg.m**2,
        "s34": sigma[2][3],
        "s35": sigma[2][4],
        "s44": sigma[3][3],
        "s45": sigma[3][4],
        "s55": sigma[4][4],
    }

    beam_distribution = Distribution.from_5d_sigma_matrix(
        n=int(1e3), x=x, px=px, y=y, py=py, dpp=dpp, **sigma_matrix
    )

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
