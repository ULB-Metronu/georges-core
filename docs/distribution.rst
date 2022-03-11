****************************
Generate a beam distribution
****************************

Georges-core provides methods to load and analyze beam distributions.
These methods return a distribution object that can be used or analyzed.
Supported format for external files are *csv* and *parquet*::

    beam_distribution = Distribution.from_csv('beam.csv')
    beam_distribution = Distribution.from_parquet('beam.tar.gz')

A beam distribution can be defined from his parameters. Different methods are implemented, such as:

* *from_5d_sigma_matrix*

::

    pass

* *from_5d_multigaussian_distribution*

::

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


Analysis :

* mean
* std
* emit
* coupling
* halo