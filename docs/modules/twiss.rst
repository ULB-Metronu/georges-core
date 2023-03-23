************
Twiss module
************

This module allows to calculate the different parameters of Twiss as well as the dispersion based on a propagation matrix. It is possible either to compute the periodic Twiss function of a lattice or to compute the propagation of an initial :code:`BetaBlock`. The standard uncoupled Twiss parametrization (including off-momentum effects, aka. dispersion) is the default option. Additional formalisms for the parametrization of fully coupled transfer matrices are also available (Teng, Ripken, etc.). The implementation of the coupled transfert matrix is detailed in this paper <link URL>`_

Herebelow, there is some examples on how to use the Twiss module.

.. jupyter-execute::
    :hide-output:

    import os
    import pandas as pd
    import georges_core
    from georges_core.units import ureg as _ureg
    from georges_core.sequences import BetaBlock
    from georges_core.twiss import Twiss


.. jupyter-execute::
    :hide-output:

    twiss_init = BetaBlock(BETA11 = 1*_ureg.m,
                           BETA22 = 2*_ureg.m)

    tw = Twiss(twiss_init=twiss_init, with_phase_unrolling=False)

.. jupyter-execute::
    :hide-output:
    :hide-code:

    import cpymad.madx
    m = cpymad.madx.Madx(stdout=False)
    m.input(
        """
        BEAM, PARTICLE=PROTON, ENERGY = 0.250+0.938, PARTICLE = PROTON, EX=1e-6, EY=1e-6;

        RHO:=1.35;
        KQ := +0.9;
        LCELL:=4.;
        LQ:= 0.3;
        LB:= 2.0;
        L2:=0.5*(LCELL-LB-LQ);
        L3:= 0.5;
        EANG:=10.*TWOPI/360;
        KQ1:= -0.9;

        D1: DRIFT, L=L3;
        D2: DRIFT, L=0.2;
        D3: DRIFT, L=0.2;
        D4: DRIFT, L=0.1;

        BD : SBEND,L=LB, ANGLE=EANG;
        MQF1 : QUADRUPOLE,L=0.5*LQ, K1=+KQ1;
        MQF2: MQF1;
        MQD : QUADRUPOLE,L=LQ, K1=-KQ1;

        ACHROM: LINE=(MQF1, D1, MQD, D2, BD,D3, MQF2, D4);
        RING: LINE=(ACHROM);
        USE, sequence=RING;
        """,
    )

    m.command.twiss(
        sequence="RING",
        FILE="twiss.tfs",
        rmatrix=True,
    )
    twiss_madx = georges_core.codes_io.load_mad_twiss_table("twiss.tfs", lines=51)
    matrix = twiss_madx.filter(regex=("RE([1-9][0-9]*|0)"))
    matrix = matrix.rename(columns={element: element.replace("RE", "R") for element in matrix.columns.tolist()})
    os.remove("twiss.tfs")

.. jupyter-execute::
    :hide-output:

    # Load a transfer matrix which can computed with MAD-X or Zgoubidoo.
    results_twiss = tw(matrix=matrix)

You can now have acces to the Twiss functions along the line as well as the dispersion:

.. jupyter-execute::

    print(results_twiss)