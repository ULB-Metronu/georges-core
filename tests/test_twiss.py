import os

import cpymad.madx
import numpy as _np

import georges_core
from georges_core.sequences import BetaBlock
from georges_core.twiss import Twiss
from georges_core.units import ureg as _ureg


def get_madx_input():
    m = cpymad.madx.Madx(stdout=False)
    m.input(
        """
        beam;

        b:     sbend,l=35.09, angle = 0.011306116;
        qf:    quadrupole,l=1.6,k1=-0.02268553;
        qd:    quadrupole,l=1.6,k1=0.022683642;
        sf:    sextupole,l=0.4,k2=-0.13129;
        sd:    sextupole,l=0.76,k2=0.26328;

        ! define the cell as a sequence:
        sequ:  sequence,l=79;
        b1:    b,      at=19.115;
        sf1:   sf,     at=37.42;
        qf1:   qf,     at=38.70;
        b2:    b,      at=58.255,angle=b1->angle;
        sd1:   sd,     at=76.74;
        qd1:   qd,     at=78.20;
        endm:  marker, at=79.0;
        endsequence;


        ! skew quadrupole
        qfs:	quadrupole,l=0, k1s=-5e-8;
        qds:	quadrupole,l=0,k1s=3e-7;

        ! sequence with a skew quadrupole
        sequSkew: sequence,l=79;
        b1:	b,	at=19.115;
        sf1:   sf,     at=37.42;
        qf1:   qf,     at=38.70;
        qfs1:  qfs,    at=38.70+1.6/2;
        b2:    b,      at=58.255,angle=b1->angle;
        sd1:   sd,     at=76.74;
        !qd1:   qd,     at=78.20;
        qds1:  qds,    at=78.20+1.6/2;
        endm:  marker, at=79.0;
        endsequence;

        """,
    )

    return m


def test_periodic_twiss():
    m = get_madx_input()
    m.use(sequence="sequ")
    m.command.twiss(
        sequence="sequ",
        file="twiss.tfs",
        rmatrix=True,
    )
    twiss_madx = georges_core.codes_io.load_mad_twiss_table("twiss.tfs", lines=51)
    matrix = twiss_madx.filter(regex=("RE([1-9][0-9]*|0)"))
    matrix = matrix.rename(columns={element: element.replace("RE", "R") for element in matrix.columns.tolist()})
    results_twiss = Twiss()(matrix=matrix)

    # We only check that the Twiss init are the same as the transfert matrix with MAD-X is periodic.
    _np.testing.assert_approx_equal(results_twiss.iloc[0]["BETA11"], twiss_madx.iloc[0]["BETX"], significant=4)
    _np.testing.assert_approx_equal(results_twiss.iloc[0]["BETA22"], twiss_madx.iloc[0]["BETY"], significant=4)
    _np.testing.assert_approx_equal(results_twiss.iloc[0]["ALPHA11"], twiss_madx.iloc[0]["ALFX"], significant=4)
    _np.testing.assert_approx_equal(results_twiss.iloc[0]["ALPHA22"], twiss_madx.iloc[0]["ALFY"], significant=4)
    _np.testing.assert_approx_equal(results_twiss.iloc[0]["DISP1"], twiss_madx.iloc[0]["DX"], significant=4)
    _np.testing.assert_approx_equal(results_twiss.iloc[0]["DISP2"], twiss_madx.iloc[0]["DPX"])
    _np.testing.assert_approx_equal(results_twiss.iloc[0]["DISP3"], twiss_madx.iloc[0]["DY"])
    _np.testing.assert_approx_equal(results_twiss.iloc[0]["DISP4"], twiss_madx.iloc[0]["DPY"])
    os.remove("twiss.tfs")


def test_nonperiodic_twiss():
    m = get_madx_input()
    m.use(sequence="sequ")
    m.command.twiss(
        sequence="sequ",
        file="twiss.tfs",
        rmatrix=True,
        betx=1,
        bety=2,
        alfx=0.5,
        alfy=-1,
        dx=0.1,
    )
    twiss_madx = georges_core.codes_io.load_mad_twiss_table("twiss.tfs", lines=51)
    matrix = twiss_madx.filter(regex=("RE([1-9][0-9]*|0)"))
    matrix = matrix.rename(columns={element: element.replace("RE", "R") for element in matrix.columns.tolist()})
    results_twiss = Twiss(
        twiss_init=BetaBlock(
            BETA11=1 * _ureg.m,
            BETA22=2 * _ureg.m,
            ALPHA11=0.5,
            ALPHA22=-1,
            DISP1=0.1 * _ureg.m,
        ),
    )(matrix=matrix)
    _np.testing.assert_array_almost_equal(results_twiss["BETA11"].values, twiss_madx["BETX"].values, decimal=4)
    _np.testing.assert_array_almost_equal(results_twiss["BETA22"].values, twiss_madx["BETY"].values, decimal=4)
    _np.testing.assert_array_almost_equal(results_twiss["ALPHA11"].values, twiss_madx["ALFX"].values, decimal=4)
    _np.testing.assert_array_almost_equal(results_twiss["ALPHA22"].values, twiss_madx["ALFY"].values, decimal=4)
    _np.testing.assert_array_almost_equal(results_twiss["DISP1"].values, twiss_madx["DX"].values, decimal=4)
    _np.testing.assert_array_almost_equal(results_twiss["DISP2"].values, twiss_madx["DPX"].values, decimal=4)
    _np.testing.assert_array_almost_equal(results_twiss["DISP3"].values, twiss_madx["DY"].values, decimal=4)
    _np.testing.assert_array_almost_equal(results_twiss["DISP4"].values, twiss_madx["DPY"].values, decimal=4)
    os.remove("twiss.tfs")


# def test_lebedev_twiss():
# m = get_madx_input()
# m.use(sequence="sequSkew")
# m.command.ptc_create_universe()
# m.command.ptc_create_layout(model=2, method=6, nst=10, exact=True)
# m.command.select(flag="ptc_twiss")
# m.command.ptc_twiss(deltap=0.0, closed_orbit=True, icase=5, file="coupledTwiss.tfs", no=3)
# m.command.ptc_end()

# twiss_madx = georges_core.codes_io.load_mad_twiss_table("coupledTwiss.tfs", lines=89)
# os.remove("coupledTwiss.tfs")
