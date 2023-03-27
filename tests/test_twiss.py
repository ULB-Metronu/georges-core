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

    return m


def test_periodic_twiss():
    m = get_madx_input()
    m.command.twiss(
        sequence="RING",
        FILE="twiss.tfs",
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
    m.command.twiss(
        sequence="RING",
        FILE="twiss.tfs",
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
    _np.testing.assert_array_almost_equal(results_twiss["BETA11"].values, twiss_madx["BETX"].values)
    _np.testing.assert_array_almost_equal(results_twiss["BETA22"].values, twiss_madx["BETY"].values)
    _np.testing.assert_array_almost_equal(results_twiss["ALPHA11"].values, twiss_madx["ALFX"].values)
    _np.testing.assert_array_almost_equal(results_twiss["ALPHA22"].values, twiss_madx["ALFY"].values)
    _np.testing.assert_array_almost_equal(results_twiss["DISP1"].values, twiss_madx["DX"].values)
    _np.testing.assert_array_almost_equal(results_twiss["DISP2"].values, twiss_madx["DPX"].values)
    _np.testing.assert_array_almost_equal(results_twiss["DISP3"].values, twiss_madx["DY"].values)
    _np.testing.assert_array_almost_equal(results_twiss["DISP4"].values, twiss_madx["DPY"].values)
    os.remove("twiss.tfs")


def test_lebedev_twiss():
    pass
