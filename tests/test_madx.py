import os

import cpymad.madx

import georges_core
import georges_core.madx
import georges_core.sequences
from georges_core.sequences import Element, PlacementSequence, SequenceMetadata
from georges_core.units import ureg as _ureg


def get_madx_twiss(filename):
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
        L3:= 1.5;
        EANG:=10.*TWOPI/360;
        ANG := TWOPI/4;
        KQ1:= -0.9;

        D1: DRIFT, L=L3;
        D2: DRIFT, L=0.2;

        OO : DRIFT,L=L2;
        BD : SBEND,L=LB, ANGLE=ANG, E1=EANG,E2=EANG, K2=0.;
        MQ : QUADRUPOLE,L=LQ, K1=KQ;
        MQF : QUADRUPOLE,L=LQ, K1=+KQ1;
        MQD : QUADRUPOLE,L=LQ, K1=-KQ1;
        OBS1: MARKER;
        OBS2: MARKER;

        ACHROM: LINE=(MQD,D2,MQF,D2,BD,OO,MQ,OO,BD,D2,MQF,D2,MQD,OBS1);
        RING: LINE=(ACHROM, D1,ACHROM,D1);

        USE, sequence=RING;
        """,
    )
    m.command.twiss(sequence="RING", FILE=filename)


def test_madx_sequence():
    filename = "twiss.tfs"
    get_madx_twiss(filename)
    madx_line = georges_core.sequences.TwissSequence(
        path=".",
        filename=filename,
        lines=51,
        with_units=True,
        with_beam=True,
        nparticles=100,
    )
    assert georges_core.madx.MadX(sequence=madx_line)
    os.remove("twiss.tfs")


def test_from_placement():
    aper1 = 5 * _ureg.cm
    aper2 = 3.5 * _ureg.cm
    d1 = Element.Drift(
        NAME="D1",
        L=0.3 * _ureg.m,
        APERTYPE="RECTANGULAR",
        APERTURE=[aper1, aper2],
    )

    qf = Element.Quadrupole(
        NAME="Q1",
        L=0.3 * _ureg.m,
        K1=2 * _ureg.m**-2,
        APERTYPE="RECTANGULAR",
        APERTURE=[aper1, aper2],
    )

    d2 = Element.Drift(
        NAME="D2",
        L=0.3 * _ureg.m,
        APERTYPE="CIRCULAR",
        APERTURE=[aper1],
    )

    b1 = Element.SBend(
        NAME="B1",
        L=1 * _ureg.m,
        ANGLE=30 * _ureg.degrees,
        K1=0 * _ureg.m**-2,
        APERTYPE="CIRCULAR",
        APERTURE=[aper1, aper1],
    )

    d3 = Element.Drift(
        NAME="D3",
        L=0.3 * _ureg.m,
        APERTYPE="CIRCULAR",
        APERTURE=[aper1, aper1],
    )

    qd = Element.Quadrupole(
        NAME="Q2",
        L=0.3 * _ureg.m,
        K1=-2 * _ureg.m**-2,
        APERTYPE="RECTANGULAR",
        APERTURE=[aper1, aper2],
    )

    d4 = Element.Drift(
        NAME="D4",
        L=0.3 * _ureg.m,
        APERTYPE="CIRCULAR",
        APERTURE=[aper1, aper1],
    )

    b2 = Element.SBend(
        NAME="B2",
        L=1 * _ureg.m,
        ANGLE=-30 * _ureg.degrees,
        K1=0 * _ureg.m**-2,
        APERTYPE="RECTANGULAR",
        APERTURE=[aper1, aper2],
    )

    d5 = Element.Drift(
        NAME="D5",
        L=0.3 * _ureg.m,
        APERTYPE="CIRCULAR",
        APERTURE=[aper1, aper1],
    )

    sequence = PlacementSequence(
        name="SEQ",
        metadata=SequenceMetadata(kinematics=georges_core.Kinematics(230 * _ureg.MeV)),
    )

    sequence.place(d1, at_entry=0 * _ureg.m)
    sequence.place_after_last(qf)
    sequence.place_after_last(d2)
    sequence.place_after_last(b1)
    sequence.place_after_last(d3)
    sequence.place_after_last(qd)
    sequence.place_after_last(d4)
    sequence.place_after_last(b2)
    sequence.place_after_last(d5)

    mad_input = georges_core.madx.MadX(sequence=sequence)
    assert mad_input
