from typing import Dict, Callable, Tuple, Any, List, AnyStr
import os
from .. import ureg as _ureg
import georges_core
import numpy as _np

MANZONI_FLAVOR = {"Srotation": "SRotation",
                  "Hkicker": "HKicker",
                  "Vkicker": "VKicker",
                  }

CSV_TO_SEQUENCE = {
    "MARKER": (lambda _: georges_core.sequences.Element.Marker(_[0], APERTYPE=None)),
    "DRIFT": (lambda _: georges_core.sequences.Element.Drift(_[0], L=_[1]['L'], APERTYPE=None)),
    "QUADRUPOLE": (lambda _: georges_core.sequences.Element.Quadrupole(_[0],
                                                                       L=_[1]['L'],
                                                                       K1=_[1]['K1'],
                                                                       APERTURE=_[1]['APERTURE'],
                                                                       APERTYPE=_[1]['APERTYPE'],
                                                                       CHAMBER=_[1]['CHAMBER'])),
    "SBEND": (lambda _: georges_core.sequences.Element.SBend(_[0],
                                                             L=_[1]['L'],
                                                             K1=_[1]['K1'],
                                                             ANGLE=_[1]['ANGLE'],
                                                             E1=_[1]["E1"],
                                                             E2=_[1]["E2"],
                                                             APERTURE=_[1]['APERTURE'],
                                                             APERTYPE=_[1]['APERTYPE'],
                                                             CHAMBER=_[1]['CHAMBER'])),

    "CIRCULARCOLLIMATOR": (lambda _: georges_core.sequences.Element.CircularCollimator(_[0],
                                                                                       L=_[1]['L'],
                                                                                       APERTURE=_[1]['APERTURE'],
                                                                                       APERTYPE="CIRCULAR")),

    "RECTANGULARCOLLIMATOR": (lambda _: georges_core.sequences.Element.RectangularCollimator(_[0],
                                                                                             L=_[1]['L'],
                                                                                             APERTURE=_[1]['APERTURE'],
                                                                                             APERTYPE="RECTANGULAR")),
    "ELLIPTICALCOLLIMATOR": (lambda _: georges_core.sequences.Element.EllipticalCollimator(_[0],
                                                                                           L=_[1]['L'],
                                                                                           APERTURE=_[1]['APERTURE'],
                                                                                           APERTYPE="ELLIPTICAL")),
    "SCATTERER": (lambda _: georges_core.sequences.Element.Scatterer(_[0],
                                                                     L=_[1]["L"],
                                                                     MATERIAL=_[1]["MATERIAL"] if isinstance(
                                                                         _[1]["MATERIAL"], str) else "Air",
                                                                     KINETIC_ENERGY=0 * _ureg.MeV,
                                                                     APERTYPE=None)),

    "DEGRADER": (lambda _: georges_core.sequences.Element.Degrader(_[0],
                                                                   L=_[1]["L"],
                                                                   KINETIC_ENERGY=0 * _ureg.MeV,
                                                                   MATERIAL=_[1]["MATERIAL"] if isinstance(
                                                                       _[1]["MATERIAL"], str) else "Beryllium",
                                                                   APERTYPE=None)),

    "SROTATION": (lambda _: georges_core.sequences.Element.SRotation(_[0],
                                                                     ANGLE=0 * _ureg.radians,
                                                                     APERTYPE=None)),

    "HKICKER": (lambda _: georges_core.sequences.Element.HKicker(_[0],
                                                                 L=_[1]['L'],
                                                                 KICK=0,
                                                                 APERTURE=_[1]['APERTURE'],
                                                                 APERTYPE=_[1]['APERTYPE'],
                                                                 CHAMBER=_[1]['CHAMBER'])),
    "VKICKER": (lambda _: georges_core.sequences.Element.VKicker(_[0],
                                                                 L=_[1]['L'],
                                                                 KICK=0,
                                                                 APERTURE=_[1]['APERTURE'],
                                                                 APERTYPE=_[1]['APERTYPE'],
                                                                 CHAMBER=_[1]['CHAMBER'])),
    "FRINGEIN": (lambda _: georges_core.sequences.Element.Fringein(_[0],
                                                                   L=_[1]['L'],
                                                                   KICK=0,
                                                                   ANGLE=_[1]['ANGLE'],
                                                                   APERTURE=_[1]['APERTURE'],
                                                                   APERTYPE=_[1]['APERTYPE'],
                                                                   CHAMBER=_[1]['CHAMBER'])),
    "FRINGEOUT": (lambda _: georges_core.sequences.Element.Fringeout(_[0],
                                                                     L=_[1]['L'],
                                                                     KICK=0,
                                                                     ANGLE=_[1]['ANGLE'],
                                                                     APERTURE=_[1]['APERTURE'],
                                                                     APERTYPE=_[1]['APERTYPE'],
                                                                     CHAMBER=_[1]['CHAMBER'])),
}


def csv_element_factory(d):
    if d[1]['TYPE'] in CSV_TO_SEQUENCE.keys():
        res = CSV_TO_SEQUENCE[d[1]['TYPE']](d)
    else:
        element_class = MANZONI_FLAVOR.get(d[1]['TYPE'].capitalize(), d[1]['TYPE'].capitalize())
        res = georges_core.sequences.Element.make_subclass(element_class)(d[0],
                                                                          L=d[1]["L"],
                                                                          APERTURE=d[1]['APERTURE'],
                                                                          APERTYPE=d[1]['APERTYPE'])
    return res
