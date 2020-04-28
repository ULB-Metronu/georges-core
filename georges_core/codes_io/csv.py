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
    "QUADRUPOLE": (lambda _: georges_core.sequences.Element.Quadrupole(_[0],
                                                                       L=_[1]['LENGTH'],
                                                                       K1=0*_ureg.m**-2,
                                                                       APERTURE=_[1]['APERTURE'],
                                                                       APERTYPE=_[1]['APERTYPE'])),

    "SBEND": (lambda _: georges_core.sequences.Element.SBend(_[0],
                                                             L=_[1]['LENGTH'],
                                                             K1=0*_ureg.m**-2,
                                                             E1=_[1]["E1"] if not _np.isnan(_[1]["E1"]) else 0 * _ureg.radians,
                                                             E2=_[1]["E2"] if not _np.isnan(_[1]["E2"]) else 0 * _ureg.radians,
                                                             APERTURE=_[1]['APERTURE'],
                                                             APERTYPE=_[1]['APERTYPE'])),

    "COLLIMATOR": (lambda _: georges_core.sequences.Element.Collimator(_[0],
                                                L=_[1]['LENGTH'],
                                                APERTURE=_[1]['APERTURE'],
                                                APERTYPE=_[1]['APERTYPE'])),

}


def csv_element_factory(d):

    if d[1]['TYPE'] in CSV_TO_SEQUENCE.keys():
        res = CSV_TO_SEQUENCE[d[1]['TYPE']](d)
    else:
        element_class = MANZONI_FLAVOR.get(d[1]['TYPE'].capitalize(), d[1]['TYPE'].capitalize())
        res = georges_core.sequences.Element.make_subclass(element_class)(d[0],
                                                                          L=d[1]["LENGTH"],
                                                                          APERTURE=d[1]['APERTURE'],
                                                                          APERTYPE=d[1]['APERTYPE'])
    return res
