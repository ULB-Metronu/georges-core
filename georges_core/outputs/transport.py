"""
TODO
"""
from typing import Mapping, Callable
import os
from .. import ureg as _ureg
from ..sequences import Element

TRANSPORT_TYPE_CODES: Mapping[int, Callable] = {
    1: lambda _: Element.Beam(),
    2: lambda _: Element.Face(),
    3: lambda _: Element.Drift(L=float(_[1]) * _ureg.m),
    4: lambda _: Element.SBend(ANGLE=float(_[1]) * _ureg.degree,
                               L=float(_[2]) * _ureg.m,
                               N=float(_[3])
                               ),
    5: lambda _: Element.Quadrupole(L=float(_[1]) * _ureg.m,
                                    K1=(float(_[2]) * _ureg.T) / (float(_[3]) * _ureg.mm)
                                    ),
    6: lambda _: Element.Collimator(),
    7: lambda _: Element.Steerer(),
    10: lambda _: Element.Fit(),
    12: lambda _: Element.Beam_correlations(),
    100: lambda _: Element.Window(),
}


def load_transport_input_file(filename: str, path: str = '.'):
    with open(os.path.join(path, filename), 'r') as f:
        return f.readlines()


def transport_element_factory(d):
    d[0] = float(d[0]) if '.' in d[0] else int(d[0])
    return TRANSPORT_TYPE_CODES[d[0]](d)
