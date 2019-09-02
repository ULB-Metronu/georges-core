"""
TODO
"""
from typing import Dict, Callable, Tuple, Any
import os
from .. import ureg as _ureg
from ..sequences import Element
from ..kinematics import Kinematics as _Kinematics


def _process_beam_code(code, sequence_metadata):
    sequence_metadata.kinematics = _Kinematics(float(code[7]) * _ureg.MeV)


def _process_beam_correlations_code(code, sequence_metadata):
    pass


TRANSPORT_TYPE_CODES: Dict[int, Tuple[Callable[[Any], Any], Callable[[Any, Any], Any]]] = {
    1: (lambda _: None,
        _process_beam_code),
    2: (lambda _: None,
        lambda _, __: None),
    3: (lambda _: Element.Drift(L=float(_[1]) * _ureg.m),
        lambda _, __: None),
    4: (lambda _: Element.SBend(ANGLE=float(_[1]) * _ureg.degree, L=float(_[2]) * _ureg.m, N=float(_[3])),
        lambda _, __: None),
    5: (lambda _: Element.Quadrupole(L=float(_[1]) * _ureg.m, K1=(float(_[2]) * _ureg.T) / (float(_[3]) * _ureg.mm)),
        lambda _, __: None),
    6: (lambda _: Element.Collimator(),
        lambda _, __: None),
    7: (lambda _: Element.Steerer(),
        lambda _, __: None),
    10: (lambda _: None,
         lambda _, __: None),
    12: (lambda _: None,
         _process_beam_correlations_code),
    100: (lambda _: None,
          lambda _, __: None),
}


def load_transport_input_file(filename: str, path: str = '.'):
    with open(os.path.join(path, filename), 'r') as f:
        return f.readlines()


def transport_element_factory(d, sequence_metadata):
    d[0] = float(d[0]) if '.' in d[0] else int(d[0])
    _ = TRANSPORT_TYPE_CODES[d[0]]
    return _[0](d), _[1](d, sequence_metadata)
