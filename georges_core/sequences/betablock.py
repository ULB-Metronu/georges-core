"""TODO"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import pandas as _pd
from .. import ureg as _ureg
from .. import Q_ as _Q


class BetaBlockType(type):
    """TODO"""
    pass


@dataclass
class BetaBlock(metaclass=BetaBlockType):
    """TODO"""
    BETA11: _Q = 1.0 * _ureg.m
    ALPHA11: float = 0.0
    GAMMA11: Optional[float] = None
    BETA22: _Q = 1.0 * _ureg.m
    ALPHA22: float = 0.0
    GAMMA22: Optional[float] = None
    DISP1: _Q = 0.0 * _ureg.m
    DISP2: _Q = 0.0 * _ureg.radians
    DISP3: _Q = 0.0 * _ureg.m
    DISP4: _Q = 0.0 * _ureg.radians
    EMIT1: _Q = 1E-9 * _ureg('m * radians')
    EMIT2: _Q = 1E-9 * _ureg('m * radians')
    EMIT3: float = 1E-9
    MU1: float = 0.0
    MU2: float = 0.0
    CMU1: float = 1.0
    CMU2: float = 1.0
    DY: float = 0.0
    DX: float = 0.0
    DYP: float = 0.0
    DXP: float = 0.0
    DZ: float = 0.0
    DZP: float = 0.0

    def __post_init__(self):
        if self.GAMMA11 is None:
            self.GAMMA11 = (1 + self.ALPHA11**2) / self.BETA11
        if self.GAMMA22 is None:
            self.GAMMA22 = (1 + self.ALPHA22**2) / self.BETA22

    def __getitem__(self, item):
        return getattr(self, item)

    def __repr__(self):
        return _pd.Series(data={'BETA11': self.BETA11,
                                'ALPHA11': self.ALPHA11,
                                'GAMMA11': self.GAMMA11,
                                'BETA22': self.BETA22,
                                'ALPHA22': self.ALPHA22,
                                'GAMMA22': self.GAMMA22,
                                'DISP1': self.DISP1,
                                'DISP2': self.DISP2,
                                'DISP3': self.DISP3,
                                'DISP4': self.DISP4,
                                'EMIT1': self.EMIT1,
                                'EMIT2': self.EMIT2,
                                'EMIT3': self.EMIT3,
                                'MU1': self.MU1,
                                'MU2': self.MU2,
                                'CMU1': self.CMU1,
                                'CMU2': self.CMU2,
                                'DY': self.DY,
                                'DX': self.DX,
                                'DYP': self.DYP,
                                'DXP': self.DXP,
                                'DZ': self.DZ,
                                'DZP': self.DZP}
                          ).__repr__()
