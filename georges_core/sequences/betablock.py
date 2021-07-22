"""TODO"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from .. import ureg as _ureg


class BetaBlockType(type):
    """TODO"""
    pass


@dataclass
class BetaBlock(metaclass=BetaBlockType):
    """TODO"""
    BETA11: float = 1.0 * _ureg.m
    ALPHA11: float = 0.0
    GAMMA11: Optional[float] = None
    BETA22: float = 1.0 * _ureg.m
    ALPHA22: float = 0.0
    GAMMA22: Optional[float] = None
    DISP1: float = 0.0 * _ureg.m
    DISP2: float = 0.0 * _ureg.radians
    DISP3: float = 0.0 * _ureg.m
    DISP4: float = 0.0 * _ureg.radians
    EMIT1: float = 1E-9 * _ureg('m * radians')
    EMIT2: float = 1E-9 * _ureg('m * radians')
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
