"""
TODO
"""
from __future__ import annotations

from abc import ABCMeta
from collections import UserDict
from typing import Any, Dict, Optional, Tuple, Type, Union

from .. import ureg as _ureg


class ElementClass(ABCMeta):
    """
    TODO
    """

    def __new__(
        mcs, name: str, bases: Optional[Union[type, Tuple[type]]] = None, dct: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Element:
        dct = dct or {}
        if not isinstance(bases, tuple):
            bases = (bases,)
        c = super().__new__(mcs, name, bases, {**dct, **kwargs})
        if name != "Element":
            setattr(bases[0], name, c)
            setattr(c, "metaclass", type(name + "Class", (mcs,), {}))
        return c

    def __init__(
        cls, name: str, bases: Optional[Union[type, Tuple[type]]] = None, dct: Optional[Dict[str, Any]] = None, **kwargs
    ):
        dct = dct or {}
        if not isinstance(bases, tuple):
            bases = (bases,)
        super().__init__(name, bases, {**dct, **kwargs})
        if name != "Element":
            for b in bases:
                cls.parameters = {**{"KEYWORD": cls.__name__.upper()}, **b.parameters, **dct, **kwargs}


class Element(UserDict, metaclass=ElementClass):
    """
    TODO
    """

    parameters = {}
    metaclass = ElementClass

    def __init__(self, name: Optional[str] = None, **kwargs: Any):
        super().__init__(**{**kwargs, **{"NAME": name, "CLASS": self.__class__.__name__, "L": 0 * _ureg.m}})
        self.data = {**self.data, **self.__class__.parameters, **kwargs}

    def __getattr__(self, k: str) -> Any:
        """Provides attribute-like access to the dictionary elements."""
        return self.get(k)

    @classmethod
    def make_subclass(
        cls,
        name: str,
        bases: Optional[Union[type, Tuple[type]]] = None,
        dct: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Type[ElementClass]:
        """

        Args:
            name:
            bases:
            dct:
            **kwargs:

        Returns:

        """
        if bases is None:
            bases = cls
        return cls.metaclass(name, bases, dct, **kwargs)

    subclass = make_subclass


Element.make_subclass("Marker")
Element.make_subclass("Instrument")
Element.make_subclass("Solenoid")
Element.make_subclass("Drift")
Element.make_subclass(
    "Quadrupole",
    K1=0.0 * _ureg.m**-2,
    K1L=0.0 * _ureg.m**-1,
    E1=0.0 * _ureg.radian,
    E2=0.0 * _ureg.radian,
    TILT=0.0 * _ureg.radian,
)
Element.make_subclass("Sextupole")
Element.make_subclass("Octupole")
Element.make_subclass("Decapole")
Element.make_subclass("Multipole")
Element.make_subclass("Dipole")
Element.make_subclass("Bend")
Element.make_subclass(
    "RBend",
    TILT=0.0 * _ureg.radian,
)
Element.make_subclass("Face", L=0.0 * _ureg.meter, E1=0.0 * _ureg.degrees)
Element.make_subclass(
    "SBend",
    E1=0.0 * _ureg.radian,
    E2=0.0 * _ureg.radian,
    ANGLE=0.0 * _ureg.radian,
    TILT=0.0 * _ureg.radian,
)
Element.make_subclass("HKicker", L=0.0 * _ureg.m, KICK=0.0 * _ureg.radian, TILT=0.0 * _ureg.radian)
Element.make_subclass("VKicker", L=0.0 * _ureg.m, KICK=0.0 * _ureg.radian, TILT=0.0 * _ureg.radian)
Element.make_subclass("Cavity")
Element.make_subclass("Steerer")
Element.make_subclass("CircularCollimator")
Element.make_subclass("RectangularCollimator")
Element.make_subclass("EllipticalCollimator")
Element.make_subclass("Scatterer")
Element.make_subclass("Degrader")
Element.make_subclass("SRotation")
Element.make_subclass("Fringein", TILT=0.0 * _ureg.radian)
Element.make_subclass("Fringeout", TILT=0.0 * _ureg.radian)
