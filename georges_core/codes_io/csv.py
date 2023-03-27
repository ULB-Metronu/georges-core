from typing import Any, Dict

from georges_core.sequences import Element

from .. import ureg as _ureg

MANZONI_FLAVOR = {
    "Srotation": "SRotation",
    "Hkicker": "HKicker",
    "Vkicker": "VKicker",
}

CSV_TO_SEQUENCE: Dict[str, Any] = {
    "MARKER": (lambda _: Element.Marker(_[0], APERTYPE=None)),
    "DRIFT": (
        lambda _: Element.Drift(
            _[0],
            L=_[1]["L"],
            APERTYPE=_[1]["APERTYPE"],
            APERTURE=_[1]["APERTURE"],
        )
    ),
    "QUADRUPOLE": (
        lambda _: Element.Quadrupole(
            _[0],
            L=_[1]["L"],
            K1=_[1]["K1"],
            APERTURE=_[1]["APERTURE"],
            APERTYPE=_[1]["APERTYPE"],
            CHAMBER=_[1]["CHAMBER"],
        )
    ),
    "SBEND": (
        lambda _: Element.SBend(
            _[0],
            L=_[1]["L"],
            K1=_[1]["K1"],
            ANGLE=_[1]["ANGLE"],
            E1=_[1]["E1"],
            E2=_[1]["E2"],
            APERTURE=_[1]["APERTURE"],
            APERTYPE=_[1]["APERTYPE"],
            CHAMBER=_[1]["CHAMBER"],
        )
    ),
    "CIRCULARCOLLIMATOR": (
        lambda _: Element.CircularCollimator(
            _[0],
            L=_[1]["L"],
            APERTURE=_[1]["APERTURE"],
            APERTYPE="CIRCULAR",
        )
    ),
    "RECTANGULARCOLLIMATOR": (
        lambda _: Element.RectangularCollimator(
            _[0],
            L=_[1]["L"],
            APERTURE=_[1]["APERTURE"],
            APERTYPE="RECTANGULAR",
        )
    ),
    "ELLIPTICALCOLLIMATOR": (
        lambda _: Element.EllipticalCollimator(
            _[0],
            L=_[1]["L"],
            APERTURE=_[1]["APERTURE"],
            APERTYPE="ELLIPTICAL",
        )
    ),
    "SCATTERER": (
        lambda _: Element.Scatterer(
            _[0],
            L=_[1]["L"],
            MATERIAL=_[1]["MATERIAL"] if isinstance(_[1]["MATERIAL"], str) else "Air",
            KINETIC_ENERGY=0 * _ureg.MeV,
            APERTYPE=None,
        )
    ),
    "DEGRADER": (
        lambda _: Element.Degrader(
            _[0],
            L=_[1]["L"],
            KINETIC_ENERGY=0 * _ureg.MeV,
            MATERIAL=_[1]["MATERIAL"] if isinstance(_[1]["MATERIAL"], str) else "Beryllium",
            APERTYPE=None,
        )
    ),
    "SROTATION": (lambda _: Element.SRotation(_[0], ANGLE=0 * _ureg.radians, APERTYPE=None)),
    "HKICKER": (
        lambda _: Element.HKicker(
            _[0],
            L=_[1]["L"],
            KICK=0,
            APERTURE=_[1]["APERTURE"],
            APERTYPE=_[1]["APERTYPE"],
            CHAMBER=_[1]["CHAMBER"],
        )
    ),
    "VKICKER": (
        lambda _: Element.VKicker(
            _[0],
            L=_[1]["L"],
            KICK=0,
            APERTURE=_[1]["APERTURE"],
            APERTYPE=_[1]["APERTYPE"],
            CHAMBER=_[1]["CHAMBER"],
        )
    ),
    "FRINGEIN": (
        lambda _: Element.Fringein(
            _[0],
            L=_[1]["L"],
            KICK=0,
            ANGLE=_[1]["ANGLE"],
            APERTURE=_[1]["APERTURE"],
            APERTYPE=_[1]["APERTYPE"],
            CHAMBER=_[1]["CHAMBER"],
        )
    ),
    "FRINGEOUT": (
        lambda _: Element.Fringeout(
            _[0],
            L=_[1]["L"],
            KICK=0,
            ANGLE=_[1]["ANGLE"],
            APERTURE=_[1]["APERTURE"],
            APERTYPE=_[1]["APERTYPE"],
            CHAMBER=_[1]["CHAMBER"],
        )
    ),
}


def csv_element_factory(d: Any) -> Any:
    if d[1]["TYPE"] in CSV_TO_SEQUENCE.keys():
        res = CSV_TO_SEQUENCE[d[1]["TYPE"]](d)
    else:
        element_class = MANZONI_FLAVOR.get(d[1]["TYPE"].capitalize(), d[1]["TYPE"].capitalize())
        res = Element.make_subclass(element_class)(
            d[0],
            L=d[1]["L"],
            APERTURE=d[1]["APERTURE"],
            APERTYPE=d[1]["APERTYPE"],
        )
    return res
