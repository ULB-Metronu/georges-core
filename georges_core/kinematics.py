"""Georges-core relativistic' physics module.

This module provides a collection of functions and classes to deal with relativistic physics computations. This mainly
concerns conversions between kinematic quantities. Full support for units (via ``pint``) is provided. Additionnally, a
helper class (``Kinematics``) provides automatic construction and conversion of kinematics quantities.

Examples:
    >>> Kinematics(230 *_ureg.MeV) #doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    Proton
    (.etot) Total energy: 1168.2720299999999 megaelectronvolt
    (.ekin) Kinetic energy: 230 megaelectronvolt
    (.momentum) Momentum: 696.0640299570144 megaelectronvolt_per_c
    (.brho): Magnetic rigidity: 2.321819896553311 meter * tesla
    (.range): Range in water (protons only): 32.9424672323197 centimeter
    (.pv): Relativistic pv: 414.71945005821954 megaelectronvolt
    (.beta): Relativistic beta: 0.5958064663732595
    (.gamma): Relativistic gamma: 1.2451314678963625
    <BLANKLINE>
"""

from __future__ import annotations

from functools import partial as _partial
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as _np

from georges_core.particles import Proton as _Proton
from georges_core.units import Q_ as _Q
from georges_core.units import ureg as _ureg

if TYPE_CHECKING:
    from georges_core.particles import ParticuleType as _ParticuleType


class KinematicsException(Exception):
    """Exception raised for errors in the Kinematics module."""

    def __init__(self, m: str = "") -> None:
        self.message = m


class Kinematics:
    """TODO"""

    def __init__(self, q: Union[float, _Q], particle: _ParticuleType = _Proton, kinetic: Optional[bool] = None):
        """

        Args:
            q:
            particle:
            kinetic:
        """
        self._q: Union[float, _Q] = q
        self._p: _ParticuleType = particle
        self._type: Optional[str] = None

        if _Q(q).dimensionality == _ureg.cm.dimensionality:
            self._type = "range"
        elif _Q(q).dimensionality == _ureg.eV.dimensionality:
            if kinetic is True:
                self._type = "ekin"
            elif kinetic is False:
                self._type = "etot"
            else:
                if _Q(q) < particle.M * _ureg.c**2:
                    self._type = "ekin"
                else:
                    self._type = "etot"
        elif _Q(q).dimensionality == _ureg.eV_c.dimensionality:
            self._type = "momentum"
        elif _Q(q).dimensionality == (_ureg.tesla * _ureg.m).dimensionality:
            self._type = "brho"
        elif _Q(q).dimensionless:
            if q < 1:
                self._type = "beta"
            else:
                self._type = "gamma"
        else:
            raise KinematicsException("Invalid kinematic quantity.")

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"""
        {self._p.__name__}
        (.etot) Total energy: {self.etot}
        (.ekin) Kinetic energy: {self.ekin}
        (.momentum) Momentum: {self.momentum}
        (.brho): Magnetic rigidity: {self.brho.to('tesla meter')}
        (.range): Range in water (protons only): {self.range if self._p == _Proton else _np.nan}
        (.pv): Relativistic pv: {self.pv}
        (.beta): Relativistic beta: {self.beta}
        (.gamma): Relativistic gamma: {self.gamma}
        """

    @property
    def particule(self) -> _ParticuleType:
        """Associated particle type."""
        return self._p

    def to(self, quantity: str) -> Union[float, _Q]:
        """

        Args:
            quantity:

        Returns:

        """
        if self._type == quantity:
            return self._q
        c = f"{self._type}_to_{quantity}"
        try:
            return globals()[c](self._q, particle=self._p)
        except KeyError:
            raise KinematicsException(f"Invalid conversion attempted: {c}.")

    def to_range(self, magnitude: bool = False) -> Union[float, _Q]:
        """

        Args:
            magnitude:

        Returns:

        """
        if self._p != _Proton:
            return 0.0
        _ = self.to("range").to("cm")  # type: ignore[union-attr]
        if magnitude:
            return _.magnitude
        return _

    range = property(to_range)
    """TODO"""

    range_ = property(_partial(to_range, magnitude=True))
    """TODO"""

    def to_ekin(self, magnitude: bool = False) -> Union[float, _Q]:
        """

        Args:
            magnitude:

        Returns:

        """
        _ = self.to("ekin").to("MeV")  # type: ignore[union-attr]
        if magnitude:
            return _.magnitude
        return _

    ekin = property(to_ekin)
    """TODO"""

    ekin_ = property(_partial(to_ekin, magnitude=True))
    """TODO"""

    def to_etot(self, magnitude: bool = False) -> Union[float, _Q]:
        """

        Args:
            magnitude:

        Returns:

        """
        _ = self.to("etot").to("MeV")  # type: ignore[union-attr]
        if magnitude:
            return _.magnitude
        return _

    etot = property(to_etot)
    """TODO"""

    etot_ = property(_partial(to_etot, magnitude=True))
    """TODO"""

    def to_momentum(self, magnitude: bool = False) -> Union[float, _Q]:
        """

        Args:
            magnitude:

        Returns:

        """
        _ = self.to("momentum").to("MeV_c")  # type: ignore[union-attr]
        if magnitude:
            return _.magnitude
        return _

    momentum = property(to_momentum)
    """Provides the *momentum*."""

    momentum_ = property(_partial(to_momentum, magnitude=True))
    """Provides the *momentum* (magnitude only)."""

    def to_brho(self, magnitude: bool = False) -> Union[float, _Q]:
        """

        Args:
            magnitude:

        Returns:

        """
        _ = self.to("brho").to("tesla meter")  # type: ignore[union-attr]
        if magnitude:
            return _.magnitude
        return _

    brho = property(to_brho)
    """Provides *brho*."""

    brho_ = property(_partial(to_brho, magnitude=True))
    """Provides *brho* (magnitude only)."""

    def to_pv(self) -> _Q:
        """

        Returns:

        """
        return self.to("pv")

    pv = property(to_pv)
    """Provides *pv*."""

    def to_beta(self) -> float:
        """

        Returns:

        """
        return self.to("beta")

    beta = property(to_beta)
    """Provides *beta*."""

    def to_gamma(self) -> float:
        """

        Returns:

        """
        return self.to("gamma")

    gamma = property(to_gamma)
    """Provides *gamma*."""


def etot_to_ekin(e: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts total energy to kinetic energy.

    >>> etot_to_ekin(1168 * _ureg.MeV)
    <Quantity(229.72797, 'megaelectronvolt')>

    Args:
        e: Total energy of the particle
        particle: the particle type (default: proton)

    Returns:
         Kinetic energy of the particle
    """
    return e - particle.M * _ureg.c**2


def etot_to_momentum(e: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts total energy to momentum.

    >>> etot_to_momentum(1168 * _ureg.MeV).to('MeV_c')
    <Quantity(695.607359, 'megaelectronvolt_per_c')>

    Args:
        e: Total energy of the particle
        particle: the particle type (default: proton)

    Returns:
         Momentum of the particle
    """
    return ekin_to_momentum(etot_to_ekin(e, particle), particle)


def etot_to_brho(e: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts total energy to magnetic rigidity (brho).

    >>> etot_to_brho(1168 * _ureg.MeV).to('T m')
    <Quantity(2.32029661, 'meter * tesla')>

    Args:
        e: Total energy of the particle
        particle: the particle type (default: proton)

    Returns:
         Magnetic rigidity of the particle
    """
    return ekin_to_brho(etot_to_ekin(e, particle), particle)


def etot_to_range(e: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts total energy to proton range in water; following IEC-60601.

    >>> etot_to_range(1168 * _ureg.MeV).to('cm')
    <Quantity(32.8760931, 'centimeter')>

    Args:
        e: Total energy of the particle
        particle: the particle type (default: proton)

    Returns:
         Proton range in water
    """
    return ekin_to_range(etot_to_ekin(e, particle), particle)


def etot_to_pv(e: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts total energy to relativistic pv.

    >>> etot_to_pv(1168 * _ureg.MeV)
    <Quantity(414.271916, 'megaelectronvolt')>

    Args:
        e: Total energy of the particle
        particle: the particle type (default: proton)

    Returns:
         Relativistic pv of the particle
    """
    return ekin_to_pv(etot_to_ekin(e, particle), particle)


def etot_to_beta(e: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts total energy to relativistic beta.

    >>> etot_to_beta(1168 * _ureg.MeV)
    0.595554245611312

    Args:
        e: Total energy of the particle
        particle: the particle type (default: proton)

    Returns:
         Relativistic beta of the particle
    """
    return ekin_to_beta(etot_to_ekin(e, particle), particle)


def etot_to_gamma(e: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts total energy to relativistic gamma.

    >>> etot_to_beta(1168 * _ureg.MeV)
    0.595554245611312

    Args:
        e: Total energy of the particle
        particle: the particle type (default: proton)

    Returns:
         Relativistic gamma of the particle
    """
    return ekin_to_gamma(etot_to_ekin(e, particle), particle)


def ekin_to_etot(e: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts kinetic energy to total energy

    >>> ekin_to_etot(100 * _ureg.MeV)
    <Quantity(1038.27203, 'megaelectronvolt')>

    Args:
        e: kinetic energy
        particle: the particle type (default: proton)

    Returns:
        Total energy of the particle

    """
    return e + particle.M * _ureg.c**2


def ekin_to_momentum(e: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts kinetic energy to momentum

    >>> ekin_to_momentum(100 * _ureg.MeV).to('MeV_c')
    <Quantity(444.583407, 'megaelectronvolt_per_c')>
    >>> ekin_to_momentum(230 * _ureg.MeV).to('MeV_c')
    <Quantity(696.06403, 'megaelectronvolt_per_c')>

    Args:
        e: kinetic energy
        particle: the particle type (default: proton)

    Returns:
        Momentum of the particle
    """
    return _np.sqrt(ekin_to_etot(e, particle) ** 2 - particle.M**2 * _ureg.c**4) / _ureg.c


def ekin_to_brho(e: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts kinetic energy to magnetic rigidity (brho).

    >>> ekin_to_brho(100 * _ureg.MeV).to('T m')
    <Quantity(1.48297076, 'meter * tesla')>
    >>> ekin_to_brho(230 * _ureg.MeV).to('T m')
    <Quantity(2.3218199, 'meter * tesla')>

    Args:
        e: kinetic energy
        particle: the particle type (default: proton)

    Returns:
        Magnetic rigidity of the particle
    """
    return momentum_to_brho(ekin_to_momentum(e, particle), particle)


def ekin_to_range(e: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts kinetic energy to proton range in water; following IEC-60601.

    >>> ekin_to_range(100 * _ureg.MeV).to('cm')
    <Quantity(7.72696361, 'centimeter')>
    >>> ekin_to_range(230 * _ureg.MeV).to('cm')
    <Quantity(32.9424672, 'centimeter')>

    Args:
        e: kinetic energy
        particle: the particle type (default: proton)

    Returns:
        Proton range in water
    """
    if particle is not _Proton:
        raise KinematicsException("Conversion to range only works for protons.")

    b = 0.008539
    c = 0.5271
    d = 3.4917
    e = e.m_as("MeV")
    return _np.exp((-c + _np.sqrt(c**2 - 4 * b * (d - _np.log(e)))) / (2 * b)) * _ureg.cm


def ekin_to_pv(e: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts kinetic energy to relativistic pv.

    >>> ekin_to_pv(230 * _ureg.MeV)
    <Quantity(414.71945, 'megaelectronvolt')>

    Args:
        e: kinetic energy
        particle: the particle type (default: proton)

    Returns:
        Relativistic pv of the particle
    """
    return ((e + particle.M * _ureg.c**2) ** 2 - (particle.M * _ureg.c**2) ** 2) / (e + particle.M * _ureg.c**2)


def ekin_to_beta(e: _Q, particle: _ParticuleType = _Proton) -> float:
    """
    Converts the kinetic energy to relativistic beta.

    >>> ekin_to_beta(230 * _ureg.MeV)
    0.5958064663732595

    Args:
        e: kinetic energy
        particle: the particle type (default: proton)

    Returns:
         Relativistic beta of the particle
    """
    gamma = (particle.M * _ureg.c**2 + e) / (particle.M * _ureg.c**2)
    return float((_np.sqrt((gamma**2 - 1) / gamma**2)).magnitude)


def ekin_to_gamma(e: _Q, particle: _ParticuleType = _Proton) -> float:
    """
    Converts the kinetic energy to relativistic gamma.

    >>> ekin_to_gamma(230 * _ureg.MeV)
    1.2451314678963625

    Args:
        e: kinetic energy
        particle: the particle type (default: proton)

    Returns:
         Relativistic gamma of the particle
    """
    return float(((particle.M * _ureg.c**2 + e) / (particle.M * _ureg.c**2)).magnitude)


def momentum_to_etot(p: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts momentum to total energy.

    >>> momentum_to_etot(100 * _ureg.MeV_c).to('MeV')
    <Quantity(943.585927, 'megaelectronvolt')>

    Args:
        p: relativistic momentum
        particle: the particle type (default: proton)

    Returns:
         Total energy of the particle
    """
    return _np.sqrt((p**2 * _ureg.c**2) + ((particle.M * _ureg.c**2) ** 2))


def momentum_to_ekin(p: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts momentum to kinetic energy.

    >>> momentum_to_ekin(100 * _ureg.MeV_c).to('MeV')
    <Quantity(5.31389734, 'megaelectronvolt')>

    Args:
        p: relativistic momentum
        particle: the particle type (default: proton)

    Returns:
         Kinetic energy of the particle
    """
    return momentum_to_etot(p, particle) - particle.M * _ureg.c**2


def momentum_to_brho(p: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts momentum to magnetic rigidity (brho).

    >>> momentum_to_brho(100 * _ureg.MeV_c).to('tesla * meter')
    <Quantity(0.333564126, 'meter * tesla')>

    Args:
        p: relativistic momentum
        particle: the particle type (default: proton)

    Returns:
         Magnetic rigidity of the particle
    """
    return p / particle.Q


def momentum_to_range(p: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts momentum to proton range in water; following IEC-60601.

    >>> momentum_to_range(696 * _ureg.MeV_c).to('cm')
    <Quantity(32.9331561, 'centimeter')>

    Args:
        p: relativistic momentum
        particle: the particle type (default: proton)

    Returns:
         Range of the particle
    """
    return ekin_to_range(momentum_to_ekin(p, particle), particle)


def momentum_to_pv(p: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts momentum to relativistic pv.

    >>> momentum_to_pv(696 * _ureg.MeV_c).to('megaelectronvolt')
    <Quantity(414.656695, 'megaelectronvolt')>

    Args:
        p: relativistic momentum
        particle: the particle type (default: proton)

    Returns:
         Relativistic pv of the particle
    """
    return ekin_to_pv(momentum_to_ekin(p, particle), particle)


def momentum_to_beta(p: _Q, particle: _ParticuleType = _Proton) -> float:
    """
    Converts momentum to relativistic beta.

    >>> momentum_to_beta(696 * _ureg.MeV_c)
    0.5957711130629324

    Args:
        p: relativistic momentum
        particle: the particle type (default: proton)

    Returns:
         Relativistic beta of the particle
    """
    return float(ekin_to_beta(momentum_to_ekin(p, particle), particle))


def momentum_to_gamma(p: _Q, particle: _ParticuleType = _Proton) -> float:
    """
    Converts momentum to relativistic gamma.

    >>> momentum_to_gamma(696 * _ureg.MeV_c)
    1.2450908098255749

    Args:
        p: relativistic momentum
        particle: the particle type (default: proton)

    Returns:
         Relativistic gamma of the particle
    """
    return ekin_to_gamma(momentum_to_ekin(p, particle), particle)


def brho_to_etot(brho: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts magnetic rigidity (brho) to total energy.

    >>> brho_to_etot(3 * _ureg.tesla * _ureg.meter).to('MeV')
    <Quantity(1299.70532, 'megaelectronvolt')>

    Args:
        brho: the magnetic rigidity
        particle: the particle type (default: proton)

    Returns:
        the total energy of the particle.
    """
    return momentum_to_etot(brho_to_momentum(brho, particle), particle)


def brho_to_ekin(brho: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts magnetic rigidity (brho) to kinetic energy.

    >>> brho_to_ekin(3 * _ureg.tesla * _ureg.meter).to('MeV')
    <Quantity(361.433288, 'megaelectronvolt')>

    Args:
        brho: the magnetic rigidity
        particle: the particle type (default: proton)

    Returns:
        the kinetic energy of the particle.
    """
    return momentum_to_ekin(brho_to_momentum(brho, particle))


def brho_to_momentum(brho: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts magnetic rigidity (brho) to momentum.

    >>> brho_to_momentum(3 * _ureg.T * _ureg.m).to('MeV/c')
    <Quantity(899.377291, 'megaelectronvolt / speed_of_light')>

    Args:
        brho: the magnetic rigidity
        particle: the particle type (default: proton)

    Returns:
        the momentum of the particle.
    """
    return brho * particle.Q


def brho_to_range(brho: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts magnetic rigidity (brho) to range.

    >>> brho_to_range(3 * _ureg.tesla * _ureg.meter).to('cm')
    <Quantity(70.5706415, 'centimeter')>

    Args:
        brho: the magnetic rigidity
        particle: the particle type (default: proton)

    Returns:
        the range of the particle.
    """
    return momentum_to_range(brho_to_momentum(brho, particle), particle)


def brho_to_pv(brho: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts magnetic rigidity (brho) to relativistic pv.

    >>> brho_to_pv(3 * _ureg.tesla * _ureg.meter).to('MeV')
    <Quantity(622.356084, 'megaelectronvolt')>

    Args:
        brho: the magnetic rigidity
        particle: the particle type (default: proton)

    Returns:
        Relativistic pv of the particle
    """
    return momentum_to_pv(brho_to_momentum(brho, particle), particle)


def brho_to_beta(brho: _Q, particle: _ParticuleType = _Proton) -> float:
    """
    Converts magnetic rigidity (brho) to relativistic beta.

    >>> brho_to_beta(3 * _ureg.tesla * _ureg.meter)
    0.6919855437534259

    Args:
        brho: the magnetic rigidity
        particle: the particle type (default: proton)

    Returns:
        Relativistic beta of the particle
    """
    return momentum_to_beta(brho_to_momentum(brho, particle), particle)


def brho_to_gamma(brho: _Q, particle: _ParticuleType = _Proton) -> float:
    """
    Converts magnetic rigidity (brho) to relativistic gamma.

    >>> brho_to_gamma(3 * _ureg.tesla * _ureg.meter)
    1.385211619719756

    Args:
        brho: the magnetic rigidity
        particle: the particle type (default: proton)

    Returns:
        Relativistic gamma of the particle
    """
    return momentum_to_gamma(brho_to_momentum(brho, particle), particle)


def range_to_etot(r: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Examples:
        >>> range_to_etot(32 * _ureg.cm).to('MeV')
        <Quantity(1164.40114, 'megaelectronvolt')>

    Args:
        r:  proton range in water
        particle: the particle type (default: proton)

    Returns:
        Total energy of the particle.

    """
    return ekin_to_etot(range_to_ekin(r, particle), particle)


def range_to_ekin(r: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts proton range in water to kinetic energy following IEC60601.

    Examples:
        >>> range_to_ekin(32 * _ureg.cm).to('MeV')
        <Quantity(226.129112, 'megaelectronvolt')>

    Args:
        r: proton range in water
        particle: the particle type (default: proton)

    Returns:
        the kinetic energy of the particle.
    """
    if particle != _Proton:
        raise KinematicsException("Conversion from range only works for protons.")

    a = 0.00169
    b = -0.00490
    c = 0.56137
    d = 3.46405
    r = r.to("cm").magnitude
    return _np.exp(a * _np.log(r) ** 3 + b * _np.log(r) ** 2 + c * _np.log(r) + d) * _ureg.MeV


def range_to_momentum(r: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts proton range in water to momentum.

    >>> range_to_momentum(32 * _ureg.cm).to('MeV / c')
    <Quantity(689.5474, 'megaelectronvolt / speed_of_light')>

    Args:
        r: proton range in water
        particle: the particle type (default: proton)

    Returns:
        the momentum of the particle.
    """

    return ekin_to_momentum(range_to_ekin(r, particle), particle)


def range_to_brho(r: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts proton range in water to magnetic rigidity (brho).

    >>> range_to_brho(32 * _ureg.cm).to('T m')
    <Quantity(2.30008276, 'meter * tesla')>

    Args:
        r: proton range in water
        particle: the particle type (default: proton)

    Returns:
        The magnetic rigidity (brho) of the particle.
    """
    return ekin_to_brho(range_to_ekin(r, particle), particle)


def range_to_pv(r: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts proton range in water to relativistic pv.

    >>> range_to_pv(32 * _ureg.cm).to('MeV')
    <Quantity(408.343482, 'megaelectronvolt')>

    Args:
        r: proton range in water
        particle: the particle type (default: proton)

    Returns:
        the relativistic pv of the particle.
    """

    return ekin_to_pv(range_to_ekin(r, particle), particle)


def range_to_beta(r: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts proton range in water to relativistic beta.

    >>> range_to_beta(32 * _ureg.cm)
    0.5921905906552516

    Args:
        r: proton range in water
        particle: the particle type (default: proton)

    Returns:
        Relativistic beta of the particle.
    """

    return ekin_to_beta(range_to_ekin(r, particle), particle)


def range_to_gamma(r: _Q, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts proton range in water to relativistic beta.

    >>> range_to_gamma(32 * _ureg.cm)
    1.2410059178641932

    Args:
        r: proton range in water
        particle: the particle type (default: proton)

    Returns:
        Relativistic beta of the particle.
    """

    return ekin_to_gamma(range_to_ekin(r, particle), particle)


def beta_to_etot(beta: float, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts relativistic beta to total energy.

    >>> beta_to_etot(0.3).to('MeV')
    <Quantity(983.576342, 'megaelectronvolt')>
    >>> beta_to_etot(0.9).to('MeV')
    <Quantity(2152.54366, 'megaelectronvolt')>

    Args:
        beta: relativistic beta
        particle: the particle type (default: proton)

    Returns:
        Total energy of the particle
    """
    return beta_to_gamma(beta) * (particle.M * _ureg.c**2)


def beta_to_ekin(beta: float, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts relativistic beta to kinetic energy.

    >>> beta_to_ekin(0.3).to('MeV')
    <Quantity(45.3043118, 'megaelectronvolt')>
    >>> beta_to_ekin(0.9).to('MeV')
    <Quantity(1214.27163, 'megaelectronvolt')>

    Args:
        beta: relativistic beta
        particle: the particle type (default: proton)

    Returns:
        Kinetic energy of the particle
    """
    return (beta_to_gamma(beta) - 1) * (particle.M * _ureg.c**2)


def beta_to_momentum(beta: float, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts relativistic beta to momentum.

    >>> beta_to_momentum(0.3).to('MeV/c')
    <Quantity(295.072903, 'megaelectronvolt / speed_of_light')>
    >>> beta_to_momentum(0.9).to('MeV/c')
    <Quantity(1937.2893, 'megaelectronvolt / speed_of_light')>

    Args:
        beta: relativistic beta
        particle: the particle type (default: proton)

    Returns:
        Momentum of the particle
    """
    return ekin_to_momentum(beta_to_ekin(beta), particle=particle)


def beta_to_brho(beta: float, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts relativistic beta to magnetic rigidity.

    >>> beta_to_brho(0.3).to('T m')
    <Quantity(0.984257348, 'meter * tesla')>
    >>> beta_to_brho(0.9).to('T m')
    <Quantity(6.46210211, 'meter * tesla')>

    Args:
        beta: relativistic beta
        particle: the particle type (default: proton)

    Returns:
        Magnetic rigidity of the particle
    """
    return ekin_to_brho(beta_to_ekin(beta), particle=particle)


def beta_to_range(beta: float, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts relativistic beta to range.

    >>> beta_to_range(0.3).to('cm')
    <Quantity(1.83016632, 'centimeter')>
    >>> beta_to_range(0.9).to('cm')
    <Quantity(503.718206, 'centimeter')>

    Args:
        beta: relativistic beta
        particle: the particle type (default: proton)

    Returns:
        Range of the particle
    """
    return ekin_to_range(beta_to_ekin(beta), particle=particle)


def beta_to_pv(beta: float, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts relativistic beta to relativistic pv.

    >>> beta_to_pv(0.15).to('MeV')
    <Quantity(21.3527053, 'megaelectronvolt')>
    >>> beta_to_pv(0.9).to('MeV')
    <Quantity(1743.56037, 'megaelectronvolt')>

    Args:
        beta: relativistic beta
        particle: the particle type (default: proton)

    Returns:
        Relativistic pv of the particle
    """
    return ekin_to_pv(beta_to_ekin(beta), particle=particle)


def beta_to_gamma(beta: float, **_) -> float:  # type: ignore[no-untyped-def]
    """
    Converts relativistic beta to relativistic gamma.

    >>> beta_to_gamma(0.5)
    1.1547005383792517
    >>> beta_to_gamma(0.9)
    2.294157338705618

    Args:
        beta: relativistic beta

    Returns:
        Relativistic gamma of the particle
    """
    return 1 / (_np.sqrt(1 - beta**2))  # type: ignore[no-any-return]


def gamma_to_etot(gamma: float, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts relativistic gamma to total energy.

    >>> gamma_to_etot(1.5).to('MeV')
    <Quantity(1407.40804, 'megaelectronvolt')>
    >>> gamma_to_etot(1.2).to("MeV")
    <Quantity(1125.92644, 'megaelectronvolt')>

    Args:
        gamma: relativistic gamma
        particle: the particle type (default: proton)

    Returns:
        Total energy of the particle
    """
    return gamma_to_ekin(gamma, particle) + particle.M * _ureg.c**2


def gamma_to_ekin(gamma: float, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts relativistic gamma to kinetic energy.

    >>> gamma_to_ekin(100.0).to('MeV')
    <Quantity(92888.931, 'megaelectronvolt')>

    Args:
        gamma: relativistic gamma
        particle: the particle type (default: proton)

    Returns:
        Kinetic energy of the particle
    """
    return (gamma - 1) * (particle.M * _ureg.c**2)


def gamma_to_momentum(gamma: float, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts relativistic gamma to momentum.

    >>> gamma_to_momentum(100.0).to('MeV/c')
    <Quantity(93822.5115, 'megaelectronvolt / speed_of_light')>

    Args:
        gamma: relativistic gamma
        particle: the particle type (default: proton)

    Returns:
        Momentum of the particle
    """
    return ekin_to_momentum(gamma_to_ekin(gamma, particle), particle)


def gamma_to_brho(gamma: float, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts relativistic gamma to magnetic rigidity.

    >>> gamma_to_brho(100.0).to('T m')
    <Quantity(312.95824, 'meter * tesla')>

    Args:
        gamma: relativistic gamma
        particle: the particle type (default: proton)

    Returns:
        Magnetic rigidity of the particle
    """
    return ekin_to_brho(gamma_to_ekin(gamma, particle), particle)


def gamma_to_range(gamma: float, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts relativistic gamma to range (protons only).

    >>> gamma_to_range(100.0)
    <Quantity(277356.143, 'centimeter')>

    Args:
        gamma: relativistic gamma
        particle: the particle type (default: proton)

    Returns:
        Range of the particle
    """
    return ekin_to_range(gamma_to_ekin(gamma, particle), particle)


def gamma_to_pv(gamma: float, particle: _ParticuleType = _Proton) -> _Q:
    """
    Converts relativistic gamma to relativistic pv.

    >>> gamma_to_pv(100.0)
    <Quantity(93817.8203, 'megaelectronvolt_per_c2 * speed_of_light ** 2')>

    Args:
        gamma: relativistic gamma
        particle: the particle type (default: proton)

    Returns:
        Relativistic pv of the particle
    """
    return ekin_to_pv(gamma_to_ekin(gamma, particle), particle)


def gamma_to_beta(gamma: float, **_: Any) -> float:
    """
    Converts relativistic gamma to relativistic beta.

    >>> gamma_to_beta(1.5)
    0.7453559924999299

    Args:
        gamma: relativistic gamma

    Returns:
        Relativistic beta of the particle
    """
    return _np.sqrt((gamma**2 - 1) / gamma**2)  # type: ignore[no-any-return]
