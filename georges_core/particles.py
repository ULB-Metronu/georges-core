"""TODO

"""
from . import ureg as _ureg
import scipy.constants


class ParticuleType(type):
    pass


class Particule(metaclass=ParticuleType):
    """Particle characteristics."""

    M = None
    Q = None
    G = None
    tau = None

    @property
    def mass(self):
        """Mass of the particle."""
        return self.M

    @property
    def charge(self):
        """Charge of the particle."""
        return self.Q

    @property
    def lifetime(self):
        """Lifetime constant of the particle."""
        return self.tau

    @property
    def gyro(self):
        """Gyromagnetic factor of the particle."""
        return self.G

    @property
    def name(self):
        """Gyromagnetic factor of the particle."""
        return self.name


class Electron(Particule):
    """An electron."""
    M = 0.5109989461 * _ureg.MeV_c2
    Q = -1.6021766208e-19 * _ureg.coulomb
    G = (-2.0023193043622 - 2) / 2
    name = 'e-'


class Proton(Particule):
    """A proton."""
    M = 938.27203 * _ureg.MeV_c2
    Q = 1.602176487e-19 * _ureg.coulomb
    G = (5.585694701 - 2) / 2
    name = 'Proton'


class AntiMuon(Particule):
    """An anti-muon particle."""
    M =  scipy.constants.physical_constants['muon mass'][0] * _ureg.kg
    Q = scipy.constants.elementary_charge * _ureg.coulomb
    G = (scipy.constants.physical_constants['muon g factor'][0] - 2) / 2
    tau = 2.197029e-6 * _ureg.s


Posmuon = AntiMuon
