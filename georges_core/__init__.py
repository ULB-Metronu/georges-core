__version__ = "2019.1"


# Manipulation of physical quantities (with units, etc.)
# https://pint.readthedocs.io/en/latest/
from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('electronvolt = e * volt = eV')
ureg.define('electronvolt_per_c = eV / c = eV_c')
ureg.define('electronvolt_per_c2 = eV / c**2 = eV_c2')

from .kinematics import Kinematics, ZgoubiKinematicsException
