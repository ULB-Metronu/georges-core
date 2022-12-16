import numpy as _np
import numpy.testing
import pytest

from georges_core import Q_ as _Q
from georges_core import ureg
from georges_core.kinematics import Kinematics, KinematicsException, ekin_to_range, range_to_ekin
from georges_core.particles import Electron as _Electron
from georges_core.particles import ParticuleType as _ParticuleType
from georges_core.particles import Proton as _Proton


def compute_kinematics(quantity: _Q = None, kinetic: bool = False, particle: _ParticuleType = _Proton):
    return Kinematics(quantity, kinetic=kinetic, particle=particle)


def test_single_proton():
    kin = compute_kinematics(1.0 * ureg.GeV, False, _Proton)
    _np.testing.assert_approx_equal(kin.etot.m_as("GeV"), 1.0, significant=4)
    _np.testing.assert_approx_equal(kin.ekin.m_as("MeV"), 61.72797, significant=4)
    _np.testing.assert_approx_equal(kin.momentum.m_as("GeV_c"), 0.34589824763892485, significant=4)
    _np.testing.assert_approx_equal(kin.brho.m_as("meter * tesla"), 1.153792360043043, significant=4)
    _np.testing.assert_approx_equal(kin.range.m_as("cm"), 3.23758370, significant=4)
    _np.testing.assert_approx_equal(kin.beta, 0.345898, significant=4)
    _np.testing.assert_approx_equal(kin.gamma, 1.0657889, significant=4)
    _np.testing.assert_approx_equal(kin.pv.m_as("GeV"), 0.11964559771967898, significant=4)
    _np.testing.assert_approx_equal(kin.particule.M.m_as("MeV_c2"), 938.27203, significant=4)
    _np.testing.assert_approx_equal(kin.particule.Q.m_as("C"), 1.602176487e-19, significant=4)
    _np.testing.assert_approx_equal(kin.particule.G, (5.585694701 - 2) / 2, significant=4)


def test_single_electron():
    kin = compute_kinematics(1.0 * ureg.GeV, False, _Electron)
    _np.testing.assert_approx_equal(kin.etot_, 1000.0, significant=4)
    _np.testing.assert_approx_equal(kin.ekin_, 999.4890010539, significant=4)
    _np.testing.assert_approx_equal(kin.momentum_, 999.9998694400298, significant=4)
    _np.testing.assert_approx_equal(kin.brho_, -3.335640543961986, significant=4)
    _np.testing.assert_approx_equal(kin.range, 0.0, significant=4)
    _np.testing.assert_approx_equal(kin.beta, 0.99999986944003, significant=4)
    _np.testing.assert_approx_equal(kin.gamma, 1956.9511984948497, significant=4)
    _np.testing.assert_approx_equal(kin.pv.m_as("GeV"), 0.9999997388800771, significant=4)
    _np.testing.assert_approx_equal(kin.particule.M.m_as("MeV_c2"), 0.5109989461, significant=4)
    _np.testing.assert_approx_equal(kin.particule.Q.m_as("C"), -1.602176487e-19, significant=4)
    _np.testing.assert_approx_equal(kin.particule.G, -2.0011596521810997, significant=4)


@pytest.mark.parametrize(
    "quantity, kinetic",
    [
        (1200 * ureg.MeV, False),
        (100 * ureg.MeV, True),
        (100 * ureg.MeV, None),
        (5 * ureg.cm, True),
        (0.4 * ureg.GeV_c, False),
        (2.3 * ureg("T * m"), True),
        (0.5, True),
        (1.14, True),
    ],
)
def test_proton(quantity, kinetic):
    kin = compute_kinematics(quantity=quantity, kinetic=kinetic, particle=_Proton)
    print(kin)
    assert kin.particule == _Proton


@pytest.mark.parametrize(
    "quantity, kinetic",
    [
        (1200 * ureg.MeV, False),
        (100 * ureg.MeV, True),
        (0.4 * ureg.GeV_c, False),
        (2.3 * ureg("T * m"), True),
        (0.5, True),
        (1.14, True),
    ],
)
def test_electron(quantity, kinetic):
    kin = compute_kinematics(quantity=quantity, kinetic=kinetic, particle=_Electron)
    print(kin)
    assert kin.particule == _Electron


def test_exceptions():
    with pytest.raises(KinematicsException):
        _ = Kinematics(230 * ureg.kg)
    _np.testing.assert_approx_equal(Kinematics(230 * ureg.MeV, particle=_Electron).range, 0.0)

    with pytest.raises(KinematicsException):
        _ = ekin_to_range(230 * ureg.MeV, particle=_Electron)
    with pytest.raises(KinematicsException):
        _ = range_to_ekin(30 * ureg.cm, particle=_Electron)
