import pytest
from georges_core import ureg
from georges_core.kinematics import Kinematics, KinematicsException
from georges_core.particles import Proton as _Proton
from georges_core.particles import Electron as _Electron


def test_proton_total_energy():
    kin = Kinematics(1 * ureg.GeV)
    assert kin.etot.m_as('GeV') == pytest.approx(1)
    assert kin.ekin.m_as('MeV') == pytest.approx(61.72797)
    assert kin.momentum.m_as('GeV_c') == pytest.approx(0.34589824763892485)
    assert kin.brho.m_as('meter * tesla') == pytest.approx(1.153792360043043)
    assert kin.range.m_as('cm') == pytest.approx(3.23758370)
    assert kin.beta == pytest.approx(0.345898)
    assert kin.gamma == pytest.approx(1.0657889)


def test_proton_kinetic_energy():
    kin = Kinematics(61.72797 * ureg.MeV)
    assert kin.etot.m_as('GeV') == pytest.approx(1)
    assert kin.ekin.m_as('MeV') == pytest.approx(61.72797)
    assert kin.momentum.m_as('GeV_c') == pytest.approx(0.34589824763892485)
    assert kin.brho.m_as('meter * tesla') == pytest.approx(1.153792360043043)
    assert kin.range.m_as('cm') == pytest.approx(3.23758370)
    assert kin.beta == pytest.approx(0.345898)
    assert kin.gamma == pytest.approx(1.0657889)


@pytest.mark.parametrize("etot, particle", [(1200 * ureg.MeV, _Proton),
                                            (100 * ureg.MeV, _Electron)])
def test_etot(etot, particle):
    k = Kinematics(etot, kinetic=False, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except KinematicsException:
        assert False


@pytest.mark.parametrize("ekin, particle", [(1200 * ureg.MeV, _Proton), (100 * ureg.MeV, _Electron)])
def test_ekin(ekin, particle):
    k = Kinematics(ekin, kinetic=True, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except KinematicsException:
        assert False


@pytest.mark.parametrize("momentum, particle", [(696 * ureg('MeV/c'), _Proton), (1500 * ureg('MeV/c'), _Electron)])
def test_momentum(momentum, particle):
    k = Kinematics(momentum, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except KinematicsException:
        assert False


@pytest.mark.parametrize("brho, particle", [(0.5 * ureg('T m'), _Proton), (3 * ureg('T m'), _Electron)])
def test_brho(brho, particle):
    k = Kinematics(brho, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except KinematicsException:
        assert False


@pytest.mark.parametrize("r, particle", [(0.5 * ureg('m'), _Proton), (3 * ureg('m'), _Electron)])
def test_range(r, particle):
    k = Kinematics(r, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except KinematicsException:
        assert False


@pytest.mark.parametrize("beta, particle", [(0.1 * ureg(''), _Proton), (0.9 * ureg(''), _Electron)])
def test_beta(beta, particle):
    k = Kinematics(beta, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except KinematicsException:
        assert False


@pytest.mark.parametrize("gamma, particle", [(1.1, _Proton), (1.9, _Electron)])
def test_gamma(gamma, particle):
    k = Kinematics(gamma, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except KinematicsException:
        assert False
