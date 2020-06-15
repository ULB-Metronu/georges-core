import pytest
from georges_core import ureg
from georges_core.kinematics import Kinematics, KinematicsException
from georges_core.particles import *


@pytest.mark.parametrize("etot, particle", [(1200 * ureg.MeV, Proton),
                                            (100 * ureg.MeV, Electron)])
def test_etot(etot, particle):
    k = Kinematics(etot, kinetic=False, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except:
        assert False


@pytest.mark.parametrize("ekin, particle", [(1200 * ureg.MeV, Proton), (100 * ureg.MeV, Electron)])
def test_ekin(ekin, particle):
    k = Kinematics(ekin, kinetic=True, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except:
        assert False


@pytest.mark.parametrize("momentum, particle", [(696 * ureg('MeV/c'), Proton), (1500 * ureg('MeV/c'), Electron)])
def test_momentum(momentum, particle):
    k = Kinematics(momentum, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except:
        assert False
        

@pytest.mark.parametrize("brho, particle", [(0.5 * ureg('T m'), Proton), (3 * ureg('T m'), Electron)])
def test_brho(brho, particle):
    k = Kinematics(brho, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except:
        assert False


@pytest.mark.parametrize("r, particle", [(0.5 * ureg('m'), Proton), (3 * ureg('m'), Electron)])
def test_range(r, particle):
    k = Kinematics(r, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except:
        assert False


@pytest.mark.parametrize("beta, particle", [(0.1 * ureg(''), Proton), (0.9 * ureg(''), Electron)])
def test_beta(beta, particle):
    k = Kinematics(beta, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except:
        assert False


@pytest.mark.parametrize("gamma, particle", [(1.1, Proton), (1.9, Electron)])
def test_gamma(gamma, particle):
    k = Kinematics(gamma, particle=particle)

    try:
        [k.etot, k.ekin, k.momentum, k.beta, k.brho, k.gamma, k.range]
    except:
        assert False