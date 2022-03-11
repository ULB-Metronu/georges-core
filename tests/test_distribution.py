import pytest
from georges_core import ureg
from georges_core.distribution import Distribution


@pytest.mark.parametrize("x, px, y, py, dpp, "
                         "betax, alphax, betay, alphay,"
                         "emitx, emity, dispx, dispy, "
                         "dispxp, dispyp, dpprms", [
                             (0.0 * ureg.m, 0.0, 0.0 * ureg.m, 0.0, 0.0,
                              1.0 * ureg.m, 0.0, 1.0 * ureg.m, 0.0,
                              1e-6 * ureg.m, 1e-6 * ureg.m,
                              0.0 * ureg.m, 0.0 * ureg.m, 0.0, 0.0, 0.0),
                             (1.0 * ureg.cm, 0.0, 0.0 * ureg.m, 0.0, 0.0,
                              1.0 * ureg.m, 0.0, 1.0 * ureg.m, 0.0,
                              1e-6 * ureg.m, 1e-6 * ureg.m,
                              0.0 * ureg.m, 0.0 * ureg.m, 0.0, 0.0, 0.0),
                             (0.0 * ureg.cm, 5.0 * ureg.milliradians, 0.0 * ureg.m, 0.0, 0.0,
                              1.0 * ureg.m, 0.0, 1.0 * ureg.m, 0.0,
                              1e-6 * ureg.m, 1e-6 * ureg.m,
                              0.0 * ureg.m, 0.0 * ureg.m, 0.0, 0.0, 0.0),
                             (0.0 * ureg.cm, 0.0, 0.0 * ureg.m, 0.0, 0.02,
                              1.0 * ureg.m, 0.0, 1.0 * ureg.m, 0.0,
                              1e-6 * ureg.m, 1e-6 * ureg.m,
                              0.0 * ureg.m, 0.0 * ureg.m, 0.0, 0.0, 0.0),
                         ])
def test_from_twiss(x, px, y, py, dpp, betax, alphax, betay, alphay,
                    emitx, emity, dispx, dispy, dispxp, dispyp, dpprms):
    beam_distribution = Distribution.from_twiss_parameters(n=int(1e7),
                                                           x=x,
                                                           px=px,
                                                           y=y,
                                                           py=py,
                                                           dpp=dpp,
                                                           betax=betax,
                                                           alphax=alphax,
                                                           betay=betay,
                                                           alphay=alphay,
                                                           emitx=emitx,
                                                           emity=emity,
                                                           dispx=dispx,
                                                           dispy=dispy,
                                                           dispxp=dispxp,
                                                           dispyp=dispyp,
                                                           dpprms=dpprms)

    mean_distribution = beam_distribution.mean
    twiss_distribution = beam_distribution.twiss
    emit_distribution = beam_distribution.emit
    print(mean_distribution, x)
    assert x.m_as('m') == pytest.approx(mean_distribution['X'], abs=1e-4)
    assert y.m_as('m') == pytest.approx(mean_distribution['Y'], abs=1e-4)
    assert px == pytest.approx(mean_distribution['PX'], abs=1e-4)
    assert py == pytest.approx(mean_distribution['PY'], abs=1e-4)
    assert dpp == pytest.approx(mean_distribution['DPP'], abs=1e-4)
    assert betax.m_as('m') == pytest.approx(twiss_distribution['beta_x'], abs=1e-3)
    assert alphax == pytest.approx(twiss_distribution['alpha_x'], abs=1e-3)
    assert betay.m_as('m') == pytest.approx(twiss_distribution['beta_y'], abs=1e-3)
    assert alphay == pytest.approx(twiss_distribution['alpha_y'], abs=1e-3)
    assert emitx.m_as('m radians') == pytest.approx(emit_distribution['X'], abs=1e-3)
    assert emity.m_as('m radians') == pytest.approx(emit_distribution['Y'], abs=1e-3)


@pytest.mark.parametrize("x, px, y, py, dpp,"
                         "xrms, pxrms, yrms, pyrms,dpprms", [
                             (0.0 * ureg.m, 0.0, 0.0 * ureg.m, 0.0, 0.0,
                              0.1 * ureg.cm, 0.0, 0.5 * ureg.cm, 0.0, 0.0),
                             (0.1 * ureg.m, 0.0, 0.0 * ureg.m, 0.0, 0.0,
                              0.0 * ureg.m, 0.001, 0.003 * ureg.m, 0.001, 0.001),
                             (0 * ureg.m, 5 * ureg.milliradians, 0.0 * ureg.m, 0.0, 0.0,
                              0.6 * ureg.cm, 0.1, 0.0 * ureg.m, 0.0, 0.01),
                         ])
def test_from_multigaussian_distribution(x, px, y, py, dpp, xrms, pxrms, yrms, pyrms, dpprms):
    beam_distribution = Distribution.from_5d_multigaussian_distribution(n=int(1e7),
                                                                        x=x,
                                                                        px=px,
                                                                        y=y,
                                                                        py=py,
                                                                        dpp=dpp,
                                                                        xrms=xrms,
                                                                        pxrms=pxrms,
                                                                        yrms=yrms,
                                                                        pyrms=pyrms,
                                                                        dpprms=dpprms)

    mean_distribution = beam_distribution.mean
    std_distribution = beam_distribution.std

    assert x.m_as('m') == pytest.approx(mean_distribution['X'], abs=1e-4)
    assert y.m_as('m') == pytest.approx(mean_distribution['Y'], abs=1e-4)
    assert px == pytest.approx(mean_distribution['PX'], abs=1e-4)
    assert py == pytest.approx(mean_distribution['PY'], abs=1e-4)
    assert dpp == pytest.approx(mean_distribution['DPP'], abs=1e-4)
    assert xrms.m_as('m') == pytest.approx(std_distribution['X'], abs=1e-3)
    assert pxrms == pytest.approx(std_distribution['PX'], abs=1e-3)
    assert yrms.m_as('m') == pytest.approx(std_distribution['Y'], abs=1e-3)
    assert pyrms == pytest.approx(std_distribution['PY'], abs=1e-3)
    assert dpprms == pytest.approx(std_distribution['DPP'], abs=1e-3)


@pytest.mark.parametrize("x, px, y, py, dpp,"
                         "s11, s12, s13, s14, s15, s22, s23, s24, s25, s33, s34, s35, s44, s45, s55", [
                             (0.0 * ureg.m, 0.0, 0.0 * ureg.m, 0.0, 0.0,
                              0.1 * ureg.cm ** 2, 0.0, 0.0, 0.0, 0.0, 0.2 ** 2, 0.0, 0.0, 0.0,
                              0.2 * ureg.cm ** 2, 0.0, 0.0, 0.2 ** 2, 0.0, 0.01),
                             (0.0 * ureg.m, 0.0, 0.0 * ureg.m, 0.0, 0.0,
                              10 * ureg.cm ** 2, 0.001, 0.001, 0.0, 0.001, 0.2 ** 2, 0.001, 0.001, 0.001,
                              20 * ureg.cm ** 2, 0.001, 0.001, 0.2 ** 2, 0.001, 0.01),
                         ])
def test_from_sigma_matrix(x, px, y, py, dpp,
                           s11, s12, s13, s14, s15, s22, s23, s24, s25, s33, s34, s35, s44, s45, s55):
    beam_distribution = Distribution.from_5d_sigma_matrix(n=int(1e7),
                                                          x=x,
                                                          px=px,
                                                          y=y,
                                                          py=py,
                                                          dpp=dpp,
                                                          s11=s11,
                                                          s12=s12,
                                                          s13=s13,
                                                          s14=s14,
                                                          s15=s15,
                                                          s22=s22,
                                                          s23=s23,
                                                          s24=s24,
                                                          s25=s25,
                                                          s33=s33,
                                                          s34=s34,
                                                          s35=s35,
                                                          s44=s44,
                                                          s45=s45,
                                                          s55=s55)

    mean_distribution = beam_distribution.mean
    sigma_distribution = beam_distribution.sigma

    assert x.m_as('m') == pytest.approx(mean_distribution['X'], abs=1e-3)
    assert y.m_as('m') == pytest.approx(mean_distribution['Y'], abs=1e-3)
    assert px == pytest.approx(mean_distribution['PX'], abs=1e-3)
    assert py == pytest.approx(mean_distribution['PY'], abs=1e-3)
    assert dpp == pytest.approx(mean_distribution['DPP'], abs=1e-3)
    assert s11.m_as('m**2') == pytest.approx(sigma_distribution['X']['X'], abs=1e-3)
    assert s12 == pytest.approx(sigma_distribution['X']['PX'], abs=1e-3)
    assert s13 == pytest.approx(sigma_distribution['X']['Y'], abs=1e-3)
    assert s14 == pytest.approx(sigma_distribution['X']['PY'], abs=1e-3)
    assert s15 == pytest.approx(sigma_distribution['X']['DPP'], abs=1e-3)
    assert s22 == pytest.approx(sigma_distribution['PX']['PX'], abs=1e-3)
    assert s23 == pytest.approx(sigma_distribution['PX']['Y'], abs=1e-3)
    assert s24 == pytest.approx(sigma_distribution['PX']['PY'], abs=1e-3)
    assert s25 == pytest.approx(sigma_distribution['PX']['DPP'], abs=1e-3)
    assert s33.m_as('m**2') == pytest.approx(sigma_distribution['Y']['Y'], abs=1e-3)
    assert s34 == pytest.approx(sigma_distribution['Y']['PY'], abs=1e-3)
    assert s35 == pytest.approx(sigma_distribution['Y']['DPP'], abs=1e-3)
    assert s44 == pytest.approx(sigma_distribution['PY']['PY'], abs=1e-3)
    assert s45 == pytest.approx(sigma_distribution['PY']['DPP'], abs=1e-3)
    assert s55 == pytest.approx(sigma_distribution['DPP']['DPP'], abs=1e-3)
