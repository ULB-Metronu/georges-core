import os

import pytest
import numpy as _np
import numpy.testing
import pandas as _pd
from georges_core import ureg
from georges_core.distribution import Distribution, DistributionException


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
                             (0.0 * ureg.cm, 5.0, 0.0 * ureg.m, 0.0, 0.0,
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

    _np.testing.assert_equal(1e7, beam_distribution.n_particles)
    _np.testing.assert_equal(6, beam_distribution.dims)
    _np.testing.assert_allclose(beam_distribution.compute_twiss(beam_distribution.distribution.values),
                                _np.array([emitx.m_as('m radians'), betax.m_as('m'), alphax, dispx.m_as('m'), dispxp,
                                           emity.m_as('m radians'), betay.m_as('m'), alphay, dispy.m_as('m'), dispyp]),
                                atol=5e-3
                                )
    _np.testing.assert_allclose(x.m_as('m'), mean_distribution['X'], atol=1e-4)
    _np.testing.assert_allclose(y.m_as('m'), mean_distribution['Y'], atol=1e-4)
    _np.testing.assert_allclose(px, mean_distribution['PX'], atol=1e-4)
    _np.testing.assert_allclose(py, mean_distribution['PY'], atol=1e-4)
    _np.testing.assert_allclose(dpp, mean_distribution['DPP'], atol=1e-4)
    _np.testing.assert_allclose(betax.m_as('m'), twiss_distribution['beta_x'], atol=5e-3)
    _np.testing.assert_allclose(alphax, twiss_distribution['alpha_x'], atol=5e-3)
    _np.testing.assert_allclose(betay.m_as('m'), twiss_distribution['beta_y'], atol=5e-3)
    _np.testing.assert_allclose(alphay, twiss_distribution['alpha_y'], atol=5e-3)
    _np.testing.assert_allclose(emitx.m_as('m radians'), emit_distribution['X'], atol=5e-3)
    _np.testing.assert_allclose(emity.m_as('m radians'), emit_distribution['Y'], atol=5e-3)


@pytest.mark.parametrize("x, px, y, py, dpp,"
                         "xrms, pxrms, yrms, pyrms,dpprms", [
                             (0.0 * ureg.m, 0.0, 0.0 * ureg.m, 0.0, 0.0,
                              0.1 * ureg.cm, 0.0, 0.5 * ureg.cm, 0.0, 0.0),
                             (0.1 * ureg.m, 0.0, 0.0 * ureg.m, 0.0, 0.0,
                              0.0 * ureg.m, 0.001, 0.003 * ureg.m, 0.001, 0.001),
                             (0 * ureg.m, 5, 0.0 * ureg.m, 0.0, 0.0,
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

    _np.testing.assert_equal(1e7, beam_distribution.n_particles)
    _np.testing.assert_equal(6, beam_distribution.dims)
    _np.testing.assert_allclose(x.m_as('m'), mean_distribution['X'], atol=1e-4)
    _np.testing.assert_allclose(y.m_as('m'), mean_distribution['Y'], atol=1e-4)
    _np.testing.assert_allclose(px, mean_distribution['PX'], atol=1e-4)
    _np.testing.assert_allclose(py, mean_distribution['PY'], atol=1e-4)
    _np.testing.assert_allclose(dpp, mean_distribution['DPP'], atol=1e-4)
    _np.testing.assert_allclose(xrms.m_as('m'), std_distribution['X'], atol=5e-3)
    _np.testing.assert_allclose(pxrms, std_distribution['PX'], atol=5e-3)
    _np.testing.assert_allclose(yrms.m_as('m'), std_distribution['Y'], atol=5e-3)
    _np.testing.assert_allclose(pyrms, std_distribution['PY'], atol=5e-3)
    _np.testing.assert_allclose(dpprms, std_distribution['DPP'], atol=5e-3)


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

    _np.testing.assert_equal(1e7, beam_distribution.n_particles)
    _np.testing.assert_equal(6, beam_distribution.dims)
    _np.testing.assert_allclose(x.m_as('m'), mean_distribution['X'], atol=5e-3)
    _np.testing.assert_allclose(y.m_as('m'), mean_distribution['Y'], atol=5e-3)
    _np.testing.assert_allclose(px, mean_distribution['PX'], atol=5e-3)
    _np.testing.assert_allclose(py, mean_distribution['PY'], atol=5e-3)
    _np.testing.assert_allclose(dpp, mean_distribution['DPP'], atol=5e-3)
    _np.testing.assert_allclose(s11.m_as('m**2'), sigma_distribution['X']['X'], atol=5e-3)
    _np.testing.assert_allclose(s12, sigma_distribution['X']['PX'], atol=5e-3)
    _np.testing.assert_allclose(s13, sigma_distribution['X']['Y'], atol=5e-3)
    _np.testing.assert_allclose(s14, sigma_distribution['X']['PY'], atol=5e-3)
    _np.testing.assert_allclose(s15, sigma_distribution['X']['DPP'], atol=5e-3)
    _np.testing.assert_allclose(s22, sigma_distribution['PX']['PX'], atol=5e-3)
    _np.testing.assert_allclose(s23, sigma_distribution['PX']['Y'], atol=5e-3)
    _np.testing.assert_allclose(s24, sigma_distribution['PX']['PY'], atol=5e-3)
    _np.testing.assert_allclose(s25, sigma_distribution['PX']['DPP'], atol=5e-3)
    _np.testing.assert_allclose(s33.m_as('m**2'), sigma_distribution['Y']['Y'], atol=5e-3)
    _np.testing.assert_allclose(s34, sigma_distribution['Y']['PY'], atol=5e-3)
    _np.testing.assert_allclose(s35, sigma_distribution['Y']['DPP'], atol=5e-3)
    _np.testing.assert_allclose(s44, sigma_distribution['PY']['PY'], atol=5e-3)
    _np.testing.assert_allclose(s45, sigma_distribution['PY']['DPP'], atol=5e-3)
    _np.testing.assert_allclose(s55, sigma_distribution['DPP']['DPP'], atol=5e-3)


@pytest.mark.parametrize("x, px, y, py, dpp,"
                         "matrix", [
                             (0.0 * ureg.m, 0.0, 0.0 * ureg.m, 0.0, 0.0,
                              _np.random.rand(5, 5)),
                             (0.01 * ureg.m, 0.0, 0.0 * ureg.m, 0.0, 0.0,
                              _np.random.rand(5, 5))
                         ])
def test_from_matrix(x, px, y, py, dpp, matrix):
    matrix = _np.dot(matrix, matrix.transpose())
    beam_distribution = Distribution.from_5d_sigma_matrix(n=int(1e7),
                                                          x=x,
                                                          px=px,
                                                          y=y,
                                                          py=py,
                                                          dpp=dpp,
                                                          matrix=matrix)

    mean_distribution = beam_distribution.mean
    sigma_matrix = beam_distribution.sigma.drop(labels='DT', axis=0).drop(labels='DT', axis=1).values

    _np.testing.assert_equal(1e7, beam_distribution.n_particles)
    _np.testing.assert_equal(6, beam_distribution.dims)
    _np.testing.assert_allclose(x.m_as('m'), mean_distribution['X'], atol=5e-3)
    _np.testing.assert_allclose(y.m_as('m'), mean_distribution['Y'], atol=5e-3)
    _np.testing.assert_allclose(px, mean_distribution['PX'], atol=5e-3)
    _np.testing.assert_allclose(py, mean_distribution['PY'], atol=5e-3)
    _np.testing.assert_allclose(dpp, mean_distribution['DPP'], atol=5e-3)
    _np.testing.assert_allclose(sigma_matrix, matrix, atol=1e-2)


def test_from_file():
    # generate a distribution and save it to csv and parquet file
    distrib = Distribution.from_5d_multigaussian_distribution(n=100,
                                                              xrms=0.1 * ureg.m,
                                                              yrms=0.1 * ureg.m)
    distrib.distribution.to_csv("./beam.csv", index=None)
    distrib.distribution.to_parquet("./beam.tar.gz", index=None, compression='gzip')

    distrib_csv = Distribution.from_csv(path='.', filename='beam.csv')
    distrib_parquet = Distribution.from_parquet(path='.', filename='beam.tar.gz')
    os.remove('beam.csv')
    os.remove('beam.tar.gz')
    _np.testing.assert_equal(100, distrib_csv.n_particles)
    _np.testing.assert_equal(6, distrib_csv.dims)
    _pd.testing.assert_frame_equal(distrib.distribution, distrib_csv.distribution, check_dtype=False)
    _np.testing.assert_equal(100, distrib_parquet.n_particles)
    _np.testing.assert_equal(6, distrib_parquet.dims)
    _pd.testing.assert_frame_equal(distrib.distribution, distrib_parquet.distribution, check_dtype=False)


def test_halo():
    beam_distribution = Distribution.from_5d_multigaussian_distribution(n=int(1e7),
                                                                        xrms=0.01 * ureg.m,
                                                                        pxrms=0.01,
                                                                        yrms=0.01 * ureg.m,
                                                                        pyrms=0.01)

    halo_distribution = beam_distribution.halo

    _np.testing.assert_allclose(halo_distribution.loc['X', :].values,
                                _np.percentile(beam_distribution.distribution['X'].values, [1, 5, 20, 80, 95, 99]),
                                atol=5e-3)
    _np.testing.assert_allclose(halo_distribution.loc['Y', :].values,
                                _np.percentile(beam_distribution.distribution['Y'].values, [1, 5, 20, 80, 95, 99]),
                                atol=5e-3)
    _np.testing.assert_allclose(halo_distribution.loc['PX', :].values,
                                _np.percentile(beam_distribution.distribution['PX'].values, [1, 5, 20, 80, 95, 99]),
                                atol=5e-3)
    _np.testing.assert_allclose(halo_distribution.loc['PY', :].values,
                                _np.percentile(beam_distribution.distribution['PY'].values, [1, 5, 20, 80, 95, 99]),
                                atol=5e-3)


def test_exceptions():
    # distribution is None
    dist = Distribution()
    _np.testing.assert_equal(dist.distribution, _np.zeros((1, 6)))

    # Missing columns
    beam = _pd.DataFrame({'X': [1], 'Y': [2]})
    dist = Distribution(distribution=beam)
    _np.testing.assert_equal(dist.distribution, _np.zeros((1, 6)))

    # Acces to invalid date
    distrib = Distribution.from_5d_multigaussian_distribution(n=100,
                                                              xrms=0.1 * ureg.m,
                                                              yrms=0.1 * ureg.m)
    _ = distrib['X']
    with pytest.raises(DistributionException):
        _ = distrib['U']

    # No particles
    with pytest.raises(DistributionException):
        _ = Distribution(distribution=_pd.DataFrame(columns=['X', 'PX', 'Y', 'PY']))
