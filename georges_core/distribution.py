import pandas as pd
import numpy as np
import os
from .units import ureg as _ureg

PARTICLE_TYPES = {'proton', 'antiproton', 'electron', 'positron'}
PHASE_SPACE_DIMENSIONS = ['X', 'PX', 'Y', 'PY', 'DPP', 'DT']
DEFAULT_N_PARTICLES = 1e5


class DistributionException(Exception):
    """Exception raised for errors in the Beam module."""

    def __init__(self, m):
        self.message = m


class Distribution:
    """Particle beam to be tracked in a beamline or accelerator model.

    The internal representation is essentially a `pandas` `DataFrame`.
    """

    def __init__(self, distribution=None, *args, **kwargs):
        """
        Initialize a beam object from various sources of particle beam distribution.
        :param distribution: distribution of particles to initialize the beam with. Should be pandas.DataFrame() friendly.
        :param particle: the particle type (default: 'proton', must be 'proton', 'antiproton', 'electron' or 'positron').
        :param energy: the reference energy of the beam
        :param args: optional parameters.
        :param kwargs: optional keyword parameters.
        """
        try:
            self.__initialize_distribution(distribution, *args, **kwargs)
        except DistributionException:
            self.__dims = 5
            self.__distribution = pd.DataFrame(np.zeros((1, 5)))
            self.__distribution.columns = PHASE_SPACE_DIMENSIONS[:self.__dims]
        self._halo = None  # To force the first computation of the halo

    @property
    def distribution(self):
        """Return a dataframe containing the beam's particles distribution."""
        return self.__distribution

    @property
    def dims(self):
        """Return the dimensions of the beam's phase-space."""
        return self.__dims

    @property
    def n_particles(self):
        """Return the number of particles in the beam's distribution."""
        return self.__n_particles

    @property
    def mean(self):
        """Return a dataframe containing the first order moments of each dimensions."""
        return self.__distribution.mean()

    @property
    def std(self):
        """Return a dataframe containing the second order moments of each dimensions."""
        return self.__distribution.std()

    @property
    def emit(self):
        """Return the emittance of the beam in both planes"""
        return {
            'X': np.sqrt(np.linalg.det(self.__distribution.head(len(self.__distribution))[['X', 'PX']].cov())),
            'Y': np.sqrt(np.linalg.det(self.__distribution.head(len(self.__distribution))[['Y', 'PY']].cov()))
        }

    @property
    def sigma(self):
        """Return the sigma matrix of the beam"""
        return self.__distribution.cov()

    covariance = sigma

    @property
    def twiss(self):
        """Return the Twiss parameters of the beam"""
        s11 = self.sigma['X']['X']
        s12 = self.sigma['X']['PX']
        s22 = self.sigma['PX']['PX']
        s33 = self.sigma['Y']['Y']
        s34 = self.sigma['Y']['PY']
        s44 = self.sigma['PY']['PY']
        return {
            'beta_x': s11 / self.emit['X'],
            'alpha_x': -s12 / self.emit['X'],
            'gamma_x': s22 / self.emit['X'],
            'beta_y': s33 / self.emit['Y'],
            'alpha_y': -s34 / self.emit['Y'],
            'gamma_y': s44 / self.emit['Y'],
        }

    @property
    def halo(self, dimensions=['X', 'Y', 'PX', 'PY']):
        """Return a dataframe containing the 1st, 5th, 95th and 99th percentiles of each dimensions."""
        if self._halo is None:
            self._halo = pd.concat([
                self.__distribution[dimensions].quantile(0.01),
                self.__distribution[dimensions].quantile(0.05),
                self.__distribution[dimensions].quantile(1.0 - 0.842701),
                self.__distribution[dimensions].quantile(0.842701),
                self.__distribution[dimensions].quantile(0.95),
                self.__distribution[dimensions].quantile(0.99)
            ], axis=1).rename(columns={0.01: '1%',
                                       0.05: '5%',
                                       1.0 - 0.842701: '20%',
                                       0.842701: '80%',
                                       0.95: '95%',
                                       0.99: '99%'
                                       }
                              )
        return self._halo

    @property
    def coupling(self):
        """Return a dataframe containing the covariances (coupling) between each dimensions."""
        return self.__distribution.cov()

    def __getitem__(self, item):
        if item not in PHASE_SPACE_DIMENSIONS[:self.__dims]:
            raise DistributionException("Trying to access an invalid data from a beam.")
        return self.__distribution[item]

    def from_csv(self, path='.', filename=''):
        """Read a beam distribution from a csv file."""
        filename = os.path.join(path, filename)
        self.__initialize_distribution(distribution=pd.read_csv(filename)[['X', 'PX', 'Y', 'PY', 'DPP']])
        self.__distribution.columns = PHASE_SPACE_DIMENSIONS[:self.__dims]
        return self

    def from_parquet(self, fname):
        """Read a beam distribution from a parquet file."""
        self.__initialize_distribution(distribution=pd.read_parquet(fname)[['X', 'PX', 'Y', 'PY', 'DPP']])
        self.__distribution.columns = PHASE_SPACE_DIMENSIONS[:self.__dims]
        return self

    def __initialize_distribution(self, distribution=None, *args, **kwargs):
        """Try setting the internal pandas.DataFrame with a distribution."""
        if distribution is not None:
            self.__distribution = distribution
        else:
            try:
                self.__distribution = pd.DataFrame(args[0])
            except (IndexError, ValueError):
                if kwargs.get("filename") is not None:
                    self.__distribution = Distribution.from_file(kwargs.get('filename'), path=kwargs.get('path', ''))
                else:
                    return
        self.__n_particles = self.__distribution.shape[0]
        if self.__n_particles <= 0:
            raise DistributionException("Trying to initialize a beam distribution with invalid number of particles.")
        self.__dims = self.__distribution.shape[1]
        if self.__dims < 2 or self.__dims > 6:
            raise DistributionException("Trying to initialize a beam distribution with invalid dimensions.")
        self.__distribution.columns = PHASE_SPACE_DIMENSIONS[:self.__dims]

    def from_5d_multigaussian_distribution(self, **kwargs):
        """Initialize a beam with a 5D particle distribution."""
        self.from_5d_sigma_matrix(n=kwargs.get('n', DEFAULT_N_PARTICLES),
                                  x=kwargs.get('X', 0 * _ureg.m).m_as("m"),
                                  px=kwargs.get('PX', 0 * _ureg.radians).m_as("radians"),
                                  y=kwargs.get('Y', 0 * _ureg.m).m_as("m"),
                                  py=kwargs.get('PY', 0 * _ureg.radians).m_as("radians"),
                                  dpp=kwargs.get('DPP', 0),
                                  dpprms=kwargs.get('DPPRMS', 0),
                                  s11=kwargs.get('XRMS', 0 * _ureg.m).m_as("m") ** 2,
                                  s12=0,
                                  s22=kwargs.get('PXRMS', 0 * _ureg.radians).m_as("radians") ** 2,
                                  s33=kwargs.get('YRMS', 0 * _ureg.m).m_as("m") ** 2,
                                  s34=0,
                                  s44=kwargs.get('PYRMS', 0 * _ureg.radians).m_as("radians") ** 2
                                  )
        return self

    def from_twiss_parameters(self, **kwargs):
        """Initialize a beam with a 5D particle distribution from Twiss parameters."""
        keys = {'n', 'X', 'PX', 'Y', 'PY', 'DPP', 'DPPRMS',
                'BETAX', 'ALPHAX', 'DISPX',
                'BETAY', 'ALPHAY', 'DISPY',
                'EMITX', 'EMITY'}
        if any([k not in keys for k in kwargs.keys()]):
            raise DistributionException("Invalid argument for a twiss distribution.")

        betax = kwargs.get('BETAX', 1 * _ureg.m).m_as('m')
        alphax = kwargs.get('ALPHAX', 0)
        gammax = (1 + alphax ** 2) / betax
        betay = kwargs.get('BETAY', 1 * _ureg.m).m_as('m')
        alphay = kwargs.get('ALPHAY', 0)
        gammay = (1 + alphay ** 2) / betay
        emitx = kwargs.get('EMITX', 1e-6 * _ureg.m * _ureg.radians).m_as('m rad')
        emity = kwargs.get('EMITY', 1e-6 * _ureg.m * _ureg.radians).m_as('m rad')
        dispx = kwargs.get('DISPX', 0 * _ureg.m).m_as('m')
        dispy = kwargs.get('DISPY', 0 * _ureg.m).m_as('m')
        dispxp = kwargs.get('DISPXP', 0 * _ureg.radians).m_as('radians')
        dispyp = kwargs.get('DISPYP', 0 * _ureg.radians).m_as('radians')
        dpprms = kwargs.get('DPPRMS', 0)

        self.from_5d_sigma_matrix(n=kwargs.get('n', DEFAULT_N_PARTICLES),
                                  x=kwargs.get('X', 0 * _ureg.m).m_as('m'),
                                  px=kwargs.get('PX', 0 * _ureg.radians).m_as('radians'),
                                  y=kwargs.get('Y', 0 * _ureg.m).m_as('m'),
                                  py=kwargs.get('PY', 0 * _ureg.radians).m_as('radians'),
                                  dpp=kwargs.get('DPP', 0),
                                  s11=betax * emitx + (dispx * dpprms) ** 2,
                                  s12=-alphax * emitx + dispx * dispxp * dpprms ** 2,
                                  s22=gammax * emitx + (dispxp * dpprms) ** 2,
                                  s33=betay * emity + (dispy * dpprms) ** 2,
                                  s34=-alphay * emity + dispy * dispyp * dpprms ** 2,
                                  s44=gammay * emity + (dispyp * dpprms) ** 2,
                                  s13=dispx * dispy * dpprms ** 2,
                                  s23=dispxp * dispy * dpprms ** 2,
                                  s14=dispx * dispyp * dpprms ** 2,
                                  s24=dispxp * dispyp * dpprms ** 2,
                                  s55=dpprms ** 2,
                                  )
        return self

    @staticmethod
    def generate_from_5d_sigma_matrix(n: int,
                                      x: float = 0,
                                      px: float = 0,
                                      y: float = 0,
                                      py: float = 0,
                                      dpp: float = 0,
                                      s11: float = 0,
                                      s12: float = 0,
                                      s13: float = 0,
                                      s14: float = 0,
                                      s15: float = 0,
                                      s22: float = 0,
                                      s23: float = 0,
                                      s24: float = 0,
                                      s25: float = 0,
                                      s33: float = 0,
                                      s34: float = 0,
                                      s35: float = 0,
                                      s44: float = 0,
                                      s45: float = 0,
                                      s55: float = 0,
                                      matrix=None,
                                      ):
        """

        Args:
            n:
            x:
            px:
            y:
            py:
            dpp:
            s11:
            s12:
            s13:
            s14:
            s15:
            s22:
            s23:
            s24:
            s25:
            s33:
            s34:
            s35:
            s44:
            s45:
            s55:
            matrix:

        Returns:

        """
        # For performance considerations, see
        # https://software.intel.com/en-us/blogs/2016/06/15/faster-random-number-generation-in-intel-distribution-for-python
        try:
            import numpy.random_intel
            generator = numpy.random_intel.multivariate_normal
        except ModuleNotFoundError:
            import numpy.random
            generator = numpy.random.multivariate_normal

        s21 = s12
        s31 = s13
        s32 = s23
        s41 = s14
        s42 = s24
        s43 = s34
        s51 = s15
        s52 = s25
        s53 = s35
        s54 = s45

        if matrix is not None:
            assert matrix.shape == (5, 5)
            return generator(
                [x, px, y, py, dpp],
                matrix,
                int(n)
            )
        else:
            return generator(
                [x, px, y, py, dpp],
                np.array([
                    [s11, s12, s13, s14, s15],
                    [s21, s22, s23, s24, s25],
                    [s31, s32, s33, s34, s35],
                    [s41, s42, s43, s44, s45],
                    [s51, s52, s53, s54, s55]
                ]),
                int(n)
            )

    def from_5d_sigma_matrix(self, **kwargs):
        """Initialize a beam with a 5D particle distribution from a Sigma matrix."""
        distribution = Distribution.generate_from_5d_sigma_matrix(**kwargs)
        self.__initialize_distribution(pd.DataFrame(distribution))
        self.__distribution.columns = PHASE_SPACE_DIMENSIONS[:self.__dims]
        return self
