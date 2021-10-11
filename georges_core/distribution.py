import pandas as _pd
import numpy as _np
import os
from .units import ureg as _ureg
from .units import Q_ as _Q

PARTICLE_TYPES = {'proton', 'antiproton', 'electron', 'positron'}
PHASE_SPACE_DIMENSIONS = ['X', 'PX', 'Y', 'PY', 'DPP', 'DT']
DEFAULT_N_PARTICLES = 1e5


# Define all methods to generate the beam
def load_from_file(path: str = '', filename='', file_format='csv') -> _pd.DataFrame:
    if file_format == 'csv':
        return _pd.read_csv(os.path.join(path, filename))[['X', 'PX', 'Y', 'PY', 'DPP']]
    elif file_format == 'parquet':
        return _pd.read_parquet(os.path.join(path, filename))[['X', 'PX', 'Y', 'PY', 'DPP']]
    else:
        raise DistributionException(f"{file_format} is not a valid format. Please use csv or parquet instead")


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
            _np.array([
                [s11, s12, s13, s14, s15],
                [s21, s22, s23, s24, s25],
                [s31, s32, s33, s34, s35],
                [s41, s42, s43, s44, s45],
                [s51, s52, s53, s54, s55]
            ]),
            int(n)
        )


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
        Args:
            distribution: distribution of particles to initialize the beam with. Should be pandas.DataFrame() friendly.
            *args: optional parameters.
            **kwargs: optional keyword parameters.
        """
        try:
            self.__initialize_distribution(distribution, *args, **kwargs)
        except DistributionException:
            self.__dims = 5
            self.__distribution = _pd.DataFrame(_np.zeros((1, 5)))
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
            'X': _np.sqrt(_np.linalg.det(self.__distribution.head(len(self.__distribution))[['X', 'PX']].cov())),
            'Y': _np.sqrt(_np.linalg.det(self.__distribution.head(len(self.__distribution))[['Y', 'PY']].cov()))
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
    def halo(self, dimensions=None):
        """Return a dataframe containing the 1st, 5th, 95th and 99th percentiles of each dimensions."""
        if dimensions is None:
            dimensions = ['X', 'Y', 'PX', 'PY']
        if self._halo is None:
            self._halo = _pd.concat([
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

    def __initialize_distribution(self, distribution=None, *args, **kwargs):
        """Try setting the internal pandas.DataFrame with a distribution."""
        if distribution is not None:
            self.__distribution = distribution
        else:
            try:
                self.__distribution = _pd.DataFrame(args[0])
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

    @classmethod
    def from_csv(cls, path: str = '', filename: str = ''):
        return cls(distribution=load_from_file(path, filename, file_format='csv'))

    @classmethod
    def from_parquet(cls, path: str = '', filename: str = ''):
        return cls(distribution=load_from_file(path, filename, file_format='parquet'))

    @classmethod
    def from_5d_sigma_matrix(cls,
                             n: int,
                             x: float = 0,
                             px: float = 0,
                             y: float = 0,
                             py: float = 0,
                             dpp: float = 0,
                             s11: _Q = 0 * _ureg.m ** 2,
                             s12: float = 0,
                             s13: float = 0,
                             s14: float = 0,
                             s15: float = 0,
                             s22: _Q = 0 * _ureg.radians ** 2,
                             s23: float = 0,
                             s24: float = 0,
                             s25: float = 0,
                             s33: _Q = 0 * _ureg.m ** 2,
                             s34: float = 0,
                             s35: float = 0,
                             s44: _Q = 0 * _ureg.radians ** 2,
                             s45: float = 0,
                             s55: float = 0,
                             matrix=None):

        return cls(distribution=_pd.DataFrame(generate_from_5d_sigma_matrix(n=int(n),
                                                                            x=x,
                                                                            px=px,
                                                                            y=y,
                                                                            py=py,
                                                                            dpp=dpp,
                                                                            s11=s11.m_as('m**2'),
                                                                            s12=s12,
                                                                            s13=s13,
                                                                            s14=s14,
                                                                            s15=s15,
                                                                            s22=s22.m_as('radians**2'),
                                                                            s23=s23,
                                                                            s24=s24,
                                                                            s25=s25,
                                                                            s33=s33.m_as('m**2'),
                                                                            s34=s34,
                                                                            s35=s35,
                                                                            s44=s44.m_as('radians**2'),
                                                                            s45=s45,
                                                                            s55=s55,
                                                                            matrix=matrix)))

    @classmethod
    def from_5d_multigaussian_distribution(cls,
                                           n: int = DEFAULT_N_PARTICLES,
                                           x: _Q = 0 * _ureg.m,
                                           px: _Q = 0 * _ureg.radians,
                                           y: _Q = 0 * _ureg.m,
                                           py: _Q = 0 * _ureg.radians,
                                           dpp: _Q = 0,
                                           xrms: _Q = 0 * _ureg.m,
                                           pxrms: _Q = 0 * _ureg.radians,
                                           yrms: _Q = 0 * _ureg.m,
                                           pyrms: _Q = 0 * _ureg.radians,
                                           dpprms=0):
        return cls(distribution=_pd.DataFrame(generate_from_5d_sigma_matrix(n=int(n),
                                                                            x=x.m_as('m'),
                                                                            px=px.m_as("radians"),
                                                                            y=y.m_as("m"),
                                                                            py=py.m_as("radians"),
                                                                            dpp=dpp,
                                                                            s11=xrms.m_as("m") ** 2,
                                                                            s12=0,
                                                                            s22=pxrms.m_as("radians") ** 2,
                                                                            s33=yrms.m_as("m") ** 2,
                                                                            s34=0,
                                                                            s44=pyrms.m_as("radians") ** 2,
                                                                            s55=dpprms ** 2)))

    @classmethod
    def from_twiss_parameters(cls,
                              n: int = DEFAULT_N_PARTICLES,
                              x: _Q = 0 * _ureg.m,
                              px: _Q = 0 * _ureg.radians,
                              y: _Q = 0 * _ureg.m,
                              py: _Q = 0 * _ureg.radians,
                              dpp: _Q = 0,
                              betax: _Q = 1 * _ureg.m,
                              alphax: _Q = 0,
                              betay: _Q = 1 * _ureg.m,
                              alphay: _Q = 0,
                              emitx: _Q = 1e-6 * _ureg.m * _ureg.radians,
                              emity: _Q = 1e-6 * _ureg.m * _ureg.radians,
                              dispx: _Q = 0 * _ureg.m,
                              dispy: _Q = 0 * _ureg.m,
                              dispxp: _Q = 0,
                              dispyp: _Q = 0,
                              dpprms: _Q = 0

                              ):
        """Initialize a beam with a 5D particle distribution from Twiss parameters."""

        gammax = (1 + alphax ** 2) / betax.m_as('m')
        gammay = (1 + alphay ** 2) / betay.m_as('m')

        s11 = betax.m_as('m') * emitx.m_as('m rad') + (dispx.m_as('m') * dpprms) ** 2
        s12 = -alphax * emitx.m_as('m rad') + dispx.m_as('m') * dispxp * dpprms ** 2
        s22 = gammax * emitx.m_as('m rad') + (dispxp * dpprms) ** 2
        s33 = betay.m_as('m') * emity.m_as('m rad') + (dispy.m_as('m') * dpprms) ** 2
        s34 = -alphay * emity.m_as('m rad') + dispy.m_as('m') * dispyp * dpprms ** 2
        s44 = gammay * emity.m_as('m rad') + (dispyp * dpprms) ** 2
        s13 = dispx.m_as('m') * dispy.m_as('m') * dpprms ** 2
        s23 = dispxp * dispy.m_as('m') * dpprms ** 2
        s14 = dispx.m_as('m') * dispyp * dpprms ** 2
        s24 = dispxp * dispyp * dpprms ** 2
        s55 = dpprms ** 2

        return cls(distribution=_pd.DataFrame(generate_from_5d_sigma_matrix(n=int(n),
                                                                            x=x.m_as('m'),
                                                                            px=px.m_as("radians"),
                                                                            y=y.m_as("m"),
                                                                            py=py.m_as("radians"),
                                                                            dpp=dpp,
                                                                            s11=s11,
                                                                            s12=s12,
                                                                            s22=s22,
                                                                            s33=s33,
                                                                            s34=s34,
                                                                            s44=s44,
                                                                            s13=s13,
                                                                            s23=s23,
                                                                            s14=s14,
                                                                            s24=s24,
                                                                            s55=s55)))
