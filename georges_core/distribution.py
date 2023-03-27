"""
Georges-core beam distribution module.

This module provides a collection of functions and classes to deal with a beam distribution. After loading or
generating a distribution, methods are available to compute beam properties, such as mean, standard deviation,
Twiss parameters, emittance or the beam halo.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as _np
import numpy.typing as _npt
import pandas as _pd
from numba import njit

from .units import Q_ as _Q
from .units import ureg as _ureg

PARTICLE_TYPES = {"proton", "antiproton", "electron", "positron"}
PHASE_SPACE_DIMENSIONS = ["X", "PX", "Y", "PY", "DPP"]
DEFAULT_N_PARTICLES = int(1e5)


# Define all methods to generate the beam
def load_from_file(path: str = "", filename: str = "", file_format: str = "csv") -> _pd.DataFrame:
    """Load a distribution from a file

    Args:
        path (str, optional): Path to the file. Defaults to "".
        filename (str, optional): Name of the file. Defaults to "".
        file_format (str, optional): Format of the file. Defaults to "csv".

    Returns:
        _pd.DataFrame: Pandas dataframe with the distributions.
    """
    if file_format == "csv":
        return _pd.read_csv(os.path.join(path, filename))
    if file_format == "parquet":
        return _pd.read_parquet(os.path.join(path, filename))
    raise DistributionException("Format of the file is incorrect. Only csv or parquet are available.")


def generate_from_5d_sigma_matrix(
    n: int,
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
    matrix: Optional[_npt.NDArray[_np.float_]] = None,
) -> _npt.NDArray[_np.float_]:
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
        return generator([x, px, y, py, dpp], matrix, int(n))  # type: ignore[no-any-return]

    return generator(  # type: ignore[no-any-return]
        [x, px, y, py, dpp],
        _np.array(
            [
                [s11, s12, s13, s14, s15],
                [s21, s22, s23, s24, s25],
                [s31, s32, s33, s34, s35],
                [s41, s42, s43, s44, s45],
                [s51, s52, s53, s54, s55],
            ],
        ),
        int(n),
    )


class DistributionException(Exception):
    """Exception raised for errors in the Beam module."""

    def __init__(self, m: str = ""):
        self.message = m


class Distribution:
    """Particle beam to be tracked in a beamline or accelerator model.

    The internal representation is essentially a `pandas` `DataFrame`.
    """

    def __init__(self, distribution: Optional[_pd.DataFrame] = None):
        """
        Initialize a beam object from various sources of particle beam distribution.
        Args:
            distribution: distribution of particles to initialize the beam with. Should be pandas.DataFrame() friendly.
        """
        try:
            self.__initialize_distribution(distribution)
            self.__dims = self.__distribution.shape[1]  # type: ignore[has-type]
        except DistributionException:
            self.__dims = len(PHASE_SPACE_DIMENSIONS)
            self.__distribution = _pd.DataFrame(_np.zeros((1, self.__dims)))
            self.__distribution.columns = PHASE_SPACE_DIMENSIONS[: self.__dims]
        self.__n_particles = self.__distribution.shape[0]
        if self.__n_particles <= 0:
            raise DistributionException("Error, no particles in the beam.")
        self._halo = None  # To force the first computation of the halo

    @property
    def distribution(self) -> _pd.DataFrame:
        """Return a dataframe containing the beam's particles distribution."""
        return self.__distribution

    @property
    def dims(self) -> int:
        """Return the dimensions of the beam's phase-space."""
        return self.__dims  # type: ignore[no-any-return]

    @property
    def n_particles(self) -> int:
        """Return the number of particles in the beam's distribution."""
        return self.__n_particles  # type: ignore[no-any-return]

    @property
    def mean(self) -> _pd.Series:
        """Return a dataframe containing the first order moments of each dimension."""
        return self.__distribution.mean()

    @property
    def std(self) -> _pd.Series:
        """Return a dataframe containing the second order moments of each dimension."""
        return self.__distribution.std()

    @property
    def emit(self) -> Dict[str, float]:
        """Return the emittance of the beam in both planes"""
        tw = njit(self.compute_twiss)(self.__distribution.values)
        return {"X": tw[0], "Y": tw[5]}

    @property
    def sigma(self) -> _pd.Series:
        """Return the sigma matrix of the beam"""
        return self.__distribution.cov()

    covariance = sigma

    @property
    def twiss(self) -> Dict[str, float]:
        """Return the Twiss parameters of the beam"""
        tw = njit(self.compute_twiss)(self.__distribution.values)
        return {
            "emit_x": tw[0],
            "beta_x": tw[1],
            "alpha_x": tw[2],
            "disp_x": tw[3],
            "disp_xp": tw[4],
            "emit_y": tw[5],
            "beta_y": tw[6],
            "alpha_y": tw[7],
            "disp_y": tw[8],
            "disp_yp": tw[9],
        }

    @property
    def halo(self, dimensions: Optional[Union[str, List[str]]] = None) -> _pd.DataFrame:
        """Return a dataframe containing the 1st, 5th, 95th and 99th percentiles of each dimensions."""
        if dimensions is None:
            dimensions = ["X", "Y", "PX", "PY"]
        if self._halo is None:
            self._halo = _pd.concat(
                [
                    self.__distribution[dimensions].quantile(0.01),
                    self.__distribution[dimensions].quantile(0.05),
                    self.__distribution[dimensions].quantile(0.2),
                    self.__distribution[dimensions].quantile(0.8),
                    self.__distribution[dimensions].quantile(0.95),
                    self.__distribution[dimensions].quantile(0.99),
                ],
                axis=1,
            ).rename(columns={0.01: "1%", 0.05: "5%", 0.2: "20%", 0.8: "80%", 0.95: "95%", 0.99: "99%"})
        return self._halo

    def __getitem__(self, item: str) -> _pd.Series:
        if item not in PHASE_SPACE_DIMENSIONS[: self.__dims]:
            raise DistributionException("Trying to access an invalid data from a beam.")
        return self.__distribution[item]

    def __initialize_distribution(self, distribution: _pd.DataFrame = None) -> None:
        """Try setting the internal pandas.DataFrame with a distribution."""
        if distribution is not None:
            self.__distribution = distribution
        else:
            logging.warning("Distribution is None: generate a default beam")
            raise DistributionException("")
        self.__dims = self.__distribution.shape[1]
        if self.__dims < 4 or self.__dims > len(PHASE_SPACE_DIMENSIONS):
            missing_key = list({"X", "Y", "PX", "PY"} - set(distribution.columns.values))
            logging.warning(
                "Trying to initialize a beam distribution with invalid dimensions. "
                f"{missing_key} are missing. Generate a default beam",
            )
            raise DistributionException("")
        self.__distribution[list(set(PHASE_SPACE_DIMENSIONS) - set(self.__distribution.columns.values))] = 0

    @staticmethod
    def compute_twiss(beam: _npt.NDArray[_np.float_]) -> _npt.NDArray[_np.float_]:
        """Compute Twiss parameters of a beam
        From http://nicadd.niu.edu/~syphers/tutorials/analyzeTrack.html

        Args:
            beam (_np.ndarray): beam input distribution

        Returns:
            _np.array: An array with the Twiss parameters
        """

        s11 = _np.var(beam[:, 0])
        s22 = _np.var(beam[:, 1])
        s33 = _np.var(beam[:, 2])
        s44 = _np.var(beam[:, 3])
        s55 = _np.var(beam[:, 4])

        if s55 == 0:
            s12 = _np.cov(beam[:, 0], beam[:, 1])[0, 1]
            s34 = _np.cov(beam[:, 2], beam[:, 3])[0, 1]

            emit_x = _np.sqrt(_np.linalg.det(_np.cov(beam[:, 0], beam[:, 1])))
            emit_y = _np.sqrt(_np.linalg.det(_np.cov(beam[:, 2], beam[:, 3])))

            beta_x = s11 / emit_x
            alpha_x = -s12 / emit_x
            disp_x = 0
            disp_xp = 0

            beta_y = s33 / emit_y
            alpha_y = -s34 / emit_y
            disp_y = 0
            disp_yp = 0

        else:
            a_xxp = _np.mean((beam[:, 0] - _np.mean(beam[:, 0])) * (beam[:, 1] - _np.mean(beam[:, 1])))
            a_xd = _np.mean((beam[:, 0] - _np.mean(beam[:, 0])) * (beam[:, 4] - _np.mean(beam[:, 4])))
            a_xpd = _np.mean((beam[:, 1] - _np.mean(beam[:, 1])) * (beam[:, 4] - _np.mean(beam[:, 4])))

            a_yyp = _np.mean((beam[:, 2] - _np.mean(beam[:, 2])) * (beam[:, 3] - _np.mean(beam[:, 3])))
            a_yd = _np.mean((beam[:, 2] - _np.mean(beam[:, 2])) * (beam[:, 4] - _np.mean(beam[:, 4])))
            a_ypd = _np.mean((beam[:, 3] - _np.mean(beam[:, 3])) * (beam[:, 4] - _np.mean(beam[:, 4])))

            disp_x = a_xd / s55
            disp_xp = a_xpd / s55

            ebeta_x = s11 - a_xd**2 / s55
            egamma_x = s22 - a_xpd**2 / s55
            ealpha_x = -a_xxp + a_xpd * a_xd / s55

            emit_x = _np.sqrt(ebeta_x * egamma_x - ealpha_x**2)
            beta_x = ebeta_x / emit_x
            alpha_x = ealpha_x / emit_x

            disp_y = a_yd / s55
            disp_yp = a_ypd / s55

            ebeta_y = s33 - a_yd**2 / s55
            egamma_y = s44 - a_ypd**2 / s55
            ealpha_y = -a_yyp + a_ypd * a_yd / s55

            emit_y = _np.sqrt(ebeta_y * egamma_y - ealpha_y**2)
            beta_y = ebeta_y / emit_y
            alpha_y = ealpha_y / emit_y

        return _np.array([emit_x, beta_x, alpha_x, disp_x, disp_xp, emit_y, beta_y, alpha_y, disp_y, disp_yp])

    @classmethod
    def from_csv(cls, path: str = "", filename: str = "") -> Distribution:
        """

        Args:
            path (str): Path to the csv file
            filename (str): filename

        Returns:
            An instance of the class with the distribution
        """
        return cls(distribution=load_from_file(path, filename, file_format="csv"))

    @classmethod
    def from_parquet(cls, path: str = "", filename: str = "") -> Distribution:
        """

        Args:
            path (str): path to the parquet file
            filename (str): filename

        Returns:
            An instance of the class with the distribution
        """
        return cls(distribution=load_from_file(path, filename, file_format="parquet"))

    @classmethod
    def from_5d_sigma_matrix(
        cls,
        n: int,
        x: _Q = 0 * _ureg.m,
        px: float = 0,
        y: _Q = 0 * _ureg.m,
        py: float = 0,
        dpp: float = 0,
        s11: _Q = 0 * _ureg.m**2,
        s12: float = 0,
        s13: float = 0,
        s14: float = 0,
        s15: float = 0,
        s22: float = 0,
        s23: float = 0,
        s24: float = 0,
        s25: float = 0,
        s33: _Q = 0 * _ureg.m**2,
        s34: float = 0,
        s35: float = 0,
        s44: float = 0,
        s45: float = 0,
        s55: float = 0,
        matrix: Optional[Any] = None,
    ) -> Distribution:
        """
        Initialize a beam with a 5D particle distribution from a Sigma matrix.
        Args:
            n (int): number of particles
            x (_Q): Horizontal position [m]
            px (float): Horizontal component momentum of unit vector
            y (_Q): Vertical position [m]
            py (float): Vertical component momentum of unit vector
            dpp (float): Momentum spread
            s11 ():
            s12 ():
            s13 ():
            s14 ():
            s15 ():
            s22 ():
            s23 ():
            s24 ():
            s25 ():
            s33 ():
            s34 ():
            s35 ():
            s44 ():
            s45 ():
            s55 ():
            matrix ():

        Returns:
            An instance of the class with the distribution
        """
        return cls(
            distribution=_pd.DataFrame(
                generate_from_5d_sigma_matrix(
                    n=int(n),
                    x=x.m_as("m"),
                    px=px,
                    y=y.m_as("m"),
                    py=py,
                    dpp=dpp,
                    s11=s11.m_as("m**2"),
                    s12=s12,
                    s13=s13,
                    s14=s14,
                    s15=s15,
                    s22=s22,
                    s23=s23,
                    s24=s24,
                    s25=s25,
                    s33=s33.m_as("m**2"),
                    s34=s34,
                    s35=s35,
                    s44=s44,
                    s45=s45,
                    s55=s55,
                    matrix=matrix,
                ),
                columns=["X", "PX", "Y", "PY", "DPP"],
            ),
        )

    @classmethod
    def from_5d_multigaussian_distribution(
        cls,
        n: int = DEFAULT_N_PARTICLES,
        x: _Q = 0 * _ureg.m,
        px: float = 0,
        y: _Q = 0 * _ureg.m,
        py: float = 0,
        dpp: float = 0,
        xrms: _Q = 0 * _ureg.m,
        pxrms: float = 0,
        yrms: _Q = 0 * _ureg.m,
        pyrms: float = 0,
        dpprms: float = 0,
    ) -> Distribution:
        """
        Initialize a beam with a 5D particle distribution from rms quantities.
        Args:
            n (int): number of particles
            x (_Q): Horizontal position [m]
            px (): Horizontal component momentum of unit vector
            y (_Q): Vertical position [m]
            py (float): Vertical component momentum of unit vector
            dpp (float): Momentum spread
            xrms (_Q): Horizontal Gaussian sigma [m]
            pxrms (float): Sigma of the horizontal component of unit momentum
            yrms (_Q): Vertical Gaussian sigma [m]
            pyrms (float): Sigma of the vertical component of unit momentum
            dpprms (float): Relative momentum spread

        Returns:
            An instance of the class with the distribution
        """
        return cls(
            distribution=_pd.DataFrame(
                generate_from_5d_sigma_matrix(
                    n=int(n),
                    x=x.m_as("m"),
                    px=px,
                    y=y.m_as("m"),
                    py=py,
                    dpp=dpp,
                    s11=xrms.m_as("m") ** 2,
                    s12=0,
                    s22=pxrms**2,
                    s33=yrms.m_as("m") ** 2,
                    s34=0,
                    s44=pyrms**2,
                    s55=dpprms**2,
                ),
                columns=["X", "PX", "Y", "PY", "DPP"],
            ),
        )

    @classmethod
    def from_twiss_parameters(
        cls,
        n: int = DEFAULT_N_PARTICLES,
        x: _Q = 0 * _ureg.m,
        px: float = 0,
        y: _Q = 0 * _ureg.m,
        py: float = 0,
        dpp: float = 0,
        betax: _Q = 1 * _ureg.m,
        alphax: float = 0,
        betay: _Q = 1 * _ureg.m,
        alphay: float = 0,
        emitx: _Q = 1e-6 * _ureg.m * _ureg.radians,
        emity: _Q = 1e-6 * _ureg.m * _ureg.radians,
        dispx: _Q = 0 * _ureg.m,
        dispy: _Q = 0 * _ureg.m,
        dispxp: float = 0,
        dispyp: float = 0,
        dpprms: float = 0,
    ) -> Distribution:
        """
        Initialize a beam with a 5D particle distribution from Twiss parameters.
        Args:
            n (int): number of particles
            x (_Q): Horizontal position [m]
            px (float): Horizontal component momentum of unit vector
            y (_Q): Vertical position [m]
            py (float): Vertical component momentum of unit vector
            dpp (float): Momentum spread
            betax (_Q): Horizontal beta function [m]
            alphax (): Horizontal alpha function
            betay (_Q): Vertical beta function [m]
            alphay (float): Vertical alpha function
            emitx (_Q): Horizontal emittance [m rad]
            emity (_Q): Vertical emittance [m rad]
            dispx (_Q): Horizontal dispersion function [m]
            dispy (_Q): Vertical dispersion function [m]
            dispxp (float): Horizontal angular dispersion function
            dispyp (float): Vertical angular dispersion function
            dpprms (float): Relative momentum spread

        Returns:
            An instance of the class with the distribution
        """

        gammax = (1 + alphax**2) / betax.m_as("m")
        gammay = (1 + alphay**2) / betay.m_as("m")

        s11 = betax.m_as("m") * emitx.m_as("m rad") + (dispx.m_as("m") * dpprms) ** 2
        s12 = -alphax * emitx.m_as("m rad") + dispx.m_as("m") * dispxp * dpprms**2
        s22 = gammax * emitx.m_as("m rad") + (dispxp * dpprms) ** 2
        s33 = betay.m_as("m") * emity.m_as("m rad") + (dispy.m_as("m") * dpprms) ** 2
        s34 = -alphay * emity.m_as("m rad") + dispy.m_as("m") * dispyp * dpprms**2
        s44 = gammay * emity.m_as("m rad") + (dispyp * dpprms) ** 2
        s13 = dispx.m_as("m") * dispy.m_as("m") * dpprms**2
        s23 = dispxp * dispy.m_as("m") * dpprms**2
        s14 = dispx.m_as("m") * dispyp * dpprms**2
        s24 = dispxp * dispyp * dpprms**2
        s15 = dispx.m_as("m") * dpprms**2
        s25 = dispxp * dpprms**2
        s35 = dispy.m_as("m") * dpprms**2
        s45 = dispyp * dpprms**2
        s55 = dpprms**2

        return cls(
            distribution=_pd.DataFrame(
                generate_from_5d_sigma_matrix(
                    n=int(n),
                    x=x.m_as("m"),
                    px=px,
                    y=y.m_as("m"),
                    py=py,
                    dpp=dpp,
                    s11=s11,
                    s12=s12,
                    s15=s15,
                    s25=s25,
                    s22=s22,
                    s33=s33,
                    s34=s34,
                    s35=s35,
                    s44=s44,
                    s45=s45,
                    s13=s13,
                    s23=s23,
                    s14=s14,
                    s24=s24,
                    s55=s55,
                ),
                columns=["X", "PX", "Y", "PY", "DPP"],
            ),
        )
