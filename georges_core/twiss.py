"""Module for the computation of Twiss parametrizations from transfer matrices.

The standard uncoupled Twiss parametrization (including off-momentum effects, aka. dispersion) is the default option.
Additional formalisms for the parametrization of fully coupled transfer matrices are also available (Teng, Ripken,
etc.).
"""
import copy
import warnings
from logging import warning
from typing import Optional, Tuple, Union

import numpy as _np
import pandas as _pd

from . import Kinematics as _Kinematics
from . import ureg as _ureg
from .sequences import BetaBlock as _BetaBlock


def _get_matrix_elements_block(
    m: _pd.DataFrame,
    twiss: Optional[_BetaBlock],
    block: int = 1,
) -> Union[
    Tuple[_pd.Series, _pd.Series, _pd.Series, _pd.Series],
    Tuple[_pd.Series, _pd.Series, _pd.Series, _pd.Series, Optional[float], Optional[float], Optional[float]],
]:
    """Extract parameters from the DataFrame."""
    p = 1 if block == 1 else 3
    v = 1 if block == 1 else 2
    r11: _pd.Series = m[f"R{p}{p}"]
    r12: _pd.Series = m[f"R{p}{p + 1}"]
    r21: _pd.Series = m[f"R{p + 1}{p}"]
    r22: _pd.Series = m[f"R{p + 1}{p + 1}"]
    if twiss is not None:
        alpha: float = twiss[f"ALPHA{v}{v}"]
        beta: float = twiss[f"BETA{v}{v}"].m_as("m")
        gamma: float = twiss[f"GAMMA{v}{v}"].m_as("m**-1")
        return r11, r12, r21, r22, alpha, beta, gamma
    else:
        return r11, r12, r21, r22


class ParametrizationType(type):
    pass


class Parametrization(metaclass=ParametrizationType):
    def compute_canonical_transformation_matrix(self, matrix_row: _pd.Series, kin: _Kinematics) -> _pd.Series:
        b_s = matrix_row["BX"] * _ureg.kG
        r_s = b_s / kin.brho
        r_s = r_s.to("1/(m)").magnitude
        matrix_rs = _np.array([[1, 0, 0, 0], [0, 1, -r_s / 2, 0], [0, 0, 1, 0], [r_s / 2, 0, 0, 1]])
        matrix_row["matrix_rs"] = matrix_rs
        matrix_row["RS"] = r_s
        matrix_row["BS"] = b_s.magnitude
        return matrix_row

    @staticmethod
    def compute_canonical_transfer_matrices(matrix_row: _pd.Series, matrix_rs1: _np.ndarray) -> _pd.Series:
        mat = (
            matrix_row[
                [
                    "R11",
                    "R12",
                    "R13",
                    "R14",
                    "R21",
                    "R22",
                    "R23",
                    "R24",
                    "R31",
                    "R32",
                    "R33",
                    "R34",
                    "R41",
                    "R42",
                    "R43",
                    "R44",
                ]
            ]
            .apply(float)
            .values.reshape(4, 4)
        )

        matrix_rs = matrix_row["matrix_rs"]
        m_canon = matrix_rs @ mat @ _np.linalg.inv(matrix_rs1)
        matrix_row["m_canon"] = m_canon

        return matrix_row

    @staticmethod
    def compute_one_turn_transfer_matrix(matrix_row: _pd.Series, mat_tot: _np.ndarray) -> _pd.Series:
        m_i = matrix_row["m_canon"]
        m = m_i @ mat_tot @ _np.linalg.inv(m_i)
        matrix_row["m"] = m

        return matrix_row

    def compute_eigenvectors(self, matrix_row: _pd.Series) -> _pd.Series:
        # Eigenvalues and eigenvectors of the one period transfer matrix
        v1, v1_, v2, v2_ = self.compute_orderded_turned_normalized_eigenvectors()
        matrix_row["v1"] = v1
        matrix_row["v2"] = v2
        matrix_row["v1_"] = v1_
        matrix_row["v2_"] = v2_

        return matrix_row

    def compute_orderded_turned_normalized_eigenvectors(self):
        # Fonction redéfinie dans les classes-filles !
        v1, v1_, v2, v2_ = _np.identidy(4)
        return v1, v1_, v2, v2_

    @staticmethod
    def phase_unrolling(phi, s):
        """TODO"""
        if phi[0] < 0:
            phi[0] += 1 * _np.pi
        for i in range(1, phi.shape[0]):
            if phi[i] < 0:
                phi[i] += 1 * _np.pi
            if phi[i - 1] - phi[i] > 0.5 and s[i - 1] - s[i] < 0.0:
                phi[i:] += 1 * _np.pi
        return phi

    @staticmethod
    def compute_turned_eigvec(v1: _np.ndarray, v1_: _np.ndarray, plane: int = 1):
        j = 1j

        phi_v1 = _np.arctan(_np.imag(v1[plane * 2 - 2]) / _np.real(v1[plane * 2 - 2]))
        theta_1 = -phi_v1
        v1 = v1 * (_np.cos(theta_1) + j * _np.sin(theta_1))
        v1_ = v1_ * (_np.cos(theta_1) - j * _np.sin(theta_1))

        if _np.real(v1[plane * 2 - 2] < 0):  # Permet d'assurer le signe du beta pour la propagation
            v1 = v1 * (_np.cos(_np.pi) + j * _np.sin(_np.pi))
            v1_ = v1_ * (_np.cos(_np.pi) - j * _np.sin(_np.pi))

        return v1, v1_


class Twiss(Parametrization):
    def __init__(self, twiss_init: Optional[_BetaBlock] = None, with_phase_unrolling: bool = True):
        """

        Args:
            twiss_init: the initial values for the Twiss computation (if None, periodic conditions are assumed and the
            Twiss parameters are computed from the transfer matrix).
            with_phase_unrolling: TODO
        """
        self._twiss_init = twiss_init
        self._with_phase_unrolling = with_phase_unrolling

    def __call__(self, matrix: _pd.DataFrame, end: Union[int, str] = -1) -> _pd.DataFrame:
        """
        Uses a step-by-step transfer matrix to compute the Twiss parameters (uncoupled). The phase advance and the
        determinants of the jacobians are computed as well.

        Args:
            matrix: the input step-by-step transfer matrix

        Returns:
            the same DataFrame as the input, but with added columns for the computed quantities.
        """
        if self._twiss_init is None:
            twiss_init = self.compute_periodic_twiss(matrix, end)
        else:
            twiss_init = self._twiss_init

        matrix["BETA11"] = self.compute_beta_from_matrix(matrix, twiss_init)
        matrix["BETA22"] = self.compute_beta_from_matrix(matrix, twiss_init, plane=2)
        matrix["ALPHA11"] = self.compute_alpha_from_matrix(matrix, twiss_init)
        matrix["ALPHA22"] = self.compute_alpha_from_matrix(matrix, twiss_init, plane=2)
        matrix["GAMMA11"] = self.compute_gamma_from_matrix(matrix, twiss_init)
        matrix["GAMMA22"] = self.compute_gamma_from_matrix(matrix, twiss_init, plane=2)
        matrix["MU1"] = self.compute_mu_from_matrix(matrix, twiss_init)
        matrix["MU2"] = self.compute_mu_from_matrix(matrix, twiss_init, plane=2)
        matrix["DET1"] = self.compute_jacobian_from_matrix(matrix)
        matrix["DET2"] = self.compute_jacobian_from_matrix(matrix, plane=2)
        matrix["DISP1"] = self.compute_dispersion_from_matrix(matrix, twiss_init)
        matrix["DISP2"] = self.compute_dispersion_prime_from_matrix(matrix, twiss_init)
        matrix["DISP3"] = self.compute_dispersion_from_matrix(matrix, twiss_init, plane=2)
        matrix["DISP4"] = self.compute_dispersion_prime_from_matrix(matrix, twiss_init, plane=2)

        def phase_unrolling(phi: _np.ndarray) -> _np.ndarray:  # type: ignore[type-arg]
            """TODO"""
            if phi[0] < 0:
                phi[0] += 2 * _np.pi
            for i in range(1, phi.shape[0]):
                if phi[i] < 0:
                    phi[i] += 2 * _np.pi
                if phi[i - 1] - phi[i] > 0.5:
                    phi[i:] += 2 * _np.pi
            return phi

        try:
            from numba import njit

            phase_unrolling = njit(phase_unrolling)
        except ModuleNotFoundError:  # pragma: no cover
            pass

        if self._with_phase_unrolling:
            matrix["MU1U"] = phase_unrolling(matrix["MU1"].values)
            matrix["MU2U"] = phase_unrolling(matrix["MU2"].values)

        return matrix

    @staticmethod
    def compute_alpha_from_matrix(m: _pd.DataFrame, twiss: _BetaBlock, plane: int = 1) -> _pd.Series:
        """
        Computes the Twiss alpha values at every steps of the input step-by-step transfer matrix.

        Args:
            m: the step-by-step transfer matrix for which the alpha values should be computed
            twiss: the initial Twiss values
            plane: an integer representing the block (1 or 2)

        Returns:
            a Pandas Series with the alpha values computed at all steps of the input step-by-step transfer matrix
        """
        r11, r12, r21, r22, alpha, beta, gamma = _get_matrix_elements_block(m, twiss, plane)  # type: ignore[misc]
        return -r11 * r21 * beta + (r11 * r22 + r12 * r21) * alpha - r12 * r22 * gamma

    @staticmethod
    def compute_beta_from_matrix(
        m: _pd.DataFrame,
        twiss: _BetaBlock,
        plane: int = 1,
        strict: bool = False,
    ) -> _pd.Series:
        """
        Computes the Twiss beta values at every steps of the input step-by-step transfer matrix.

        Args:
            m: the step-by-step transfer matrix for which the beta values should be computed
            twiss: the initial Twiss values
            plane: an integer representing the block (1 or 2)
            strict: flag to activate the strict mode: checks and ensures that all computed beta are positive

        Returns:
            a Pandas Series with the beta values computed at all steps of the input step-by-step transfer matrix
        """
        r11, r12, r21, r22, alpha, beta, gamma = _get_matrix_elements_block(m, twiss, plane)  # type: ignore[misc]
        _ = r11**2 * beta - 2.0 * r11 * r12 * alpha + r12**2 * gamma
        if strict:  # pragma: no cover
            assert (_ > 0).all(), "Not all computed beta are positive."
        return _

    @staticmethod
    def compute_gamma_from_matrix(m: _pd.DataFrame, twiss: _BetaBlock, plane: int = 1) -> _pd.Series:
        """
        Computes the Twiss gamma values at every steps of the input step-by-step transfer matrix.

        Args:
            m: the step-by-step transfer matrix for which the beta values should be computed
            twiss: the initial Twiss values
            plane: an integer representing the block (1 or 2)

        Returns:
            a Pandas Series with the gamma values computed at all steps of the input step-by-step transfer matrix
        """
        r11, r12, r21, r22, alpha, beta, gamma = _get_matrix_elements_block(m, twiss, plane)  # type: ignore[misc]
        return r21**2 * beta - 2.0 * r21 * r22 * alpha + r22**2 * gamma

    @staticmethod
    def compute_mu_from_matrix(m: _pd.DataFrame, twiss: _BetaBlock, plane: int = 1) -> _pd.Series:
        """
        Computes the phase advance values at every steps of the input step-by-step transfer matrix.

        Args:
            m: the step-by-step transfer matrix for which the beta values should be computed
            twiss: the initial Twiss values
            plane: an integer representing the block (1 or 2)

        Returns:
            a Pandas Series with the phase advance computed at all steps of the input step-by-step transfer matrix
        """
        r11, r12, r21, r22, alpha, beta, gamma = _get_matrix_elements_block(m, twiss, plane)  # type: ignore[misc]
        return _np.arctan2(r12, r11 * beta - r12 * alpha)

    @staticmethod
    def compute_jacobian_from_matrix(m: _pd.DataFrame, plane: int = 1) -> _pd.Series:
        """
        Computes the jacobian of the 2x2 transfer matrix (useful to verify the simplecticity).

        Args:
            m: the step-by-step transfer matrix for which the jacobians should be computed
            plane: an integer representing the block (1 or 2)

        Returns:
            a Pandas Series with the jacobian computed at all steps of the input step-by-step transfer matrix
        """
        r11, r12, r21, r22 = _get_matrix_elements_block(m, None, plane)  # type: ignore[misc]
        return r11 * r22 - r12 * r21

    @staticmethod
    def compute_dispersion_from_matrix(m: _pd.DataFrame, twiss: _BetaBlock, plane: int = 1) -> _pd.Series:
        """
        Computes the dispersion function at every steps of the input step-by-step transfer matrix.

        Args:
            m: the step-by-step transfer matrix for which the dispersion function should be computed
            twiss: initial values for the Twiss parameters
            plane: an integer representing the block (1 or 2)

        Returns:
            a Pandas Series with the dispersion function computed at all steps of the input step-by-step transfer matrix

        """
        p = 1 if plane == 1 else 3
        if p == 1:
            d0 = twiss["DISP1"].m_as("m")
            dp0 = twiss["DISP2"]
        else:
            d0 = twiss["DISP3"].m_as("m")
            dp0 = twiss["DISP4"]
        r11: _pd.Series = m[f"R{p}{p}"]
        r12: _pd.Series = m[f"R{p}{p + 1}"]
        r16: _pd.Series = m[f"R{p}6"]
        return d0 * r11 + dp0 * r12 + r16

    @staticmethod
    def compute_dispersion_prime_from_matrix(m: _pd.DataFrame, twiss: _BetaBlock, plane: int = 1) -> _pd.Series:
        """
        Computes the dispersion prime function at every steps of the input step-by-step transfer matrix.

        Args:
            m: the step-by-step transfer matrix for which the dispersion prime function should be computed
            twiss: initial values for the Twiss parameters
            plane: an integer representing the block (1 or 2)

        Returns:
            a Pandas Series with the dispersion prime function computed at all steps of the input step-by-step transfer
            matrix

        Example:

        """
        p = 1 if plane == 1 else 3
        if p == 1:
            d0 = twiss["DISP1"].m_as("m")
            dp0 = twiss["DISP2"]
        else:
            d0 = twiss["DISP3"].m_as("m")
            dp0 = twiss["DISP4"]
        r21: _pd.Series = m[f"R{p + 1}{p}"]
        r22: _pd.Series = m[f"R{p + 1}{p + 1}"]
        r26: _pd.Series = m[f"R{p + 1}6"]
        return d0 * r21 + dp0 * r22 + r26

    @staticmethod
    def compute_periodic_twiss(matrix: _pd.DataFrame, end: Union[int, str] = -1) -> _BetaBlock:
        """
        Compute twiss parameters from a transfer matrix which is assumed to be a periodic transfer matrix.

        Args:
            matrix: the (periodic) transfer matrix
            end:

        Returns:
            a Series object with the values of the periodic Twiss parameters.
        """
        m = matrix
        if isinstance(end, int):
            m = matrix.iloc[end]
        elif isinstance(end, str):
            m = matrix[matrix.LABEL1 == end].iloc[-1]
        twiss = dict(
            {
                "CMU1": (m["R11"] + m["R22"]) / 2.0,
                "CMU2": (m["R33"] + m["R44"]) / 2.0,
            },
        )
        if twiss["CMU1"] < -1.0 or twiss["CMU1"] > 1.0:  # pragma: no cover
            warning(f"Horizontal motion is unstable; proceed with caution (cos(mu) = {twiss['CMU1']}).")
        with warnings.catch_warnings():  # pragma: no cover
            warnings.simplefilter("ignore")
            twiss["MU1"] = _np.arccos(twiss["CMU1"])
        if twiss["CMU2"] < -1.0 or twiss["CMU2"] > 1.0:  # pragma: no cover
            warning(f"Vertical motion is unstable; proceed with caution (cos(mu) = {twiss['CMU2']}).")
        with warnings.catch_warnings():  # pragma: no cover
            warnings.simplefilter("ignore")
            twiss["MU2"] = _np.arccos(twiss["CMU2"])
        twiss["BETA11"] = m["R12"] / _np.sin(twiss["MU1"]) * _ureg.m
        if twiss["BETA11"] < 0.0:
            twiss["BETA11"] *= -1
            twiss["MU1"] *= -1
        twiss["BETA22"] = m["R34"] / _np.sin(twiss["MU2"]) * _ureg.m
        if twiss["BETA22"] < 0.0:
            twiss["BETA22"] *= -1
            twiss["MU2"] *= -1
        twiss["ALPHA11"] = (m["R11"] - m["R22"]) / (2.0 * _np.sin(twiss["MU1"]))
        twiss["ALPHA22"] = (m["R33"] - m["R44"]) / (2.0 * _np.sin(twiss["MU2"]))
        twiss["GAMMA11"] = -m["R21"] / _np.sin(twiss["MU1"]) * _ureg.m**-1
        twiss["GAMMA22"] = -m["R43"] / _np.sin(twiss["MU2"]) * _ureg.m**-1
        m44 = (
            m[
                [
                    "R11",
                    "R12",
                    "R13",
                    "R14",
                    "R21",
                    "R22",
                    "R23",
                    "R24",
                    "R31",
                    "R32",
                    "R33",
                    "R34",
                    "R41",
                    "R42",
                    "R43",
                    "R44",
                ]
            ]
            .apply(float)
            .values.reshape(4, 4)
        )
        r6 = m[["R16", "R26", "R36", "R46"]].apply(float).values.reshape(4, 1)
        disp = _np.dot(_np.linalg.inv(_np.identity(4) - m44), r6).reshape(4)
        twiss["DY"] = disp[0] * _ureg.m
        twiss["DYP"] = disp[1]
        twiss["DZ"] = disp[2] * _ureg.m
        twiss["DZP"] = disp[3]
        twiss["DISP1"] = twiss["DY"]
        twiss["DISP2"] = twiss["DYP"]
        twiss["DISP3"] = twiss["DZ"]
        twiss["DISP4"] = twiss["DZP"]

        return _BetaBlock(**twiss)


class Parzen(Parametrization):
    def __init__(
        self,
        twiss_init: Optional[_BetaBlock] = None,
        with_phase_unrolling: bool = True,
    ):
        """
        Args:
            twiss_init: the initial values for the Twiss computation (if None, periodic conditions are assumed and the
            Twiss parameters are computed from the transfer matrix).
            with_phase_unrolling: TODO
        """

        self._twiss_init = twiss_init
        self._with_phase_unrolling = with_phase_unrolling

    def __call__(
        self,
        matrix: _pd.DataFrame,
        tracks: _pd.DataFrame,
        kin: _Kinematics,
    ) -> _pd.DataFrame:
        """
        Uses a step-by-step transfer matrix to compute the generalized Twiss parameters (coupled motions)
        with the parametrization of Edwards and Teng using the method with eigenvectors presented in the paper
        of G. Parzen. The phase advances are computed as well.

        Args:
            matrix: the input step-by-step transfer matrix
            tracks: tracks_global for the centered particle 'O' of the BeamTwiss
            kin : Kinematics object
        Returns:
            the same DataFrame as the matrix input DataFrame, but with added columns for the computed quantities.
        """
        matrix["BX"] = tracks["BX"]

        # Calculation of the matrix for the transformation of geometric coordinates into the canonical ones
        matrix = matrix.apply(lambda row: self.compute_canonical_transformation_matrix(row, kin), axis=1)
        matrix_rs1 = matrix.iloc[0]["matrix_rs"]
        matrix = matrix.apply(lambda row: self.compute_canonical_transfer_matrices(row, matrix_rs1), axis=1)

        if self._twiss_init is not None:
            twiss_init = self._twiss_init
            u1, r_decoupling = self.get_initial_parametrisation(twiss_init)
            vec1 = u1 * _np.sqrt(-2j)
            x_1 = r_decoupling @ copy.deepcopy(vec1)
            # Calculation of eigenvectors
            matrix = matrix.apply(lambda row: self.compute_eigenvectors_from_initial_eigvecs(row, x_1), axis=1)

        else:
            # Total transfer matrix and one-turn transfer matrices
            mat_tot = matrix.iloc[-1]["m_canon"]
            matrix = matrix.apply(lambda row: self.compute_one_turn_transfer_matrix(row, mat_tot), axis=1)

            # Calculation of the ordered and normalized eigenvectors
            eigvals_init, eigvec_init = _np.linalg.eig(mat_tot)
            lambda1_0 = eigvals_init[0]
            matrix = matrix.apply(lambda row: self.compute_eigenvectors(row, lambda1_0), axis=1)

            # Parametrisation
        # beta, alpha, gamma, decoupling matrix R
        matrix = matrix.apply(self.compute_parametrisation_from_eigenvectors, axis=1)
        matrix = matrix.apply(self.compute_decoupling_matrix_from_eigenvectors, axis=1)
        r_decoupling_0 = matrix.iloc[0]["R"]
        matrix = matrix.apply(lambda row: self.compute_decoupled_transfer_matrix(row, r_decoupling_0), axis=1)

        # Phase advances
        u_0 = matrix.iloc[0]["U_"]
        eigvec_init = _np.linalg.inv(u_0)  # Vecteurs propres initiaux dans l'espace découplé
        matrix = matrix.apply(lambda row: self.compute_phase_advances(row, eigvec_init), axis=1)
        matrix["MU1"] = round(matrix["MU1"] - matrix.iloc[0]["MU1"], 10)
        matrix["MU2"] = round(matrix["MU2"] - matrix.iloc[0]["MU2"], 10)

        try:
            from numba import njit

            self.phase_unrolling = njit(self.phase_unrolling)
        except ModuleNotFoundError:
            pass

        if self._with_phase_unrolling:
            matrix["MU1"] = self.phase_unrolling(matrix["MU1"].values, matrix["S"].values)
            matrix["MU2"] = self.phase_unrolling(matrix["MU2"].values, matrix["S"].values)

        self.check_tunes(matrix.iloc[-1])

        return matrix

    @staticmethod
    def _get_twiss_elements(twiss: Optional[_BetaBlock], block: int = 1) -> Tuple:
        v = 1 if block == 1 else 2

        alpha: float = twiss[f"ALPHA{v}{v}"]
        beta: float = twiss[f"BETA{v}{v}"].m_as("m")
        gamma: float = twiss[f"GAMMA{v}{v}"].m_as("m**-1")

        r_matrix: _np.ndarray = twiss["R"]

        return alpha, beta, gamma, r_matrix

    def get_initial_parametrisation(self, twiss_init: Optional[_BetaBlock]) -> Tuple:
        alpha_1, beta_1, gamma_1, r_decoupling = self._get_twiss_elements(twiss_init)
        alpha_2, beta_2, gamma_2, r_decoupling = self._get_twiss_elements(twiss_init, 2)

        j = 1j

        phi_1_ok = 0.0
        phi_2_ok = 0.0

        u_1 = _np.array(
            [
                [
                    ((-alpha_1 - j) * _np.exp(-j * phi_1_ok)) / (_np.sqrt(beta_1)),
                    -_np.sqrt(beta_1) * _np.exp(-j * phi_1_ok),
                    0,
                    0,
                ],
                [
                    -((-alpha_1 + j) * _np.exp(j * phi_1_ok)) / (_np.sqrt(beta_1)),
                    _np.sqrt(beta_1) * _np.exp(j * phi_1_ok),
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    (-alpha_2 - j) * _np.exp(-j * phi_2_ok) / (_np.sqrt(beta_2)),
                    -_np.sqrt(beta_2) * _np.exp(-j * phi_2_ok),
                ],
                [
                    0,
                    0,
                    -(-alpha_2 + j) * _np.exp(j * phi_2_ok) / (_np.sqrt(beta_2)),
                    _np.sqrt(beta_2) * _np.exp(j * phi_2_ok),
                ],
            ],
        )
        u_1 = u_1 / _np.sqrt(-2j)

        u1 = _np.linalg.inv(u_1)

        return u1, r_decoupling

    @staticmethod
    def compute_normalized_eigenvectors(v1, v1_):
        # Normalisation des vecteurs propres: leur invariant de lagrange = 2i
        U = _np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
        ortho1 = v1_.T @ U @ v1
        ratio = 2j / ortho1

        ratio = abs(_np.real(ratio))
        v1 = v1 * _np.sqrt(ratio)
        v1_ = v1_ * _np.sqrt(ratio)
        return v1, v1_

    @staticmethod
    def compute_orderded_complex_conjugate_vectors_pair(
        phi_1: float,
        phi_2: float,
        eigvec: _np.ndarray,
        plane: int = 1,
    ):
        if _np.abs(phi_1) == _np.abs(phi_2):
            if phi_1 > 0:
                v1, v1_ = eigvec[:, plane * 2 - 2], eigvec[:, plane * 2 - 2 + 1]
            else:
                v1_, v1 = eigvec[:, plane * 2 - 2], eigvec[:, plane * 2 - 2 + 1]
        else:
            print("ERREUR")

        return v1, v1_

    def compute_orderded_turned_normalized_eigenvectors(
        self,
        eigvec: _np.ndarray = None,
        lambda1_0: float = None,
        eigvals: _np.ndarray = None,
    ):
        lambda1 = eigvals[0]
        phi_1, phi_2, phi_3, phi_4 = (
            _np.angle(eigvals[0]),
            _np.angle(eigvals[1]),
            _np.angle(eigvals[2]),
            _np.angle(
                eigvals[3],
            ),
        )

        # On vérifie l'ordre des vecteurs popres dans la paire de complexe conjugués
        v1, v1_ = self.compute_orderded_complex_conjugate_vectors_pair(phi_1, phi_2, eigvec)
        v2, v2_ = self.compute_orderded_complex_conjugate_vectors_pair(phi_3, phi_4, eigvec, plane=2)

        # On vérifie que les vecteurs propres sont bien ordonnés en fonction du mode propre
        if _np.round(_np.real(lambda1), 2) != _np.round(_np.real(lambda1_0), 2):
            v1, v1_, v2, v2_ = v2, v2_, v1, v1_

        v1, v1_ = self.compute_turned_eigvec(v1, v1_)
        v2, v2_ = self.compute_turned_eigvec(v2, v2_, plane=2)

        u1 = -_np.imag(v1[1] * v1[0])
        if u1 > 0:
            v1, v1_ = v1_, v1

        u4 = -_np.imag(v2[3] * v2[2])
        if u4 > 0:
            v2, v2_ = v2_, v2

        # On normalise les vecteurs propres avec la condition donnée dans Parzen
        v1, v1_ = self.compute_normalized_eigenvectors(v1, v1_)
        v2, v2_ = self.compute_normalized_eigenvectors(v2, v2_)

        return v1, v1_, v2, v2_

    def compute_eigenvectors_from_initial_eigvecs(self, matrix_row: _pd.Series, x_1: _np.ndarray) -> _pd.Series:
        matt_i = matrix_row["m_canon"]
        eigvec = matt_i @ x_1
        v1 = eigvec[:, 0]
        v1_ = eigvec[:, 1]
        v2 = eigvec[:, 2]
        v2_ = eigvec[:, 3]

        matrix_row["v1"] = v1
        matrix_row["v2"] = v2
        matrix_row["v1_"] = v1_
        matrix_row["v2_"] = v2_

        return matrix_row

    @staticmethod
    def compute_parametrisation_from_eigenvectors(matrix_row: _pd.Series) -> _pd.Series:
        j = 1j
        v1, v2 = matrix_row[["v1", "v2"]]

        # Generalized Twiss parameters alphas, betas and gammas
        beta_1 = 1 / (_np.imag(v1[1] / v1[0]))
        beta_2 = 1 / (_np.imag(v2[3] / v2[2]))

        sign = 0
        if beta_1 < 0:
            beta_1 *= -1
            sign += 1
        if beta_2 < 0:
            beta_2 *= -1
            sign += 2

        alpha_1 = -beta_1 * _np.real(v1[1] / v1[0])
        alpha_2 = -beta_2 * _np.real(v2[3] / v2[2])

        gamma_1 = (1 + alpha_1**2) / beta_1
        gamma_2 = (1 + alpha_2**2) / beta_2

        q1 = _np.linalg.norm(v1[0]) / _np.sqrt(beta_1)
        q2 = _np.linalg.norm(v2[2]) / _np.sqrt(beta_2)

        phi_1_ok = _np.arctan(_np.imag(v1[0]) / _np.real(v1[0]))
        phi_2_ok = _np.arctan(_np.imag(v2[2]) / _np.real(v2[2]))

        u_ = _np.array(
            [
                [
                    ((-alpha_1 - j) * _np.exp(-j * phi_1_ok)) / (_np.sqrt(beta_1)),
                    -_np.sqrt(beta_1) * _np.exp(-j * phi_1_ok),
                    0,
                    0,
                ],
                [
                    -((-alpha_1 + j) * _np.exp(j * phi_1_ok)) / (_np.sqrt(beta_1)),
                    _np.sqrt(beta_1) * _np.exp(j * phi_1_ok),
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    (-alpha_2 - j) * _np.exp(-j * phi_2_ok) / (_np.sqrt(beta_2)),
                    -_np.sqrt(beta_2) * _np.exp(-j * phi_2_ok),
                ],
                [
                    0,
                    0,
                    -(-alpha_2 + j) * _np.exp(j * phi_2_ok) / (_np.sqrt(beta_2)),
                    _np.sqrt(beta_2) * _np.exp(j * phi_2_ok),
                ],
            ],
        )

        u_ = u_ / (_np.sqrt(-2j))

        matrix_row["BETA1"] = beta_1
        matrix_row["BETA2"] = beta_2

        matrix_row["SIGN"] = sign
        matrix_row["ALPHA1"] = alpha_1
        matrix_row["ALPHA2"] = alpha_2

        matrix_row["GAMMA1"] = gamma_1
        matrix_row["GAMMA2"] = gamma_2

        matrix_row["Q1"] = q1
        matrix_row["Q2"] = q2

        matrix_row["PHI1"] = phi_1_ok
        matrix_row["PHI2"] = phi_2_ok

        matrix_row["U_"] = u_
        return matrix_row

    @staticmethod
    def compute_decoupling_matrix_from_eigenvectors(matrix_row: _pd.Series) -> _pd.Series:
        [v1, v1_, v2, v2_] = matrix_row[["v1", "v1_", "v2", "v2_"]]

        # Calcul de la matrice X
        x = _np.array([v1, v1_, v2, v2_]).T
        x = x / (_np.sqrt(-2j))

        # Matrice U^-1 and R
        u_ = matrix_row["U_"]
        r_decoupling = x @ u_
        r_decoupling = _np.real(
            r_decoupling,
        )
        matrix_row["R"] = r_decoupling

        return matrix_row

    @staticmethod
    def compute_decoupled_transfer_matrix(matrix_row: _pd.Series, r_decoupling_0: _np.ndarray) -> _pd.Series:
        matt_i = matrix_row["m_canon"]
        r_decoupling = matrix_row["R"]

        # Decoupled transfer matrix
        p = _np.linalg.inv(r_decoupling) @ matt_i @ r_decoupling_0
        p = _np.real(p)
        matrix_row["p"] = p

        return matrix_row

    @staticmethod
    def compute_phase_advances(matrix_row: _pd.Series, eigvec_init) -> _pd.Series:
        p = matrix_row["p"]
        eigvec_aligned = p @ eigvec_init

        v1_aligned = eigvec_aligned[:, 0]
        v3_aligned = eigvec_aligned[:, 2]

        mu1 = _np.arctan(_np.imag(v1_aligned[0]) / _np.real(v1_aligned[0]))
        mu2 = _np.arctan(_np.imag(v3_aligned[2]) / _np.real(v3_aligned[2]))

        sign = matrix_row["SIGN"]

        matrix_row["MU1"] = mu1 * (-1) ** sign
        if sign == 2 or sign == 3:
            mu2 = -mu2
        matrix_row["MU2"] = mu2

        return matrix_row

    @staticmethod
    def check_tunes(matrix_row: _pd.Series):
        # Check le tune autrement qu'avec les valeurs propres
        mat_tot = matrix_row["m_canon"]

        S2 = _np.array([[0, 1], [-1, 0]])
        t1 = 1 / 2 * (mat_tot[0, 0] + mat_tot[1, 1])
        t2 = 1 / 2 * (mat_tot[2, 2] + mat_tot[3, 3])

        c12 = mat_tot[0:2, 2:4] + -S2 @ _np.transpose(copy.deepcopy(mat_tot[2:4, 0:2])) @ S2
        c12_det = _np.linalg.det(c12) / 4
        cos_mu1 = 1 / 2 * (t1 + t2) + 1 / 2 * _np.sqrt((t1 - t2) ** 2 + 4 * c12_det)
        cos_mu2 = 1 / 2 * (t1 + t2) - 1 / 2 * _np.sqrt((t1 - t2) ** 2 + 4 * c12_det)

        tune1 = _np.arccos(cos_mu1) / (2 * _np.pi)
        tune2 = _np.arccos(cos_mu2) / (2 * _np.pi)
        tune1_bis = matrix_row["MU1"] / (2 * _np.pi)
        tune2_bis = matrix_row["MU2"] / (2 * _np.pi)

        # En supposant qu'on fait le phase_unrolling
        if tune1_bis > 0.5:
            tune1_bis = -(tune1_bis - 1)
        if tune2_bis > 0.5:
            tune2_bis = -(tune2_bis - 1)

        print("tune1 = ", tune1)
        print("tune2 = ", tune2)
        check = round(tune1 - tune1_bis + tune2 - tune2_bis, 3)
        print("check", check)
        return check


EdwardsTengTwiss = Parzen


class LebedevTwiss(Parametrization):
    def __init__(
        self,
        twiss_init: Optional[_BetaBlock] = None,
        with_phase_unrolling: bool = True,
        all_periodic: bool = False,
    ):
        """
        Args:
            twiss_init: the initial values for the Twiss computation (if None, periodic conditions are assumed and the
            Twiss parameters are computed from the transfer matrix).
            with_phase_unrolling: TODO
        """

        self._twiss_init = twiss_init
        self._with_phase_unrolling = with_phase_unrolling
        self._all_periodic = all_periodic

    def __call__(
        self,
        matrix: _pd.DataFrame,
        tracks: _pd.DataFrame,
        kin: _Kinematics,
    ) -> _pd.DataFrame:
        """
        Uses a step-by-step transfer matrix to compute the generalized Twiss parameters (coupled motions)
        with the parametrization of V.A. Lebedev and S.A Bogacz. The phase advances are computed as well.

        Args:
            matrix: the input step-by-step transfer matrix
            tracks: tracks_global for the centered particle 'O' of the BeamTwiss
        Returns:
            the same DataFrame as the matrix input DataFrame, but with added columns for the computed quantities.
        """
        if self._twiss_init is not None:
            twiss_init = self._twiss_init

        else:
            periodic_twiss = self.compute_periodic_LebedevTwiss(
                copy.deepcopy(matrix).iloc[-1].to_frame().T,
                tracks,
                kin,
            )
            periodic_twiss.rename(
                columns={
                    "BETA1X": "BETA11",
                    "BETA2Y": "BETA22",
                    "BETA2X": "BETA21",
                    "BETA1Y": "BETA12",
                    "ALPHA1X": "ALPHA11",
                    "ALPHA2Y": "ALPHA22",
                    "ALPHA2X": "ALPHA21",
                    "ALPHA1Y": "ALPHA12",
                },
                inplace=True,
            )
            periodic_twiss = dict(
                periodic_twiss[
                    [
                        "BETA11",
                        "ALPHA11",
                        "BETA22",
                        "ALPHA22",
                        "MU1",
                        "MU2",
                        "BETA12",
                        "BETA21",
                        "ALPHA12",
                        "ALPHA21",
                        "NU1",
                        "NU2",
                        "U",
                    ]
                ].iloc[0],
            )
            for k in ["BETA11", "BETA22", "BETA12", "BETA21"]:
                periodic_twiss[k] = periodic_twiss[k] * _ureg.m
            twiss_init = _BetaBlock(**periodic_twiss)

        if self._all_periodic:
            matrix = self.compute_periodic_LebedevTwiss(matrix, tracks, kin)

        else:
            matrix["BX"] = tracks["BX"]
            V1 = self.get_initial_normalisation_matix(twiss_init)

            # Calculation of the matrix for the transformation of geometric coordinates into the canonical ones
            matrix = matrix.apply(lambda row: self.compute_canonical_transformation_matrix(row, kin), axis=1)
            matrix_rs1 = matrix.iloc[0]["matrix_rs"]
            matrix = matrix.apply(lambda row: self.compute_canonical_transfer_matrices(row, matrix_rs1), axis=1)

            # Calculation of the the normalisation matrix
            matrix = matrix.apply(lambda row: self.compute_turned_normalisation_matrix(row, V1), axis=1)
            matrix["MU1_BIS"] = matrix["MU1_BIS"] - matrix.iloc[0]["MU1_BIS"]
            matrix["MU2_BIS"] = matrix["MU2_BIS"] - matrix.iloc[0]["MU2_BIS"]
            matrix = matrix.apply(self.compute_normalisation_matrix_from_V2_turned, axis=1)

            # Parametrisation
            # beta, alpha, nu and u
            matrix = matrix.apply(self.compute_parametrisation_from_normalisation_matrix, axis=1)

            # Phase advances
            matrix = matrix.apply(
                lambda row: self.compute_phase_advances(row, matrix.iloc[0]["Normalisation_matrix"]),
                axis=1,
            )

        try:
            from numba import njit

            self.phase_unrolling = njit(self.phase_unrolling)
        except ModuleNotFoundError:
            pass

        if self._with_phase_unrolling:
            matrix["MU1"] = self.phase_unrolling(matrix["MU1"].values, matrix["S"].values)
            matrix["MU2"] = self.phase_unrolling(matrix["MU2"].values, matrix["S"].values)
            matrix["MU1_BIS"] = self.phase_unrolling(matrix["MU1_BIS"].values, matrix["S"].values)
            matrix["MU2_BIS"] = self.phase_unrolling(matrix["MU2_BIS"].values, matrix["S"].values)

        return matrix

    @staticmethod
    def _get_twiss_elements(twiss: Optional[_BetaBlock], block: int = 1) -> Tuple:
        """Extract parameters from the coupled _BetaBlock."""

        v = 1 if block == 1 else 2
        p = 2 if block == 1 else 1

        alpha: float = twiss[f"ALPHA{v}{v}"]
        beta: float = twiss[f"BETA{v}{v}"].m_as("m")
        gamma: float = twiss[f"GAMMA{v}{v}"].m_as("m**-1")

        alpha_: float = twiss[f"ALPHA{v}{p}"]
        beta_: float = twiss[f"BETA{v}{p}"].m_as("m")
        nu: float = twiss[f"NU{v}"]
        u: float = twiss["U"]

        return alpha, beta, gamma, alpha_, beta_, nu, u

    def get_initial_normalisation_matix(self, twiss_init: Optional[_BetaBlock]) -> _np.ndarray:
        alpha_11, beta_11, gamma_11, alpha1y, beta1y, nu1, u = self._get_twiss_elements(twiss_init)
        alpha_22, beta_22, gamma_22, alpha2x, beta2x, nu2, u = self._get_twiss_elements(twiss_init, 2)
        if beta1y == 0.0 or beta2x == 0:
            v = _np.array(
                [
                    [_np.sqrt(beta_11), 0, 0, 0],
                    [-alpha_11 / _np.sqrt(beta_11), (1) / _np.sqrt(beta_11), 0, 0],
                    [0, 0, _np.sqrt(beta_22), 0],
                    [0, 0, -alpha_22 / _np.sqrt(beta_22), (1) / _np.sqrt(beta_22)],
                ],
            )

        else:
            v = _np.array(
                [
                    [_np.sqrt(beta_11), 0, _np.sqrt(beta2x) * _np.cos(nu2), -_np.sqrt(beta2x) * _np.sin(nu2)],
                    [
                        -alpha_11 / _np.sqrt(beta_11),
                        (1 - u) / _np.sqrt(beta_11),
                        (u * _np.sin(nu2) - alpha2x * _np.cos(nu2)) / _np.sqrt(beta2x),
                        (u * _np.cos(nu2) + alpha2x * _np.sin(nu2)) / _np.sqrt(beta2x),
                    ],
                    [_np.sqrt(beta1y) * _np.cos(nu1), -_np.sqrt(beta1y) * _np.sin(nu1), _np.sqrt(beta_22), 0],
                    [
                        (u * _np.sin(nu1) - alpha1y * _np.cos(nu1)) / _np.sqrt(beta1y),
                        (u * _np.cos(nu1) + alpha1y * _np.sin(nu1)) / _np.sqrt(beta1y),
                        -alpha_22 / _np.sqrt(beta_22),
                        (1 - u) / _np.sqrt(beta_22),
                    ],
                ],
            )

        return v

    @staticmethod
    def compute_turned_normalisation_matrix(matrix_row: _pd.Series, v1: _np.ndarray) -> _pd.Series:
        matt_i = matrix_row["m_canon"]
        v2_turned = matt_i @ v1

        phi_v1_2 = _np.arctan(v2_turned[0, 1] / v2_turned[0, 0])
        phi_v2_2 = _np.arctan(v2_turned[2, 3] / v2_turned[2, 2])

        matrix_row["V2_turned"] = v2_turned
        matrix_row["MU1_BIS"] = phi_v1_2
        matrix_row["MU2_BIS"] = phi_v2_2

        return matrix_row

    @staticmethod
    def compute_normalisation_matrix_from_V2_turned(matrix_row: _pd.Series) -> _pd.Series:
        v2_turned = matrix_row["V2_turned"]
        dmu1 = matrix_row["MU1_BIS"]
        dmu2 = matrix_row["MU2_BIS"]

        S = _np.array(
            [
                [_np.cos(dmu1), _np.sin(dmu1), 0, 0],
                [-_np.sin(dmu1), _np.cos(dmu1), 0, 0],
                [0, 0, _np.cos(dmu2), _np.sin(dmu2)],
                [0, 0, -_np.sin(dmu2), _np.cos(dmu2)],
            ],
        )
        v2_ok = v2_turned @ _np.linalg.inv(S)
        matrix_row["Normalisation_matrix"] = v2_ok

        return matrix_row

    @staticmethod
    def compute_normalized_eigenvectors(v1, v1_):
        U = _np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
        ortho1 = v1_.T @ U @ v1
        ratio = -2j / ortho1
        ratio = abs(_np.real(ratio))
        v1 = v1 * _np.sqrt(ratio)
        v1_ = v1_ * _np.sqrt(ratio)
        return v1, v1_

    def compute_orderded_turned_normalized_eigenvectors(
        self,
        eigvec: _np.ndarray = None,
        lambda1_0: float = None,
        eigvals: _np.ndarray = None,
    ):
        [v1, v1_, v2, v2_] = eigvec.T
        lambda1 = eigvals[0]

        # On vérifie qu'on a les vecteurs propres sont bien ordonnés en fonction du mode propre
        if _np.round(_np.real(lambda1), 2) != _np.round(_np.real(lambda1_0), 2):
            v1, v1_, v2, v2_ = v2, v2_, v1, v1_

        v1, v1_ = self.compute_turned_eigvec(v1, v1_)
        v2, v2_ = self.compute_turned_eigvec(v2, v2_, plane=2)

        # On vérifie que u1 et u4 tels que définis dans le papier de Bogacz soient >0
        u1 = -_np.imag(v1[1] * v1[0])
        if u1 < 0:
            v1, v1_ = v1_, v1

        u4 = -_np.imag(v2[3] * v2[2])
        if u4 < 0:
            v2, v2_ = v2_, v2

        # On normalise les vecteurs propres avc la condition donnée dans Bogacz and Lebedev
        v1, v1_ = self.compute_normalized_eigenvectors(v1, v1_)
        v2, v2_ = self.compute_normalized_eigenvectors(v2, v2_)

        return v1, v1_, v2, v2_

    @staticmethod
    def compute_normalisation_matrix_from_eigenvectors(matrix_row: _pd.Series) -> _pd.Series:
        v1 = matrix_row["v1"]
        v2 = matrix_row["v2"]
        v = _np.zeros((4, 4))
        v[:, 0] = _np.real(v1)
        v[:, 1] = -_np.imag(v1)
        v[:, 2] = _np.real(v2)
        v[:, 3] = -_np.imag(v2)

        matrix_row["Normalisation_matrix"] = v
        return matrix_row

    @staticmethod
    def compute_parametrisation_from_normalisation_matrix(matrix_row: _pd.Series) -> _pd.Series:
        v = matrix_row["Normalisation_matrix"]

        # Generalized Twiss parameters alphas and betas from V elements
        # 8 Parameters to describe the 4x4 symplectic normalisation matrix (lattice parameters)
        beta_1x = v[0, 0] ** 2
        beta_2y = v[2, 2] ** 2
        beta_1y = v[2, 0] ** 2 + v[2, 1] ** 2
        beta_2x = v[0, 2] ** 2 + v[0, 3] ** 2

        alpha_1x = -v[1, 0] * v[0, 0]
        alpha_2y = -v[3, 2] * v[2, 2]

        alpha_1y = -(v[3, 0] * v[2, 0] + v[3, 1] * v[2, 1])
        alpha_2x = -(v[1, 2] * v[0, 2] + v[1, 3] * v[0, 3])

        # Other dependent real functions that appears in the parametrization
        u_coupling = 1 - v[0, 0] * v[1, 1]
        u_coupling_bis = 1 - v[2, 2] * v[3, 3]
        if v[2, 0] != 0:
            nu_1 = -_np.arctan(v[2, 1] / v[2, 0])
        else:
            nu_1 = 0

        if v[0, 2] != 0:
            nu_2 = -_np.arctan(v[0, 3] / v[0, 2])
        else:
            nu_2 = 0

        if _np.sign(v[3, 0]) != _np.sign((u_coupling * _np.sin(nu_1) - alpha_1y * _np.cos(nu_1)) / _np.sqrt(beta_1y)):
            nu_1 = _np.pi + nu_1

        if _np.sign(v[1, 2]) != _np.sign(
            (u_coupling_bis * _np.sin(nu_2) - alpha_2x * _np.cos(nu_2)) / _np.sqrt(beta_2x),
        ):
            nu_2 = _np.pi + nu_2

        matrix_row["BETA1X"] = beta_1x
        matrix_row["BETA2X"] = beta_2x
        matrix_row["BETA1Y"] = beta_1y
        matrix_row["BETA2Y"] = beta_2y

        matrix_row["ALPHA1X"] = alpha_1x
        matrix_row["ALPHA2X"] = alpha_2x
        matrix_row["ALPHA1Y"] = alpha_1y
        matrix_row["ALPHA2Y"] = alpha_2y

        matrix_row["NU1"] = nu_1
        matrix_row["NU2"] = nu_2

        matrix_row["U"] = 1 - v[0, 0] * v[1, 1]
        matrix_row["U_BIS"] = 1 - v[2, 2] * v[3, 3]
        matrix_row["U_BIS2"] = v[3, 1] * v[2, 0] - v[3, 0] * v[2, 1]
        matrix_row["U_BIS3"] = v[1, 3] * v[0, 2] - v[1, 2] * v[0, 3]

        return matrix_row

    @staticmethod
    def compute_phase_advances(matrix_row: _pd.Series, initial_normalisation_matrix) -> _pd.Series:
        v1 = initial_normalisation_matrix
        v = matrix_row["Normalisation_matrix"]
        matt_i = matrix_row["m_canon"]
        R = _np.linalg.inv(v) @ matt_i @ v1

        matrix_row["MU1"] = _np.round(_np.arctan(R[0, 1] / R[0, 0]), 8)
        matrix_row["MU2"] = _np.round(_np.arctan(R[2, 3] / R[2, 2]), 8)

        return matrix_row

    @staticmethod
    def compute_phase_advances_bis(matrix_row: _pd.Series, eigvec_init) -> _pd.Series:
        matt_i = matrix_row["m_canon"]
        eigvec_align = matt_i @ eigvec_init

        v1_align = eigvec_align[:, 0]
        phi_aligned = _np.arctan(_np.imag(v1_align[0]) / _np.real(v1_align[0]))
        v2_align = eigvec_align[:, 2]
        phi_aligned_2 = _np.arctan(_np.imag(v2_align[2]) / _np.real(v2_align[2]))

        matrix_row["MU1_BIS"] = -phi_aligned  # Peut-être un petit problème d'arrondis pour la première valeurs
        matrix_row["MU2_BIS"] = -phi_aligned_2

        return matrix_row

    def compute_periodic_LebedevTwiss(
        self,
        matrix: _pd.DataFrame,
        tracks: _pd.DataFrame,
        kin: _Kinematics,
    ) -> _pd.DataFrame:
        """
        Args:
            matrix: the input step-by-step transfer matrix
            tracks: tracks_global for the centered particle 'O' of the BeamTwiss
        Returns:
            the same DataFrame as the matrix input DataFrame, but with added columns for the computed quantities.
        """
        matrix["BX"] = tracks["BX"]
        matrix["BY"] = tracks["BY"]
        matrix["BZ"] = tracks["BZ"]
        matrix["P"] = tracks["P"]
        matrix["T"] = tracks["T"]

        # Calculation of the matrix for the transformation of geometric coordinates into the canonical ones
        matrix = matrix.apply(lambda row: self.compute_canonical_transformation_matrix(row, kin), axis=1)
        matrix_rs1 = matrix.iloc[0]["matrix_rs"]
        matrix = matrix.apply(lambda row: self.compute_canonical_transfer_matrices(row, matrix_rs1), axis=1)

        # Total transfer matrix and one-turn transfer matrices
        mat_tot = matrix.iloc[-1]["m_canon"]  # Seems not symplectic when we take the last transfer matrix (changeref)?
        matrix = matrix.apply(lambda row: self.compute_one_turn_transfer_matrix(row, mat_tot), axis=1)

        # Calculation of the rotated and normalized eigenvectors and the normalisation matrix
        eigvals_init, eigvec_init = _np.linalg.eig(mat_tot)
        lambda1_0 = eigvals_init[0]
        matrix = matrix.apply(lambda row: self.compute_eigenvectors(row, lambda1_0), axis=1)
        matrix = matrix.apply(self.compute_normalisation_matrix_from_eigenvectors, axis=1)

        # Parametrisation
        # beta, alpha, nu and u
        matrix = matrix.apply(self.compute_parametrisation_from_normalisation_matrix, axis=1)

        # Phase advances
        eigvec_init = _np.array(
            [matrix.iloc[0]["v1"], matrix.iloc[0]["v1_"], matrix.iloc[0]["v2"], matrix.iloc[0]["v2"]],
        ).T
        matrix = matrix.apply(lambda row: self.compute_phase_advances_bis(row, eigvec_init), axis=1)
        matrix = matrix.apply(
            lambda row: self.compute_phase_advances(row, matrix.iloc[0]["Normalisation_matrix"]),
            axis=1,
        )

        return matrix


class RipkenTwiss(Parametrization):
    ...


class WolskiTwiss(Parametrization):
    def __init__(self):  # type: ignore[no-untyped-def]
        ...

    def __call__(self):  # type: ignore[no-untyped-def]
        ...
