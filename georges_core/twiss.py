"""Module for the computation of Twiss parametrizations from transfer matrices.

The standard uncoupled Twiss parametrization (including off-momentum effects, aka. dispersion) is the default option.
Additional formalisms for the parametrization of fully coupled transfer matrices are also available (Teng, Ripken,
etc.).
"""
from typing import Optional, Tuple, Union
from logging import warning
import warnings
import numpy as _np
import pandas as _pd
from . import ureg as _ureg
from .sequences import BetaBlock as _BetaBlock
from . import Kinematics as _Kinematics
import cmath

def _get_matrix_elements_block(m: _pd.DataFrame, twiss: Optional[_BetaBlock], block: int = 1) -> Tuple:
    """Extract parameters from the DataFrame."""
    p = 1 if block == 1 else 3
    v = 1 if block == 1 else 2
    r11: _pd.Series = m[f"R{p}{p}"]
    r12: _pd.Series = m[f"R{p}{p + 1}"]
    r21: _pd.Series = m[f"R{p + 1}{p}"]
    r22: _pd.Series = m[f"R{p + 1}{p + 1}"]
    if twiss is not None:
        alpha: float = twiss[f"ALPHA{v}{v}"]
        beta: float = twiss[f"BETA{v}{v}"].m_as('m')
        gamma: float = twiss[f"GAMMA{v}{v}"].m_as('m**-1')
        return r11, r12, r21, r22, alpha, beta, gamma
    else:
        return r11, r12, r21, r22


class ParametrizationType(type):
    pass


class Parametrization(metaclass=ParametrizationType):
    pass


class Twiss(Parametrization):
    def __init__(self,
                 twiss_init: Optional[_BetaBlock] = None,
                 with_phase_unrolling: bool = True):
        """

        Args:
            twiss_init: the initial values for the Twiss computation (if None, periodic conditions are assumed and the
            Twiss parameters are computed from the transfer matrix).
            with_phase_unrolling: TODO
        """
        self._twiss_init = twiss_init
        self._with_phase_unrolling = with_phase_unrolling

    def __call__(self,
                 matrix: _pd.DataFrame,
                 end: Union[int, str] = -1
                 ) -> _pd.DataFrame:
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

        matrix['BETA11'] = self.compute_beta_from_matrix(matrix, twiss_init)
        matrix['BETA22'] = self.compute_beta_from_matrix(matrix, twiss_init, plane=2)
        matrix['ALPHA11'] = self.compute_alpha_from_matrix(matrix, twiss_init)
        matrix['ALPHA22'] = self.compute_alpha_from_matrix(matrix, twiss_init, plane=2)
        matrix['GAMMA11'] = self.compute_gamma_from_matrix(matrix, twiss_init)
        matrix['GAMMA22'] = self.compute_gamma_from_matrix(matrix, twiss_init, plane=2)
        matrix['MU1'] = self.compute_mu_from_matrix(matrix, twiss_init)
        matrix['MU2'] = self.compute_mu_from_matrix(matrix, twiss_init, plane=2)
        matrix['DET1'] = self.compute_jacobian_from_matrix(matrix)
        matrix['DET2'] = self.compute_jacobian_from_matrix(matrix, plane=2)
        matrix['DISP1'] = self.compute_dispersion_from_matrix(matrix, twiss_init)
        matrix['DISP2'] = self.compute_dispersion_prime_from_matrix(matrix, twiss_init)
        matrix['DISP3'] = self.compute_dispersion_from_matrix(matrix, twiss_init, plane=2)
        matrix['DISP4'] = self.compute_dispersion_prime_from_matrix(matrix, twiss_init, plane=2)

        def phase_unrolling(phi):
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
        except ModuleNotFoundError:
            pass

        if self._with_phase_unrolling:
            matrix['MU1U'] = phase_unrolling(matrix['MU1'].values)
            matrix['MU2U'] = phase_unrolling(matrix['MU2'].values)

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
        r11, r12, r21, r22, alpha, beta, gamma = _get_matrix_elements_block(m, twiss, plane)
        return -r11 * r21 * beta + (r11 * r22 + r12 * r21) * alpha - r12 * r22 * gamma

    @staticmethod
    def compute_beta_from_matrix(m: _pd.DataFrame, twiss: _BetaBlock, plane: int = 1,
                                 strict: bool = False) -> _pd.Series:
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
        r11, r12, r21, r22, alpha, beta, gamma = _get_matrix_elements_block(m, twiss, plane)
        _ = r11 ** 2 * beta - 2.0 * r11 * r12 * alpha + r12 ** 2 * gamma
        if strict:
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
        r11, r12, r21, r22, alpha, beta, gamma = _get_matrix_elements_block(m, twiss, plane)
        return r21 ** 2 * beta - 2.0 * r21 * r22 * alpha + r22 ** 2 * gamma

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
        r11, r12, r21, r22, alpha, beta, gamma = _get_matrix_elements_block(m, twiss, plane)
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
        r11, r12, r21, r22 = _get_matrix_elements_block(m, None, plane)
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
            d0 = twiss['DISP1'].m_as('m')
            dp0 = twiss['DISP2'].m_as('radians')
        else:
            d0 = twiss['DISP3'].m_as('m')
            dp0 = twiss['DISP4'].m_as('radians')
        r11: _pd.Series = m[f"R{p}{p}"]
        r12: _pd.Series = m[f"R{p}{p + 1}"]
        r15: _pd.Series = m[f"R{p}5"]
        return d0 * r11 + dp0 * r12 + r15

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
            d0 = twiss['DISP1'].m_as('m')
            dp0 = twiss['DISP2'].m_as('radians')
        else:
            d0 = twiss['DISP3'].m_as('m')
            dp0 = twiss['DISP4'].m_as('radians')
        r21: _pd.Series = m[f"R{p + 1}{p}"]
        r22: _pd.Series = m[f"R{p + 1}{p + 1}"]
        r25: _pd.Series = m[f"R{p + 1}5"]
        return d0 * r21 + dp0 * r22 + r25

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
        twiss = dict({
            'CMU1': (m['R11'] + m['R22']) / 2.0,
            'CMU2': (m['R33'] + m['R44']) / 2.0,
        })
        if twiss['CMU1'] < -1.0 or twiss['CMU1'] > 1.0:
            warning(f"Horizontal motion is unstable; proceed with caution (cos(mu) = {twiss['CMU1']}).")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            twiss['MU1'] = _np.arccos(twiss['CMU1'])
        if twiss['CMU2'] < -1.0 or twiss['CMU2'] > 1.0:
            warning(f"Vertical motion is unstable; proceed with caution (cos(mu) = {twiss['CMU2']}).")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            twiss['MU2'] = _np.arccos(twiss['CMU2'])
        twiss['BETA11'] = m['R12'] / _np.sin(twiss['MU1']) * _ureg.m
        if twiss['BETA11'] < 0.0:
            twiss['BETA11'] *= -1
            twiss['MU1'] *= -1
        twiss['BETA22'] = m['R34'] / _np.sin(twiss['MU2']) * _ureg.m
        if twiss['BETA22'] < 0.0:
            twiss['BETA22'] *= -1
            twiss['MU2'] *= -1
        twiss['ALPHA11'] = (m['R11'] - m['R22']) / (2.0 * _np.sin(twiss['MU1']))
        twiss['ALPHA22'] = (m['R33'] - m['R44']) / (2.0 * _np.sin(twiss['MU2']))
        twiss['GAMMA11'] = -m['R21'] / _np.sin(twiss['MU1']) * _ureg.m ** -1
        twiss['GAMMA22'] = -m['R43'] / _np.sin(twiss['MU2']) * _ureg.m ** -1
        m44 = m[['R11', 'R12', 'R13', 'R14',
                 'R21', 'R22', 'R23', 'R24',
                 'R31', 'R32', 'R33', 'R34',
                 'R41', 'R42', 'R43', 'R44']].apply(float).values.reshape(4, 4)
        r6 = m[['R15', 'R25', 'R35', 'R45']].apply(float).values.reshape(4, 1)
        disp = _np.dot(_np.linalg.inv(_np.identity(4) - m44), r6).reshape(4)
        twiss['DY'] = disp[0] * _ureg.m
        twiss['DYP'] = disp[1] * _ureg.radians
        twiss['DZ'] = disp[2] * _ureg.m
        twiss['DZP'] = disp[3] * _ureg.radians
        twiss['DISP1'] = twiss['DY']
        twiss['DISP2'] = twiss['DYP']
        twiss['DISP3'] = twiss['DZ']
        twiss['DISP4'] = twiss['DZP']

        return _BetaBlock(**twiss)


class TengEdwardsTwiss(Parametrization):
    ...


class RipkenTwiss(Parametrization):
    ...


class WolskiTwiss(Parametrization):
    def __init__(self):
        ...

    def __call__(self):
        ...


class LebedevTwiss(Parametrization):
    def __init__(self,
                 with_phase_unrolling: bool = True):

        self._with_phase_unrolling = with_phase_unrolling

    def __call__(self,
                 matrix: _pd.DataFrame,
                 t: _pd.DataFrame,
                 kin: _Kinematics) -> _pd.DataFrame:
        """
        Uses a step-by-step transfer matrix to compute the generalized Twiss parameters (coupled motions)
        with the parametrization of V.A. Lebedev and S.A Bogacz. The phase advances are computed as well.

        Args:
            matrix: the input step-by-step transfer matrix
            t: tracks_global for the centered particle 'O' of the BeamTwiss
        Returns:
            the same DataFrame as the matrix input DataFrame, but with added columns for the computed quantities.
        """
        matrix['BX'] = t['BX']
        matrix['BY'] = t['BY']
        matrix['BZ'] = t['BZ']
        matrix['P'] = t['P']
        matrix['T'] = t['T']

        # Calculation of the matrix for the transformation of geometric coordinates into the canonical ones
        matrix = matrix.apply(lambda row: self.compute_canonical_transformation_matrix(row, kin), axis=1)
        matrix_rs1 = matrix.iloc[0]["matrix_rs"]
        matrix = matrix.apply(lambda row: self.compute_canonical_transfer_matrices(row, matrix_rs1), axis=1)

        # Total transfer matrix and one-turn transfer matrices
        mat_tot = matrix.iloc[-1][
            'm_canon']  # Seems not symplectic when we take the last transfer matrix (changeref)?
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
            [matrix.iloc[0]['v1'], matrix.iloc[0]['v1_'], matrix.iloc[0]['v2'], matrix.iloc[0]['v2']]).T
        matrix = matrix.apply(lambda row: self.compute_phase_advances_bis(row, eigvec_init), axis=1)
        matrix = matrix.apply(lambda row: self.compute_phase_advances(row, matrix.iloc[0]['Normalisation_matrix']),
                              axis=1)

        def phase_unrolling(phi):
            """TODO"""
            if phi[0] < 0:
                phi[0] += 1 * _np.pi
            for i in range(1, phi.shape[0]):
                if phi[i] < 0:
                    phi[i] += 1 * _np.pi
                if phi[i - 1] - phi[i] > 0.5:
                    phi[i:] += 1 * _np.pi
            return phi

        try:
            from numba import njit
            phase_unrolling = njit(phase_unrolling)
        except ModuleNotFoundError:
            pass

        if self._with_phase_unrolling:
            matrix['MU1'] = phase_unrolling(matrix['MU1'].values)
            matrix['MU2'] = phase_unrolling(matrix['MU2'].values)
            matrix['MU1_BIS'] = phase_unrolling(matrix['MU1_BIS'].values)
            matrix['MU2_BIS'] = phase_unrolling(matrix['MU2_BIS'].values)

        return matrix

    @staticmethod
    def get_B_rotated(t):
        frame = georges_core.frame.Frame()
        frame_rotated = frame.rotate([0.0 * _.rad, _np.arctan(t['P']) * _.rad, _np.arctan(-t['T']) * _.rad])
        element_rotation = frame_rotated.get_rotation_matrix()
        t['SREF_'] = 1.0
        return _np.dot(element_rotation, t[['BX', 'BY', 'BZ']].values.T)

    def compute_canonical_transformation_matrix(self, matrix_row: _pd.Series, kin: _Kinematics) -> _pd.Series:
        # b_s = matrix_row['BX'] * _ureg.kG
        B_s_bis = self.get_B_rotated(matrix_row)
        b_s = B_s_bis[0] * _ureg.kG
        # print(b_s)
        r_s = b_s / kin.brho
        r_s = r_s.to('1/(m)').magnitude
        matrix_rs = _np.array([[1, 0, 0, 0], [0, 1, -r_s / 2, 0], [0, 0, 1, 0], [r_s / 2, 0, 0, 1]])
        matrix_row["matrix_rs"] = matrix_rs
        matrix_row["RS"] = r_s
        matrix_row["BS"] = b_s.magnitude
        return matrix_row

    @staticmethod
    def compute_canonical_transfer_matrices(matrix_row: _pd.Series, matrix_rs1: _np.ndarray) -> _pd.Series:
        mat = matrix_row[['R11', 'R12', 'R13', 'R14',
                          'R21', 'R22', 'R23', 'R24',
                          'R31', 'R32', 'R33', 'R34',
                          'R41', 'R42', 'R43', 'R44']].apply(float).values.reshape(4, 4)

        matrix_rs = matrix_row['matrix_rs']
        m_canon = matrix_rs @ mat @ _np.linalg.inv(matrix_rs1)
        matrix_row['m_canon'] = m_canon

        # Pour DEBUG : à enlever après
        U = _np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
        U_1 = m_canon.T @ U @ m_canon  #
        U_2 = mat.T @ U @ mat  #
        matrix_row['U_canon'] = U_1  #
        matrix_row['U_geom'] = U_2  #

        return matrix_row

    @staticmethod
    def compute_one_turn_transfer_matrix(matrix_row: _pd.Series, mat_tot: _np.ndarray) -> _pd.Series:
        m_i = matrix_row['m_canon']
        m = m_i @ mat_tot @ _np.linalg.inv(m_i)
        matrix_row['m'] = m

        # Pour DEBUG : à enlever après
        U = _np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
        U_3 = m.T @ U @ m  #
        matrix_row['U_m'] = U_3  #

        return matrix_row

    @staticmethod
    def compute_turned_eigvec(v1: _np.ndarray, v1_: _np.ndarray, plane: int = 1):
        j = 1j

        phi_v1 = _np.arctan(_np.imag(v1[plane * 2 - 2]) / _np.real(v1[plane * 2 - 2]))
        theta_1 = - phi_v1
        v1 = v1 * (_np.cos(theta_1) + j * _np.sin(theta_1))
        v1_ = v1_ * (_np.cos(theta_1) - j * _np.sin(theta_1))

        if (_np.real(v1[plane * 2 - 2] < 0)):  # Permet d'assurer le signe du beta pour la propagation
            v1 = v1 * (_np.cos(_np.pi) + j * _np.sin(_np.pi))
            v1_ = v1_ * (_np.cos(_np.pi) - j * _np.sin(_np.pi))

        return v1, v1_

    @staticmethod
    def compute_normalized_eigenvectors(v1, v1_):
        U = _np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
        ortho1 = v1_.T @ U @ v1
        ratio = -2j / ortho1
        ratio = abs(_np.real(ratio))
        v1 = v1 * _np.sqrt(ratio)
        v1_ = v1_ * _np.sqrt(ratio)
        return v1, v1_

    def compute_orderded_turned_normalized_eigenvectors(self, eigvec: _np.ndarray, lambda1: float, lambda1_0: float):
        [v1, v1_, v2, v2_] = eigvec.T

        # On vérifie qu'on a les vecteurs propres sont bien ordonnés en fonction du mode propre
        if (_np.round(_np.real(lambda1), 2) != _np.round(_np.real(lambda1_0), 2)):
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

    def compute_eigenvectors(self, matrix_row: _pd.Series, lambda1_0) -> _pd.Series:
        U = _np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])

        eigvals, eigvec = _np.linalg.eig(matrix_row['m'])
        lambda1 = eigvals[0]
        v1, v1_, v2, v2_ = self.compute_orderded_turned_normalized_eigenvectors(eigvec, lambda1_0, lambda1)
        matrix_row['v1'] = v1
        matrix_row['v2'] = v2
        matrix_row['v1_'] = v1_
        matrix_row['v2_'] = v2_

        cond5 = v2.T @ U @ v1
        cond6 = v2_.T @ U @ v1

        matrix_row['COND5'] = cmath.polar(cond5)[0]
        matrix_row['COND6'] = cmath.polar(cond6)[0]

        return matrix_row

    @staticmethod
    def compute_normalisation_matrix_from_eigenvectors(matrix_row: _pd.Series) -> _pd.Series:
        v1 = matrix_row['v1']
        v2 = matrix_row['v2']
        v = _np.zeros((4, 4))
        v[:, 0] = _np.real(v1)
        v[:, 1] = -_np.imag(v1)
        v[:, 2] = _np.real(v2)
        v[:, 3] = -_np.imag(v2)

        matrix_row['Normalisation_matrix'] = v
        return matrix_row

    @staticmethod
    def compute_parametrisation_from_normalisation_matrix(matrix_row: _pd.Series) -> _pd.Series:
        v = matrix_row['Normalisation_matrix']

        # Generalized Twiss parameters alphas and betas from V elements
        # 8 Parameters to describe the 4x4 symplectic normalisation matrix (lattice parameters)
        beta_1x = v[0, 0] ** 2
        beta_2y = v[2, 2] ** 2
        beta_1y = v[2, 0] ** 2 + v[2, 1] ** 2
        beta_2x = v[0, 2] ** 2 + v[0, 3] ** 2

        alpha_1x = - v[1, 0] * v[0, 0]
        alpha_2y = - v[3, 2] * v[2, 2]

        alpha_1y = -(v[3, 0] * v[2, 0] + v[3, 1] * v[2, 1])
        alpha_2x = -(v[1, 2] * v[0, 2] + v[1, 3] * v[0, 3])

        # Other dependent real functions that appears in the parametrization
        u_coupling = 1 - v[0, 0] * v[1, 1]
        u_coupling_bis = 1 - v[2, 2] * v[3, 3]
        nu_1 = -_np.arctan(v[2, 1] / v[2, 0])
        nu_2 = -_np.arctan(v[0, 3] / v[0, 2])

        if _np.sign(v[3, 0]) != _np.sign((u_coupling * _np.sin(nu_1) - alpha_1y * _np.cos(nu_1)) / _np.sqrt(beta_1y)):
            nu_1 = _np.pi + nu_1

        if _np.sign(v[1, 2]) != _np.sign(
                (u_coupling_bis * _np.sin(nu_2) - alpha_2x * _np.cos(nu_2)) / _np.sqrt(beta_2x)):
            nu_2 = _np.pi + nu_2

        matrix_row['BETA1X'] = beta_1x
        matrix_row['BETA2X'] = beta_2x
        matrix_row['BETA1Y'] = beta_1y
        matrix_row['BETA2Y'] = beta_2y

        matrix_row['ALPHA1X'] = alpha_1x
        matrix_row['ALPHA2X'] = alpha_2x
        matrix_row['ALPHA1Y'] = alpha_1y
        matrix_row['ALPHA2Y'] = alpha_2y

        matrix_row['NU1'] = nu_1
        matrix_row['NU2'] = nu_2

        matrix_row['U'] = 1 - v[0, 0] * v[1, 1]
        matrix_row['U_BIS'] = 1 - v[2, 2] * v[3, 3]
        matrix_row['U_BIS2'] = v[3, 1] * v[2, 0] - v[3, 0] * v[2, 1]
        matrix_row['U_BIS3'] = v[1, 3] * v[0, 2] - v[1, 2] * v[0, 3]

        return matrix_row

    @staticmethod
    def compute_phase_advances_bis(matrix_row: _pd.Series, eigvec_init) -> _pd.Series:
        matt_i = matrix_row['m_canon']
        eigvec_align = matt_i @ eigvec_init

        v1_align = eigvec_align[:, 0]
        phi_aligned = _np.arctan(_np.imag(v1_align[0]) / _np.real(v1_align[0]))
        v2_align = eigvec_align[:, 2]
        phi_aligned_2 = _np.arctan(_np.imag(v2_align[2]) / _np.real(v2_align[2]))

        matrix_row['MU1_BIS'] = - phi_aligned  # Peut-être un petit problème d'arrondis pour la première valeurs
        matrix_row['MU2_BIS'] = - phi_aligned_2

        return matrix_row

    @staticmethod
    def compute_phase_advances(matrix_row: _pd.Series, initial_normalisation_matrix) -> _pd.Series:
        v1 = initial_normalisation_matrix
        v = matrix_row['Normalisation_matrix']
        matt_i = matrix_row['m_canon']
        R = _np.linalg.inv(v) @ matt_i @ v1

        matrix_row['MU1'] = _np.round(_np.arctan(R[0, 1] / R[0, 0]), 8)
        matrix_row['MU2'] = _np.round(_np.arctan(R[2, 3] / R[2, 2]), 8)

        return matrix_row
