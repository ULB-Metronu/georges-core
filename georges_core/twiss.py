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
    def __init__(self):
        ...

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
        e = 1.6 * 1e-19 * _ureg.C
        mom = kin.momentum

        # 10 Parameters to describe the 4x4 symplectic transfer matrix
        betas_1x = []
        betas_2x = []
        betas_1y = []
        betas_2y = []

        alphas_1x = []
        alphas_2x = []
        alphas_1y = []
        alphas_2y = []

        mus_1 = []
        mus_2 = []

        # Other dependent real functions that appears in the parametrization
        nus_1 = []
        nus_2 = []

        # Calculation of the matrix for the transformation of geometric coordinates into the canonical ones
        b_s = t['BX'].iloc[-1] * _ureg.kG
        r_s = e * b_s / (mom * _ureg.c)
        r_s = r_s.to('1/(m*c)').magnitude
        matrix_rs_tot = _np.array([[1, 0, 0, 0], [0, 1, -r_s / 2, 0], [0, 0, 1, 0], [r_s / 2, 0, 0, 1]])

        b_s = t['BX'].iloc[0] * _ureg.kG
        r_s = e * b_s / (mom * _ureg.c)
        r_s = r_s.to('1/(m*c)').magnitude
        matrix_rs1 = _np.array([[1, 0, 0, 0], [0, 1, -r_s / 2, 0], [0, 0, 1, 0], [r_s / 2, 0, 0, 1]])

        # Total transfer matrix
        mat = matrix[['R11', 'R12', 'R13', 'R14', 'R15', 'R21', 'R22', 'R23', 'R24',
                      'R25', 'R31', 'R32', 'R33', 'R34', 'R35', 'R41', 'R42', 'R43', 'R44',
                      'R45', 'R51', 'R52', 'R53', 'R54', 'R55']]
        mat_tot = mat.iloc[-1, :]

        matt_tot_geom = _np.array(mat_tot).reshape(5, 5)[:4, :4]
        matt_tot = matrix_rs_tot @ matt_tot_geom @ _np.linalg.inv(matrix_rs1)

        # Step by step generalized twiss parameters calculation
        for i in matrix.index:

            # Transformation matrix from geometric coordinates to canonical ones
            b_s = t['BX'].iloc[i] * _ureg.kG
            r_s = e * b_s / (mom * _ureg.c)
            r_s = r_s.to('1/(m*c)').magnitude
            matrix_rs = _np.array([[1, 0, 0, 0], [0, 1, -r_s / 2, 0], [0, 0, 1, 0], [r_s / 2, 0, 0, 1]])

            mat_i = mat.iloc[i, :]
            matt_i_geom = _np.array(mat_i).reshape(5, 5)[:4, :4]
            matt_i = matrix_rs @ matt_i_geom @ _np.linalg.inv(matrix_rs1)
            m = matt_i @ matt_tot @ _np.linalg.inv(matt_i)

            # Symplectic unit matrix
            U = _np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])

            # Eigenvalues and eigenvectors
            eigvals, eigvec = _np.linalg.eig(m)

            v1 = eigvec[:, 0]
            v1_ = eigvec[:, 1]
            v2 = eigvec[:, 2]
            v2_ = eigvec[:, 3]

            ortho1 = v1_.T @ U @ v1
            ratio = -2j / ortho1
            if _np.real(ratio) < 0:
                v1 = eigvec[:, 1]
                v1_ = eigvec[:, 0]
            ratio = abs(_np.real(ratio))
            v1 = v1 * _np.sqrt(ratio)
            v1_ = v1_ * _np.sqrt(ratio)

            ortho2 = v2_.T @ U @ v2
            ratio = -2j / ortho2
            if _np.real(ratio) < 0:
                v2 = eigvec[:, 3]
                v2_ = eigvec[:, 2]
            ratio = abs(_np.real(ratio))
            v2 = v2 * _np.sqrt(ratio)
            v2_ = v2_ * _np.sqrt(ratio)

            if _np.imag(v1[0]) != 0:
                save = v1
                save_ = v1_
                v1 = v2
                v1_ = v2_
                v2 = save
                v2_ = save_

            # Normalization condition
            cond1 = v1_.T @ U @ v1
            cond2 = v2_.T @ U @ v2
            cond3 = v1.T @ U @ v1
            cond4 = v2.T @ U @ v2
            cond5 = v2.T @ U @ v1
            cond6 = v2_.T @ U @ v1
            #print(cond1)
            #print(cond2)
            #print(cond3)
            #print(cond4)
            #print(cond5)
            #print(cond6)

            # Matrix V, obtained with real and imaginary parts of eigenvectors
            v = _np.zeros((4, 4))
            v[:, 0] = _np.real(v1)
            v[:, 1] = -_np.imag(v1)
            v[:, 2] = _np.real(v2)
            v[:, 3] = -_np.imag(v2)

            if i == 0:
                V1 = v

            # Generalized Twiss parameters alphas and betas from V elements
            beta_1x = v[0, 0] ** 2
            beta_2y = v[2, 2] ** 2
            beta_1y = v[2, 0] ** 2 + v[2, 1] ** 2
            beta_2x = v[0, 2] ** 2 + v[0, 3] ** 2

            alpha_1x = - v[1, 0] * v[0, 0]
            alpha_2y = - v[3, 2] * v[3, 3]
            alpha_1y = -(v[3, 0] * v[2, 0] + v[3, 1] * v[2, 1])
            alpha_2x = -(v[1, 2] * v[0, 2] + v[1, 3] * v[0, 3])

            nu_1 = -_np.arctan(v[2, 1] / v[2, 0])
            nu_2 = -_np.arctan(v[0, 3] / v[0, 2])

            betas_1x.append(beta_1x)
            betas_2x.append(beta_2x)
            betas_1y.append(beta_1y)
            betas_2y.append(beta_2y)

            alphas_1x.append(alpha_1x)
            alphas_2x.append(alpha_2x)
            alphas_1y.append(alpha_1y)
            alphas_2y.append(alpha_2y)

            nus_1.append(nu_1)
            nus_2.append(nu_2)

            # Calculation of phase advances
            R = _np.linalg.inv(v) @ matt_i @ V1
            try:
                mu_1 = _np.arccos(R[0, 0])
                mu1_b = -_np.arcsin(R[1, 0])
                mu_2 = _np.arccos(R[2, 2])
                mu2_b = -_np.arcsin(R[3, 2])
                if mu1_b < 0:
                    mu_1 = _np.pi - mu_1
                if mu2_b < 0:
                    mu_2 = _np.pi - mu_2
                mus_1.append(mu_1)
                mus_2.append(mu_2)
            except ValueError:
                print("erreur")
                mus_1.append(0)
                mus_2.append(0)

        matrix['BETA1X'] = _np.array(betas_1x)
        matrix['BETA2X'] = _np.array(betas_2x)
        matrix['BETA1Y'] = _np.array(betas_1y)
        matrix['BETA2Y'] = _np.array(betas_2y)

        matrix['ALPHA1X'] = _np.array(alphas_1x)
        matrix['ALPHA2X'] = _np.array(alphas_2x)
        matrix['ALPHA1Y'] = _np.array(alphas_1y)
        matrix['ALPHA2Y'] = _np.array(alphas_2y)

        matrix['NU1'] = _np.array(nus_1)
        matrix['NU2'] = _np.array(nus_2)

        matrix['MU1'] = _np.array(mus_1)
        matrix['MU2'] = _np.array(mus_2)

        R = _np.linalg.inv(v) @ matt_tot @ v
        mu1 = _np.arccos(R[0, 0]) / (2 * _np.pi)
        mu2 = _np.arccos(R[2, 2]) / (2 * _np.pi)
        print("mu1 = ", mu1)
        print("mu2 = ", mu2)

        return matrix

