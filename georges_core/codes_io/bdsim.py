"""
TODO
"""
from typing import Mapping
import os
from .. import ureg as _ureg
import pandas as _pd
import georges.bdsim
from .. import particles as _particles
from ..kinematics import Kinematics as _Kinematics


_BDSIM_TO_MAD_CONVENTION: Mapping[str, str] = {
    'Rcol': 'RectangularCollimator',
    'Ecol': 'EllipticalCollimator',
}


def load_bdsim_model(filename: str, path: str = '.', with_units: bool = True) -> _pd.DataFrame:
    """

    Args:
        filename:
        path:
        with_units:

    Returns:

    """
    _: _pd.DataFrame = georges.bdsim.BDSimOutput(os.path.join(path, filename)).model.df

    for c in _.columns:
        try:
            _[c] = _[c].apply(float)
        except ValueError:
            pass

    _['CLASS'] = _['TYPE'].apply(str.capitalize)
    _['CLASS'] = _['CLASS'].apply(lambda e: _BDSIM_TO_MAD_CONVENTION.get(e, e))

    ## TMP
    for i, line in _.iterrows():
        if line["CLASS"] == "RectangularCollimator" or line["CLASS"] == "Dump":
            _.at[i, "APERTYPE"] = "rectangular"
        if line["CLASS"] == "EllipticalCollimator":
            _.at[i, "APERTYPE"] = "elliptical"

    if with_units:
        _['L'] = _['L'].apply(lambda e: e * _ureg.m)
        _['APERTURE1'] = _['APERTURE1'].apply(lambda e: e * _ureg.m)
        _['APERTURE2'] = _['APERTURE2'].apply(lambda e: e * _ureg.m)
        _['APERTURE'] = _[['APERTURE1', 'APERTURE2']].apply(list, axis=1)
        _['K1'] = _['K1'].apply(lambda e: e * _ureg.m ** -2)
        _['K1S'] = _['K1S'].apply(lambda e: e * _ureg.m ** -2)
        _['K2'] = _['K2'].apply(lambda e: e * _ureg.m ** -3)
        _['E1'] = _['E1'].apply(lambda e: e * _ureg.radian)
        _['E2'] = _['E2'].apply(lambda e: e * _ureg.radian)
        _['HGAP'] = _['HGAP'].apply(lambda e: e * _ureg.meter)
        _['TILT'] = _['TILT'].apply(lambda e: e * _ureg.radian)
        _['B'] = _['B'].apply(lambda e: e * _ureg.T)

    return _


def load_bdsim_kinematics(filename: str, path: str = '.') -> _Kinematics:
    """

    Args:
        filename:
        path:

    Returns:

    """
    _: _pd.DataFrame = georges.bdsim.BDSimOutput(os.path.join(path, filename)).beam.df
    particle_name = _["particleName"].values[0].capitalize()
    p = getattr(_particles, particle_name if particle_name != 'Default' else 'Proton')
    return _Kinematics(_["E0"].values[0] * _ureg.GeV, kinetic=False, particle=p)


def load_bdsim_beam_distribution(filename: str, path: str = '.') -> _pd.DataFrame:
    """

    Args:
        filename:
        path:
        with_units:

    Returns:

    """
    beam_distribution: _pd.DataFrame = georges.bdsim.BDSimOutput(os.path.join(path, filename)).event.primary.df

    for c in beam_distribution.columns:
        try:
            beam_distribution[c] = beam_distribution[c].apply(float)
        except ValueError:
            pass

    beam_distribution['kineticEnergy'] = beam_distribution['kineticEnergy'].apply(lambda e: e * _ureg.GeV)
    beam_distribution['p'] = beam_distribution['kineticEnergy'].apply(
        lambda e: _Kinematics(e, kinetic=True).to_momentum()
    )
    beam_distribution.rename(columns={
        "xp": "px",
        "yp": "py"
    }, inplace=True)
    beam_distribution.columns = map(str.upper, beam_distribution.columns)
    return beam_distribution
