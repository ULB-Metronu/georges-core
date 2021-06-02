from typing import List
import os
import pandas as _pd
from .. import ureg as _ureg

MADNG_TWISS_TABLE_HEADER_ROWS: int = 35
"""MAD-X Twiss table header rows (lines to be skipped when reading the table's content."""

MADNG_TWISS_HEADERS: List[str] = [
    'name', 'kind', 's', 'l',
    'id',
    'x', 'px', 'y', 'py', 't', 'pt', 'pc',
    'slc', 'turn', 'tdir', 'eidx', 'status',
    'alfa11', 'beta11', 'gama11', 'mu1', 'dmu1', 'dx', 'dpx', 'Dx', 'Dpx', 'ddx', 'ddpx', 'wx', 'phix',
    'alfa22', 'beta22', 'gama22', 'mu2', 'dmu2', 'dy', 'dpy', 'Dy', 'Dpy', 'ddy', 'ddpy', 'wy', 'phiy',
    'alfa33', 'beta33', 'gama33', 'mu3', 'dmu3'
]

# MADNG_TWISS_HEADERS: List[str] = [
#     'name', 'kind', 's', 'l',
#     'id',
#     'x', 'px', 'y', 'py', 't', 'pt',
#     'slc', 'turn', 'tdir', 'eidx', 'status',
#     'alfa11', 'beta11', 'gama11', 'mu1', 'dmu1', 'dx', 'dpx', 'ddx', 'ddpx', 'wx', 'phix',
#     'alfa22', 'beta22', 'gama22', 'mu2', 'dmu2', 'dy', 'dpy', 'ddy', 'ddpy', 'wy', 'phiy',
#     'alfa33', 'beta33', 'gama33', 'mu3', 'dmu3'
# ]
"""MAD-X Twiss headers (by default, when no columns are selected)."""


def load_madng_twiss_headers(filename: str = 'twiss.outx', path: str = '.', lines: int = None) -> _pd.Series:
    """

    Args:
        filename: name of the Twiss table file
        path: path to the Twiss table file
        lines: number of lines in the Twiss table file

    Returns:

    """
    # TODO change according to MAD-NG
    lines = lines or MADNG_TWISS_TABLE_HEADER_ROWS
    _ = _pd.read_csv(os.path.join(path, filename),
                     sep=r'\s+',
                     usecols=['KEY', 'VALUE'],
                     squeeze=True,
                     index_col=0,
                     names=['@', 'KEY', '_', 'VALUE'],
                     )[0:lines]
    for c in ('MASS', 'CHARGE', 'ENERGY', 'PC', 'GAMMA', 'KBUNCH', 'BCURRENT', 'SIGE', 'SIGT', 'NPART', 'EX', 'EY',
              'ET', 'BV_FLAG', 'LENGTH', 'ALFA', 'ORBIT5', 'GAMMATR', 'Q1', 'Q2', 'DXMAX', 'DYMAX', 'XCOMAX', 'YCOMAX',
              'BETXMAX', 'BETYMAX', 'XCORMS', 'YCORMS', 'DXRMS', 'DYRMS', 'DELTAP', 'SYNCH_1', 'SYNCH_2', 'SYNCH_3',
              'SYNCH_4', 'SYNCH_5',
              ):
        try:
            _[c] = _pd.to_numeric(_[c])
        except KeyError:
            pass
    return _


def load_madng_twiss_table(filename: str = 'twiss.outx',
                           path: str = '.',
                           columns: List = [],
                           lines: int = None,
                           with_units: bool = True,
                           ) -> _pd.DataFrame:
    """

    Args:
        filename: name of the Twiss table file
        path: path to the Twiss table file
        columns: the list of columns in the Twiss file
        lines: number of lines in the Twiss table file
        with_units:

    Returns:
        A DataFrame representing the Twiss table.
    """
    columns = MADNG_TWISS_HEADERS + columns
    lines = lines or MADNG_TWISS_TABLE_HEADER_ROWS
    _: _pd.DataFrame = _pd \
        .read_csv(os.path.join(path, filename),
                  skiprows=lines,
                  sep=r'\s+',
                  index_col=False,
                  names=columns,
                  ) \
        .drop(0)
    for c in _.columns:
        try:
            _[c] = _pd.to_numeric(_[c])
        except ValueError:
            pass
    if with_units:
        _['l'] = _['l'].apply(lambda e: e * _ureg.m)
        _['e1'] = _['e1'].apply(lambda e: e * _ureg.radian)
        _['e2'] = _['e2'].apply(lambda e: e * _ureg.radian)
        _['angle'] = _['angle'].apply(lambda e: e * _ureg.radian)
        _['k1'] = _['k1'].apply(lambda e: e / _ureg.m ** 2)
        _['k2'] = _['k2'].apply(lambda e: e / _ureg.m ** 3)
        _['k3'] = _['k3'].apply(lambda e: e / _ureg.m ** 4)
        _['k0l'] = _['k0l'].apply(lambda e: e)
        _['k1l'] = _['k1l'].apply(lambda e: e / _ureg.m)
        _['k2l'] = _['k2l'].apply(lambda e: e / _ureg.m ** 2)
        _['k3l'] = _['k3l'].apply(lambda e: e / _ureg.m ** 3)
        _['k4l'] = _['k4l'].apply(lambda e: e / _ureg.m ** 4)
        _['tilt'] = _['tilt'].apply(lambda e: e * _ureg.radian)

    _.rename(columns={'kind': 'keyword', 'name': 'NAME'}, inplace=True)
    return _.set_index('NAME')


def get_twiss_values(table: _pd.DataFrame, location: int = 0) -> _pd.Series:
    """Extract the initial Twiss parameters from a Twiss table

    Args:
        table: a MAD-X twiss table read as a DataFrame
        location: the location at which the parameters need to be extracted

    Returns:
        A Pandas Series containing the extracted Twiss parameters.
    """
    return _pd.Series({
        'MU1': 0,
        'MU2': 0,
        'BETA11': table.iloc[location]['BETX'],
        'BETA22': table.iloc[location]['BETY'],
        'ALPHA11': table.iloc[location]['ALFX'],
        'ALPHA22': table.iloc[location]['ALFY'],
        'GAMMA11': (1 + table.iloc[location]['ALFX'] ** 2) / table.iloc[location]['BETX'],
        'GAMMA22': (1 + table.iloc[location]['ALFY'] ** 2) / table.iloc[location]['BETY'],
        'DY': table.iloc[location]['DX'],
        'DYP': table.iloc[location]['DPX'],
        'DZ': table.iloc[location]['DY'],
        'DZP': table.iloc[location]['DPY'],
    })
