"""
Import sequence that can be generated using MAD-X or MAD-NG.
"""

import os
from typing import Optional

import pandas as _pd

from .. import Q_ as _Q
from .. import ureg as _ureg

MADX_TWISS_TABLE_HEADER_ROWS: int = 47
"""MAD-X Twiss table header rows (lines to be skipped when reading the table's content."""


def load_mad_twiss_headers(filename: str = "twiss.outx", path: str = ".", lines: Optional[int] = None) -> _pd.Series:
    """

    Args:
        filename: name of the Twiss table file
        path: path to the Twiss table file
        lines: number of lines in the Twiss table file

    Returns:

    """
    lines = lines or MADX_TWISS_TABLE_HEADER_ROWS
    _ = _pd.read_csv(
        os.path.join(path, filename),
        sep=r"\s+",
        index_col=0,
        names=["@", "KEY", "_", "VALUE"],
        usecols=["KEY", "VALUE"],
        nrows=lines - 1,
    ).squeeze()
    _.index = list(map(str.upper, _.index))
    for c in _.index:
        try:
            _[c] = _pd.to_numeric(_[c])
        except (KeyError, ValueError):
            pass
    return _


def load_mad_twiss_table(
    filename: str = "twiss.outx",
    path: str = ".",
    lines: Optional[int] = None,
    with_units: bool = True,
) -> _pd.DataFrame:
    """

    Args:
        filename: name of the Twiss table file
        path: path to the Twiss table file
        lines: number of lines in the Twiss table file to skip
        with_units:

    Returns:
        A DataFrame representing the Twiss table.
    """
    lines = lines or MADX_TWISS_TABLE_HEADER_ROWS
    _ = _pd.read_csv(os.path.join(path, filename), skiprows=lines - 1, sep=r"\s+", index_col=False)
    names = _.columns[1:]  # extract the columns in the Twiss table
    _ = _.drop(columns=_.columns[-1], axis=1).drop(index=0, axis=0)
    _.columns = list(map(str.upper, names))

    # TODO check if it is correct, there is two same columns in MAD-NG.
    _ = _.loc[:, ~_.columns.duplicated()]

    for c in _.columns:
        try:
            _[c] = _pd.to_numeric(_[c])
        except ValueError:
            pass
    if "KIND" in _.columns:
        _.rename(columns={"KIND": "KEYWORD"}, inplace=True)
    _["KEYWORD"] = _["KEYWORD"].apply(lambda e: e.upper())

    # Compute strength of magnetics elements K1, K2, K3
    idx = _.query("L > 0").index
    if "K1" not in _.columns:
        _["K1"] = 0
        _.loc[idx, "K1"] = _.loc[idx, "K1L"].values / _.loc[idx, "L"].values
    if "K2" not in _.columns:
        _["K2"] = 0
        _.loc[idx, "K2"] = _.loc[idx, "K2L"].values / _.loc[idx, "L"].values
    if "K3" not in _.columns:
        _["K3"] = 0
        _.loc[idx, "K3"] = _.loc[idx, "K3L"].values / _.loc[idx, "L"].values

    if with_units:

        def set_unit(df: _pd.DataFrame, column: str, unit: _Q) -> _pd.DataFrame:
            try:
                df[column] = df[column].apply(lambda e: e * unit)
            except KeyError:
                pass
            return df

        _ = set_unit(_, "S", _ureg.m)
        _ = set_unit(_, "L", _ureg.m)
        _ = set_unit(_, "E1", _ureg.radians)
        _ = set_unit(_, "E2", _ureg.radians)
        _ = set_unit(_, "ANGLE", _ureg.radians)
        _ = set_unit(_, "K1", _ureg.m**-2)
        _ = set_unit(_, "K2", _ureg.m**-3)
        _ = set_unit(_, "K3", _ureg.m**-4)
        _ = set_unit(_, "K4", _ureg.m**-5)
        _ = set_unit(_, "K0L", _ureg(""))
        _ = set_unit(_, "K1L", _ureg.m**-1)
        _ = set_unit(_, "K2L", _ureg.m**-2)
        _ = set_unit(_, "K3L", _ureg.m**-3)
        _ = set_unit(_, "K4L", _ureg.m**-4)
        _ = set_unit(_, "TILT", _ureg.radians)
        _ = set_unit(_, "HGAP", _ureg.m)

    _.rename(columns={"KIND": "KEYWORD"}, inplace=True)
    return _.set_index("NAME")


def get_twiss_values(table: _pd.DataFrame, element: str = "$start") -> _pd.Series:
    """Extract the initial Twiss parameters from a Twiss table

    Args:
        table: a MAD twiss table read as a DataFrame
        element: the name of the element at which the parameters need to be extracted

    Returns:
        A Pandas Series containing the extracted Twiss parameters.
    """
    return _pd.Series(
        {
            "MU1": table.loc[element]["MU1"],
            "MU2": table.loc[element]["MU2"],
            "BETA11": table.loc[element]["BETA11"],
            "BETA22": table.loc[element]["BETA22"],
            "ALPHA11": table.loc[element]["ALFA11"],
            "ALPHA22": table.loc[element]["ALFA22"],
            "GAMMA11": (1 + table.loc[element]["ALFA11"] ** 2) / table.loc[element]["BETA11"],
            "GAMMA22": (1 + table.loc[element]["ALFA22"] ** 2) / table.loc[element]["BETA22"],
            "DY": table.loc[element]["DX"],
            "DYP": table.loc[element]["DPX"],
            "DZ": table.loc[element]["DY"],
            "DZP": table.loc[element]["DPY"],
        },
    )
