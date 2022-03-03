"""High-level interface for Zgoubi using sequences.

"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, List, Tuple, Mapping, Union, Dict
from dataclasses import dataclass
import numpy as _np
import pandas as _pd
import logging
from itertools import compress
import os
import copy
import pandas as pd
from .. import particles as _particles
from ..particles import Proton as _Proton
from ..kinematics import Kinematics as _Kinematics
from .elements import Element as _Element
from .elements import ElementClass as _ElementClass
from .betablock import BetaBlock as _BetaBlock
from ..codes_io import load_mad_twiss_table, load_mad_twiss_headers, \
    load_transport_input_file, transport_element_factory, \
    csv_element_factory
from ..distribution import Distribution as _Distribution
from .. import ureg as _ureg
from .. import Q_ as _Q
from ..frame import Frame

if TYPE_CHECKING:
    from ..particles import ParticuleType as _ParticuleType

__all__ = ['SequenceException',
           'SequenceMetadata',
           'Sequence',
           'PlacementSequence',
           'TwissSequence',
           'SurveySequence',
           'TransportSequence',
           'BDSIMSequence',
           ]

_BDSIM_TO_MAD_CONVENTION: Mapping[str, str] = {
    'Rcol': 'RectangularCollimator',
    'Ecol': 'EllipticalCollimator',
}


class SequenceException(Exception):
    """Exception raised for errors when using zgoubidoo.Sequence"""

    def __init__(self, m):
        self.message = m


class SequenceMetadataType(type):
    """TODO"""
    pass


@dataclass
class SequenceMetadata(metaclass=SequenceMetadataType):
    """TODO"""
    data: _pd.Series = None
    kinematics: _Kinematics = None
    particle: _ParticuleType = _Proton
    n_particles: int = 1

    def __getitem__(self, item):
        return self.data[item]

    def __post_init__(self):
        # Try to infer the particle type from the metadata
        if self.data is None:
            return
        try:
            self.particle = self.particle or getattr(_particles, str(self.data['PARTICLE'].capitalize()))
        except KeyError:
            self.particle = _Proton

        # Try to infer the kinematics from the metadata
        try:
            self.kinematics = self.kinematics or _Kinematics(self.data['PC'] * _ureg.GeV_c,
                                                             particle=self.particle)
        except KeyError:
            pass
        try:
            self.kinematics = self.kinematics or _Kinematics(self.data['ENERGY'] * _ureg.GeV,
                                                             particle=self.particle)
        except KeyError:
            pass
        try:
            self.kinematics = self.kinematics or _Kinematics(self.data['GAMMA'],
                                                             particle=self.particle)
        except KeyError:
            pass

        try:
            self.n_particles = self.n_particles or int(self.data['NPART'])
        except KeyError:
            pass


class SequenceType(type):
    """TODO"""
    pass


class Sequence(metaclass=SequenceType):
    """Sequence.

    """

    def __init__(self,
                 name: str = '',
                 data=None,
                 metadata: Optional[SequenceMetadata] = None,
                 element_keys: Optional[Mapping[str, str]] = None,
                 ):
        """

        Args:
            name: the name of the physics
            data:
            metadata:
            element_keys:
        """
        self._name: str = name
        self._data: Any = data
        self._metadata = metadata or SequenceMetadata()
        self._element_keys = element_keys or {
            k: k for k in [
                'L',
            ]
        }

    def __repr__(self):
        return repr(self._data)

    @property
    def name(self) -> str:
        """Provides the name of the sequence."""
        return self._name

    @property
    def metadata(self) -> SequenceMetadata:
        """Provides the metadata associated with the sequence."""
        return self._metadata

    @property
    def kinematics(self) -> _Kinematics:
        """Provides the kinematics data associated with the sequence metadata."""
        return self.metadata.kinematics

    @property
    def particle(self) -> _ParticuleType:
        """Provides the particle type associated with the sequence metadata."""
        return self.metadata.particle

    @property
    def betablock(self) -> _BetaBlock:
        """TODO"""
        return _BetaBlock()

    def set_parameters(self, element: str, parameters: Dict):
        if isinstance(self._data, _pd.DataFrame):
            self._data.loc[element, parameters.keys()] = parameters.values()
        else:
            for el in self._data:
                if el[0]['NAME'] == element:
                    for param in parameters.keys():
                        el[0][param] = parameters[param]

    def set_position(self, elements: str, value: _Q):
        """
        Set a new position of the center for an element. The parameter at_entry and at_exit are re-computed.
        Args:
            elements:
            value:

        Returns:

        """
        for k, el in enumerate(self._data):
            if el[0]['NAME'] == elements:
                at = list(el[0:4])
                at[2] = value
                at[1] = at[2] - 0.5 * at[0]['L']
                at[3] = at[2] + 0.5 * at[0]['L']
                self._data[k] = tuple(at)

    def get_parameters(self, element: str, parameters: List):
        if isinstance(self._data, _pd.DataFrame):
            return dict(self._data.loc[element, parameters])
        else:
            for el in self._data:
                if el[0]['NAME'] == element:
                    return dict(zip(parameters, list(map(el[0].get, parameters))))

    def to_df(self, df: Optional[_pd.DataFrame] = None, strip_units: bool = False) -> _pd.DataFrame:
        """TODO"""
        if self._data is None and df is None:
            return _pd.DataFrame()
        else:
            df = df if df is not None else _pd.DataFrame(self._data)
            if strip_units:
                def safe_convert(unit: str):
                    def do(_):
                        if _np.isnan(_):
                            return _
                        else:
                            return _.m_as(unit)

                    return do

                df['AT_ENTRY'] = df['AT_ENTRY'].apply(safe_convert('meter'))
                df['AT_CENTER'] = df['AT_CENTER'].apply(safe_convert('meter'))
                df['AT_EXIT'] = df['AT_EXIT'].apply(safe_convert('meter'))
                try:
                    df['L'] = df['L'].apply(safe_convert('meter'))
                except KeyError:
                    pass
                try:
                    df['ANGLE'] = df['ANGLE'].apply(safe_convert('radian'))
                except KeyError:
                    pass
                try:
                    df['K1'] = df['K1'].apply(safe_convert('1/m**2'))
                except KeyError:
                    pass
                try:
                    df['K2'] = df['K2'].apply(safe_convert('1/m**3'))
                except KeyError:
                    pass
                try:
                    df['E1'] = df['E1'].apply(safe_convert('radian'))
                except KeyError:
                    pass
                try:
                    df['E2'] = df['E2'].apply(safe_convert('radian'))
                except KeyError:
                    pass
                try:
                    df['TILT'] = df['TILT'].apply(safe_convert('radian'))
                except KeyError:
                    pass

            return df

    df = property(to_df)

    def apply(self, func, axis=0):
        """

        Args:
            func:
            axis:

        Returns:

        """
        return self.df.apply(func, axis)

    @staticmethod
    def from_madx_twiss(filename: str = 'twiss.outx',
                        path: str = '.',
                        kinematics: _Kinematics = None,
                        lines: int = None,
                        with_units: bool = True,
                        from_element: str = None,
                        to_element: str = None,
                        element_keys: Optional[Mapping[str, str]] = None,
                        ) -> Sequence:
        """
        TODO

        Args:
            element_keys:
            lines:
            kinematics:
            with_units:
            filename: name of the Twiss table file
            path: path to the Twiss table file
            from_element:
            to_element:

        Returns:

        Examples:
            TODO
        """
        return TwissSequence(filename=filename,
                             path=path,
                             kinematics=kinematics,
                             lines=lines,
                             with_units=with_units,
                             from_element=from_element,
                             to_element=to_element,
                             element_keys=element_keys)

    @staticmethod
    def from_transport(filename: str = 'transport.txt',
                       path: str = '.',
                       ) -> Sequence:
        """
        TODO

        Args:
            filename:
            path:

        Returns:

        """
        return TransportSequence(filename=filename,
                                 path=path)

    @staticmethod
    def from_survey(filename: str = 'survey.csv',
                    path: str = '.',
                    kinematics: _Kinematics = None
                    ):
        """
        TODO

        Returns:

        """
        return SurveySequence(filename=filename, path=path, kinematics=kinematics)

    @staticmethod
    def from_bdsim(filename: str = 'output.root',
                   path: str = '.',
                   ) -> Sequence:
        """
        TODO

        Returns:

        """
        return BDSIMSequence(filename=filename,
                             path=path,
                             from_element=None,
                             to_element=None
                             )


class PlacementSequence(Sequence):
    """Placement Sequence.

    """

    def __init__(self,
                 name: str = '',
                 data: Optional[List[Tuple[_Element,
                                           _ureg.Quantity,
                                           _ureg.Quantity,
                                           _ureg.Quantity]]] = None,
                 metadata: Optional[SequenceMetadata] = None,
                 reference_placement: str = 'ENTRY',
                 element_keys: Optional[Mapping[str, str]] = None,
                 ):
        """

        Args:
            name: the name of the physics
            data: the list of commands composing the physics
            metadata:
            reference_placement:
            element_keys:
        """
        super().__init__(name=name, data=data or [], metadata=metadata, element_keys=element_keys)
        self._reference_placement = reference_placement
        self._betablock: Optional[_BetaBlock] = None
        self._expanded = False

    @property
    def expanded(self):
        return self._expanded

    @property
    def betablock(self) -> _BetaBlock:
        return self._betablock

    @betablock.setter
    def betablock(self, betablock: _BetaBlock):
        self._betablock = betablock

    def add(self,
            element_or_sequence: Union[_Element, Sequence]):
        """

        Args:
            element_or_sequence:

        Returns:

        """
        self.place(element_or_sequence,
                   at_entry=0,
                   after=self._data[-1][0])

    def place(self,
              element_or_sequence: Union[_Element, Sequence],
              at: Optional[_ureg.Quantity] = None,
              at_entry: Optional[_ureg.Quantity] = None,
              at_center: Optional[_ureg.Quantity] = None,
              at_exit: Optional[_ureg.Quantity] = None,
              after: Optional[str] = None,
              before: Optional[str] = None,
              ) -> PlacementSequence:
        """

        Args:
            element_or_sequence:
            at:
            at_center:
            at_entry:
            at_exit:
            after:
            before:

        Returns:

        """
        if before is not None and after is not None:
            raise SequenceException("'preceeding' and 'following' cannot be defined at the same time.")
        ats = locals()
        if after is not None:
            for e in self._data:
                if e[0]['NAME'] == after:
                    for k in ats:
                        if k.startswith('at') and ats[k] is not None:
                            ats[k] += e[3]
        if before is not None:
            for e in self._data:
                if e[0]['NAME'] == before:
                    for k in ats:
                        if k.startswith('at') and ats[k] is not None:
                            ats[k] *= -1
                            ats[k] += e[1] - element_or_sequence.data['L']
        if ats['at'] is not None:
            ats[f"at_{self._reference_placement.lower()}"] = ats['at']

        def compute(d):
            """Compute placement quantities."""
            if d['at_entry'] is None:
                if d['at_center'] is not None:
                    d['at_entry'] = d['at_center'] - element_or_sequence.data[self._element_keys['L']] / 2.0
                elif d['at_exit'] is not None:
                    d['at_entry'] = d['at_exit'] - element_or_sequence.data[self._element_keys['L']]
            if d['at_center'] is None:
                if d['at_entry'] is not None:
                    d['at_center'] = d['at_entry'] + element_or_sequence.data[self._element_keys['L']] / 2.0
                elif d['at_exit'] is not None:
                    d['at_center'] = d['at_exit'] - element_or_sequence.data[self._element_keys['L']] / 2.0
            if d['at_exit'] is None:
                if d['at_entry'] is not None:
                    d['at_exit'] = d['at_entry'] + element_or_sequence.data[self._element_keys['L']]
                elif d['at_center'] is not None:
                    d['at_exit'] = d['at_center'] + element_or_sequence.data[self._element_keys['L']] / 2.0
            return d

        tmp = ats
        tmp2 = tmp
        while True:
            _ = compute(tmp)
            tmp, tmp2 = tmp2, _
            if tmp == tmp2:
                break  # Fixed point
        ats = tmp2
        self._data.append((element_or_sequence, ats['at_entry'], ats['at_center'], ats['at_exit']))
        return self

    def place_after_last(self,
                         element_or_sequence: Union[_Element, Sequence],
                         at: Optional[_ureg.Quantity] = None,
                         at_entry: Optional[_ureg.Quantity] = None,
                         at_center: Optional[_ureg.Quantity] = None,
                         at_exit: Optional[_ureg.Quantity] = None,
                         ) -> PlacementSequence:
        """

        Args:
            element_or_sequence:
            at:
            at_center:
            at_entry:
            at_exit:

        Returns:

        """
        self._data.sort(key=lambda _: _[1])
        offset = self._data[-1][3]
        if at is None and at_entry is None and at_center is None and at_exit is None:
            at = 0.0 * _ureg.m
        if at is not None:
            at += offset
        if at_entry is not None:
            at_entry += offset
        if at_center is not None:
            at_center += offset
        if at_exit is not None:
            at_exit += offset
        return self.place(element_or_sequence=element_or_sequence,
                          at=at,
                          at_entry=at_entry,
                          at_center=at_center,
                          at_exit=at_exit)

    def place_before_first(self,
                           element_or_sequence: Union[_Element, Sequence],
                           at: Optional[_ureg.Quantity] = None,
                           at_entry: Optional[_ureg.Quantity] = None,
                           at_center: Optional[_ureg.Quantity] = None,
                           at_exit: Optional[_ureg.Quantity] = None,
                           ) -> PlacementSequence:
        """

        Args:
            element_or_sequence:
            at:
            at_center:
            at_entry:
            at_exit:

        Returns:

        """
        self._data.sort(key=lambda _: _[1])
        offset = self._data[0][1]
        if at is not None:
            at = offset - at - element_or_sequence.data['L']
        if at_entry is not None:
            at_entry = offset - at_entry - element_or_sequence['L']
        if at_center is not None:
            at_center = offset - at_center - element_or_sequence['L']
        if at_exit is not None:
            at_exit = offset - at_exit - element_or_sequence['L']
        return self.place(element_or_sequence=element_or_sequence,
                          at=at,
                          at_entry=at_entry,
                          at_center=at_center,
                          at_exit=at_exit)

    def expand(self, drift_element: _ElementClass = _Element.Drift) -> PlacementSequence:
        """
        TODO Use namedtuples

        Args:
            drift_element:

        Returns:

        """
        self._data.sort(key=lambda _: _[1])
        at = 0 * _ureg.m
        expanded = []
        for e in self._data:
            length = (e[1] - at).m_as('m')
            if length > 1e-6:
                expanded.append((drift_element(f"D_{e[0].NAME}", L=length * _ureg.m, APERTYPE=None),
                                 at,
                                 at + length * _ureg.m / 2,
                                 at + length * _ureg.m,
                                 ))
            expanded.append(e)
            at = e[3]
        self._data = expanded
        self._expanded = True
        return self

    def reverse(self) -> PlacementSequence:
        """

        Returns:

        """
        length = self._data[-1][3]
        self._data = self._data[::-1]
        self._data = [
            (e, length - at_entry, length - at_center, length - at_exit) for e, at_entry, at_center, at_exit in
            self._data
        ]
        return self

    def sort(self, reverse: bool = False) -> PlacementSequence:
        """

        Args:
            reverse:

        Returns:

        """
        self._data.sort(key=lambda e: e[2], reverse=reverse)
        return self

    def join(self, other):
        pass

    def to_df(self, df: Optional[_pd.DataFrame] = None, strip_units: bool = False) -> _pd.DataFrame:
        """

        Args:
            df:
            strip_units:

        Returns:

        """
        if len(self._data) == 0:
            return _pd.DataFrame()
        df = _pd.DataFrame([{**e[0].data, **{
            'AT_ENTRY': e[1],
            'AT_CENTER': e[2],
            'AT_EXIT': e[3]
        }} for e in self._data])
        df.name = self.name
        df.set_index('NAME', inplace=True)
        return super().to_df(df=df, strip_units=strip_units)

    df = property(to_df)


class TwissSequence(Sequence):
    """
    TODO
    """

    def __init__(self,
                 filename: str = 'twiss.outx',
                 path: str = '.',
                 *,
                 kinematics: _Kinematics = None,
                 lines: int = None,
                 with_units: bool = True,
                 from_element: str = None,
                 to_element: str = None,
                 with_beam: bool = False,
                 nparticles: int = 1,
                 element_keys: Optional[Mapping[str, str]] = None,
                 ):
        """

        Args:
            filename: the name of the Twiss table
            path: path to the Twiss table
            lines: number of lines in the header (default: 47)
            kinematics: kinematics of the particle. Must be specified for MAD-NG
            with_units: Set units to columns
            from_element: Name of the first element
            to_element: Name of the last element
            with_beam: Generate a Gaussian beam from Twiss parameters
            nparticles: Number of particles in the beam (default 1)
            element_keys:

        """
        twiss_headers = load_mad_twiss_headers(filename, path, lines)
        twiss_table = load_mad_twiss_table(filename, path, lines, with_units).loc[from_element:to_element]

        # Add some columns
        twiss_table['CLASS'] = twiss_table['KEYWORD'].apply(str.capitalize)
        twiss_table['AT_CENTER'] = twiss_table['S']
        twiss_table["AT_ENTRY"] = twiss_table["AT_CENTER"] - 0.5 * twiss_table["L"]
        twiss_table["AT_EXIT"] = twiss_table["AT_CENTER"] + 0.5 * twiss_table["L"]

        try:  # For MAD-X
            particle_name = twiss_headers['PARTICLE'].capitalize()
            p = getattr(_particles, particle_name if particle_name != 'Default' else 'Proton')
            k = _Kinematics(float(twiss_headers['PC']) * _ureg.GeV_c, particle=p)
        except KeyError:  # For MAD-NG
            # TODO check with MAD-NG changes. 
            p = kinematics.particule
            k = kinematics

        super().__init__(name=twiss_headers['NAME'],
                         data=twiss_table,
                         metadata=SequenceMetadata(data=twiss_headers, kinematics=k, particle=p),
                         element_keys=element_keys
                         )
        if with_beam:
            twiss_init = self.betablock
            beam_distribution = _Distribution.from_twiss_parameters(n=nparticles,
                                                                    betax=twiss_init['BETA11'],
                                                                    alphax=twiss_init['ALPHA11'],
                                                                    dispx=twiss_init['DISP1'],
                                                                    dispxp=twiss_init['DISP2'],
                                                                    betay=twiss_init['BETA22'],
                                                                    alphay=twiss_init['ALPHA22'],
                                                                    dispy=twiss_init['DISP3'],
                                                                    dispyp=twiss_init['DISP4'],
                                                                    emitx=twiss_init['EMIT1'],
                                                                    emity=twiss_init['EMIT2'],
                                                                    dpp=twiss_headers['DELTAP']).distribution
            beam_distribution['T'] = 0
            self._metadata = SequenceMetadata(data=beam_distribution, kinematics=k, particle=p)

    @property
    def betablock(self) -> _BetaBlock:
        """TODO"""
        # Keep in this order
        try:  # For MAD-X
            return _BetaBlock(
                BETA11=self.df.iloc[0]['BETX'] * _ureg.m,
                ALPHA11=self.df.iloc[0]['ALFX'],
                BETA22=self.df.iloc[0]['BETY'] * _ureg.m,
                ALPHA22=self.df.iloc[0]['ALFY'],
                DISP1=self.df.iloc[0]['DX'] * _ureg.m,
                DISP2=self.df.iloc[0]['DPX'],
                DISP3=self.df.iloc[0]['DY'] * _ureg.m,
                DISP4=self.df.iloc[0]['DPY'],
                EMIT1=self.metadata['EX'] * _ureg('m * radians'),
                EMIT2=self.metadata['EY'] * _ureg('m * radians'),
                EMIT3=self.metadata['ET'],
            )
        except KeyError:
            try:  # For MAD-NG
                return _BetaBlock(
                    BETA11=self.df.iloc[0]['BETA11'] * _ureg.m,
                    ALPHA11=self.df.iloc[0]['ALFA11'],
                    BETA22=self.df.iloc[0]['BETA22'] * _ureg.m,
                    ALPHA22=self.df.iloc[0]['ALFA22'],
                    DISP1=self.df.iloc[0]['DX'] * _ureg.m,
                    DISP2=self.df.iloc[0]['DPX'],
                    DISP3=self.df.iloc[0]['DY'] * _ureg.m,
                    DISP4=self.df.iloc[0]['DPY'],
                    EMIT1=1e-9 * _ureg('m * radians'),  # self.metadata['EMIT1'] * _ureg('m * radians') not yet in MADNG
                    EMIT2=1e-9 * _ureg('m * radians'),  # self.metadata['EMIT2'] * _ureg('m * radians'),
                    EMIT3=1e-9 * _ureg('m * radians'),  # self.metadata['EMIT3'] * _ureg('m * radians')
                )
            except KeyError:
                logging.warning('Setting BetaBlock by default')
                return _BetaBlock()

    def to_df(self, df: Optional[_pd.DataFrame] = None, strip_units: bool = False) -> _pd.DataFrame:
        """TODO"""
        return super().to_df(self._data, strip_units=strip_units)

    df = property(to_df)


class TransportSequence(Sequence):
    """
    TODO
    """
    from ..codes_io.transport import TransportInputFlavor, TransportInputOriginalFlavor

    def __init__(self,
                 filename: str,
                 path: str = '.',
                 flavor: TransportInputFlavor = TransportInputOriginalFlavor,
                 ):
        """

        Args:
            filename: the name of the physics
            path:
            flavor:
        """
        transport_input = load_transport_input_file(filename, path)

        sequence_metadata = SequenceMetadata()
        data = []
        at_entry = 0 * _ureg.meter
        for line in transport_input:
            if len(line.strip()) == 0:
                continue
            d = line.rsplit(';', 1)[0].split()
            if d[0].startswith('-'):
                continue
            try:
                float(d[0])
            except ValueError:
                continue

            transport_element = transport_element_factory(d, sequence_metadata, flavor)[0]

            if transport_element is not None:
                transport_element = self.process_element(transport_element,
                                                         sequence_metadata.kinematics.brho,
                                                         at_entry)
                data.append(transport_element)
                at_entry += transport_element['L']

        data = self.process_face_angle(data)
        super().__init__(name='TRANSPORT',
                         data=data,
                         metadata=sequence_metadata,
                         )

    @staticmethod
    def process_element(ele, brho, at_entry):
        ele['AT_ENTRY'] = at_entry
        ele['AT_CENTER'] = at_entry + 0.5 * ele['L']
        ele['AT_EXIT'] = at_entry + ele['L']
        if ele["CLASS"] == "Quadrupole":
            ele["K1"] = ((ele["B1"] / ele["R"]) / brho).to("meter**-2")  # For manzoni
            ele["TILT"] = 0 * _ureg.radians
        if ele["CLASS"] == "SBend" or ele["CLASS"] == "RBend":
            ele['R'] = (ele['L'] / ele['ANGLE']).to('m')
            ele["K1"] = -ele['N'] / ele['R'] ** 2  # For manzoni
            ele["B"] = ((ele["ANGLE"] * brho) / ele["L"]).to("T")
        return ele

    @staticmethod
    def process_face_angle(line):
        for idx in range(len(line) - 1):
            element = line[idx]
            previous = line[idx - 1]
            after = line[idx + 1]
            if element["CLASS"] == "SBend" or element["CLASS"] == "RBend":
                if previous["CLASS"] == "Face":
                    element["E1"] = previous["E1"]
                if after["CLASS"] == "Face":
                    element["E2"] = after["E1"]

        # Remove the faces from the list
        t = [not isinstance(val, _Element.Face) for val in line]
        return list(compress(line, t))

    def to_df(self, df: Optional[_pd.DataFrame] = None, strip_units: bool = False) -> pd.DataFrame:
        dicts = list(map(dict, self._data))
        counters = {}
        for d in dicts:
            if d['NAME'] is None:
                counters[d['KEYWORD']] = counters.get(d['KEYWORD'], 0) + 1
                d['NAME'] = f"{d['KEYWORD']}_{counters[d['KEYWORD']]}"
        return super().to_df(_pd.DataFrame(dicts).set_index('NAME'), strip_units=strip_units)

    df = property(to_df)


class SurveySequence(Sequence):
    def __init__(self,
                 filename: str,
                 path: str = '.',
                 kinematics: _Kinematics = None,
                 metadata: Optional[SequenceMetadata] = None
                 ):
        """

        Args:
            filename: Name of the file, must be a csv.
            path: Path to the file
            kinematics: Kinematics of the particle
            metadata: metadata of the sequence
        """

        def get_entrance_exit_frame(e):
            center_frame = Frame().translate([e['X'] * _ureg.meter, e['Y'] * _ureg.meter, e['Z'] * _ureg.meter])
            if e['TYPE'] == 'SBEND':
                # TODO This is not working in 3D and avoid copy.
                Angle = -_np.pi / 2 - e['ANGLE'] / 2 - e['CUMULATIVE_ANGLE']
                radius = _np.abs(e['L'] / e['ANGLE'])
                x_c = e['X'] + _np.sign(e['ANGLE'].m_as('radians')) * radius.m_as('m') * _np.cos(Angle.m_as('radians'))
                y_c = e['Y'] + _np.sign(e['ANGLE'].m_as('radians')) * radius.m_as('m') * _np.sin(Angle.m_as('radians'))
                dx = e['X'] - x_c
                dy = e['Y'] - y_c
                frame_bend_center = Frame().translate([x_c * _ureg.m, y_c * _ureg.m, 0 * _ureg.m])
                f1 = copy.deepcopy(Frame(frame_bend_center).translate([dx * _ureg.m, dy * _ureg.m, 0 * _ureg.m]))
                f2 = copy.deepcopy(Frame(frame_bend_center).translate([dx * _ureg.m, dy * _ureg.m, 0 * _ureg.m]))

                frame_entrance = Frame().translate(
                    [f1.rotate_z(e['ANGLE'] / 2 * _ureg.radians).x, f1.rotate_z(e['ANGLE'] / 2 * _ureg.radians).y,
                     f1.z])
                frame_exit = Frame().translate(
                    [f2.rotate_z(-e['ANGLE'] / 2 * _ureg.radians).x, f2.rotate_z(-e['ANGLE'] / 2 * _ureg.radians).y,
                     f2.z])
            else:
                frame_entrance = Frame(center_frame).translate_x(-e['L'] / 2).rotate_z(
                    -e['CUMULATIVE_ANGLE'])
                frame_exit = Frame(center_frame).translate_x(e['L'] / 2).rotate_z(
                    -e['CUMULATIVE_ANGLE'])
            return frame_entrance, frame_exit

        sequence = _pd.read_csv(os.path.join(path, filename), index_col='NAME', sep=',')
        sequence['L'] = sequence['L'].fillna(0).apply(lambda e: e * _ureg.meter)
        sequence["ANGLE"] = sequence["ANGLE"].fillna(0).apply(lambda e: e * _ureg.radian)

        # check if the survey is (AT_CENTER, L) or (X, Y, Z)
        if sequence.get(['AT_CENTER']) is None and sequence.get(['X', 'Y', 'Z']) is not None:
            sequence['CUMULATIVE_ANGLE'] = sequence['ANGLE'].cumsum().shift(1)
            sequence.loc[sequence.iloc[0].name, 'CUMULATIVE_ANGLE'] = [0.0] * _ureg.radians
            sequence[['F_ENTRANCE', 'F_EXIT']] = sequence.apply(lambda e: get_entrance_exit_frame(e), axis=1,
                                                                result_type='expand')
            at = 0
            for i, j in sequence.iterrows():
                d = _np.linalg.norm(_np.array([(j['F_ENTRANCE'].x - j['F_EXIT'].x).m_as('m'),
                                               (j['F_ENTRANCE'].y - j['F_EXIT'].y).m_as('m'),
                                               (j['F_ENTRANCE'].z - j['F_EXIT'].z).m_as('m')]))
                at += (d + j['L'].m_as('m') / 2)
                sequence.at[i, 'AT_CENTER'] = at
                at += j['L'].m_as('m') / 2

        elif sequence.get(['AT_CENTER']) is not None and sequence.get(['X', 'Y', 'Z']) is None:
            pass
        else:
            raise SequenceException("Sequence must be (AT_CENTER, L) or (X,Y,Z)")

        sequence['AT_CENTER'] = sequence['AT_CENTER'].apply(lambda e: e * _ureg.meter)
        sequence["AT_ENTRY"] = sequence["AT_CENTER"] - 0.5 * sequence["L"]
        sequence["AT_EXIT"] = sequence["AT_CENTER"] + 0.5 * sequence["L"]

        # Set units to columns
        try:
            sequence['K1'] = sequence['K1'].fillna(0)
        except KeyError:
            sequence['K1'] = 0
        try:
            sequence['E1'] = sequence['E1'].fillna(0)
        except KeyError:
            sequence['E1'] = 0
        try:
            sequence['E2'] = sequence['E2'].fillna(0)
        except KeyError:
            sequence['E2'] = 0

        sequence['K1'] = sequence['K1'].apply(lambda e: e*_ureg.m**-2)
        sequence['E1'] = sequence['E1'].apply(lambda e: e*_ureg.radians)
        sequence['E2'] = sequence['E2'].apply(lambda e: e*_ureg.radians)

        idx = sequence.query("TYPE == 'COLLIMATOR'").index
        sequence.loc[idx, "TYPE"] = [f"{sequence.loc[i, 'APERTYPE']}COLLIMATOR" for i in idx]

        def check_apertures(e):
            if isinstance(e, float):
                return [e * _ureg.m]
            else:
                return [float(k) * _ureg.m for k in e.replace('[', '').replace(']', '').split(';')]

        sequence['APERTURE'] = sequence['APERTURE'].apply(lambda e: check_apertures(e))

        data = []
        sequence_metadata = metadata or SequenceMetadata(kinematics=kinematics,
                                                         particle=kinematics.particule)

        extra_columns = list(set(sequence.columns.values) - {'APERTYPE', 'CLASS', 'L', 'KEYWORD', 'AT_ENTRY',
                                                             'AT_CENTER', 'AT_EXIT', 'K1', 'APERTURE', 'K1L', 'E1',
                                                             'E2', 'TILT', 'MATERIAL', 'KINETIC_ENERGY', 'ANGLE',
                                                             'KICK'})
        for element in sequence.iterrows():
            if extra_columns:
                ele = {**csv_element_factory(element), **element[1][extra_columns]}
            else:
                ele = {**csv_element_factory(element)}
            data.append((ele,
                         element[1]['AT_ENTRY'],
                         element[1]['AT_CENTER'],
                         element[1]['AT_EXIT']
                         ))
        super().__init__(name='SURVEY',
                         data=data,
                         metadata=sequence_metadata)

    def to_df(self, df: Optional[_pd.DataFrame] = None, strip_units=False):
        df = _pd.DataFrame([{**e[0], **{
            'AT_ENTRY': e[1],
            'AT_CENTER': e[2],
            'AT_EXIT': e[3]
        }} for e in self._data])
        df.name = self.name
        df.set_index('NAME', inplace=True)
        return super().to_df(df=df, strip_units=strip_units)

    df = property(to_df)


# TODO use with pybdsim
class BDSIMSequence(Sequence):

    def __init__(self,
                 filename: str = 'output.root',
                 path: str = '.',
                 from_element: str = None,
                 to_element: str = None,
                 ):
        """
        Args:
            filename: the name of the physics
            path:
            from_element:
            to_element:
        """
        # The pybdsim import is made inside the class init to avoid a pybdsim (and then ROOT)
        # dependence when it is not needed.
        from pybdsim.Analysis import BDSimOutput

        # Load the model
        bdsim_data = BDSimOutput(filename=filename, path=path)
        bdsim_model = bdsim_data.model.df.loc[from_element:to_element]
        bdsim_model.rename(columns={"TYPE": "KEYWORD"}, inplace=True)
        self.set_units(bdsim_model)

        # Load the beam properties
        # FIXME Change the method to access beam base. Do not change in pybdsim

        bdsim_beam = bdsim_data.beam.beam_base.pandas(branches=['beamEnergy', 'particle'])
        particle_name = bdsim_beam["particle"].values[0].capitalize()
        particle_energy = bdsim_beam["beamEnergy"].values[0] * _ureg.GeV
        p = getattr(_particles, particle_name if particle_name != 'Default' else 'Proton')
        kin = _Kinematics(particle_energy, kinetic=False, particle=p)

        # Load the beam distribution
        beam_distribution = bdsim_data.event.primary.df.copy()
        beam_distribution['dpp'] = beam_distribution['p'].apply(lambda e: ((e / kin.momentum.m_as("GeV/c")) - 1))
        beam_distribution = beam_distribution[["x", 'y', 'xp', 'yp', 'dpp']]
        beam_distribution.rename(columns={
            "xp": "px",
            "yp": "py"
        }, inplace=True)
        beam_distribution.columns = map(str.upper, beam_distribution.columns)
        beam_distribution['T'] = 0
        beam_distribution.reset_index(inplace=True)  # Remove the multi index
        super().__init__(name="BDSIM",
                         data=bdsim_model,
                         metadata=SequenceMetadata(
                             data=_pd.Series({
                                 'BEAM_DISTRIBUTION': beam_distribution[['X', 'PX', 'Y', 'PY', 'T', 'DPP']],
                             }),
                             kinematics=kin,
                             particle=p)
                         )

    @staticmethod
    def set_units(model: _pd.DataFrame = None):
        # Specify the units
        for c in model.columns:
            try:
                model[c] = model[c].apply(float)
            except ValueError:
                pass

        model['CLASS'] = model['KEYWORD'].apply(str.capitalize)
        model['CLASS'] = model['CLASS'].apply(lambda e: _BDSIM_TO_MAD_CONVENTION.get(e, e))

        model.loc[model["CLASS"] == "RectangularCollimator", "APERTYPE"] = "rectangular"
        model.loc[model["CLASS"] == "Dump", "APERTYPE"] = "rectangular"
        model.loc[model["CLASS"] == "EllipticalCollimator", "APERTYPE"] = "elliptical"

        model['L'] = model['L'].apply(lambda e: e * _ureg.m)
        model['AT_ENTRY'] = model['AT_ENTRY'].apply(lambda e: e * _ureg.m)
        model['AT_CENTER'] = model['AT_CENTER'].apply(lambda e: e * _ureg.m)
        model['AT_EXIT'] = model['AT_EXIT'].apply(lambda e: e * _ureg.m)
        model['ANGLE'] = model['ANGLE'].apply(lambda e: e * _ureg.radians)
        model['APERTURE1'] = model['APERTURE1'].apply(lambda e: e * _ureg.m)
        model['APERTURE2'] = model['APERTURE2'].apply(lambda e: e * _ureg.m)
        model['APERTURE'] = model[['APERTURE1', 'APERTURE2']].apply(list, axis=1)
        model['K1'] = model['K1'].apply(lambda e: e * _ureg.m ** -2)
        model['K1S'] = model['K1S'].apply(lambda e: e * _ureg.m ** -2)
        model['K2'] = model['K2'].apply(lambda e: e * _ureg.m ** -3)
        model['E1'] = model['E1'].apply(lambda e: e * _ureg.radian)
        model['E2'] = model['E2'].apply(lambda e: e * _ureg.radian)
        model['HGAP'] = model['HGAP'].apply(lambda e: e * _ureg.meter)
        model['TILT'] = model['TILT'].apply(lambda e: e * _ureg.radian)
        model['B'] = model['B'].apply(lambda e: e * _ureg.T)

    def to_df(self, df: Optional[_pd.DataFrame] = None, strip_units: bool = False) -> _pd.DataFrame:
        return self._data
