"""High-level interface for Zgoubi using sequences.

"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, List, Tuple, Mapping, Union
from dataclasses import dataclass
import numpy as _np
import pandas as _pd
from .. import particles as _particles
from ..particles import Proton as _Proton
from ..kinematics import Kinematics as _Kinematics
from .elements import Element as _Element
from .elements import ElementClass as _ElementClass
from .betablock import BetaBlock as _BetaBlock
from ..codes_io import load_madx_twiss_headers, load_madx_twiss_table, load_transport_input_file, transport_element_factory
from ..codes_io import load_bdsim_model, load_bdsim_beam_distribution, load_bdsim_kinematics
from .. import ureg as _ureg
if TYPE_CHECKING:
    from ..particles import ParticuleType as _ParticuleType

__all__ = ['SequenceException',
           'SequenceMetadata',
           'Sequence',
           'PlacementSequence',
           'TwissSequence',
           'SurveySequence',
           'TransportSequence',
           ]


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
                        columns: List = None,
                        from_element: str = None,
                        to_element: str = None, ) -> Sequence:
        """
        TODO

        Args:
            filename: name of the Twiss table file
            path: path to the Twiss table file
            columns: the list of columns in the Twiss file
            from_element:
            to_element:

        Returns:

        Examples:
            TODO
        """
        return TwissSequence(filename=filename,
                             path=path,
                             columns=columns,
                             from_element=from_element,
                             to_element=to_element,
                             )

    @staticmethod
    def from_transport(filename: str = 'transport.txt',
                       path: str = '.',
                       ):
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
    def from_survey():
        """
        TODO

        Returns:

        """
        return SurveySequence()

    @staticmethod
    def from_bdsim(filename: str = 'output.root',
                   path: str = '.',
                   ):
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

    def to_df(self, df: Optional[_pd.DataFrame] = None, strip_units: bool = False) -> _pd.DataFrame:
        """

        Args:
            df:
            strip_units:

        Returns:

        """
        if len(self._data) == 0:
            return _pd.DataFrame()
        if df is None:
            df = _pd.DataFrame([{**e[0].data, **{
                'AT_ENTRY': e[1],
                'AT_CENTER': e[2],
                'AT_EXIT': e[3]
            }} for e in self._data])
            df.name = self.name
            df.set_index('NAME', inplace=True)
        return super().to_df(df, strip_units=strip_units)

    df = property(to_df)

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
                expanded.append((drift_element(f"D_{e[0].NAME}", L=length * _ureg.m),
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
            (e, length-at_entry, length-at_center, length-at_exit) for e, at_entry, at_center, at_exit in self._data
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


class TwissSequence(Sequence):
    """
    TODO
    """

    def __init__(self,
                 filename: str = 'twiss.outx',
                 path: str = '.',
                 *,
                 columns: List = None,
                 from_element: str = None,
                 to_element: str = None,
                 element_keys: Optional[Mapping[str, str]] = None,
                 ):
        """

        Args:
            filename: the name of the physics
            path:
            columns:
            from_element:
            to_element:
            element_keys:
        """
        twiss_headers = load_madx_twiss_headers(filename, path)
        twiss_table = load_madx_twiss_table(filename, path, columns).loc[from_element:to_element]
        particle_name = twiss_headers['PARTICLE'].capitalize()
        p = getattr(_particles, particle_name if particle_name != 'Default' else 'Proton')
        k = _Kinematics(float(twiss_headers['PC']) * _ureg.GeV_c, particle=p)
        super().__init__(name=twiss_headers['NAME'],
                         data=twiss_table,
                         metadata=SequenceMetadata(data=twiss_headers, kinematics=k, particle=p),
                         element_keys=element_keys
                         )

    @property
    def betablock(self) -> _BetaBlock:
        """TODO"""
        try:
            return _BetaBlock(
                beta11=self.df.iloc[0]['BETA11'] * _ureg.m,
                alpha11=self.df.iloc[0]['ALPHA11'],
                beta22=self.df.iloc[0]['BETA22'] * _ureg.m,
                alpha22=self.df.iloc[0]['ALPHA22'],
                disp1=self.df.iloc[0]['DISP1'] * _ureg.m,
                disp2=self.df.iloc[0]['DISP2'],
                disp3=self.df.iloc[0]['DISP3'] * _ureg.m,
                disp4=self.df.iloc[0]['DISP4'],
                emit1=self.metadata['EX'],
                emit2=self.metadata['EY'],
                emit3=self.metadata['ET'],
            )
        except KeyError:
            try:
                return _BetaBlock(
                    beta11=self.df.iloc[0]['BETX'] * _ureg.m,
                    alpha11=self.df.iloc[0]['ALFX'],
                    beta22=self.df.iloc[0]['BETY'] * _ureg.m,
                    alpha22=self.df.iloc[0]['ALFY'],
                    disp1=self.df.iloc[0]['DX'] * _ureg.m,
                    disp2=self.df.iloc[0]['DPX'],
                    disp3=self.df.iloc[0]['DY'] * _ureg.m,
                    disp4=self.df.iloc[0]['DPY'],
                    emit1=self.metadata['EX'],
                    emit2=self.metadata['EY'],
                    emit3=self.metadata['ET'],
                )
            except KeyError:
                return _BetaBlock()

    def to_df(self) -> _pd.DataFrame:
        """TODO"""
        return self._data

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

        data = []
        sequence_metadata = SequenceMetadata()
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
            data.append(transport_element_factory(d, sequence_metadata, flavor)[0])

        super().__init__(name='TRANSPORT',
                         data=[d for d in data if d is not None],
                         metadata=sequence_metadata,
                         )

    def to_df(self):
        dicts = list(map(dict, self._data))
        counters = {}
        for d in dicts:
            if d['NAME'] is None:
                counters[d['KEYWORD']] = counters.get(d['KEYWORD'], 0) + 1
                d['NAME'] = f"{d['KEYWORD']}_{counters[d['KEYWORD']]}"
        return _pd.DataFrame(dicts).set_index('NAME')

    df = property(to_df)


class SurveySequence(PlacementSequence):
    def __init__(self,
                 filename: str,
                 path: str = '.',
                 ):
        """

        Args:
            filename: the name of the physics
            path:
        """
        transport_input = load_transport_input_file(filename, path)

        data = []
        sequence_metadata = SequenceMetadata()
        for line in transport_input:
            if len(line.strip()) == 0:
                continue
            d = line.split()
            if d[0].startswith('-'):
                continue
            data.append(transport_element_factory(d, sequence_metadata)[0])

        if sequence_metadata.kinematics is None:
            raise SequenceException("Invalid kinematics - Beam not found in input")

        super().__init__(name='TRANSPORT',
                         data=[d for d in data if d is not None],
                         metadata=sequence_metadata,
                         )

    def to_df(self):
        dicts = list(map(dict, self._data))
        counters = {}
        for d in dicts:
            if d['NAME'] is None:
                counters[d['KEYWORD']] = counters.get(d['KEYWORD'], 0) + 1
                d['NAME'] = f"{d['KEYWORD']}_{counters[d['KEYWORD']]}"
        return _pd.DataFrame(dicts).set_index('NAME')

    df = property(to_df)


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
        bdsim_model = load_bdsim_model(filename, path, with_units=True).loc[from_element:to_element]

        bdsim_kinematics = load_bdsim_kinematics(filename, path)
        particle = bdsim_kinematics.particule

        bdsim_beam_distribution = load_bdsim_beam_distribution(filename, path)
        bdsim_beam_distribution['DPP'] = bdsim_beam_distribution['P'].apply(
            lambda e: ((e/bdsim_kinematics.momentum)-1).magnitude)

        super().__init__(name="BDSIM",
                         data=bdsim_model,
                         metadata=SequenceMetadata(
                             data=_pd.Series({
                                 'BEAM_DISTRIBUTION': bdsim_beam_distribution[['X', 'PX', 'Y', 'PY', 'T', 'DPP']],
                             }),
                             kinematics=bdsim_kinematics,
                             particle=particle)
                         )

    def to_df(self, df: Optional[_pd.DataFrame] = None, strip_units: bool = False) -> _pd.DataFrame:
        return self._data

    df = property(to_df)
