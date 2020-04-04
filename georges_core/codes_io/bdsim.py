"""
A Pythonic way to analyze and work with Beam Delivery SIMulation (BDSIM) ROOT output files.

Design goals:
 - No dependency on (py)ROOT(py) is needed. The module uses `uproot` instead.
 - Enables and favors exploration of the ROOT files. No prior knowledge of the content should be required
 to explore and discover the data structure.
 - provide analysis tools exploiting the new Awkward 1.0 library (https://arxiv.org/pdf/2001.06307.pdf)
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Mapping
from collections import UserDict
import logging
import os
try:
    import uproot as _uproot
    if TYPE_CHECKING:
        import uproot.source.compressed
except (ImportError, ImportWarning):
    logging.error("Uproot is required for this module to work.")
import numpy as _np
import pandas as _pd

__all__ = [
    'Output',
    'BDSimOutput',
    'ReBDSimOutput',
    'ReBDSimOpticsOutput',
]


class BDSimOutputException(Exception):
    pass


class OutputType(type):
    """A generic type for BDSIM output classes."""
    pass


class Output(metaclass=OutputType):
    def __init__(self, filename: str = 'output.root', path: str = '.', *, open_file: bool = True):
        """
        Create a representation of a BDSIM output using uproot to read the root file.

        The root file is opened with uproot, so a valid path and filename must be provided.

        Args:
            filename: the name of the root file to read
            path: the path to the root file
            open_file: attempts to open the master file
        """
        self._file = os.path.join(path, filename)
        if open_file:
            self._root_directory: _uproot.rootio.ROOTDirectory = _uproot.open(self._file)

    @classmethod
    def from_root_directory(cls, directory: _uproot.rootio.ROOTDirectory) -> Output:
        """Create an `Output` object directly attached to an existing ROOT directory.

        Args:
            directory: an existing `uproot` `ROOTDirectory`
        """
        o = cls(open_file=False)
        o._file = None
        o._root_directory = directory
        return o

    def __getitem__(self, item: str):
        """Read an object from the ROOT file or directory by name."""
        return self._root_directory[item]

    @property
    def compression(self) -> _uproot.source.compressed.Compression:
        """The compression algorithm used for the root file or directory."""
        return self._root_directory.compression

    @property
    def directory(self) -> _uproot.rootio.ROOTDirectory:
        """Return the master directory attached to this parent."""
        return self._root_directory

    class Directory:
        def __init__(self, parent: Output, directory: _uproot.rootio.ROOTDirectory):
            """
            A representation of a (nested) structure of ROOT directories.

            Args:
                parent: the `Output` to which the directory structure is attached
                directory: the top-level ROOT directory
            """
            def _build(n, c):
                if c.__name__.endswith('Directory'):
                    return Output.Directory(parent, directory=self._directory[n])
                else:
                    return self._directory[n]

            self._output: Output = parent
            self._directory: _uproot.rootio.ROOTDirectory = directory
            for name, cls in self._directory.iterclasses():
                setattr(self, name.decode('utf-8').split(';')[0].replace('-', '_'), _build(name, cls))

        def __getitem__(self, item):
            return self._directory[item]

        @property
        def compression(self) -> _uproot.source.compressed.Compression:
            """The compression algorithm used for the directory."""
            return self._directory.compression

        @property
        def parent(self) -> Output:
            """The parent Output to which the directory structure is attached."""
            return self._output

    class Tree:
        def __init__(self, parent: Output):
            """
            A representation of a ROOT TTree structure.

            Args:
                parent: the `Output` to which the parent is attached
            """
            self._parent = parent
            self._tree: uproot.rootio.TTree = parent[self.__class__.__name__]
            self._df: Optional[_pd.DataFrame] = None
            self._np: Optional[_np.ndarray] = None

        def __getitem__(self, item):
            try:
                return self._tree[item]
            except KeyError:
                return self._tree[item + '.']

        def __getattr__(self, b):
            branch_class = getattr(self.__class__, b.title().replace('_', ''), None)
            if branch_class is not None:
                setattr(self, b, branch_class(parent=self))
                return getattr(self, b)
            else:
                raise AttributeError(f"Branch {b} does not exist for {self.__class__.__name__}")

        def array(self, branch=None, **kwargs) -> _np.ndarray:
            """A proxy for the `uproot` `array` method."""
            return self.tree.array(branch=branch, **kwargs)

        def arrays(self, branches=None, **kwargs):
            """A proxy for the `uproot` `arrays` method."""
            return self.tree.arrays(branches=branches, **kwargs)

        def pandas(self, branches=None, **kwargs):
            """A proxy for the `uproot` `pandas` method."""
            return self._tree.pandas.df(branches=branches, **kwargs)

        def to_df(self) -> _pd.DataFrame:
            pass

        def to_np(self) -> _np.ndarray:
            pass

        @property
        def parent(self):
            """The parent Output to which the parent structure is attached."""
            return self._parent

        @property
        def tree(self) -> _uproot.rootio.TTree:
            """The associated uproot parent."""
            return self._tree

        @property
        def branches_names(self) -> List[str]:
            return [b.decode('utf-8') for b in self.tree.keys()]

        @property
        def branches(self) -> List[uproot.rootio.TBranchElement]:
            return [b for b in self.tree.values()]

        @property
        def numentries(self) -> int:
            """Provides the number of entries in the parent (without reading the entire file)."""
            return self.tree.numentries

        @property
        def df(self) -> _pd.DataFrame:
            if self._df is None:
                return self.to_df()
            return self._df

        @property
        def np(self) -> _np.ndarray:
            if self._np is None:
                return self.to_np()
            return self._np

    class Branch:
        BRANCH_NAME: Optional[str] = None
        DEFAULT_LEAVES: Dict[str, Tuple[bool, Optional[str]]] = {}

        def __new__(cls, *args, **kwargs):
            def toggle(leave):
                def do_toggle(self):
                    self._active_leaves[leave][1] = not self._active_leaves[leave][1]
                    return self
                return do_toggle
            instance = super().__new__(cls)
            for k in cls.DEFAULT_LEAVES:
                setattr(instance, f"toggle_{k}", toggle(k).__get__(instance))
            return instance

        def __init__(self, parent: Output.Tree, branch_name: Optional[str] = None):
            """
            A representation of a ROOT Branch.

            Args:
                parent: the `Tree` to which the branch is attached
            """
            self._parent: Output.Tree = parent
            b = branch_name or self.BRANCH_NAME or self.__class__.__name__
            if len(b) > 0:
                self._branch = parent[b]
            else:
                self._branch = parent
            self._df: Optional[_pd.DataFrame] = None
            self._np: Optional[_np.ndarray] = None
            self._active_leaves: Dict[str, Tuple[bool, Optional[str]]] = self.DEFAULT_LEAVES

        def __getitem__(self, item):
            return self._branch[item]

        def array(self, branch=None, **kwargs) -> _np.ndarray:
            """A proxy for the `uproot` `array` method."""
            return self.parent.array(branch=self.branch_name + branch, **kwargs)

        def arrays(self, branches=None, **kwargs):
            """A proxy for the uproot method.
            """
            if branches is None:
                branches = self._active_leaves
            return self.parent.arrays(branches=[self.branch_name + b for b in branches], **kwargs)

        def pandas(self, branches: List[str] = None, strip_prefix: bool = True, **kwargs):
            """A proxy for the uproot method.
            """
            if branches is None:
                branches = self._active_leaves
                branches = [self.branch_name + b for b, _ in branches.items() if _[0] is True]
            else:
                branches = [self.branch_name + b for b in branches]
            df = self.parent.tree.pandas.df(branches,
                                            flatten=True,
                                            **kwargs)
            if strip_prefix:
                df.columns = [c.lstrip(self.branch_name) for c in df.columns]
            return df

        def to_df(self) -> _pd.DataFrame:
            if self._df is None:
                self._df = self.pandas()
            return self._df

        def to_np(self) -> _np.ndarray:
            pass

        @property
        def parent(self) -> Output.Tree:
            """The parent `Tree` to which the branch is attached."""
            return self._parent

        @property
        def branch(self) -> _uproot.rootio.TBranch:
            return self._branch

        @property
        def branch_name(self) -> str:
            if self.BRANCH_NAME == '':
                return ''
            name = self.branch.name.decode('utf-8')
            if not name.endswith('.'):
                return name + '.'
            else:
                return name

        @property
        def leaves(self) -> List[uproot.rootio.TBranchElement]:
            return self._branch.values()

        @property
        def leaves_names(self) -> List[str]:
            return self._branch.keys()

        @property
        def df(self) -> _pd.DataFrame:
            if self._df is None:
                return self.to_df()
            return self._df

        @property
        def np(self) -> _np.ndarray:
            if self._np is None:
                return self.to_np()
            return self._np


class BDSimOutput(Output):
    def __getattr__(self, item):
        if item in (
            'header',
            'geant4data',
            'beam',
            'options',
            'model',
            'run',
            'event',
        ):
            setattr(self,
                    item,
                    getattr(BDSimOutput, item.title())(parent=self)
                    )
            return getattr(self, item)

    class Header(Output.Tree):

        class Header(Output.Branch):
            DEFAULT_LEAVES = {
                'bdsimVersion': [True, None],
                'geant4Version': [True, None],
                'rootVersion': [True, None],
                'clhepVersion': [True, None],
                'timeStamp': [True, None],
                'fileType': [True, None],
                'dataVersion': [True, None],
                'doublePrecisionOutput': [True, None],
                'analysedFiles': [True, None],
                'combinedFiles': [True, None],
                'nTrajectoryFilters': [True, None],
                'trajectoryFilters': [True, None],
            }

    class Beam(Output.Tree):

        class BeamBase(Output.Branch):
            BRANCH_NAME = 'Beam.GMAD::BeamBase'
            DEFAULT_LEAVES = {
                'particle': [True, None],
                'beamParticleName': [True, None],
                'beamEnergy': [True, None],
                'beamKineticEnergy': [True, None],
                'beamMomentum': [True, None],
                'distrType': [True, None],
                'xDistrType': [True, None],
                'yDistrType': [True, None],
                'zDistrType': [True, None],
                'distrFile': [True, None],
                'distrFileFormat': [True, None],
                'matchDistrFileLength': [True, None],
                'nlinesIgnore': [True, None],
                'nlinesSkip': [True, None],
                'X0': [True, None],
                'Y0': [True, None],
                'Z0': [True, None],
                'S0': [True, None],
                'Xp0': [True, None],
                'Yp0': [True, None],
                'Zp0': [True, None],
                'T0': [True, None],
                'E0': [True, None],
                'Ek0': [True, None],
                'P0': [True, None],
                'tilt': [True, None],
                'sigmaT': [True, None],
                'sigmaE': [True, None],
                'sigmaEk': [True, None],
                'sigmaP': [True, None],
                'betx': [True, None],
                'bety': [True, None],
                'alfx': [True, None],
                'alfy': [True, None],
                'emitx': [True, None],
                'emity': [True, None],
                'dispx': [True, None],
                'dispy': [True, None],
                'dispxp': [True, None],
                'dispyp': [True, None],
                'emitNX': [True, None],
                'emitNY': [True, None],
                'sigmaX': [True, None],
                'sigmaXp': [True, None],
                'sigmaY': [True, None],
                'sigmaYp': [True, None],
                'envelopeX': [True, None],
                'envelopeXp': [True, None],
                'envelopeY': [True, None],
                'envelopeYp': [True, None],
                'envelopeT': [True, None],
                'envelopeE': [True, None],
                'envelopeR': [True, None],
                'envelopeRp': [True, None],
                'sigma11': [True, None],
                'sigma12': [True, None],
                'sigma13': [True, None],
                'sigma14': [True, None],
                'sigma15': [True, None],
                'sigma16': [True, None],
                'sigma22': [True, None],
                'sigma23': [True, None],
                'sigma24': [True, None],
                'sigma25': [True, None],
                'sigma26': [True, None],
                'sigma33': [True, None],
                'sigma34': [True, None],
                'sigma35': [True, None],
                'sigma36': [True, None],
                'sigma44': [True, None],
                'sigma45': [True, None],
                'sigma46': [True, None],
                'sigma55': [True, None],
                'sigma56': [True, None],
                'sigma66': [True, None],
                'shellX': [True, None],
                'shellXp': [True, None],
                'shellY': [True, None],
                'shellYp': [True, None],
                'shellXWidth': [True, None],
                'shellXpWidth': [True, None],
                'shellYWidth': [True, None],
                'shellYpWidth': [True, None],
                'Rmin': [True, None],
                'Rmax': [True, None],
                'haloNSigmaXInner': [True, None],
                'haloNSigmaXOuter': [True, None],
                'haloNSigmaYInner': [True, None],
                'haloNSigmaYOuter': [True, None],
                'haloXCutInner': [True, None],
                'haloYCutInner': [True, None],
                'haloPSWeightParameter': [True, None],
                'haloPSWeightFunction': [True, None],
                'offsetSampleMean': [True, None],
                'eventGeneratorMinX': [True, None],
                'eventGeneratorMaxX': [True, None],
                'eventGeneratorMinY': [True, None],
                'eventGeneratorMaxY': [True, None],
                'eventGeneratorMinZ': [True, None],
                'eventGeneratorMaxZ': [True, None],
                'eventGeneratorMinXp': [True, None],
                'eventGeneratorMaxXp': [True, None],
                'eventGeneratorMinYp': [True, None],
                'eventGeneratorMaxYp': [True, None],
                'eventGeneratorMinZp': [True, None],
                'eventGeneratorMaxZp': [True, None],
                'eventGeneratorMinT': [True, None],
                'eventGeneratorMaxT': [True, None],
                'eventGeneratorMinEK': [True, None],
                'eventGeneratorMaxEK': [True, None],
                'eventGeneratorParticles': [True, None],
            }

    class Geant4Data(Output.Tree):
        # https://github.com/scikit-hep/uproot/issues/468
        ...

    class Options(Output.Tree):
        ...

    class Model(Output.Tree):
        @property
        def sampler_names(self):
            return self.model.sampler_names

        @property
        def collimator_names(self):
            return self.model.collimator_names

        class Model(Output.Branch):
            @property
            def sampler_names(self):
                return [e.decode('utf-8') for e in self.array('samplerNamesUnique')[0]]

            @property
            def collimator_names(self):
                return [e.decode('utf-8') for e in self.array('collimatorBranchNamesUnique')[0]]

        def to_df(self) -> _pd.DataFrame:
            """

            Returns:

            """
            model_geometry_df = _pd.DataFrame()

            # Names and strings
            for branch, name in {'Model.componentName': 'NAME',
                                 'Model.componentType': 'TYPE',
                                 'Model.material': 'MATERIAL',
                                 'Model.beamPipeType': 'APERTYPE',
                                 }.items():
                data = [_.decode('utf-8') for _ in self.trees[0].array(branch=[branch])[0]]
                model_geometry_df[name] = data

            # Scalar
            for branch, name in {'Model.length': 'L',
                                 'Model.staS': 'AT_ENTRY',
                                 'Model.midS': 'AT_CENTER',
                                 'Model.endS': 'AT_EXIT',
                                 'Model.tilt': 'TILT',
                                 'Model.k1': 'K1',
                                 'Model.k2': 'K2',
                                 'Model.k3': 'K3',
                                 'Model.k4': 'K4',
                                 'Model.k5': 'K5',
                                 'Model.k6': 'K6',
                                 'Model.k7': 'K7',
                                 'Model.k8': 'K8',
                                 'Model.k9': 'K9',
                                 'Model.k10': 'K10',
                                 'Model.k11': 'K11',
                                 'Model.k12': 'K12',
                                 'Model.k1s': 'K1S',
                                 'Model.k2s': 'K2S',
                                 'Model.k3s': 'K3S',
                                 'Model.k4s': 'K4S',
                                 'Model.k5s': 'K5S',
                                 'Model.k6s': 'K6S',
                                 'Model.k7s': 'K7S',
                                 'Model.k8s': 'K8S',
                                 'Model.k9s': 'K9S',
                                 'Model.k10s': 'K10S',
                                 'Model.k11s': 'K11S',
                                 'Model.k12s': 'K12S',
                                 'Model.bField': 'B',
                                 'Model.eField': 'E',
                                 'Model.e1': 'E1',
                                 'Model.e2': 'E2',
                                 'Model.hgap': 'HGAP',
                                 'Model.fint': 'FINT',
                                 'Model.fintx': 'FINTX'
                                 }.items():
                model_geometry_df[name] = self.trees[0].array(branch=[branch])[0]

            # Aperture
            for branch, name in {'Model.beamPipeAper1': 'APERTURE1',
                                 'Model.beamPipeAper2': 'APERTURE2',
                                 'Model.beamPipeAper3': 'APERTURE3',
                                 'Model.beamPipeAper4': 'APERTURE4'}.items():
                model_geometry_df[name] = self.trees[0].array(branch=[branch])[0]

            # Vectors
            geometry_branches = {'Model.staPos': 'ENTRY_',
                                 'Model.midPos': 'CENTER_',
                                 'Model.endPos': 'EXIT_'}
            data = self.trees[0].pandas.df(branches=geometry_branches.keys(), flatten=True)
            for branch, name in geometry_branches.items():
                data.rename({f"{branch}.fX": f"{name}X", f"{branch}.fY": f"{name}Y", f"{branch}.fZ": f"{name}Z"},
                            axis='columns', inplace=True)

            # Concatenate
            self._df = _pd.concat([model_geometry_df, data.loc[0]], axis='columns', sort=False).set_index('NAME')

            return self._df

    class Run(Output.Tree):
        def __getattr__(self, item):
            if item in (
                    'summary',
            ):
                setattr(self,
                        item,
                        getattr(BDSimOutput.Run, item.capitalize())(branch='Summary.', tree=self)
                        )
                return getattr(self, item)

        class Summary(Output.Branch):
            DEFAULT_LEAVES = {

            }

    class Event(Output.Tree):
        def __getattr__(self, item):
            if item in (
                    'eloss',
                    'eloss_vacuum',
                    'eloss_tunnel',
                    'eloss_world',
                    'eloss_world_exit',
                    'primary',
                    'primary_first_hit',
                    'primary_last_hit',
                    'aperture_impacts',
                    'histos'
            ):
                b = ''.join([i.capitalize() for i in item.split('_')])
                setattr(self,
                        item,
                        getattr(BDSimOutput.Event, b)(parent=self)
                        )
                return getattr(self, item)

            elif item == 'samplers':
                self.samplers = BDSimOutput.Event.Samplers({
                    s.rstrip('.'):
                        BDSimOutput.Event.Sampler(parent=self, branch_name=s)
                    for s in self.parent.model.sampler_names})
                return self.samplers

            elif item == 'collimators':
                self.collimators = BDSimOutput.Event.Collimators({
                    s.rstrip('.'):
                        BDSimOutput.Event.Collimator(parent=self, branch_name=s)
                    for s in self.parent.model.collimator_names})
                return self.collimators

        class ELoss(Output.Branch):
            DEFAULT_LEAVES = {
                'Eloss.n': [True, None],
                'Eloss.energy': [True, None],
                'Eloss.S': [True, None],
                'Eloss.weight': [True, None],
                'Eloss.partID': [True, None],
                'Eloss.trackID': [True, None],
                'Eloss.parentID': [True, None],
                'Eloss.modelID': [True, None],
                'Eloss.turn': [True, None],
                'Eloss.x': [True, None],
                'Eloss.y': [True, None],
                'Eloss.z': [True, None],
                'Eloss.X': [True, None],
                'Eloss.Y': [True, None],
                'Eloss.Z': [True, None],
                'Eloss.T': [True, None],
                'Eloss.stepLength': [True, None],
                'Eloss.preStepKineticEnergy': [True, None],
                'Eloss.storeTurn': [True, None],
                'Eloss.storeLinks': [True, None],
                'Eloss.storeModelID': [True, None],
                'Eloss.storeLocal': [True, None],
                'Eloss.storeGlobal': [True, None],
                'Eloss.storeTime': [True, None],
                'Eloss.storeStepLength': [True, None],
                'Eloss.storePreStepKineticEnergy': [True, None],
            }

        class ELossVacuum(ELoss):
            pass

        class ELossTunnel(ELoss):
            pass

        class ELossWorld(ELoss):
            pass

        class ELossWorldExit(ELoss):
            pass

        class Primary(Output.Branch):
            DEFAULT_LEAVES = {
                'energy': [True, None],
                'weight': [True, None],
                'partID': [False, None],
                'trackID': [False, None],
                'parentID': [False, None],
                'modelID': [False, None],
                'turnNumber': [True, None],
                'x': [True, None],
                'y': [True, None],
                'z': [True, None],
                'xp': [True, None],
                'yp': [True, None],
                'zp': [True, None],
                'T': [True, None],
                'S': [False, None],
                'r': [False, None],
                'n': [False, None],
                'rp': [False, None],
                'phi': [False, None],
                'phip': [False, None],
                'theta': [False, None],
                'charge': [False, None],
                'kineticEnergy': [False, None],
                'mass': [False, None],
                'rigidity': [False, None],
                'isIon': [False, None],
                'ionA': [False, None],
                'ionZ': [False, None],
                'nElectrons': [False, None],
            }

        class PrimaryFirstHit(Output.Branch):
            DEFAULT_LEAVES = {
                'S': [True, None],
                'energy': [True, None],
                'weight': [True, None],
                'partID': [False, None],
                'trackID': [False, None],
                'parentID': [False, None],
                'modelID': [False, None],
                'turn': [True, None],
                'x': [True, None],
                'y': [True, None],
                'z': [True, None],
                'X': [True, None],
                'Y': [True, None],
                'Z': [True, None],
                'T': [True, None],
                'stepLength': [False, None],
                'preStepKineticEnergy': [False, None],
                'n': [False, None],
                'storeTurn': [False, None],
                'storeLinks': [False, None],
                'storeModelID': [False, None],
                'storeLocal': [False, None],
                'storeGlobal': [False, None],
                'storeTime': [False, None],
                'storeStepLength': [False, None],
                'storePreStepKineticEnergy': [False, None],
            }

        class PrimaryLastHit(PrimaryFirstHit):
            pass

        class ApertureImpacts(Output.Branch):
            DEFAULT_LEAVES = {
                'n': [True, None],
                'energy': [True, None],
                'S': [True, None],
                'weight': [True, None],
                'isPrimary': [True, None],
                'firstPrimaryImpact': [True, None],
                'partID': [True, None],
                'turn': [False, None],
                'x': [False, None],
                'y': [False, None],
                'xp': [False, None],
                'yp': [False, None],
                'T': [False, None],
                'kineticEnergy': [False, None],
                'isIon': [False, None],
                'ionA': [False, None],
                'ionZ': [False, None],
                'trackID': [False, None],
                'parentID': [False, None],
                'modelID': [False, None],
            }

        class Histos(Output.Branch):
            def read_df(self) -> _pd.DataFrame:
                pass

        class Samplers(UserDict):
            def compute_optics(self, samplers: Optional[List[str]] = None):
                return _pd.DataFrame(
                    [sampler.compute_optics() for sampler in self.data.values()]
                )

            def to_df(self, samplers: Optional[List[str]] = None, columns: Optional[List[str]] = None) -> _pd.DataFrame:
                pass

            def to_np(self, samplers: Optional[List[str]] = None, columns: Optional[List[str]] = None) -> _np.ndarray:
                pass

            @property
            def df(self) -> _pd.DataFrame:
                return self.to_df()

            @property
            def np(self) -> _np.ndarray:
                return self.to_np()

            @property
            def optics(self):
                if self._optics is None:
                    self._optics = self.compute_optics()
                return self._optics

        class Sampler(Output.Branch):
            DEFAULT_LEAVES = {
                'x': [True, None],
                'xp': [True, None],
                'y': [True, None],
                'yp': [True, None],
                'z': [True, None],
                'zp': [True, None],
                'T': [True, None],
                'energy': [True, None],
                'p': [True, None],
                'turnNumber': [True, None],
                'parentID': [True, None],
                'partID': [True, None],
                'trackID': [True, None],
                'weight': [True, None],
                'n': [True, None],
                'S': [True, None],
            }

            def to_np(self,
                      turn_number: int = -1,
                      primary_only: bool = True,
                      ) -> _np.ndarray:
                df: _pd.DataFrame = self.to_df()
                data = df[['x', 'xp', 'y', 'yp', 'T', 'energy', 'n', 'S']].values
                validity = df[['turnNumber', 'parentID']].values
                if turn_number == - 1 and primary_only is False:
                    return data
                elif turn_number == -1 and primary_only is True:
                    return data[validity[:, 1] == 0]
                elif primary_only is False:
                    return data[validity[:, 0] == turn_number]
                else:
                    return data[_np.logical_and(validity[:, 1] == 0, validity[:, 0] == turn_number), :]

            def compute_optics(self):
                """

                Returns:

                """
                data = self.to_np(turn_number=1, primary_only=True)
                cv = _np.cov(data)
                eps_x = _np.sqrt(cv[0, 0] * cv[1, 1] - cv[0, 1] * cv[1, 0])
                eps_y = _np.sqrt(cv[2, 2] * cv[3, 3] - cv[2, 3] * cv[3, 2])
                return {
                    'BETA11': (cv[0, 0] - cv[0, 5] ** 2) / eps_x,
                    'BETA22': (cv[2, 2] - cv[2, 5] ** 2) / eps_y,
                    'ALPHA11': -cv[1, 1] / eps_x,
                    'ALPHA22': -cv[3, 3] / eps_y,
                    'DISP1': cv[0, 5] / 0.001,
                    'DISP2': 0.0,
                    'DISP3': cv[2, 5] / 0.001,
                    'DISP4': 0.0,
                    'EPSX': eps_x,
                    'EPXY': eps_y,
                    'n': data[:, -2].sum(),
                    'S': data[0, -1],
                }

        class Collimators(UserDict):
            def to_df(self, samplers: Optional[List[str]] = None, columns: Optional[List[str]] = None) -> _pd.DataFrame:
                ...

            def to_np(self, samplers: Optional[List[str]] = None, columns: Optional[List[str]] = None) -> _np.ndarray:
                ...

            @property
            def df(self) -> _pd.DataFrame:
                return self.to_df()

            @property
            def np(self) -> _np.ndarray:
                return self.to_np()

        class Collimator(Output.Branch):
            DEFAULT_LEAVES = {
            }


class ReBDSimOutput(Output):
    def __getattr__(self, item):
        try:
            self._root_directory.get(item.title())
        except KeyError:
            raise BDSimOutputException(f"Key {item} is invalid.")

        if item in (
            'beam',
            'event',
            'run',
            'options'
            'model_dir'
        ):
            setattr(self,
                    item,
                    Output.Directory(parent=self, directory=self._root_directory[item.title()])
                    )
        elif item == 'model':
            setattr(self,
                    item.rstrip('_'),
                    getattr(BDSimOutput, item.title())(parent=self)
                    )

        else:
            setattr(self,
                    item,
                    getattr(BDSimOutput, item.title())(parent=self)
                    )
        return getattr(self, item)


class ReBDSimOpticsOutput(ReBDSimOutput):
    def __getattr__(self, item):
        try:
            self._root_directory.get(item.title())
        except KeyError:
            raise BDSimOutputException(f"Key {item} is invalid.")

        if item.rstrip('_') in (
            'optics',
        ):
            setattr(self,
                    item.rstrip('_'),
                    getattr(ReBDSimOpticsOutput, item.rstrip('_').title())(parent=self)
                    )
            if item.endswith('_'):
                return getattr(self, item.rstrip('_'))
            else:
                return getattr(getattr(self, item.rstrip('_')), item.rstrip('_'))
        else:
            return getattr(super(), item)

    class Optics(Output.Tree):

        class Optics(Output.Branch):
            BRANCH_NAME = ''
            DEFAULT_LEAVES = {
                'Emitt_x': [True, None],
                'Emitt_y': [True, None],
                'Alpha_x': [True, None],
                'Alpha_y': [True, None],
                'Beta_x': [True, None],
                'Beta_y': [True, None],
                'Gamma_x': [True, None],
                'Gamma_y': [True, None],
                'Disp_x': [True, None],
                'Disp_y': [True, None],
                'Disp_xp': [True, None],
                'Disp_yp': [True, None],
                'Mean_x': [True, None],
                'Mean_y': [True, None],
                'Mean_xp': [True, None],
                'Mean_yp': [True, None],
                'Sigma_x': [True, None],
                'Sigma_y': [True, None],
                'Sigma_xp': [True, None],
                'Sigma_yp': [True, None],
                'S': [True, None],
                'Npart': [True, None],
                'Sigma_Emitt_x': [True, None],
                'Sigma_Emitt_y': [True, None],
                'Sigma_Alpha_x': [True, None],
                'Sigma_Alpha_y': [True, None],
                'Sigma_Beta_x': [True, None],
                'Sigma_Beta_y': [True, None],
                'Sigma_Gamma_x': [True, None],
                'Sigma_Gamma_y': [True, None],
                'Sigma_Disp_x': [True, None],
                'Sigma_Disp_y': [True, None],
                'Sigma_Disp_xp': [True, None],
                'Sigma_Disp_yp': [True, None],
                'Sigma_Mean_x': [True, None],
                'Sigma_Mean_y': [True, None],
                'Sigma_Mean_xp': [True, None],
                'Sigma_Mean_yp': [True, None],
                'Sigma_Sigma_x': [True, None],
                'Sigma_Sigma_y': [True, None],
                'Sigma_Sigma_xp': [True, None],
                'Sigma_Sigma_yp': [True, None],
                'Mean_E': [True, None],
                'Mean_t': [True, None],
                'Sigma_E': [True, None],
                'Sigma_t': [True, None],
                'Sigma_Mean_E': [True, None],
                'Sigma_Mean_t': [True, None],
                'Sigma_Sigma_E': [True, None],
                'Sigma_Sigma_t': [True, None],
                'xyCorrelationCoefficent': [True, None],
            }
