"""
A Pythonic way to analyze and work with Beam Delivery SIMulation (BDSIM) ROOT output files.

Design goals:
 - No dependency on (py)ROOT(py) is needed. The module uses `uproot` instead.
 - Enables and favors exploration of the ROOT files. No prior knowledge of the content should be required
 to explore and discover the data structure.
 - provide analysis tools exploiting the new Awkward 1.0 library (https://arxiv.org/pdf/2001.06307.pdf)
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Union
from collections import UserDict
import logging
import os
import numpy as _np
import pandas as _pd
from scipy.interpolate import interp1d

try:
    import uproot as _uproot
except (ImportError, ImportWarning):
    logging.warning("Uproot is required for this module to have full functionalities.\n")
    raise ImportError("Uproot is required for this module to work.")

_WITH_PYBDSIM = False
try:
    try:
        import warnings

        warnings.simplefilter("ignore")
        import pybdsim

        pybdsim.Data.LoadROOTLibraries()
        warnings.simplefilter("default")
    except (ImportError, UserWarning):
        pass
    _WITH_PYBDSIM = True
except (ImportError, ImportWarning):
    logging.warning("pybdsim is required for this module to have full functionalities.\n"
                    "Not all methods will be available.")

_WITH_ROOT = False
try:
    try:
        import warnings

        warnings.simplefilter("ignore")
        import ROOT

        ROOT.gSystem.Load('librebdsim')
        warnings.simplefilter("default")
    except (ImportError, UserWarning):
        pass
    _WITH_ROOT = True
except (ImportError, ImportWarning):
    logging.warning("ROOT is required for this module to have full functionalities.\n"
                    "Not all methods will be available.")

_WITH_BOOST_HISTOGRAM = False
try:
    try:
        import warnings

        warnings.simplefilter("ignore")
        import boost_histogram as bh

        warnings.simplefilter("default")
    except (ImportError, UserWarning):
        pass
    _WITH_BOOST_HISTOGRAM = True
except (ImportError, ImportWarning):
    logging.warning("boost_histogram is required for this module to have full functionalities.\n"
                    "Not all methods will be available.")


__all__ = [
    'Output',
    'BDSimOutputException',
    'BDSimOutput',
    'ReBDSimOutput',
    'ReBDSimOpticsOutput',
    'ReBDSimCombineOutput',
    'Histogram',
    'Histogram2d',
    'Histogram3d'
]


class BDSimOutputException(Exception):
    pass


class Histogram:
    def __init__(self, h):  # h is a THxD
        self._h = h
        self.variances = h.variances()
        self._centers = None
        self._normalized_values = None
        self.normalization = 1.0
        self.coordinates_normalization = 1.0

    def __getattr__(self, item):
        return getattr(self._h, item)

    def set_normalization(self, normalization: float = 1.0):
        self.normalization = normalization
        self._normalized_values = self.normalization * self._h.values
        return self

    def set_coordinates_normalization(self, normalization: float = 1.0):
        self.coordinates_normalization = normalization
        return self

    @property
    def normalized_values(self):
        return self._normalized_values

    @property
    def xnumbins(self):
        return len(self._h.axes[0].edges())-1


class Histogram1d(Histogram):

    @property
    def centers(self):
        if self._centers is not None:
            return self._centers
        self._centers = [
            self.coordinates_normalization * (self._h.axes[0].edges()[i] + self._h.axes[0].edges()[i + 1]) / 2
            for i in range(1, len(self._h.axes[0].edges()) - 2)
        ]
        return self._centers

    @property
    def values(self):
        return self._h.values()


class Histogram2d(Histogram):
    ...


class Histogram3d(Histogram):
    def __init__(self, h, parent, name):
        Histogram.__init__(self, h)
        self._filename = parent.filename
        self._path = parent.path
        self._name = name
        self._meshname = self._name.split('-')[0]
        self._parent = parent

    @property
    def filename(self):
        return self._filename

    @property
    def path(self):
        return self._path

    @property
    def meshname(self):
        return self._meshname

    @property
    def values(self):
        return self._h.values()

    @property
    def bins_volume(self):
        return _np.diff(self._h.axes[0].edges())[0] * \
               _np.diff(self._h.axes[1].edges())[0] * \
               _np.diff(self._h.axes[2].edges())[0]

    @property
    def edges(self):
        return _np.array([list(self._h.axes[0].edges()),
                          list(self._h.axes[1].edges()),
                          list(self._h.axes[2].edges())])

    @property
    def centers(self):
        if self._centers is not None:
            return self._centers
        self._centers = [[
            self.coordinates_normalization * (self.edges[j][i] + self.edges[j][i + 1]) / 2
            for i in range(len(self.edges[j])-1)] for j in range(3)]
        return self._centers

    @property
    def ynumbins(self):
        return len(self._h.axes[1].edges())-1

    @property
    def znumbins(self):
        return len(self._h.axes[2].edges())-1

    @property
    def scoring_mesh_translations(self):
        dico = self._parent.model.scoring_mesh_translations[self._meshname]
        return [dico['fX'], dico['fY'], dico['fZ']]

    @property
    def scoring_mesh_rotations(self):
        return self._parent.model.scoring_mesh_rotations[self._meshname]

    def to_df(self):
        index = _pd.MultiIndex.from_product(self.centers, names=('X', 'Y', 'Z'))
        data = {
            'edep': self.normalized_values.flatten() / self.bins_volume,
        }
        return _pd.DataFrame(index=index, data=data)

    def to_csv(self, filename='histogram.csv', path='.', **kwargs):
        self.to_df().to_csv(os.path.join(path, filename), header=False, float_format='% 11.7E', **kwargs)


class Histogram4d:
    def __init__(self, parent, bdsbh4d_histogram, name):
        self._h = bdsbh4d_histogram
        self._bh = None
        self._bh_error = None
        self._energy_axis_type = name.split('-')[-1]
        self._filename = parent.filename
        self._path = parent.path
        self._meshname = name.split('-')[0]
        self._cache = None
        self._coordinates_normalization = 1.0
        self._parent = parent

    @property
    def scoring_mesh_translations(self):
        dico = self._parent.model.scoring_mesh_translations[self._meshname]
        return [dico['fX'], dico['fY'], dico['fZ']]

    @property
    def scoring_mesh_rotations(self):
        return self._parent.model.scoring_mesh_rotations[self._meshname]

    def get_pyboost(self, hist_type):
        if _WITH_BOOST_HISTOGRAM:
            energy_axis = None
            if self._energy_axis_type == 'log':
                energy_axis = bh.axis.Regular(self._h.h_nebins, self._h.h_emin, self._h.h_emax,
                                              transform=bh.axis.transform.log)
            elif self._energy_axis_type == 'linear':
                energy_axis = bh.axis.Regular(self._h.h_nebins, self._h.h_emin, self._h.h_emax)
            elif self._energy_axis_type == 'user':
                energy_axis = bh.axis.Variable(self._h.h_ebinsedges)

            histo4d = bh.Histogram(
                bh.axis.Regular(self._h.h_nxbins, self._h.h_xmin, self._h.h_xmax),
                bh.axis.Regular(self._h.h_nybins, self._h.h_ymin, self._h.h_ymax),
                bh.axis.Regular(self._h.h_nzbins, self._h.h_zmin, self._h.h_zmax),
                energy_axis)

            for x in range(self._h.h_nxbins):
                for y in range(self._h.h_nybins):
                    for z in range(self._h.h_nzbins):
                        for e in range(self._h.h_nebins):
                            histo4d[x, y, z, e] = getattr(self._h, hist_type).at(x, y, z, e)

            return histo4d
        else:
            raise AttributeError("Boost histograms are not available")

    def extract_spectrum(self, x=0, y=0, z=0, path='.', extract_all=False):

        if not extract_all:
            all_x = [x]
            all_y = [y]
            all_z = [z]
        else:
            all_x = range(self.xnumbins)
            all_y = range(self.ynumbins)
            all_z = range(self.znumbins)

        for _x in all_x:
            for _y in all_y:
                for _z in all_z:

                    f = open(f"{path}/fluxes_{self.meshname}_{_x}_{_y}_{_z}", 'w')
                    spectrum = list(self.bh[_x, _y, _z, :].to_numpy()[0])
                    spectrum.reverse()

                    i = 1
                    for value in spectrum:
                        f.write("  {:.4E}".format(value))
                        if i % 6 == 0:
                            f.write('\n')
                        i += 1
                    f.write('\n 1.000\n')
                    f.write(f'fluxes_{self.meshname}_{_x}_{_y}_{_z}')

                    f.close()

    def project_to_3d(self, weights=1):

        histo3d = bh.Histogram(*self.bh.axes[0:3])

        for x in range(self.bh.shape[0]):
            for y in range(self.bh.shape[1]):
                for z in range(self.bh.shape[2]):
                    tmp = self.bh[x, y, z, :].view() * weights
                    histo3d[x, y, z] = tmp.sum()

        self._cache = histo3d.view()
        return histo3d

    def compute_h10(self, conversionfactorfile):
        data = _pd.read_table(conversionfactorfile, names=["energy", "h10_coeff"])
        f = interp1d(data['energy'].values, data['h10_coeff'].values)
        self._cache = self.project_to_3d(weights=f(self.bh.axes[3].centers)).view()
        return self.project_to_3d(weights=f(self.bh.axes[3].centers))

    @property
    def filename(self):
        return self._filename

    @property
    def path(self):
        return self._path

    @property
    def meshname(self):
        return self._meshname

    @property
    def bh(self):
        if self._bh is None:
            self._bh = self.get_pyboost('h')
        return self._bh

    @property
    def bh_err(self):
        if self._bh_error is None:
            self._bh_error = self.get_pyboost('h_err')
        return self._bh_error

    @property
    def h(self):
        return self._h.h

    @property
    def h_err(self):
        return self._h.h_err

    @property
    def values(self):
        return self._cache

    @property
    def xnumbins(self):
        return self.bh.axes[0].size

    @property
    def ynumbins(self):
        return self.bh.axes[1].size

    @property
    def znumbins(self):
        return self.bh.axes[2].size

    @property
    def coordinates_normalization(self):
        return self._coordinates_normalization

    @property
    def edges(self):
        edges_x = self.bh.axes[0].edges
        edges_y = self.bh.axes[1].edges
        edges_z = self.bh.axes[2].edges
        return _np.array([edges_x, edges_y, edges_z])

    @staticmethod
    def from_root_file(parent, name):
        energy_axis_type = name.split('-')[-1]
        if energy_axis_type == "linear":
            bdsbh4d = ROOT.BDSBH4D("boost_histogram_linear")()
        elif energy_axis_type == "log":
            bdsbh4d = ROOT.BDSBH4D("boost_histogram_log")()
        elif energy_axis_type == "user":
            bdsbh4d = ROOT.BDSBH4D("boost_histogram_variable")()

        # to_PyROOT() is a BDSIM function
        bdsbh4d.to_PyROOT(parent.file, name)

        return Histogram4d(parent, bdsbh4d, name)


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
        self._filename = filename
        self._path = path
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
    def directory(self) -> _uproot.rootio.ROOTDirectory:
        """Return the master directory attached to this parent."""
        return self._root_directory

    @property
    def file(self):
        return self._file

    @property
    def filename(self):
        return self._filename

    @property
    def path(self):
        return self._path


    class Directory:
        def __init__(self, parent: Union[Output, Output.Directory], directory: _uproot.rootio.ROOTDirectory):
            """
            A representation of a (nested) structure of ROOT directories.

            Args:
                parent: the `Output` to which the directory structure is attached
                directory: the top-level ROOT directory
            """

            self._output: Output = parent
            self._directory: _uproot.rootio.ROOTDirectory = directory
            for name, cls in self._directory.iterclassnames(recursive=False):
                setattr(self, name.split(';')[0].replace('-', '_'), self.build(name, cls))

        def __getitem__(self, item):
            return self._directory[item]

        def build(self, n, c):
            item = self._directory[n]
            if c.endswith('Directory'):
                return Output.Directory(self.parent, directory=item)
            elif c.endswith('TH1D'):
                return Histogram1d(item)
            elif c.endswith('TH2D'):
                return Histogram2d(item)
            elif c.endswith('TH3D'):
                return Histogram3d(item, self.parent, n)
            elif c.endswith('TTree'):
                return Histogram4d.from_root_file(self.parent, n.split(';')[0])
            else:
                return item

        def get(self, name):
            for n, c in self._directory.iterclassnames(recursive=False):
                if n == name:
                    return self.build(n, c)

        @property
        def parent(self) -> Output:
            """The parent Output to which the directory structure is attached."""
            return self._output

        @property
        def root_directory(self) -> _uproot.rootio.ROOTDirectory:
            """The associated uproot directory."""
            return self._directory

        @property
        def keys(self) -> List[str]:
            """The content of the directory."""
            return self._directory.keys()

    class Tree:
        def __init__(self, parent: Output, tree_name: str = None):
            """
            A representation of a ROOT TTree structure.

            Args:
                parent: the `Output` to which the parent is attached
            """
            self._parent = parent
            self._tree_name = tree_name or self.__class__.__name__
            self._tree: _uproot.rootio.TTree = parent[self._tree_name]
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
                _ = branch_class(parent=self, branch_name=self.__class__.__name__)
                setattr(self, b, _)
                return getattr(self, b)
            else:
                raise AttributeError(f"Branch {b} does not exist for {self._tree_name}")

        def array(self, branch=None, **kwargs) -> _np.ndarray:
            """A proxy for the `uproot` `array` method."""
            return self._tree[self.__class__.__name__ + '.'][self.__class__.__name__ + '.' + branch].array().tolist()[0]

        def arrays(self, branches=None, **kwargs):
            """A proxy for the `uproot` `arrays` method."""
            return [self._tree[self.__class__.__name__ + '.'][self.__class__.__name__ + '.' + b].array().tolist()[0] for
                    b in branches]

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
            return [b for b in self.tree.keys() if '/' not in b]

        @property
        def branches(self) -> List[_uproot.rootio.TBranchElement]:
            return [b for b in self.tree.values() if b.name[-1] == '.']

        @property
        def numentries(self) -> int:
            """Provides the number of entries in the parent (without reading the entire file)."""
            return self.tree.num_entries

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
                    self._active_leaves[leave][0] = not self._active_leaves[leave][0]
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
            self._active_leaves: Dict[str, Tuple[bool, Optional[str]]] = self.DEFAULT_LEAVES.copy()

        def __getitem__(self, item):
            return self._branch[item]

        def array(self, branch=None, **kwargs) -> _np.ndarray:
            """A proxy for the `uproot` `array` method."""
            return self.parent.array(branch=branch, **kwargs)

        def arrays(self, branches=None, **kwargs):
            """A proxy for the uproot method.
            """
            if branches is None:
                branches = self._active_leaves
            return self.parent.arrays(branches=[b for b in branches], **kwargs)

        def pandas(self, branches: List[str] = None, strip_prefix: bool = True, **kwargs):
            """A proxy for the uproot method.
            """
            if branches is None:
                branches = [self.branch_name + b for b, _ in self._active_leaves.items() if _[0] is True]
            else:
                branches = [self.branch_name + b for b in branches]
            df = self.parent.tree.arrays(branches, library='pandas')
            if strip_prefix:
                import re
                df.columns = [re.split(self.branch_name, c)[1] for c in df.columns]
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
            name = self.branch.name
            if not name.endswith('.'):
                return name + '.'
            else:
                return name

        @property
        def leaves(self) -> List[_uproot.rootio.TBranchElement]:
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
                'event'
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
        class OptionsBase(Output.Branch):
            BRANCH_NAME = 'Options.GMAD::OptionsBase'
            DEFAULT_LEAVES = {
                'inputFileName': [True, None],
                'visMacroFileName': [True, None],
                'geant4MacroFileName': [True, None],
                'visDebug': [True, None],
                'outputFileName': [True, None],
                'outputFormat': [True, None],
                'outputDoublePrecision': [True, None],
                'survey': [True, None],
                'surveyFileName': [True, None],
                'batch': [True, None],
                'verbose': [True, None],
                'verboseRunLevel': [True, None],
                'verboseEventBDSIM': [True, None],
                'verboseEventLevel': [True, None],
                'verboseEventStart': [True, None],
                'verboseEventContinueFor': [True, None],
                'verboseTrackingLevel': [True, None],
                'verboseSteppingBDSIM': [True, None],
                'verboseSteppingLevel': [True, None],
                'verboseSteppingEventStart': [True, None],
                'verboseSteppingEventContinueFor': [True, None],
                'verboseSteppingPrimaryOnly': [True, None],
                'verboseImportanceSampling': [True, None],
                'circular': [True, None],
                'seed': [True, None],
                'nGenerate': [True, None],
                'recreate': [True, None],
                'recreateFileName': [True, None],
                'startFromEvent': [True, None],
                'writeSeedState': [True, None],
                'useASCIISeedState': [True, None],
                'seedStateFileName': [True, None],
                'generatePrimariesOnly': [True, None],
                'exportGeometry': [True, None],
                'exportType': [True, None],
                'exportFileName': [True, None],
                'bdsimPath': [True, None],
                'physicsList': [True, None],
                'physicsVerbose': [True, None],
                'physicsVerbosity': [True, None],
                'physicsEnergyLimitLow': [True, None],
                'physicsEnergyLimitHigh': [True, None],
                'g4PhysicsUseBDSIMRangeCuts': [True, None],
                'g4PhysicsUseBDSIMCutsAndLimits': [True, None],
                'eventOffset': [True, None],
                'recreateSeedState': [True, None],
                'elossHistoBinWidth': [True, None],
                'ffact': [True, None],
                'beamlineX': [True, None],
                'beamlineY': [True, None],
                'beamlineZ': [True, None],
                'beamlinePhi': [True, None],
                'beamlineTheta': [True, None],
                'beamlinePsi': [True, None],
                'beamlineAxisX': [True, None],
                'beamlineAxisY': [True, None],
                'beamlineAxisZ': [True, None],
                'beamlineAngle': [True, None],
                'beamlineAxisAngle': [True, None],
                'beamlineS': [True, None],
                'eventNumberOffset': [True, None],
                'checkOverlaps': [True, None],
                'xsize': [True, None],
                'ysize': [True, None],
                'magnetGeometryType': [True, None],
                'outerMaterialName': [True, None],
                'horizontalWidth': [True, None],
                'thinElementLength': [True, None],
                'hStyle': [True, None],
                'vhRatio': [True, None],
                'coilWidthFraction': [True, None],
                'coilHeightFraction': [True, None],
                'ignoreLocalMagnetGeometry': [True, None],
                'preprocessGDML': [True, None],
                'preprocessGDMLSchema': [True, None],
                'dontSplitSBends': [True, None],
                'yokeFields': [True, None],
                'includeFringeFields': [True, None],
                'includeFringeFieldsCavities': [True, None],
                'beampipeThickness': [True, None],
                'apertureType': [True, None],
                'aper1': [True, None],
                'aper2': [True, None],
                'aper3': [True, None],
                'aper4': [True, None],
                'beampipeMaterial': [True, None],
                'ignoreLocalAperture': [True, None],
                'vacMaterial': [True, None],
                'emptyMaterial': [True, None],
                'worldMaterial': [True, None],
                'worldGeometryFile': [True, None],
                'autoColourWorldGeometryFile': [True, None],
                'importanceWorldGeometryFile': [True, None],
                'importanceVolumeMap': [True, None],
                'worldVolumeMargin': [True, None],
                'vacuumPressure': [True, None],
                'buildTunnel': [True, None],
                'buildTunnelStraight': [True, None],
                'tunnelType': [True, None],
                'tunnelThickness': [True, None],
                'tunnelSoilThickness': [True, None],
                'tunnelMaterial': [True, None],
                'soilMaterial': [True, None],
                'buildTunnelFloor': [True, None],
                'tunnelFloorOffset': [True, None],
                'tunnelAper1': [True, None],
                'tunnelAper2': [True, None],
                'tunnelVisible': [True, None],
                'tunnelOffsetX': [True, None],
                'tunnelOffsetY': [True, None],
                'removeTemporaryFiles': [True, None],
                'samplerDiameter': [True, None],
                'turnOnOpticalAbsorption': [True, None],
                'turnOnMieScattering': [True, None],
                'turnOnRayleighScattering': [True, None],
                'turnOnOpticalSurface': [True, None],
                'scintYieldFactor': [True, None],
                'maximumPhotonsPerStep': [True, None],
                'maximumBetaChangePerStep': [True, None],
                'maximumTracksPerEvent': [True, None],
                'minimumKineticEnergy': [True, None],
                'minimumKineticEnergyTunnel': [True, None],
                'minimumRange': [True, None],
                'defaultRangeCut': [True, None],
                'prodCutPhotons': [True, None],
                'prodCutElectrons': [True, None],
                'prodCutPositrons': [True, None],
                'prodCutProtons': [True, None],
                'neutronTimeLimit': [True, None],
                'neutronKineticEnergyLimit': [True, None],
                'useLENDGammaNuclear': [True, None],
                'useElectroNuclear': [True, None],
                'useMuonNuclear': [True, None],
                'useGammaToMuMu': [True, None],
                'usePositronToMuMu': [True, None],
                'usePositronToHadrons': [True, None],
                'collimatorsAreInfiniteAbsorbers': [True, None],
                'tunnelIsInfiniteAbsorber': [True, None],
                'defaultBiasVacuum': [True, None],
                'defaultBiasMaterial': [True, None],
                'integratorSet': [True, None],
                'lengthSafety': [True, None],
                'lengthSafetyLarge': [True, None],
                'maximumTrackingTime': [True, None],
                'maximumStepLength': [True, None],
                'maximumTrackLength': [True, None],
                'chordStepMinimum': [True, None],
                'chordStepMinimumYoke': [True, None],
                'deltaIntersection': [True, None],
                'minimumEpsilonStep': [True, None],
                'maximumEpsilonStep': [True, None],
                'deltaOneStep': [True, None],
                'stopSecondaries': [True, None],
                'killNeutrinos': [True, None],
                'minimumRadiusOfCurvature': [True, None],
                'sampleElementsWithPoleface': [True, None],
                'nominalMatrixRelativeMomCut': [True, None],
                'teleporterFullTransform': [True, None],
                'sensitiveOuter': [True, None],
                'sensitiveBeamPipe': [True, None],
                'numberOfEventsPerNtuple': [True, None],
                'storeApertureImpacts': [True, None],
                'storeApertureImpactsIons': [True, None],
                'storeApertureImpactsAll': [True, None],
                'apertureImpactsMinimumKE': [True, None],
                'storeCollimatorInfo': [True, None],
                'storeCollimatorHits': [True, None],
                'storeCollimatorHitsLinks': [True, None],
                'storeCollimatorHitsIons': [True, None],
                'storeCollimatorHitsAll': [True, None],
                'collimatorHitsMinimumKE': [True, None],
                'storeEloss': [True, None],
                'storeElossHistograms': [True, None],
                'storeElossVacuum': [True, None],
                'storeElossVacuumHistograms': [True, None],
                'storeElossTunnel': [True, None],
                'storeElossTunnelHistograms': [True, None],
                'storeElossWorld': [True, None],
                'storeElossWorldContents': [True, None],
                'storeElossTurn': [True, None],
                'storeElossLinks': [True, None],
                'storeElossLocal': [True, None],
                'storeElossGlobal': [True, None],
                'storeElossTime': [True, None],
                'storeElossStepLength': [True, None],
                'storeElossPreStepKineticEnergy': [True, None],
                'storeElossModelID': [True, None],
                'storeGeant4Data': [True, None],
                'storeTrajectory': [True, None],
                'storeTrajectoryDepth': [True, None],
                'storeTrajectoryParticle': [True, None],
                'storeTrajectoryParticleID': [True, None],
                'storeTrajectoryEnergyThreshold': [True, None],
                'storeTrajectorySamplerID': [True, None],
                'storeTrajectoryELossSRange': [True, None],
                'storeTrajectoryTransportationSteps': [True, None],
                'trajNoTransportation': [True, None],
                'storeTrajectoryLocal': [True, None],
                'storeTrajectoryLinks': [True, None],
                'storeTrajectoryIon': [True, None],
                'trajectoryFilterLogicAND': [True, None],
                'storeSamplerAll': [True, None],
                'storeSamplerPolarCoords': [True, None],
                'storeSamplerCharge': [True, None],
                'storeSamplerKineticEnergy': [True, None],
                'storeSamplerMass': [True, None],
                'storeSamplerRigidity': [True, None],
                'storeSamplerIon': [True, None],
                'trajCutGTZ': [True, None],
                'trajCutLTR': [True, None],
                'trajConnect': [True, None],
                'writePrimaries': [True, None],
                'storeModel': [True, None],
                'nturns': [True, None],
                'ptcOneTurnMapFileName': [True, None],
                'printFractionEvents': [True, None],
                'printFractionTurns': [True, None],
                'printPhysicsProcesses': [True, None],
                'nSegmentsPerCircle': [True, None],
                'nbinsx': [True, None],
                'nbinsy': [True, None],
                'nbinsz': [True, None],
                'xmin': [True, None],
                'xmax': [True, None],
                'ymin': [True, None],
                'ymax': [True, None],
                'zmin': [True, None],
                'zmax': [True, None],
                'useScoringMap': [True, None],
            }

    class Model(Output.Tree):
        @property
        def component_names(self):
            return self.model.component_names

        @property
        def placement_names(self):
            return self.model.placement_names

        @property
        def sampler_names(self):
            return self.model.sampler_names

        @property
        def collimator_names(self):
            return self.model.collimator_names

        @property
        def scoring_mesh_names(self):
            return self.model.scoring_mesh_names

        @property
        def scoring_mesh_translations(self):
            return self.model.scoring_mesh_translations

        @property
        def scoring_mesh_rotations(self):
            return self.model.scoring_mesh_rotations

        class Model(Output.Branch):

            DEFAULT_LEAVES = {
                'n': [True, None],
                'samplerNamesUnique': [False, None],
                'componentName': [False, None],
                'placementName': [False, None],
                'componentType': [True, None],
                'length': [True, None],
                'staPos': [True, None],
                'midPos': [True, None],
                'endPos': [True, None],
                'staRot': [False, None],
                'midRot': [False, None],
                'endRot': [False, None],
                'staRefPos': [True, None],
                'midRefPos': [True, None],
                'endRefPos': [True, None],
                'staRefRot': [True, None],
                'midRefRot': [True, None],
                'endRefRot': [True, None],
                'tilt': [True, None],
                'offsetX': [True, None],
                'offsetY': [True, None],
                'staS': [True, None],
                'midS': [True, None],
                'endS': [True, None],
                'beamPipeType': [True, None],
                'beamPipeAper1': [True, None],
                'beamPipeAper2': [True, None],
                'beamPipeAper3': [True, None],
                'beamPipeAper4': [True, None],
                'material': [True, None],
                'k1': [True, None],
                'k2': [True, None],
                'k3': [True, None],
                'k4': [False, None],
                'k5': [False, None],
                'k6': [False, None],
                'k7': [False, None],
                'k8': [False, None],
                'k9': [False, None],
                'k10': [False, None],
                'k11': [False, None],
                'k12': [False, None],
                'k1s': [False, None],
                'k2s': [False, None],
                'k3s': [False, None],
                'k4s': [False, None],
                'k5s': [False, None],
                'k6s': [False, None],
                'k7s': [False, None],
                'k8s': [False, None],
                'k9s': [False, None],
                'k10s': [False, None],
                'k11s': [False, None],
                'k12s': [False, None],
                'ks': [False, None],
                'hkick': [True, None],
                'vkick': [True, None],
                'bField': [True, None],
                'eField': [True, None],
                'e1': [True, None],
                'e2': [True, None],
                'hgap': [True, None],
                'fint': [True, None],
                'fintx': [True, None],
                'fintk2': [True, None],
                'fintxk2': [True, None],
                'storeCollimatorInfo': [False, None],
                'collimatorIndices': [False, None],
                'collimatorIndicesByName': [False, None],
                'nCollimators': [False, None],
                'collimatorInfo': [False, None],
                'collimatorBranchNamesUnique': [False, None],
                'scoringMeshName': [False, None],
                'scoringMeshRotation': [False, None],
                'scoringMeshTranslation': [False, None]
            }

            @property
            def component_names(self):
                return self.array('componentName')

            @property
            def placement_names(self):
                return self.array('placementName')

            @property
            def sampler_names(self):
                return self.array('samplerNamesUnique')

            @property
            def collimator_names(self):
                return self.array('collimatorBranchNamesUnique')

            @property
            def scoring_mesh_names(self):
                return self.array('scoringMeshName')

            @property
            def scoring_mesh_translations(self):
                return dict(tuple(self.array('scoringMeshTranslation')))

            @property
            def scoring_mesh_rotations(self):
                return dict(tuple(self.array('scoringMeshRotation')))

        def to_df(self) -> _pd.DataFrame:
            """

            Returns:

            """
            model_geometry_df = _pd.DataFrame()

            # Names and strings
            for branch, name in {'componentName': 'NAME',
                                 'componentType': 'TYPE',
                                 'material': 'MATERIAL',
                                 'beamPipeType': 'APERTYPE',
                                 }.items():
                data = self.array(branch=branch)
                model_geometry_df[name] = data

            # Scalar
            for branch, name in {'length': 'L',
                                 'staS': 'AT_ENTRY',
                                 'midS': 'AT_CENTER',
                                 'endS': 'AT_EXIT',
                                 'tilt': 'TILT',
                                 'k1': 'K1',
                                 'k2': 'K2',
                                 'k3': 'K3',
                                 'k4': 'K4',
                                 'k5': 'K5',
                                 'k6': 'K6',
                                 'k7': 'K7',
                                 'k8': 'K8',
                                 'k9': 'K9',
                                 'k10': 'K10',
                                 'k11': 'K11',
                                 'k12': 'K12',
                                 'k1s': 'K1S',
                                 'k2s': 'K2S',
                                 'k3s': 'K3S',
                                 'k4s': 'K4S',
                                 'k5s': 'K5S',
                                 'k6s': 'K6S',
                                 'k7s': 'K7S',
                                 'k8s': 'K8S',
                                 'k9s': 'K9S',
                                 'k10s': 'K10S',
                                 'k11s': 'K11S',
                                 'k12s': 'K12S',
                                 'bField': 'B',
                                 'eField': 'E',
                                 'e1': 'E1',
                                 'e2': 'E2',
                                 'hgap': 'HGAP',
                                 'fint': 'FINT',
                                 'fintx': 'FINTX'
                                 }.items():
                model_geometry_df[name] = self.array(branch=branch)

            # Aperture
            for branch, name in {'beamPipeAper1': 'APERTURE1',
                                 'beamPipeAper2': 'APERTURE2',
                                 'beamPipeAper3': 'APERTURE3',
                                 'beamPipeAper4': 'APERTURE4'}.items():
                model_geometry_df[name] = self.array(branch=branch)

            # Vectors
            geometry_branches = {'staPos': 'ENTRY_',
                                 'midPos': 'CENTER_',
                                 'endPos': 'EXIT_'}

            data_df = _pd.DataFrame()
            for branch, name in geometry_branches.items():
                data = _pd.DataFrame(self.array(branch)).rename({"fX": f"{name}X", "fY": f"{name}Y", "fZ": f"{name}Z"},
                                                                axis='columns')
                data_df = _pd.concat([data_df, data], axis='columns')

            # Concatenate
            self._df = _pd.concat([model_geometry_df, data_df], axis='columns', sort=False).set_index('NAME')

            return self._df

    class Run(Output.Tree):

        class Summary(Output.Branch):
            DEFAULT_LEAVES = {
                'startTime': [True, None],
                'stopTime': [True, None],
                'durationWall': [True, None],
                'durationCPU': [True, None],
                'seedStateAtStart': [True, None],
            }

        class Histos(Output.Branch):
            DEFAULT_LEAVES = {

            }

    class Event(Output.Tree):
        def __getattr__(self, item):
            if item == 'samplers':
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
            else:
                return super().__getattr__(item)

        class Eloss(Output.Branch):
            DEFAULT_LEAVES = {
                'n': [True, None],
                'energy': [True, None],
                'S': [True, None],
                'weight': [True, None],
                'partID': [False, None],
                'trackID': [False, None],
                'parentID': [False, None],
                'modelID': [False, None],
                'turn': [False, None],
                'x': [False, None],
                'y': [False, None],
                'z': [False, None],
                'X': [False, None],
                'Y': [False, None],
                'Z': [False, None],
                'T': [False, None],
                'stepLength': [False, None],
                'preStepKineticEnergy': [False, None],
                'storeTurn': [True, None],
                'storeLinks': [True, None],
                'storeModelID': [True, None],
                'storeLocal': [True, None],
                'storeGlobal': [True, None],
                'storeTime': [True, None],
                'storeStepLength': [True, None],
                'storePreStepKineticEnergy': [True, None],
            }

        class ELossVacuum(Eloss):
            pass

        class ELossTunnel(Eloss):
            pass

        class ELossWorld(Eloss):
            pass

        class ELossWorldExit(Eloss):
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
                'p': [True, None],
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
                'kineticEnergy': [True, None],
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
                'primaryInteracted': [True, None],
                'primaryStopped': [True, None],
                'n': [True, None],
                'energy': [False, None],
                'energyDeposited': [False, None],
                'xln': [False, None],
                'yln': [False, None],
                'zln': [False, None],
                'xpln': [False, None],
                'ypln': [False, None],
                'T': [False, None],
                'weight': [False, None],
                'partID': [False, None],
                'parentID': [False, None],
                'turn': [False, None],
                'firstPrimaryHitThisTurn': [False, None],
                'impactParameterX': [False, None],
                'impactParameterY': [False, None],
                'isIon': [False, None],
                'ionA': [False, None],
                'ionZ': [False, None],
                'turnSet': [False, None],
                'charge': [False, None],
                'kineticEnergy': [False, None],
                'mass': [False, None],
                'rigidity': [False, None],
                'totalEnergyDeposited': [False, None],
            }


class ReBDSimOutput(Output):
    def __getattr__(self, item):
        if item in (
                'beam',
                'event',
                'run',
                'options',
                'model_dir'
        ):
            try:
                self._root_directory.get(item.split('_')[0].title())
            except KeyError:
                raise BDSimOutputException(f"Key {item} is invalid.")
            setattr(self,
                    item,
                    Output.Directory(parent=self, directory=self._root_directory[item.split('_')[0].title()])
                    )
        elif item == 'model':
            setattr(self,
                    item,
                    getattr(BDSimOutput, item.title())(parent=self, tree_name='ModelTree'),
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


class ReBDSimCombineOutput(ReBDSimOutput):
    def __getattr__(self, item):
        if item not in ('event', 'header'):
            return None
        return super().__getattr__(item)
