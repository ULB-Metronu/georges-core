"""
TODO
"""
import matplotlib.colors
from .artist import Artist as _Artist

FLUKA_COLORS = [(1.0, 1.0, 1.0), (0.9, 0.6, 0.9), (1.0, 0.4, 1.0), (0.9, 0.0, 1.0),
                (0.7, 0.0, 1.0), (0.5, 0.0, 0.8), (0.0, 0.0, 0.8),
                (0.0, 0.0, 1.0), (0.0, 0.6, 1.0), (0.0, 0.8, 1.0), (0.0, 0.7, 0.5),
                (0.0, 0.9, 0.2), (0.5, 1.0, 0.0), (0.8, 1.0, 0.0),
                (1.0, 1.0, 0.0), (1.0, 0.8, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.0),
                (0.8, 0.0, 0.0), (0.6, 0.0, 0.0), (0.0, 0.0, 0.0)]
FlukaColormap = matplotlib.colors.LinearSegmentedColormap.from_list('fluka', FLUKA_COLORS, N=300)


class MatplotlibArtist(_Artist):
    """
    TODO
    """
    pass
