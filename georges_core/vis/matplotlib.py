"""
TODO
"""
import logging

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors
from matplotlib.ticker import FixedLocator, MultipleLocator, FixedFormatter
from ..units import ureg as _ureg
from .artist import Artist as _Artist
from .artist import PALETTE

FLUKA_COLORS = [(1.0, 1.0, 1.0), (0.9, 0.6, 0.9), (1.0, 0.4, 1.0), (0.9, 0.0, 1.0),
                (0.7, 0.0, 1.0), (0.5, 0.0, 0.8), (0.0, 0.0, 0.8),
                (0.0, 0.0, 1.0), (0.0, 0.6, 1.0), (0.0, 0.8, 1.0), (0.0, 0.7, 0.5),
                (0.0, 0.9, 0.2), (0.5, 1.0, 0.0), (0.8, 1.0, 0.0),
                (1.0, 1.0, 0.0), (1.0, 0.8, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.0),
                (0.8, 0.0, 0.0), (0.6, 0.0, 0.0), (0.0, 0.0, 0.0)]
FlukaColormap = matplotlib.colors.LinearSegmentedColormap.from_list('fluka', FLUKA_COLORS, N=300)

# # Define default color palette
palette = PALETTE['solarized']

# Define "logical" colors
palette['BEND'] = palette['blue']
palette['QUADRUPOLE'] = palette['red']
palette['SEXTUPOLE'] = palette['yellow']
palette['OCTUPOLE'] = palette['violet']
palette['MULTIPOLE'] = palette['green']
palette['DEGRADER'] = palette['base02']
palette['RECTANGULARCOLLIMATOR'] = palette['goldenrod']
palette['CIRCULARCOLLIMATOR'] = palette['orange']
palette['COLLIMATOR'] = 'gold'


class MatplotlibArtist(_Artist):
    """
    TODO
    """

    def __init__(self,
                 ax=None,
                 **kwargs):
        """
            Args:
                param ax: the matplotlib ax used for plotting. If None it will be created with `init_plot` (kwargs are
                forwarded).
                with_frames: draw the entry and exit frames of each elements
                with_centers: draw the center of each polar coordinate elements
                kwargs: forwarded to `Artist` and to `init_plot`.
            """
        super().__init__(**kwargs)

        if ax is None:
            self.init_plot(**kwargs)
        else:
            self._ax = ax
        self._ax2 = self._ax.twinx()
        self._ax2.set_ylim([0, 1])
        self._ax2.axis('off')

    @property
    def ax(self):
        """Current Matplotlib ax.

        Returns:
            the Matplotlib ax.
        """
        return self._ax

    @property
    def ax2(self):
        """

        Returns:

        """
        return self._ax2

    @property
    def figure(self):
        """Current Matplotlib figure.

        Returns:
            the Matplotlib figure.
        """
        return self._fig

    @ax.setter
    def ax(self, ax):
        self._ax = ax

    def init_plot(self, figsize=(12, 8), subplots=111):
        """
        Initialize the Matplotlib figure and ax.

        Args:
            subplots: number of subplots
            figsize: figure size
        """
        self._fig = plt.figure(figsize=figsize)
        self._ax = self._fig.add_subplot(subplots)

    def plot(self, *args, **kwargs):
        """Proxy for matplotlib.pyplot.plot

        Same as `matplotlib.pyplot.plot`, forwards all arguments.
        """
        self._ax.plot(*args, **kwargs)

        # THIS IS THE OLD common.py

    @staticmethod
    def beamline_get_ticks_locations(o):
        return list(o['AT_CENTER'].apply(lambda e: e.m_as('m')).values)

    @staticmethod
    def beamline_get_ticks_labels(o):
        return list(o.index)

    def plot_beamline(self, beamline: _pd.DataFrame = None,
                      print_label: bool = False,
                      with_cartouche: bool = True,
                      with_aperture: bool = True,
                      **kwargs):

        bl_short = beamline.reset_index()
        bl_short = bl_short[[not a for a in bl_short['NAME'].str.contains("DRIFT")]]
        bl_short = bl_short.set_index("NAME")

        ticks_locations_short = self.beamline_get_ticks_locations(bl_short)
        ticks_labels_short = self.beamline_get_ticks_labels(bl_short)
        self._ax.tick_params(axis='both', which='major')
        self._ax.tick_params(axis='x', labelsize=8)
        self._ax.xaxis.set_major_locator(FixedLocator(ticks_locations_short))

        self._ax.set_xlim([bl_short.iloc[0]['AT_ENTRY'].m_as('m'), bl_short.iloc[-1]['AT_EXIT'].m_as('m')])
        self._ax.get_xaxis().set_tick_params(direction='out')
        plt.setp(self._ax.xaxis.get_majorticklabels(), rotation=-45)
        self._ax.yaxis.set_major_locator(MultipleLocator(10))
        self._ax.yaxis.set_minor_locator(MultipleLocator(5))
        self._ax.set_ylim(kwargs.get('ylim', [-60, 60]))
        self._ax.grid(True, alpha=0.25)

        if with_cartouche:
            self.draw_cartouche(beamline)

        if with_aperture:
            self.draw_aperture(beamline)

        if print_label:
            self._ax2.axis('on')
            self._ax2.get_xaxis().set_tick_params(direction='out')
            self._ax2.tick_params(axis='both', which='major')
            self._ax2.tick_params(axis='x', labelsize=8)
            plt.setp(self._ax2.xaxis.get_majorticklabels(), rotation=-90)
            self._ax2.xaxis.set_major_locator(FixedLocator(ticks_locations_short))
            self._ax2.xaxis.set_major_formatter(FixedFormatter(ticks_labels_short))

    def draw_aperture(self, bl, **kwargs):

        if 'APERTURE' not in bl:
            logging.warning("No APERTURE defined in the beamline")
            return

        bl.loc[:, 'CLASS'] = bl['CLASS'].apply(lambda e: e.upper())
        bl = bl.query("CLASS != 'MARKER' and CLASS != 'DRIFT'")  # Marker doesn't have aperture
        planes = kwargs.get('plane', 'X')

        # Set the y aperture for circular apertype
        for idx in bl.query("APERTYPE == 'CIRCULAR'").index:
            bl.loc[idx, 'APERTURE'] = _np.array([bl.loc[idx, 'APERTURE'][0],
                                                 bl.loc[idx, 'APERTURE'][0]],
                                                dtype=object)

        if planes == 'X':
            index = 0
        elif planes == 'Y':
            index = 1

        bl.loc[:, 'APERTURE_UP'] = bl['APERTURE'].apply(lambda a: a[index].m_as('mm'))
        bl.loc[:, 'APERTURE_DOWN'] = bl['APERTURE'].apply(lambda a: a[index].m_as('mm'))

        if 'CHAMBER' not in bl:
            bl.loc[:, 'CHAMBER'] = [0 * _ureg.mm]

        bl.loc[:, 'CHAMBER_UP'] = bl['CHAMBER'].apply(lambda a: a.m_as('mm'))
        bl.loc[:, 'CHAMBER_DOWN'] = bl['CHAMBER'].apply(lambda a: a.m_as('mm'))

        bl.query("CLASS == 'QUADRUPOLE'").apply(lambda e: self.draw_quad(e), axis=1)
        bl.query("CLASS == 'SBEND'").apply(lambda e: self.draw_bend(e), axis=1)
        bl.query("CLASS == 'RBEND'").apply(lambda e: self.draw_bend(e), axis=1)
        bl.query("CLASS == 'RECTANGULARCOLLIMATOR'").apply(lambda e: self.draw_coll(e), axis=1)
        bl.query("CLASS == 'CIRCULARCOLLIMATOR'").apply(lambda e: self.draw_coll(e), axis=1)

    def draw_quad(self, e):
        self._ax.add_patch(
            patches.Rectangle(
                (e['AT_ENTRY'].m_as('m'), e['APERTURE_UP'] + e['CHAMBER_UP']),  # (x,y)
                e['L'].m_as('m'),  # width
                100,
                facecolor=palette['QUADRUPOLE']
            )
        )

        self._ax.add_patch(
            patches.Rectangle(
                (e['AT_ENTRY'].m_as('m'), -e['APERTURE_DOWN'] - e['CHAMBER_UP']),  # (x,y)
                e['L'].m_as('m'),  # width
                -100,
                facecolor=palette['QUADRUPOLE']
            )
        )
        self.draw_chamber(self._ax, e)

    def draw_coll(self, e):
        self._ax.add_patch(
            patches.Rectangle(
                (e['AT_ENTRY'].m_as('m'), e['APERTURE_UP']),  # (x,y)
                e['L'].m_as('m'),  # width
                100,  # height
                facecolor=palette['COLLIMATOR']
            )
        )

        self._ax.add_patch(
            patches.Rectangle(
                (e['AT_ENTRY'].m_as('m'), -e['APERTURE_DOWN']),  # (x,y)
                e['L'].m_as('m'),  # width
                -100,  # height
                facecolor=palette['COLLIMATOR']
            )
        )

    def draw_bend(self, e):
        tmp = e['APERTURE_UP'] + e['CHAMBER_UP']
        self._ax.add_patch(
            patches.Rectangle(
                (e['AT_ENTRY'].m_as('m'), tmp if tmp < 55 else 55),  # (x,y)
                e['L'].m_as('m'),  # width
                100,  # height
                facecolor=palette['BEND']
            )
        )
        tmp = -e['APERTURE_DOWN'] - e['CHAMBER_UP']
        self._ax.add_patch(
            patches.Rectangle(
                (e['AT_ENTRY'].m_as('m'), tmp if abs(tmp) < 55 else -55),  # (x,y)
                e['L'].m_as('m'),  # width
                -100,
                facecolor=palette['BEND']
            )
        )
        self.draw_chamber(self._ax, e)

    @staticmethod
    def draw_chamber(ax, e):
        ax.add_patch(
            patches.Rectangle(
                (e['AT_ENTRY'].m_as('m'), (e['APERTURE_UP'])),  # (x,y)
                e['L'].m_as('m'),  # width
                1000 * e['CHAMBER_UP'],  # height
                hatch='', facecolor=palette['base01']
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (e['AT_ENTRY'].m_as('m'), -e['APERTURE_DOWN']),  # (x,y)
                e['L'].m_as('m'),  # width
                -1000 * e['CHAMBER_UP'],  # height
                hatch='', facecolor=palette['base01']
            )
        )

    def draw_cartouche(self, bl):

        offset = 1.15
        self._ax2.axis('on')
        self._ax2.set_yticks([])
        self._ax2.set_ylim([0, 1])
        self._ax2.hlines(offset, 0, bl.iloc[-1]['AT_EXIT'].m_as('m'), clip_on=False, colors='black', lw=1)
        for i, e in bl.iterrows():
            if e['CLASS'].upper() in ['DRIFT', 'MARKER']:
                continue

            if e['CLASS'].upper() in ['SBEND', 'RBEND']:
                if e['K1'].magnitude > 0:
                    focusing = 1.0
                elif e['K1'].magnitude < 0:
                    focusing = -1.0
                else:
                    focusing = 0.0
                self._ax2.add_patch(
                    patches.Rectangle(
                        (e['AT_ENTRY'].m_as('m'), offset - 0.05 + focusing * 0.02),
                        e['L'].m_as('m'),
                        .1,
                        hatch='',
                        facecolor=palette['BEND'],
                        clip_on=False,
                    )
                )

            elif e['CLASS'].upper() in ['SEXTUPOLE', 'QUADRUPOLE', 'MULTIPOLE']:
                self._ax2.add_patch(
                    patches.Rectangle(
                        (e['AT_ENTRY'].m_as('m'), offset - 0.05),
                        e['L'].m_as('m'),
                        .1,
                        hatch='',
                        facecolor=palette[e['CLASS'].upper()],
                        ec=palette[e['CLASS'].upper()],
                        clip_on=False,
                    )
                )

            elif e['CLASS'].upper() in ['RECTANGULARCOLLIMATOR', 'DEGRADER', 'CIRCULARCOLLIMATOR']:
                self._ax2.add_patch(
                    patches.Rectangle(
                        (e['AT_ENTRY'].m_as('m'), offset - 0.05),
                        e['L'].m_as('m'),
                        .1,
                        hatch='',
                        facecolor=palette[e['CLASS'].upper()],
                        ec=palette[e['CLASS'].upper()],
                        clip_on=False,
                    )
                )

            else:
                logging.warning(f"colors are not implemented for {e['CLASS']}")
