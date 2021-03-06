"""
TODO
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Mapping, Any
import numpy as _np
import plotly.offline
import plotly.graph_objs as go
from .artist import Artist as _Artist
if TYPE_CHECKING:
    import pandas as _pd


class PlotlyArtist(_Artist):
    """
    TODO
    """

    def __init__(self,
                 config: Optional[Mapping] = None,
                 layout: Optional[Mapping] = None,
                 width: Optional[float] = None,
                 height: Optional[float] = None,
                 **kwargs):
        """

        Args:
            config:
            layout:
            width:
            height:
            **kwargs:
        """
        super().__init__(**kwargs)
        self._data = []
        self._config = config or {
            'showLink': False,
            'scrollZoom': True,
            'displayModeBar': False,
            'editable': False,
        }
        self._layout: Mapping[Any, Any] = layout or {
            'xaxis': {
                'showgrid': True,
                'linecolor': 'black',
                'linewidth': 1,
                'mirror': True,
            },
            'yaxis': {
                'linecolor': 'black',
                'linewidth': 1,
                'mirror': True,
                'exponentformat': 'power',
            },
            'height': 600,
            'width': 600,
        }
        if height is not None:
            self._layout['height'] = height
        if width is not None:
            self._layout['width'] = width
        self._shapes = []
        self._n_y_axis = len([ax for ax in self._layout.keys() if ax.startswith('yaxis')])

    def _init_plot(self):
        pass

    @property
    def fig(self):
        """Provides the plotly figure."""
        return {
            'data': self.data,
            'layout': self.layout,
        }

    @property
    def config(self):
        return self._config

    @property
    def data(self):
        return self._data

    @property
    def layout(self):
        self._layout['shapes'] = self._shapes
        return self._layout

    @property
    def shapes(self):
        return self._shapes

    def __iadd__(self, other):
        """Add a trace to the figure."""
        self._data.append(other)
        return self

    def add_axis(self, title: str = '', axis: Optional[Mapping] = None):
        """

        Args:
            title:
            axis:

        Returns:

        """
        self._n_y_axis += 1
        self.layout[f"yaxis{self._n_y_axis if self._n_y_axis > 1 else ''}"] = axis or {
            'title': title,
            'titlefont': dict(
                color='black'
            ),
            'tickfont': dict(
                color='black'
            ),
            'linewidth': 1,
            'exponentformat': 'power',
            'overlaying': 'y',
            'side': 'left',
        }

    def add_secondary_axis(self, title: str = '', axis: Optional[Mapping] = None):
        """

        Args:
            title:
            axis:

        Returns:

        """
        self._n_y_axis += 1
        self.layout[f"yaxis{self._n_y_axis}"] = axis or {
            'title': title,
            'titlefont': dict(
                color='black'
            ),
            'tickfont': dict(
                color='black'
            ),
            'linewidth': 1,
            'exponentformat': 'power',
            'overlaying': 'y',
            'side': 'right',
        }

    def render(self):
        if len(self._data) == 0:
            self._data.append(go.Scatter(x=[0.0], y=[0.0]))
        plotly.offline.iplot(self.fig, config=self.config)

    def save(self, file: str, file_format: str = 'png'):
        plotly.io.write_image(self.fig, file=file, format=file_format)

    def save_html(self, file: str):
        return plotly.offline.plot(self.fig, config=self.config, auto_open=False, filename=file)

    def histogram(self, *args, **kwargs):
        """A proxy for plotly.graph_objs.Histogram"""
        self._data.append(go.Histogram(*args, **kwargs))

    def uproot_histogram(self, histogram, **kwargs):
        _ = histogram.numpy()
        self.bar(x=_[1], y=_[0], error_y={'array': _np.sqrt(histogram.variances)}, **kwargs)

    def histogram2d(self, *args, **kwargs):
        """A proxy for plotly.graph_objs.Histogram2d"""
        self._data.append(go.Histogram2d(*args, **kwargs))

    def bar(self, *args, **kwargs):
        self._data.append(go.Bar(*args, **kwargs))

    def scatter(self, *args, **kwargs):
        """A proxy for plotly.graph_objs.Scatter ."""
        self._data.append(go.Scatter(*args, **kwargs))

    def scatter3d(self, *args, **kwargs):
        """A proxy for plotly.graph_objs.Scatter3d ."""
        self._data.append(go.Scatter3d(*args, **kwargs))

    def surface(self, *args, **kwargs):
        """A proxy for plotly.graph_objs.Surface ."""
        self._data.append(go.Surface(*args, **kwargs))

    def plot_cartouche(self,
                       beamline_survey: _pd.DataFrame,
                       vertical_position: float = 1.2,
                       unsplit_bends: bool = True,
                       skip_elements: Optional[list] = None,
                       ):
        """

        Args:
            beamline_survey:
            vertical_position:
            unsplit_bends:
            skip_elements:

        Returns:

        """
        skip_elements = skip_elements or []

        def do_sbend(at_entry: float, at_exit: float, polarity: float):
            length = at_exit - at_entry
            if polarity >= 0.0:
                path = f"M{at_entry},{vertical_position + 0.1} " \
                       f"H{at_exit} " \
                       f"L{at_exit - 0.15 * length},{vertical_position - 0.1} " \
                       f"H{at_exit - 0.85 * length} " \
                       f"Z"
            else:
                path = f"M{at_entry + 0.15 * length},{vertical_position + 0.1} " \
                       f"H{at_exit - 0.15 * length} " \
                       f"L{at_exit},{vertical_position - 0.1} " \
                       f"H{at_entry} " \
                       f"Z"
            self.shapes.append(
                {
                    'type': 'path',
                    'xref': 'x',
                    'yref': 'paper',
                    'path': path,
                    'line': {
                        'width': 0,
                    },
                    'fillcolor': '#0000FF',
                },
            )

        colors = {
            'ELEMENT': '#AAAAAA',
            'RCOL': '#11EE11',
            'ECOL': '#1111EE'
        }
        self.shapes.append(
            {
                'type': 'line',
                'xref': 'paper',
                'yref': 'paper',
                'x0': 0,
                'y0': vertical_position,
                'x1': 1,
                'y1': vertical_position,
                'line': {
                    'color': 'rgb(150, 150, 150)',
                    'width': 2,
                },
            },
        )
        accumulate = False
        accumulator = {}
        for i, e in beamline_survey.iterrows():
            if e['TYPE'].upper() not in ('QUADRUPOLE', 'SBEND', 'ELEMENT', 'RCOL', 'ECOL'):
                continue
            if i in skip_elements:
                continue
            if unsplit_bends and accumulate and (e['TYPE'].upper() not in ('SBEND',) or i != accumulator['name']):
                accumulate = False
                do_sbend(accumulator['at_entry'], accumulator['at_exit'], accumulator['polarity'])
                accumulator = {}
            if e['TYPE'].upper() in ('ELEMENT', 'RCOL', 'ECOL'):
                self.shapes.append(
                    {
                        'type': 'rect',
                        'xref': 'x',
                        'yref': 'paper',
                        'x0': e['AT_ENTRY'],
                        'y0': vertical_position + 0.1,
                        'x1': e['AT_EXIT'],
                        'y1': vertical_position - 0.1,
                        'line': {
                            'width': 0,
                        },
                        'fillcolor': colors[e['TYPE'].upper()],
                    },
                )
            if e['TYPE'].upper() == 'QUADRUPOLE':
                self.shapes.append(
                    {
                        'type': 'rect',
                        'xref': 'x',
                        'yref': 'paper',
                        'x0': e['AT_ENTRY'],
                        'y0': vertical_position if e['K1'] > 0 else vertical_position - 0.1,
                        'x1': e['AT_EXIT'],
                        'y1': vertical_position + 0.1 if e['K1'] > 0 else vertical_position,
                        'line': {
                            'width': 0,
                        },
                        'fillcolor': '#FF0000',
                    },
                )
            if e['TYPE'].upper() == 'HKICKER' or e['TYPE'].upper() == 'VKICKER':
                self.shapes.append(
                    {
                        'type': 'rect',
                        'xref': 'x',
                        'yref': 'paper',
                        'x0': e['AT_ENTRY'],
                        'y0': vertical_position,
                        'x1': e['AT_EXIT'],
                        'y1': vertical_position + 0.1,
                        'line': {
                            'width': 0,
                        },
                        'fillcolor': 'Green',
                    },
                )
            if e['TYPE'].upper() == 'SBEND':
                if unsplit_bends:
                    if accumulate is False:
                        accumulate = True
                        accumulator['name'] = i
                        accumulator['polarity'] = e['B']
                        accumulator['at_entry'] = e['AT_ENTRY']
                        continue
                    if accumulate is True:
                        accumulator['at_exit'] = e['AT_EXIT']
                        continue
                else:
                    do_sbend(e['AT_ENTRY'], e['AT_EXIT'])
