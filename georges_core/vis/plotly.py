"""
TODO
"""
from __future__ import annotations
from typing import Optional, Mapping
import plotly.offline
import plotly.graph_objs as go
from .artist import Artist as _Artist


class PlotlyArtist(_Artist):
    """
    TODO
    """

    def __init__(self, config: Optional[Mapping] = None, layout: Optional[Mapping] = None, **kwargs):
        """

        Args:
            config:
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
        self._layout = layout or {
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
        }
        self._shapes = []
        self._n_y_axis = len([ax for ax in self._layout.keys() if ax.startswith('yaxis')])

    def _init_plot(self):
        pass

    @property
    def fig(self):
        """Provides the plotly figure."""
        return {
            'data'  : self.data,
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

    def histogram2d(self, *args, **kwargs):
        """A proxy for plotly.graph_objs.Histogram2d"""
        self._data.append(go.Histogram2d(*args, **kwargs))

    def scatter(self, *args, **kwargs):
        """A proxy for plotly.graph_objs.Scatter ."""
        self._data.append(go.Scatter(*args, **kwargs))

    def scatter3d(self, *args, **kwargs):
        """A proxy for plotly.graph_objs.Scatter3d ."""
        self._data.append(go.Scatter3d(*args, **kwargs))