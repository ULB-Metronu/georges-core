"""
TODO
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as _np
import plotly.graph_objs as go
import plotly.offline

from .artist import Artist as _Artist

if TYPE_CHECKING:
    import pandas as _pd


class PlotlyArtist(_Artist):
    """
    TODO
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        layout: Optional[Dict[str, Any]] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            config:
            layout:
            width:
            height:
            **kwargs:
        """
        super().__init__(**kwargs)
        self._data: List[Any] = []
        self._config = config or {
            "showLink": False,
            "scrollZoom": True,
            "displayModeBar": False,
            "editable": False,
        }
        self._layout: Dict[str, Any] = layout or {
            "font": {"family": "serif", "size": 18},
            "plot_bgcolor": "rgba(0,0,0,0)",
            "xaxis": {
                "showgrid": True,
                "linecolor": "black",
                "linewidth": 1,
                "mirror": True,
                "gridcolor": "grey",
                "gridwidth": 0.1,
            },
            "yaxis": {
                "linecolor": "black",
                "linewidth": 1,
                "gridcolor": "grey",
                "gridwidth": 0.1,
                "mirror": True,
                "exponentformat": "power",
            },
            "height": 600,
            "width": 600,
        }
        if height is not None:
            self._layout["height"] = height
        if width is not None:
            self._layout["width"] = width
        self._shapes: List[Any] = []
        self._n_y_axis = len([ax for ax in self._layout.keys() if ax.startswith("yaxis")])

    def _init_plot(self) -> None:
        pass

    @property
    def fig(self) -> Dict[str, Any]:  # pragma: no cover
        """Provides the plotly figure."""
        return {
            "data": self.data,
            "layout": self.layout,
        }

    @property
    def config(self) -> Dict[str, Any]:  # pragma: no cover
        return self._config

    @property
    def data(self) -> List[Any]:  # pragma: no cover
        return self._data

    @property
    def layout(self) -> Dict[str, Any]:  # pragma: no cover
        self._layout["shapes"] = self._shapes
        return self._layout

    @property
    def shapes(self) -> List[Any]:  # pragma: no cover
        return self._shapes

    def __iadd__(self, other: Any) -> PlotlyArtist:  # pragma: no cover
        """Add a trace to the figure."""
        self._data.append(other)
        return self

    def add_axis(self, title: str = "", axis: Optional[Dict[str, Any]] = None) -> None:
        """

        Args:
            title:
            axis:

        Returns:

        """
        self._n_y_axis += 1
        self.layout[f"yaxis{self._n_y_axis if self._n_y_axis > 1 else ''}"] = axis or {
            "title": title,
            "titlefont": dict(color="black"),
            "tickfont": dict(color="black"),
            "linewidth": 1,
            "exponentformat": "power",
            "overlaying": "y",
            "side": "left",
        }

    def add_secondary_axis(self, title: str = "", axis: Optional[Dict[str, Any]] = None) -> None:
        """

        Args:
            title:
            axis:

        Returns:

        """
        self._n_y_axis += 1
        self.layout[f"yaxis{self._n_y_axis}"] = axis or {
            "title": title,
            "titlefont": dict(color="black"),
            "tickfont": dict(color="black"),
            "linewidth": 1,
            "exponentformat": "power",
            "overlaying": "y",
            "side": "right",
        }

    def render(self) -> None:  # pragma: no cover
        if len(self._data) == 0:
            self._data.append(go.Scatter(x=[0.0], y=[0.0]))
        plotly.offline.iplot(self.fig, config=self.config)

    def save(self, file: str, file_format: str = "png") -> None:  # pragma: no cover
        plotly.io.write_image(self.fig, file=file, format=file_format)

    def save_html(self, file: str) -> Any:  # pragma: no cover
        return plotly.offline.plot(self.fig, config=self.config, auto_open=False, filename=file)

    def histogram(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """A proxy for plotly.graph_objs.Histogram"""
        self._data.append(go.Histogram(*args, **kwargs))

    def uproot_histogram(self, histogram: Any, **kwargs: Any) -> None:  # pragma: no cover
        _ = histogram.numpy()
        self.bar(x=_[1], y=_[0], error_y={"array": _np.sqrt(histogram.variances)}, **kwargs)

    def histogram2d(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """A proxy for plotly.graph_objs.Histogram2d"""
        self._data.append(go.Histogram2d(*args, **kwargs))

    def bar(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        self._data.append(go.Bar(*args, **kwargs))

    def scatter(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """A proxy for plotly.graph_objs.Scatter ."""
        self._data.append(go.Scatter(*args, **kwargs))

    def scatter3d(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """A proxy for plotly.graph_objs.Scatter3d ."""
        self._data.append(go.Scatter3d(*args, **kwargs))

    def surface(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """A proxy for plotly.graph_objs.Surface ."""
        self._data.append(go.Surface(*args, **kwargs))

    def heatmap(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """A proxy for plotly.graph_objs.Surface ."""
        self._data.append(go.Heatmap(*args, **kwargs))

    def plot_cartouche(
        self,
        beamline_survey: _pd.DataFrame,
        vertical_position: float = 1.2,
        unsplit_bends: bool = True,
        skip_elements: Optional[List[str]] = None,
    ) -> None:
        """

        Args:
            beamline_survey:
            vertical_position:
            unsplit_bends:
            skip_elements:

        Returns:

        """
        skip_elements = skip_elements or []

        def do_sbend(at_entry: float, at_exit: float, polarity: float) -> None:
            length = at_exit - at_entry
            if polarity >= 0.0:
                path = (
                    f"M{at_entry},{vertical_position + 0.1} "
                    f"H{at_exit} "
                    f"L{at_exit - 0.15 * length},{vertical_position - 0.1} "
                    f"H{at_exit - 0.85 * length} "
                    f"Z"
                )
            else:
                path = (
                    f"M{at_entry + 0.15 * length},{vertical_position + 0.1} "
                    f"H{at_exit - 0.15 * length} "
                    f"L{at_exit},{vertical_position - 0.1} "
                    f"H{at_entry} "
                    f"Z"
                )
            self.shapes.append(
                {
                    "type": "path",
                    "xref": "x",
                    "yref": "paper",
                    "path": path,
                    "line": {
                        "width": 0,
                    },
                    "fillcolor": "#0000FF",
                },
            )

        colors = {"ELEMENT": "#AAAAAA", "RCOL": "#11EE11", "ECOL": "#1111EE"}
        self.shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": 0,
                "y0": vertical_position,
                "x1": 1,
                "y1": vertical_position,
                "line": {
                    "color": "rgb(150, 150, 150)",
                    "width": 2,
                },
            },
        )
        accumulate = False
        accumulator: Dict[str, Any] = {}
        for i, e in beamline_survey.iterrows():
            if e["CLASS"].upper() not in ("QUADRUPOLE", "SBEND", "ELEMENT", "RCOL", "ECOL"):
                continue
            if i in skip_elements:
                continue
            if unsplit_bends and accumulate and (e["CLASS"].upper() not in ("SBEND",) or i != accumulator["name"]):
                accumulate = False
                do_sbend(accumulator["at_entry"], accumulator["at_exit"], accumulator["polarity"])
                accumulator = {}
            if e["CLASS"].upper() in ("ELEMENT", "RCOL", "ECOL"):
                self.shapes.append(
                    {
                        "type": "rect",
                        "xref": "x",
                        "yref": "paper",
                        "x0": e["AT_ENTRY"].m_as("m"),
                        "y0": vertical_position + 0.1,
                        "x1": e["AT_EXIT"].m_as("m"),
                        "y1": vertical_position - 0.1,
                        "line": {
                            "width": 0,
                        },
                        "fillcolor": colors[e["CLASS"].upper()],
                    },
                )
            if e["CLASS"].upper() == "QUADRUPOLE":
                try:
                    field_magnitude = e["K1"].magnitude
                except KeyError:
                    field_magnitude = e["K1L"].magnitude

                self.shapes.append(
                    {
                        "type": "rect",
                        "xref": "x",
                        "yref": "paper",
                        "x0": e["AT_ENTRY"].m_as("m"),
                        "y0": vertical_position if field_magnitude > 0 else vertical_position - 0.1,
                        "x1": e["AT_EXIT"].m_as("m"),
                        "y1": vertical_position + 0.1 if field_magnitude > 0 else vertical_position,
                        "line": {
                            "width": 0,
                        },
                        "fillcolor": "#FF0000",
                    },
                )
            if e["CLASS"].upper() == "HKICKER" or e["CLASS"].upper() == "VKICKER":
                self.shapes.append(
                    {
                        "type": "rect",
                        "xref": "x",
                        "yref": "paper",
                        "x0": e["AT_ENTRY"].m_as("m"),
                        "y0": vertical_position,
                        "x1": e["AT_EXIT"].m_as("m"),
                        "y1": vertical_position + 0.1,
                        "line": {
                            "width": 0,
                        },
                        "fillcolor": "Green",
                    },
                )
            if e["CLASS"].upper() == "SBEND":
                if unsplit_bends:
                    if accumulate is False:
                        accumulate = True
                        accumulator["name"] = i
                        accumulator["polarity"] = _np.sign(e["ANGLE"].m_as("radians"))
                        accumulator["at_entry"] = e["AT_ENTRY"].m_as("m")
                        accumulator["at_exit"] = e["AT_EXIT"].m_as("m")
                        continue
                    if accumulate is True:
                        accumulator["at_exit"] = e["AT_EXIT"].m_as("m")
                else:
                    do_sbend(
                        e["AT_ENTRY"].m_as("m"),
                        e["AT_EXIT"].m_as("m"),
                        polarity=_np.sign(e["ANGLE"].m_as("radians")),
                    )
