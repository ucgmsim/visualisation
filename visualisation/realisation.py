"""
Module for plotting the spatial domain and related features of realisations.

This module provides functions to visualise the domain of a
realisation, including source geometries, stations, and PGV targets.
The functions are designed to be reusable for custom plotting scripts.
"""

from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import pygmt
import shapely
import typer

from pygmt_helper import plotting
from qcore import cli
from source_modelling.sources import Fault, Plane
from visualisation import utils
from workflow.realisations import (
    DomainParameters,
    RupturePropagationConfig,
    SourceConfig,
    VelocityModelParameters,
)
from workflow.scripts import generate_velocity_model_parameters

app = typer.Typer()


def plot_stations(
    fig: pygmt.Figure,
    domain_parameters: DomainParameters,
    stations_path: Path,
    **kwargs: dict[str, Any],
) -> None:
    """Plot stations file on a figure.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure to plot on.
    domain_parameters : DomainParameters
        The simulation domain (used to count the number of stations in
        the domain).
    stations_path : Path
        Path to the stations file.
    **kwargs : dict
        Additional keyword arguments to pass to the plotting function. If empty, the default is
        - `style="t0.1c"` (triangle size)
        - `fill="red"` (triangle fill colour)
        - `pen="black"` (triangle border colour)

    Examples
    --------
    >>> import pygmt
    >>> import pandas as pd
    >>> from workflow.realisations import DomainParameters
    >>> stations_data = pd.DataFrame({'lon': [171, 171.5, 172], 'lat': [-41, -41.5, -42], 'name': ['A', 'B', 'C']})
    >>> domain = DomainParameters.read_from_realisation('realisation.json')
    >>> fig = pygmt.Figure()
    >>> plot_stations(fig, domain, stations_path)
    >>> fig.show()
    """
    kwargs = {"style": "t0.1c", "fill": "red", "pen": "black", **(kwargs or {})}
    stations = pd.read_csv(
        stations_path, delimiter=r"\s+", comment="#", names=["lon", "lat", "name"]
    )
    stations_in_domain = np.count_nonzero(
        domain_parameters.domain.contains(stations[["lat", "lon"]].to_numpy())
    )
    fig.plot(
        x=stations["lon"],
        y=stations["lat"],
        label=f"Stations ({stations_in_domain})",
        **kwargs,
    )


def plot_sources(fig: pygmt.Figure, source_config: SourceConfig, **kwargs: Any) -> None:
    """Plot the sources on the figure.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure to plot on.
    source_config : SourceConfig
        The source configuration to plot.
    **kwargs : dict
        Additional keyword arguments to pass to the plotting function. If empty, the default is
        - `pen="0.3p,black,--"` (polygon border colour)
        - trace pen is found by taking the pen and stripping the "--"

    Examples
    --------
    >>> import pygmt
    >>> from workflow.realisations import SourceConfig
    >>> source_config = SourceConfig.read_from_realisation("realisation.json")
    >>> fig = pygmt.Figure()
    >>> plot_sources(fig, source_config)
    >>> source_config.show()
    """
    pen = kwargs.get("pen", "0.3p,black,--")
    assert isinstance(pen, str)
    trace_pen = pen.removesuffix(",--")
    interior_kwargs = {"pen": pen, **(kwargs or {})}

    for source in source_config.source_geometries.values():
        utils.plot_polygon(
            fig, utils.polygon_nztm_to_pygmt(source.geometry), **interior_kwargs
        )
        if isinstance(source, Fault):
            trace = shapely.LineString(
                np.concatenate([plane.bounds[:2] for plane in source.planes])
            )
            utils.plot_polygon(fig, utils.polygon_nztm_to_pygmt(trace), pen=trace_pen)
        elif isinstance(source, Plane):
            trace = shapely.LineString(source.bounds[:2])
            utils.plot_polygon(fig, utils.polygon_nztm_to_pygmt(trace), pen=trace_pen)


def plot_domain(
    fig: pygmt.Figure,
    domain_parameters: DomainParameters,
    **kwargs: Any,
) -> None:
    """Plot the domain on a figure.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure to plot on.
    domain_parameters : DomainParameters
        The domain to plot.
    **kwargs : dict
        Additional keyword arguments to pass to the plotting function. The defaults are
        - `pen="1p,blue,-"` (polygon border colour)

    Examples
    --------
    >>> import pygmt
    >>> from workflow.realisations import DomainParameters
    >>> domain = DomainParameters.read_from_realisation("realisation.json")
    >>> fig = pygmt.Figure()
    >>> plot_domain(fig, domain)
    >>> fig.show()
    """
    kwargs = {"pen": "1p,blue,-", **(kwargs or {})}
    utils.plot_polygon(
        fig, utils.polygon_nztm_to_pygmt(domain_parameters.domain.polygon), **kwargs
    )


def plot_rrup_polygon(
    fig: pygmt.Figure,
    region: utils.Region,
    pgv_target: float,
    rrup_bounding_polygon: shapely.Polygon,
    rrup_polygon_args: dict[str, Any] | None = None,
    label_args: dict[str, Any] | None = None,
) -> None:
    """Plot the RRup bounding polygon on a figure.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure to plot on.
    region : BoundingBox
        The region of the plot.
    pgv_target : float
        The PGV target for the polygon (used as a label).
    rrup_bounding_polygon : shapely.Polygon
        The RRup bounding polygon.
    rrup_polygon_args : dict, optional
        Style arguments for the rrup polygon. See `pygmt.Figure.plot`. The defaults are
        - `pen="0.3p,black,-"` (polygon border colour)
    label_args : dict, optional
        Style arguments for the label. See `pygmt.Figure.text`. The defaults are
        - `fill="white"` (label fill colour)
        - `pen="0.3p,black"` (label border colour)

    Examples
    --------
    >>> import pygmt
    >>> from velocity_modelling.bounding_box import BoundingBox
    >>> from shapely.geometry import Polygon
    >>> from visualisation import utils
    >>> # Create dummy data
    >>> region = (170, 172, -42, -40)
    >>> rrup_polygon = Polygon([(171, -41), (171.5, -41), (171.5, -41.5), (171, -41.5)])
    >>> fig = pygmt.Figure()
    >>> plot_rrup_polygon(fig, region, 10.0, rrup_polygon)
    >>> fig.show()
    """
    rrup_polygon_args = {
        "pen": "0.3p,black,-",
        **(rrup_polygon_args or {}),
    }
    label_args = {
        "fill": "white",
        "pen": "0.3p,black",
        **(label_args or {}),
    }
    utils.plot_polygon(
        fig,
        utils.polygon_nztm_to_pygmt(rrup_bounding_polygon),
        **rrup_polygon_args,
    )
    utils.label_polygon(
        fig,
        region,
        utils.polygon_nztm_to_pygmt(rrup_bounding_polygon),
        f"{pgv_target} cm/s",
        **label_args,
    )


def plot_realisation(
    realisation_ffp: Path,
    latitude_pad: float = 0,
    longitude_pad: float = 0,
    title: str | None = None,
    subtitle: str | None = None,
    width: float = 10,
    show_geometry: bool = True,
    show_pgv_targets: bool = False,
    pgv_targets: list[float] | None = None,
    stations: Path | None = None,
) -> pygmt.Figure:
    """Plot the domain and sources of a realisation.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file to plot.
    latitude_pad : float
        Latitude padding in degrees.
    longitude_pad : float
        Longitude padding in degrees.
    title : str, optional
        Title of the plot.
    subtitle : str, optional
        Subtitle of the plot.
    width : float
        Width of the plot in cm.
    show_geometry : bool
        Show source geometry on the plot.
    show_pgv_targets : bool
        Show PGV targets on the plot.
    pgv_targets : list[float], optional
        PGV targets to plot. If None, use PGV targets from the
        realisation. A non-empty value implies `show_pgv_targets`.
    stations : Path, optional
        Path to list of stations to plot.

    Returns
    -------
    pygmt.Figure
        The figure.

    Examples
    --------
    >>> from pathlib import Path
    >>> fig = plot_realisation(
    ...     realisation_ffp=realisation_ffp,
    ...     width=5,
    ...     show_geometry=False,
    ...     show_pgv_targets=False,
    ...     stations=None,
    ... )
    >>> fig.show()
    """
    show_pgv_targets = show_pgv_targets or bool(pgv_targets)
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)

    source_config = SourceConfig.read_from_realisation(realisation_ffp)

    rrup_bounding_polygons: list[shapely.Polygon] = []

    if show_pgv_targets:
        rupture_propagation = RupturePropagationConfig.read_from_realisation(
            realisation_ffp
        )

        velocity_model_parameters = VelocityModelParameters.read_from_realisation(
            realisation_ffp
        )

        fault_pgv_targets = pgv_targets or [
            generate_velocity_model_parameters.pgv_target(
                rupture_propagation, velocity_model_parameters
            )
        ]

        for pgv_targets in fault_pgv_targets:
            rrup_bounding_polygons.append(
                shapely.union_all(
                    [
                        generate_velocity_model_parameters.find_rrup_bounding_polygon(
                            *args, pgv_target=pgv_targets
                        )
                        for args in generate_velocity_model_parameters.dict_zip(
                            source_config.source_geometries,
                            rupture_propagation.magnitudes,
                            rupture_propagation.rakes,
                        ).values()
                    ]
                )
            )

    region = utils.bounding_region_for(
        [domain_parameters.domain.polygon] + rrup_bounding_polygons,
        latitude_pad=latitude_pad,
        longitude_pad=longitude_pad,
    )

    fig = plotting.gen_region_fig(
        title,
        region,
        projection=f"M{width}c",
        subtitle=subtitle,
    )

    plot_domain(fig, domain_parameters)

    if show_geometry:
        plot_sources(fig, source_config)

    if stations:
        plot_stations(fig, domain_parameters, stations)

    if show_pgv_targets:
        for pgv_target, rrup_bounding_polygon in zip(
            fault_pgv_targets, rrup_bounding_polygons
        ):
            plot_rrup_polygon(fig, region, pgv_target, rrup_bounding_polygon)

    # Plot the legend overtop the other elements.
    if stations:
        fig.legend(position="jTR+o0.2c", box="+gwhite+p1p")

    return fig


@cli.from_docstring(app)
def plot_realisation_to_file(
    realisation_ffp: Annotated[
        Path,
        typer.Argument(dir_okay=False, exists=True, readable=True, show_default=False),
    ],
    output_ffp: Annotated[
        Path,
        typer.Argument(dir_okay=False, writable=True, show_default=False),
    ],
    latitude_pad: Annotated[float, typer.Option(min=0)] = 0,
    longitude_pad: Annotated[
        float,
        typer.Option(min=0),
    ] = 0,
    title: Annotated[
        str | None,
        typer.Option(),
    ] = None,
    subtitle: Annotated[
        str | None,
        typer.Option(),
    ] = None,
    width: Annotated[
        float,
        typer.Option(
            min=0,
        ),
    ] = 10,
    dpi: Annotated[
        float,
        typer.Option(
            min=0,
        ),
    ] = 300,
    show_geonet_stations: Annotated[
        bool,
        typer.Option(
            show_default=False,
        ),
    ] = False,
    show_geometry: Annotated[
        bool,
        typer.Option(),
    ] = True,
    show_pgv_targets: Annotated[
        bool,
        typer.Option(),
    ] = False,
    pgv_targets: Annotated[
        list[float] | None,
        # Use a different option name because --pgv-targets is in
        # plural form but only accepts one value each time it is
        # invoked:
        # --pgv-targets 0.1 --pgv-targets 0.2 vs --pgv-target 0.1 --pgv-target 0.2.
        typer.Option("--pgv-target"),
    ] = None,
    stations: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            readable=True,
        ),
    ] = None,
) -> pygmt.Figure:
    """Plot the domain and sources of a realisation to a file.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file to plot.
    output_ffp : Path
        Path to the output file.
    latitude_pad : float
        Latitude padding in degrees.
    longitude_pad : float
        Longitude padding in degrees.
    title : str, optional
        Title of the plot.
    subtitle : str, optional
        Subtitle of the plot.
    width : float
        Width of the plot in cm.
    dpi : float
        DPI of the plot (higher is better quality).
    show_geonet_stations : bool
        Show GeoNet stations on the plot.
    show_geometry : bool
        Show source geometry on the plot.
    show_pgv_targets : bool
        Show PGV targets on the plot.
    pgv_targets : list[float], optional
        PGV targets to plot. If None, use PGV targets from the
        realisation. A non-empty value implies `show_pgv_targets`.
    stations : Path, optional
        Path to list of stations to plot.

    Examples
    --------
    >>> from pathlib import Path
    >>> plot_realisation_to_file(
    ...     realisation_ffp=realisation_ffp,
    ...     width=5,
    ...     show_geometry=False,
    ...     show_pgv_targets=False,
    ...     stations=None,
    ... )
    >>> fig.show()
    """
    fig = plot_realisation(
        realisation_ffp,
        latitude_pad=latitude_pad,
        longitude_pad=longitude_pad,
        title=title,
        subtitle=subtitle,
        width=width,
        show_geometry=show_geometry,
        show_pgv_targets=show_pgv_targets,
        pgv_targets=pgv_targets,
        stations=stations,
    )
    fig.savefig(output_ffp, dpi=dpi)
