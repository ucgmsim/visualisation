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
from visualisation import utils
from workflow.realisations import (
    DomainParameters,
    RupturePropagationConfig,
    SourceConfig,
    VelocityModelParameters,
)

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


def plot_sources(
    fig: pygmt.Figure, source_config: SourceConfig, **kwargs: dict[str, Any]
) -> None:
    """Plot the sources on the figure.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure to plot on.
    source_config : SourceConfig
        The source configuration to plot.
    **kwargs : dict
        Additional keyword arguments to pass to the plotting function. If empty, the default is
        - `pen="0.3p,black"` (polygon border colour)

    Examples
    --------
    >>> import pygmt
    >>> from workflow.realisations import SourceConfig
    >>> source_config = SourceConfig.read_from_realisation("realisation.json")
    >>> fig = pygmt.Figure()
    >>> plot_sources(fig, source_config)
    >>> source_config.show()
    """
    kwargs = {"pen": "0.3p,black", **(kwargs or {})}

    for source in source_config.source_geometries.values():
        utils.plot_polygon(fig, utils.polygon_nztm_to_pygmt(source.geometry), **kwargs)


def plot_trace(
    fig: pygmt.Figure,
    source_config: SourceConfig,
    outline_kwargs: dict[str, Any] | None = None,
    trace_kwargs: dict[str, Any] | None = None,
) -> None:
    """Plot the fault plane traces on a figure.

    Each fault plane is drawn as a dashed outline representing the
    fault's 3D geometry projected onto the surface, with the surface
    trace (the top edge, where the fault meets the surface) overdrawn as
    a bold solid line. Multi-segment faults are drawn segment by
    segment. Point sources have no plane geometry and are skipped.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure to plot on.
    source_config : SourceConfig
        The source configuration to plot.
    outline_kwargs : dict, optional
        Style arguments for the dashed plane outline. See
        `pygmt.Figure.plot`. The defaults are
        - `pen="0.5p,black,-"` (dashed outline colour)
    trace_kwargs : dict, optional
        Style arguments for the bold surface trace. See
        `pygmt.Figure.plot`. The defaults are
        - `pen="0.8p,black"` (surface trace colour)

    Examples
    --------
    >>> import pygmt
    >>> from workflow.realisations import SourceConfig
    >>> source_config = SourceConfig.read_from_realisation("realisation.json")
    >>> fig = pygmt.Figure()
    >>> plot_trace(fig, source_config)
    >>> fig.show()
    """
    outline_kwargs = {"pen": "0.5p,black,-", **(outline_kwargs or {})}
    trace_kwargs = {"pen": "0.8p,black", **(trace_kwargs or {})}

    labelled = False
    for source in source_config.source_geometries.values():
        # A Fault is made up of multiple planes; a Plane is a single
        # plane. Point sources have neither a plane nor a trace and are
        # skipped by the guard below.
        for plane in getattr(source, "planes", [source]):
            if not hasattr(plane, "trace_geometry"):
                continue
            utils.plot_polygon(
                fig,
                utils.polygon_nztm_to_pygmt(plane.geometry),
                **outline_kwargs,
            )
            # Label only the first trace so the legend gets a single entry.
            plane_trace_kwargs = dict(trace_kwargs)
            if not labelled:
                plane_trace_kwargs["label"] = "Fault trace"
                labelled = True
            utils.plot_polygon(
                fig,
                utils.polygon_nztm_to_pygmt(plane.trace_geometry),
                **plane_trace_kwargs,
            )


def plot_domain(
    fig: pygmt.Figure,
    domain_parameters: DomainParameters,
    **kwargs: dict[str, Any],
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


def plot_hypocentre(
    fig: pygmt.Figure,
    source_config: SourceConfig,
    rupture_propagation: RupturePropagationConfig,
    **kwargs: dict[str, Any],
) -> None:
    """Plot the rupture hypocentre on a figure.

    The hypocentre is taken from the initial fault of the rupture
    causality tree and converted from fault-local (s, d) coordinates to
    global (lat, lon, depth) coordinates. Only the lat / lon are plotted
    as this is a map view.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure to plot on.
    source_config : SourceConfig
        The source configuration (used to locate the initial fault).
    rupture_propagation : RupturePropagationConfig
        The rupture propagation configuration containing the hypocentre.
    **kwargs : dict
        Additional keyword arguments to pass to the plotting function. If empty, the default is
        - `style="a0.3c"` (star marker)
        - `pen="0.3p,black"` (marker border colour)
        - `fill="gold"` (marker fill colour)
        - `label="Hypocentre"` (legend entry)

    Examples
    --------
    >>> import pygmt
    >>> from workflow.realisations import RupturePropagationConfig, SourceConfig
    >>> source_config = SourceConfig.read_from_realisation("realisation.json")
    >>> rup_prop_config = RupturePropagationConfig.read_from_realisation("realisation.json")
    >>> fig = pygmt.Figure()
    >>> plot_hypocentre(fig, source_config, rup_prop_config)
    >>> fig.show()
    """
    kwargs = {
        "style": "a0.3c",
        "pen": "0.3p,black",
        "fill": "gold",
        "label": "Hypocentre",
        **(kwargs or {}),
    }

    initial_fault = source_config.source_geometries[rupture_propagation.initial_fault]
    hypocentre = initial_fault.fault_coordinates_to_wgs_depth_coordinates(
        rupture_propagation.hypocentre
    )
    fig.plot(
        x=hypocentre[1],
        y=hypocentre[0],
        **kwargs,
    )


def plot_rupture_propagation(
    fig: pygmt.Figure,
    source_config: SourceConfig,
    rupture_propagation: RupturePropagationConfig,
    **kwargs: dict[str, Any],
) -> None:
    """Plot the rupture propagation on a figure.

    The rupture propagation is drawn as directed arrows from each parent
    fault to the fault it triggers, following the rupture causality
    tree. Arrows run between the representative points of the fault
    geometries. The hypocentre is not drawn here; use `plot_hypocentre`
    for that.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure to plot on.
    source_config : SourceConfig
        The source configuration (used to locate the faults).
    rupture_propagation : RupturePropagationConfig
        The rupture propagation configuration containing the causality
        tree.
    **kwargs : dict
        Additional keyword arguments to pass to the plotting function. If empty, the default is
        - `style="=0.3c+ea45+s"` (arrow from the parent to the child fault)
        - `pen="0.5p,black"` (arrow line colour)
        - `fill="black"` (arrowhead fill colour)
        - `label="Rupture propagation"` (legend entry)

    Examples
    --------
    >>> import pygmt
    >>> from workflow.realisations import RupturePropagationConfig, SourceConfig
    >>> source_config = SourceConfig.read_from_realisation("realisation.json")
    >>> rup_prop_config = RupturePropagationConfig.read_from_realisation("realisation.json")
    >>> fig = pygmt.Figure()
    >>> plot_rupture_propagation(fig, source_config, rup_prop_config)
    >>> fig.show()
    """
    kwargs = {
        "style": "=0.3c+ea45+s",
        "pen": "0.5p,black",
        "fill": "black",
        **(kwargs or {}),
    }
    # Draw the legend label on only the first arrow so the legend gets a
    # single "Rupture propagation" entry.
    label = kwargs.pop("label", "Rupture propagation")
    for fault_name, parent_name in rupture_propagation.rupture_causality_tree.items():
        if not parent_name:
            continue
        fault = source_config.source_geometries[fault_name]
        parent = source_config.source_geometries[parent_name]
        parent_point = utils.polygon_nztm_to_pygmt(
            parent.geometry
        ).representative_point()
        fault_point = utils.polygon_nztm_to_pygmt(
            fault.geometry
        ).representative_point()
        arrow_kwargs = dict(kwargs)
        if label:
            arrow_kwargs["label"] = label
            label = None
        fig.plot(
            data=[[parent_point.x, parent_point.y, fault_point.x, fault_point.y]],
            **arrow_kwargs,
        )


def plot_realisation(
    realisation_ffp: Path,
    latitude_pad: float = 0,
    longitude_pad: float = 0,
    title: str | None = None,
    subtitle: str | None = None,
    width: float = 10,
    show_geonet_stations: bool = False,
    show_geometry: bool = True,
    show_pgv_targets: bool = False,
    show_hypocentre: bool = False,
    show_traces: bool = False,
    show_rupture_propagation: bool = False,
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
    show_geonet_stations : bool
        Show GeoNet stations on the plot.
    show_geometry : bool
        Show source geometry on the plot.
    show_pgv_targets : bool
        Show PGV targets on the plot.
    show_hypocentre : bool
        Show the rupture hypocentre on the plot.
    show_traces : bool
        Show the fault plane traces (bold surface trace plus dashed 3D
        geometry) on the plot. When set, this replaces the solid source
        geometry outline.
    show_rupture_propagation : bool
        Show the rupture propagation as directed arrows from each parent
        fault to the fault it triggers. Combine with show_hypocentre to
        also mark the rupture origin.
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
    ...     show_hypocentre=True,
    ...     stations=None,
    ... )
    >>> fig.show()
    """
    show_pgv_targets = show_pgv_targets or bool(pgv_targets)
    rupture_propagation = RupturePropagationConfig.read_from_realisation(
        realisation_ffp
    )
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)

    velocity_model_parameters = VelocityModelParameters.read_from_realisation(
        realisation_ffp
    )

    source_config = SourceConfig.read_from_realisation(realisation_ffp)

    rrup_bounding_polygons: list[shapely.Polygon] = []

    if show_pgv_targets:
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

    if show_traces:
        plot_trace(fig, source_config)
    elif show_geometry:
        plot_sources(fig, source_config)

    if stations:
        plot_stations(fig, domain_parameters, stations)

    if show_pgv_targets:
        for pgv_target, rrup_bounding_polygon in zip(
            fault_pgv_targets, rrup_bounding_polygons
        ):
            plot_rrup_polygon(fig, region, pgv_target, rrup_bounding_polygon)

    if show_rupture_propagation:
        plot_rupture_propagation(fig, source_config, rupture_propagation)

    if show_hypocentre:
        plot_hypocentre(fig, source_config, rupture_propagation)

    # Plot the legend overtop the other elements.
    if stations or show_hypocentre or show_traces or show_rupture_propagation:
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
    show_hypocentre: Annotated[
        bool,
        typer.Option(),
    ] = False,
    show_traces: Annotated[
        bool,
        typer.Option(),
    ] = False,
    show_rupture_propagation: Annotated[
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
    show_hypocentre : bool
        Show the rupture hypocentre on the plot.
    show_traces : bool
        Show the fault plane traces (bold surface trace plus dashed 3D
        geometry) on the plot. When set, this replaces the solid source
        geometry outline.
    show_rupture_propagation : bool
        Show the rupture propagation as directed arrows from each parent
        fault to the fault it triggers. Combine with show_hypocentre to
        also mark the rupture origin.
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
    ...     show_hypocentre=True,
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
        show_geonet_stations=show_geonet_stations,
        show_geometry=show_geometry,
        show_pgv_targets=show_pgv_targets,
        show_hypocentre=show_hypocentre,
        show_traces=show_traces,
        show_rupture_propagation=show_rupture_propagation,
        pgv_targets=pgv_targets,
        stations=stations,
    )
    fig.savefig(output_ffp, dpi=dpi)
