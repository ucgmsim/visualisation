"""Plot rupture propagation paths for realisations."""

from pathlib import Path
from typing import Annotated

import pygmt
import typer

from pygmt_helper import plotting
from qcore import cli
from visualisation import realisation, utils
from workflow.realisations import RupturePropagationConfig, SourceConfig

app = typer.Typer()


def plot_rupture_path(
    fig: pygmt.Figure,
    source_config: SourceConfig,
    rup_prop_config: RupturePropagationConfig,
):
    """Plot a rupture path.

    The rupture is plotted as a series of directed arrows between two
    faults, from the parent fault to a subsequent fault.

    Parameters
    ----------
    fig : pygmt.Figure
        The pygmt figure to plot on.
    source_config : SourceConfig
        The definition of the sources.
    rup_prop_config : RupturePropagationConfig
        The rupture propagation containing the rupture path.

    Examples
    --------
    >>> # Create a figure and plot a rupture path
    >>> import pygmt
    >>> from workflow.realisations import RupturePropagationConfig, SourceConfig
    >>>
    >>> # Load configurations from files
    >>> source_config = SourceConfig.read_from_realisation("path/to/realisation.json")
    >>> rup_prop_config = RupturePropagationConfig.read_from_realisation("path/to/realisation.json")
    >>>
    >>> # Create a PyGMT figure with appropriate region
    >>> region = utils.bounding_region_for(
    ...     [fault.geometry for fault in source_config.source_geometries.values()],
    ...     latitude_pad=0.5,
    ...     longitude_pad=0.5
    ... )
    >>> fig = plotting.gen_region_fig("Rupture Path", region, projection="M15c")
    >>>
    >>> # Plot the rupture path on the figure
    >>> plot_rupture_path(fig, source_config, rup_prop_config)
    >>>
    >>> # Display or save the figure
    >>> fig.show()
    >>> # Or save to file
    >>> fig.savefig("rupture_path.png")
    """
    realisation.plot_sources(fig, source_config, fill="white")
    realisation.plot_rupture_propagation(fig, source_config, rup_prop_config)
    realisation.plot_hypocentre(fig, source_config, rup_prop_config)


@cli.from_docstring(app)
def plot_rupture_path_to_file(
    realisation_ffp: Annotated[Path, typer.Argument(dir_okay=False)],
    output_ffp: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
    title: Annotated[str | None, typer.Option()] = None,
    latitude_pad: Annotated[float, typer.Option(min=0)] = 0,
    longitude_pad: Annotated[float, typer.Option(min=0)] = 0,
    width: Annotated[float, typer.Option(min=0)] = 17,
    subtitle: Annotated[str | None, typer.Option()] = None,
):
    """Plot a rupture path from a realisation to a file.

    Parameters
    ----------
    realisation_ffp : Path
        The realisation to plot.
    output_ffp : Path
        The output image path.
    title : str, optional
        The title of the plot.
    latitude_pad : float
        The latitude padding to apply (in degrees).
    longitude_pad : float
        The longitude padding to apply (in degrees).
    width : float
        The width of the plot (in cm).
    subtitle : str, optional
        A plot subtitle.

    Examples
    --------
    >>> from pathlib import Path
    >>>
    >>> # Plot a rupture path from a realisation file to a PNG
    >>> plot_rupture_path_to_file(
    ...     realisation_ffp=Path("realisations/alpine_fault_M7.8.json"),
    ...     output_ffp=Path("outputs/alpine_rupture.png"),
    ...     title="Alpine Fault Rupture Simulation",
    ...     latitude_pad=0.3,
    ...     longitude_pad=0.3,
    ...     width=20,
    ...     subtitle="Northward propagation scenario"
    ... )
    >>>
    >>> # Minimal usage with default parameters
    >>> plot_rupture_path_to_file(
    ...     Path("realisations/hikurangi_M8.0.json"),
    ...     Path("outputs/hikurangi_rupture.png")
    ... )
    """
    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    rup_prop_config = RupturePropagationConfig.read_from_realisation(realisation_ffp)
    region = utils.bounding_region_for(
        [fault.geometry for fault in source_config.source_geometries.values()],
        latitude_pad=latitude_pad,
        longitude_pad=longitude_pad,
    )
    fig = plotting.gen_region_fig(
        title, region, projection=f"M{width}c", subtitle=subtitle
    )
    plot_rupture_path(fig, source_config, rup_prop_config)
    fig.savefig(output_ffp)
