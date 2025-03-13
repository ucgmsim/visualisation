#!/usr/bin/env python3
"""Plot slip-rise-rake for segments."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, NamedTuple, Optional

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pooch
import shapely
import typer
from matplotlib import pyplot as plt
from pooch import Unzip

from qcore import cli, coordinates
from source_modelling import rupture_propagation, srf
from source_modelling.sources import Fault
from visualisation import utils
from workflow.realisations import RupturePropagationConfig, SourceConfig

app = typer.Typer()

# Path on Dropbox: QuakeCoRE/Public/PlottingData/Topo/lds-nz-coastlines-and-islands-polygons-topo-150k-SHP.zip
NZ_SHP_HIGHRES = "https://www.dropbox.com/scl/fi/oal7iuvvmsmheyod8csc7/lds-nz-coastlines-and-islands-polygons-topo-150k-SHP.zip?rlkey=abnebsm06z4l5hx0jd7m1406c&st=klxhet56&dl=1"
NZ_SHP_HIGHRES_HASH = (
    "sha256:4c9547aab7d868f11b0cac4b85eafa4111d104a944d0603c4d77a6af2983108c"
)

PLOT_CONFIG = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
}


def create_grid(
    array: np.ndarray, length: float, width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Create a mesh grid based on array dimensions and physical size.

    Parameters
    ----------
    array : np.ndarray
        Input array.
    length : float
        Segment length in km.
    width : float
        Segment width in km.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Mesh grid for plotting.
    """
    x = np.linspace(0, length, array.shape[1])
    y = np.linspace(0, width, array.shape[0])
    return np.meshgrid(x, y)


def plot_contour(
    ax: plt.Axes,
    data: np.ndarray,
    length: float,
    width: float,
    levels: np.ndarray,
    cmap: str,
    label: str,
    title: str,
    description_dp: int = 2,
    extra_contour_data: np.ndarray = None,
    extra_contour_levels: np.ndarray = None,
    extra_contour_color: str = "black",
    summary: bool = True,
) -> None:
    """Plot a filled contour plot with optional additional contour lines.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on.
    data : np.ndarray
        Data to plot as a contour.
    length : float
        Segment length in km.
    width : float
        Segment width in km.
    levels : np.ndarray
        Contour levels.
    cmap : str
        Colormap to use.
    label : str
        Label for the colorbar.
    title : str
        Plot title.
    description_dp : int, optional
        Decimal places for description, by default 2.
    extra_contour_data : np.ndarray, optional
        Additional data for contour overlay, by default None.
    extra_contour_levels : np.ndarray, optional
        Contour levels for the overlay, by default None.
    extra_contour_color : str, optional
        Color for additional contours, by default "black".
    summary : bool, optional
        If True, include a summary text box, by default True.
    """
    x, y = create_grid(data, length, width)
    contours = ax.contourf(x, y, data, cmap=cmap, levels=levels)
    plt.colorbar(contours, ax=ax, label=label)
    ax.set_ylim(width, 0)
    ax.set_title(title)

    if extra_contour_data is not None:
        extra_contours = ax.contour(
            x,
            y,
            extra_contour_data,
            levels=extra_contour_levels,
            colors=extra_contour_color,
        )
        ax.clabel(extra_contours, extra_contours.levels, fontsize=6)

    if summary:
        ax.text(
            1.0,
            1.1,
            utils.format_description(data),
            transform=ax.transAxes,
            ha="center",
        )

    ax.set_xlabel("along strike (km)")
    ax.set_ylabel("W (km)")


def plot_slip(
    ax: plt.Axes,
    tinit: np.ndarray,
    slip: np.ndarray,
    length: float,
    width: float,
    levels: np.ndarray,
    t_levels: Optional[np.ndarray | int] = 15,
) -> None:
    """Plot slip distribution with optional initial time contours.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on.
    tinit : np.ndarray
        Initial time array.
    slip : np.ndarray
        Slip array.
    length : float
        Segment length in km.
    width : float
        Segment width in km.
    levels : np.ndarray
        Contour levels for slip.
    t_levels : Optional[np.ndarray | int], optional
        Contour levels for initial time, by default None.
    """
    hypocentre_index = np.argmin(tinit.ravel())
    hypocentre_y, hypocentre_x = np.unravel_index(hypocentre_index, tinit.shape)
    plot_contour(
        ax,
        slip,
        length,
        width,
        levels,
        cmap="hot_r",
        label="Slip (cm)",
        title="Slip",
        extra_contour_data=tinit,
        extra_contour_levels=t_levels,
        extra_contour_color="black",
    )
    ax.scatter(
        hypocentre_x * length / tinit.shape[1],
        hypocentre_y * width / tinit.shape[0],
        marker="*",
        s=320,
        c="white",
        edgecolors="black",
    )


def plot_rise(
    ax: plt.Axes, rise_time: np.ndarray, length: float, width: float, levels: np.ndarray
) -> None:
    """Plot rise distribution as a contour plot.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on.
    rise_time : np.ndarray
        Rise time array.
    length : float
        Segment length in km.
    width : float
        Segment width in km.
    levels : np.ndarray
        Contour levels for rise.
    """
    plot_contour(
        ax,
        rise_time,
        length,
        width,
        levels,
        cmap="cool",
        label="Rise Time (s)",
        title="Rise Time",
    )


def plot_rake(
    ax: plt.Axes,
    rake: np.ndarray,
    slip: np.ndarray,
    length: float,
    width: float,
    norm: float,
    stride: int = 3,
) -> None:
    """Plot rake quiver plot.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on.
    rake : np.ndarray
        Rake array.
    slip : np.ndarray
        Slip array, used to scale rake vectors.
    length : float
        Segment length in km.
    width : float
        Segment width in km.
    norm : float
        Scaling paramater, smaller `norm` implies smaller vectors.
    stride : int
        Sampling stride of rake array. Higher `stride` implies sparser output.
    """
    x, y = create_grid(rake, length, width)
    u, v = np.cos(np.radians(rake)), np.sin(np.radians(rake))
    scale = slip * norm
    ax.set_ylim(width, 0)
    ax.quiver(
        x[::stride, ::stride],
        y[::stride, ::stride],
        scale[::stride, ::stride] * u[::stride, ::stride],
        scale[::stride, ::stride] * v[::stride, ::stride],
        scale=3,
        color="black",
    )
    ax.set_title("Rake (deg)")
    ax.set_xlabel("along strike (km)")
    ax.set_ylabel("W (km)")


def plot_map(ax: plt.Axes, geometry: gpd.GeoDataFrame) -> None:
    """Plot segment geometry on a map of New Zealand.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on.
    geometry : gpd.GeoDataFrame
        Geometry to plot.
    """
    nz = gpd.read_file(
        next(
            file
            for file in pooch.retrieve(
                NZ_SHP_HIGHRES, known_hash=NZ_SHP_HIGHRES_HASH, processor=Unzip()
            )
            if file.endswith("shp")
        )
    )

    nz.plot(ax=ax, color="lightgrey")
    xmin, ymin, xmax, ymax = geometry.total_bounds
    pad = 0.5  # add a padding around the geometry
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    geometry["coords"] = geometry["geometry"].apply(
        lambda x: x.representative_point().coords[:]
    )
    geometry["coords"] = [coords[0] for coords in geometry["coords"]]
    geometry.plot(ax=ax, color="red", edgecolor="black")
    for idx, row in geometry.iterrows():
        txt = ax.annotate(
            text=str(idx + 1),
            xy=row["coords"],
            fontsize=24,
            horizontalalignment="center",
        )
        txt.set_bbox({"facecolor": "white"})

    ax.set_aspect("auto")  # Allow the plot to stretch vertically


def plot_slip_histogram(ax: plt.Axes, slip: np.ndarray, summary: bool = True) -> None:
    """Plot slip histogram.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on.
    slip : np.ndarray
        Slip array.
    summary : bool, optional
        Whether to include a summary text box, default True.
    """

    ax.hist(slip.ravel(), density=True)
    ax.set_xlabel("Slip (cm)")
    if summary:
        ax.text(
            1.0,
            1.1,
            utils.format_description(slip),
            transform=ax.transAxes,
            ha="center",
        )


class FaultData(NamedTuple):
    """Fault data tuple returned by `extract_fault_data`."""

    faults: list[Fault]
    slip: list[np.ndarray]
    tinit: list[np.ndarray]
    rise: list[np.ndarray]
    rake: list[np.ndarray]


def extract_fault_data(
    headers: pd.DataFrame,
    segments: list[pd.DataFrame],
    sources: SourceConfig,
    rup_prop: RupturePropagationConfig,
) -> FaultData:
    """Extract fault arrays for plotting.

    Parameters
    ----------
    headers : pd.DataFrame
        DataFrame containing fault segment metadata.
    segments : list[pd.DataFrame]
        List of DataFrames with segment-specific data.
    sources : SourceConfig
        Configuration containing source geometries.
    rup_prop : RupturePropagationConfig
        Configuration containing rupture causality tree.

    Returns
    -------
    FaultData
        The fault data extracted from the SRF.
    """
    faults, slip, tinit, rise, rake = [], [], [], [], []
    fault_names = rupture_propagation.tree_nodes_in_order(
        rup_prop.rupture_causality_tree
    )
    index = 0

    for fault_name in fault_names:
        fault = sources.source_geometries[fault_name]
        ndip = int(headers["ndip"].iloc[index])

        def segment_data(key: str) -> np.ndarray:
            return np.hstack(
                [
                    segments[i][key].values.reshape(ndip, -1)
                    for i in range(index, index + len(fault.planes))
                ]
            )

        slip.append(segment_data("slip"))
        tinit.append(segment_data("tinit"))
        rise.append(segment_data("rise"))
        rake.append(segment_data("rake"))
        faults.append(fault)
        index += len(fault.planes)

    return faults, slip, tinit, rise, rake


class PlotType(StrEnum):
    """Plot type to use."""

    slip = "slip"
    rise = "rise"
    rake = "rake"
    distribution = "dist"


@cli.from_docstring(app)
def plot_slip_rise_rake(
    realisation_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    srf_ffp: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False),
    ],
    output_ffp: Annotated[Path, typer.Argument(dir_okay=False)],
    dpi: Annotated[float, typer.Option()] = 300,
    title: Annotated[Optional[str], typer.Option()] = None,
    width: Annotated[float, typer.Option(min=0)] = 10,
    height: Annotated[float, typer.Option(min=0)] = 10,
    plot_type: Annotated[
        PlotType,
        typer.Option(),
    ] = PlotType.slip,
    segment: Annotated[
        Optional[int],
        typer.Option(),
    ] = None,
) -> None:
    """Plot slip-rise-rake for segments.

    Parameters
    ----------
    realisation_ffp : Path
        Path to ralisation file.
    srf_ffp : Path
        Path to SRF file to plot.
    output_ffp : Path
        Output plot image.
    dpi : float
        Plot output DPI (higher is better).
    title : Optional[str]
        Plot title to use.
    width : float
        Plot width (cm).
    height : float
        Plot height (cm).
    plot_type : PlotType
        Type of plot to generate.
    segment : int, optional
        The segment to plot, default will plot all segments.
    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    srf_data = srf.read_srf(srf_ffp)
    centimeters = 1 / 2.54

    headers = srf_data.header
    segments = srf_data.segments
    sources = SourceConfig.read_from_realisation(realisation_ffp)
    rup_prop = RupturePropagationConfig.read_from_realisation(realisation_ffp)

    faults, slip, tinit, rise, rake = extract_fault_data(
        headers, segments, sources, rup_prop
    )

    global_slip_max = max(segment["slip"].max() for segment in srf_data.segments)
    global_rise_max = max(segment["rise"].max() for segment in srf_data.segments)

    rows = int(np.ceil(np.sqrt(len(faults))))
    cols = int(np.ceil(len(faults) / rows))

    if segment is not None:
        i = segment - 1
        fig, axes = plt.subplots(4, figsize=(width * centimeters, height * centimeters))
        levels = np.linspace(0, global_slip_max, num=20)
        plot_slip(
            axes[0],
            tinit[i],
            slip[i],
            faults[i].length,
            faults[i].width,
            levels,
            t_levels=15,
        )
        axes[0].set_title(f"Slip (cm) on Segment {i + 1}")
        levels = np.linspace(0, global_rise_max, num=20)
        plot_rise(
            axes[1],
            rise[i],
            faults[i].length,
            faults[i].width,
            levels,
        )
        axes[1].set_title(f"Rise Time (s) on Segment {i + 1}")
        scale = 0.1 / global_slip_max
        plot_rake(
            axes[2],
            rake[i],
            slip[i],
            faults[i].length,
            faults[i].width,
            scale,
        )
        axes[2].set_title(f"Rake on Segment {i + 1}")
        plot_slip_histogram(axes[3], slip[i], summary=False)
        axes[3].set_title(f"Slip Density on Segment {i + 1}")
    else:
        plt.rcParams.update(PLOT_CONFIG)
        fig = plt.figure(figsize=(width * centimeters, height * centimeters))
        gs = fig.add_gridspec(rows, cols + 1, width_ratios=[1] * cols + [cols])

        plot_functions = {
            PlotType.slip: lambda ax, i: plot_slip(
                ax,
                tinit[i],
                slip[i],
                faults[i].length,
                faults[i].width,
                np.linspace(0, global_slip_max, num=20),
            ),
            PlotType.rise: lambda ax, i: plot_rise(
                ax,
                rise[i],
                faults[i].length,
                faults[i].width,
                np.linspace(0, global_rise_max, num=20),
            ),
            PlotType.rake: lambda ax, i: plot_rake(
                ax,
                rake[i],
                slip[i],
                faults[i].length,
                faults[i].width,
                0.1 / global_slip_max,
            ),
            PlotType.distribution: lambda ax, i: plot_slip_histogram(ax, slip[i]),
        }
        plot_names = {PlotType.distribution: "Slip distrubition"}

        for i, fault in enumerate(faults):
            ax = fig.add_subplot(gs[np.unravel_index(i, (rows, cols))])
            plot_functions[plot_type](ax, i)
            ax.set_title(
                f"{plot_names.get(plot_type, plot_type).capitalize()} on segment {i + 1}"
            )

        map_ax = fig.add_subplot(gs[:, cols])
        df = gpd.GeoDataFrame(
            data=list(range(len(sources.source_geometries))),
            geometry=[
                shapely.transform(
                    fault.geometry,
                    lambda coords: coordinates.nztm_to_wgs_depth(coords)[:, ::-1],
                )
                for fault in faults
            ],
        )

        plot_map(map_ax, df)
    plt.tight_layout()
    fig.savefig(output_ffp, dpi=dpi)
    plt.close(fig)
