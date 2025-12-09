from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy as sp
import typer
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec

from qcore import cli

app = typer.Typer()


class Units(StrEnum):
    G = "g"
    CMS = "cm/s"
    CMS2 = "cm/s^2"


_UNIT_CONVERSION_TABLE = {Units.G: 981.0, Units.CMS: 1.0, Units.CMS2: 1.0}


def plot_waveform(
    dataset: xr.Dataset,
    station: str,
    ax_x: Axes,
    ax_y: Axes,
    ax_z: Axes,
    dataset_units: Units,
    plot_units: Units | None = None,
    ylim: float | None = None,
    **kwargs,
) -> None:
    """Plot a waveform from a simulation.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to read station waveforms. It is assumed that waveform array is in cm/s^2.
    station : str
        The station to plot.
    dataset_units : Units
        The units of the dataset.
    plot_units : Units, optional
        The units to plot in. If not given, will assume it is the same
        as the dataset units.
    ylim : float or None
        Limit for y-axis.

    Returns
    -------
    Figure
        The matplotlib figure containing the waveform plot.
    Array of axes
        The figure axes.
    """
    if ylim is not None:
        ylim = abs(ylim)
    waveform = dataset.waveform.sel(station=station).values
    time = dataset.time.values
    dt = time[1] - time[0]
    plot_units = plot_units or dataset_units
    conversion = (
        _UNIT_CONVERSION_TABLE[dataset_units] / _UNIT_CONVERSION_TABLE[plot_units]
    )
    waveform *= conversion
    if plot_units == Units.CMS:
        waveform = sp.integrate.cumulative_trapezoid(waveform, dx=dt, initial=0)
    axes = [ax_x, ax_y, ax_z]
    for i, component in enumerate(dataset.component):
        axes[i].plot(time, waveform[i])
        axes[i].grid()
        axes[i].set_ylabel(f"{str(component.item())} [{plot_units}]")
        if ylim is not None:
            axes[i].set_ylim(bottom=-ylim, top=ylim)
    axes[-1].set_xlabel("time [s]")


@cli.from_docstring(app)
def plot_waveform_cli(
    dataset_path: Annotated[Path, typer.Argument()],
    dataset_units: Annotated[Units, typer.Argument()],
    stations: Annotated[list[str], typer.Argument()],
    plot_units: Annotated[Units | None, typer.Option()] = None,
    title: str | None = None,
    save: Path | None = None,
    dpi: int = 300,
    width: float = 20,
    height: float = 15,
    ylim: float | None = None,
    rows: float | None = None,
    columns: float | None = None,
) -> None:
    """Plot a station waveform.

    Parameters
    ----------
    dataset_path : Path
        Path to HDF5 waveform dataset.
    station : str
        The station to plot.
    dataset_units : Units
        The units of the dataset.
    plot_units : Units, optional
        The units to plot in. If not given, will assume it is the same
        as the dataset units.
    title : str, optional
        The title of the figure.
    save : Path, optional
        If given, save the figure to the supplied file.
    dpi : int, optional
        Figure DPI (higher is better quality). Only applies if saving
        the figure to a file.
    width : float, optional
        The figure width, in centimetres.
    height : float, optional
        The figure height, in centimetres.
    ylim : float, optional
        The maximum value for the y-axis.
    """
    dset = xr.open_dataset(dataset_path, engine="h5netcdf")
    cm = 1 / 2.54
    if not (rows or columns):
        n = len(stations)
        rows = int(np.sqrt(n))
        columns = int(np.ceil(n / rows))

    mosaic: list[list[str | None]] = [[None] * columns for _ in range(rows)]
    for i, station in enumerate(stations):
        row, column = np.unravel_index(i, (rows, columns))
        mosaic[row][column] = station

    fig, station_axes = plt.subplot_mosaic(
        mosaic, figsize=(width * cm, height * cm), sharex=True, sharey=True
    )
    for ax in station_axes.values():
        ax.remove()
    # For rescaling axes at the end
    all_station_axes = []
    for station, big_ax in station_axes.items():
        if not station:
            continue  # Skips None placeholders
        gs = GridSpecFromSubplotSpec(
            nrows=3, ncols=1, subplot_spec=big_ax.get_subplotspec(), hspace=0.1
        )
        axes = []
        for spec in gs:
            axes.append(fig.add_subplot(spec))

        for other in axes[:-1]:
            other.set_xticklabels([])

        all_station_axes.extend(axes)
        plot_waveform(
            dset,
            station,
            *axes,
            dataset_units,
            plot_units,
            ylim=ylim,
            figsize=(width * cm, height * cm),
        )
        axes[0].set_title(station)

    maxes = []

    for ax in all_station_axes:
        maxes.extend(ax.get_ylim())
    abs_max = max(abs(lim) for lim in maxes)
    global_ylim = (-abs_max, abs_max)
    for ax in all_station_axes:
        ax.set_ylim(global_ylim)
    if title:
        fig.suptitle(title)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi)
    else:
        fig.show()
        plt.show()


if __name__ == "__main__":
    app()
