from enum import StrEnum
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy.typing as npt
import scipy as sp
import typer
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qcore import cli

app = typer.Typer()


def plot_spectra(
    fig: Figure,
    ax: Axes,
    dataset: xr.Dataset,
    stations: list[str],
    component: str,
    ymax: float | None = None,
    ymin: float | None = None,
) -> None:
    """Plot a spectra from a simulation.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to read station spectras. It is assumed that spectra array is in cm/s^2.
    station : str
        The station to plot.
    units : Units
        The units to plot with.
    ymax : float or None
        Max limit for y-axis.
    ymin : float or None
        Min limit for y-axis.
    """
    ax.set_ylabel(f"pSA [{component}, g]")
    ax.set_xlabel("Period [s]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(visible=True, which="both", axis="both", lw=0.3)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    for station in stations:
        spectra = dataset.pSA.sel(station=station, component=component).values
        periods = dataset.period.values

        ax.plot(periods, spectra, label=station)

    if len(stations) > 1:
        ax.legend()


@cli.from_docstring(app)
def plot_spectra_cli(
    dataset_path: Annotated[Path, typer.Argument()],
    stations: Annotated[list[str], typer.Argument()],
    title: str | None = None,
    save: Path | None = None,
    dpi: int = 300,
    width: float = 20,
    height: float = 15,
    ymin: float | None = 1e-5,
    ymax: float | None = 1,
    component: str = "rotd50",
) -> None:
    """Plot a station spectra.

    Parameters
    ----------
    dataset_path : Path
        Path to HDF5 spectra dataset.
    station : str
        The station to plot.
    units : Units
        The units to plot in.
    """
    dset = xr.open_dataset(dataset_path, engine="h5netcdf")
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(width * cm, height * cm))
    plot_spectra(
        fig,
        ax,
        dset,
        stations,
        component,
        ymin=ymin,
        ymax=ymax,
    )

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
