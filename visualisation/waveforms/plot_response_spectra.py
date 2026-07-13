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


def setup_plot_styling(
    ax: Axes, component: str, ymax: float | None, ymin: float | None
) -> None:
    ax.set_ylabel(f"pSA [{component}, g]")
    ax.set_xlabel("Period [s]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(visible=True, which="both", axis="both", lw=0.3)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)


def plot_spectra(
    fig: Figure,
    ax: Axes,
    dataset: xr.Dataset,
    stations: list[str],
    component: str,
    ymax: float | None = None,
    ymin: float | None = None,
    labels: list[str] | None = None,
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
    labels = labels or stations
    for label, station in zip(labels, stations):
        spectra = dataset.pSA.sel(station=station, component=component).values
        periods = dataset.period.values

        ax.plot(periods, spectra, label=label)


@cli.from_docstring(app)
def plot_spectra_cli(
    dataset_paths: Annotated[list[Path], typer.Argument()],
    scenarios: Annotated[list[str] | None, typer.Option("--scenario")] = None,
    stations: Annotated[list[str] | None, typer.Option("--station")] = None,
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
    if not stations:
        raise ValueError("Require at least one station to plot.")
    elif len(dataset_paths) > 1 and (
        scenarios is None or len(scenarios) != len(dataset_paths)
    ):
        raise ValueError(
            "Require a label for each dataset, if more than one is provided."
        )

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(width * cm, height * cm))
    setup_plot_styling(ax, component, ymin=ymin, ymax=ymax)
    for i, dataset_path in enumerate(dataset_paths):
        dset = xr.open_dataset(dataset_path, engine="h5netcdf")
        for station in stations:
            label = station
            if scenarios is not None and len(scenarios) > 1 and len(stations) > 1:
                scenario = scenarios[i]
                label = f"{station} ({scenario})"
            elif scenarios is not None and len(stations) == 1:
                label = scenarios[i]

            plot_spectra(
                fig,
                ax,
                dset,
                [station],
                component,
                ymin=ymin,
                ymax=ymax,
                labels=[label],
            )

    if len(stations) * len(dataset_paths) > 1:
        ax.legend()

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi)
    else:
        fig.show()
        plt.show()
