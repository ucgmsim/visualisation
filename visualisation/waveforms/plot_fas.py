from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import re
import numpy as np
import numpy.typing as npt
import oq_wrapper as oqw
import pandas as pd
import typer
import xarray as xr
from matplotlib.axes import Axes

from qcore import cli
from visualisation import utils
from workflow.realisations import (
    Magnitudes,
    Rakes,
    RupturePropagationConfig,
    SourceConfig,
)

app = typer.Typer()


def plot_fas(
    ax: Axes,
    dataset: xr.Dataset,
    station: str,
    component: str,
    ymax: float | None = None,
    ymin: float | None = None,
    **kwargs,
) -> None:
    """Plot a Fourier Amplitude Spectrum (FAS) from a simulation.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to read station FAS.
    station : str
        The station to plot.
    component : str
        The component to plot.
    ymax : float or None
        Max limit for y-axis.
    ymin : float or None
        Min limit for y-axis.
    """
    fas = dataset.FAS.sel(station=station, component=component).values
    freqs = dataset.frequency.values
    ax.plot(freqs, fas, **kwargs)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlim(freqs.min(), 50)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(visible=True, which="both", axis="both", lw=0.3)


def plot_fas_estimate(
    ax: Axes, realisation_ffp: Path, dataset: xr.Dataset, station: str, **kwargs
) -> None:
    freqs = dataset.frequency.values
    vs30 = dataset.vs30.sel(station=station).item()
    site_properties = utils.compute_site_properties(vs30)
    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    magnitudes = Magnitudes.read_from_realisation(realisation_ffp)
    rakes = Rakes.read_from_realisation(realisation_ffp)
    rupture_propagation = RupturePropagationConfig.read_from_realisation(
        realisation_ffp
    )
    rupture_context = utils.compute_rupture_context(
        source_config, magnitudes, rakes, rupture_propagation
    )
    site = dataset.sel(station=station)
    longitude = site.longitude.item()
    latitude = site.latitude.item()
    point = np.array([latitude, longitude, 0])
    rrup = (
        min(
            source.rrup_distance(point)
            for source in source_config.source_geometries.values()
        )
        / 1000
    )
    rupture_df = pd.DataFrame(
        {
            "rrup": [rrup],
            "vs30measured": False,
            **site_properties,
            **rupture_context,
        }
    )
    fas_results = oqw.run_gmm(
        oqw.constants.GMM.BA_18,
        oqw.constants.TectType.ACTIVE_SHALLOW,
        rupture_df,
        "EAS",
        frequencies=freqs,
    )
    emp_eas = []
    emp_eas_stddev = []
    for col in fas_results.columns:
        if col.endswith("mean"):
            emp_eas.append(fas_results[col].item())
        elif col.endswith("Total"):
            emp_eas_stddev.append(fas_results[col].item())

    emp_eas = np.array(emp_eas)
    emp_eas_stddev = np.array(emp_eas_stddev)
    ax.plot(freqs, np.exp(emp_eas), **kwargs)
    colour = kwargs.get("c") or kwargs.get("color")
    ax.fill_between(
        freqs,
        np.exp(emp_eas - emp_eas_stddev),
        np.exp(emp_eas + emp_eas_stddev),
        color=colour,
        alpha=0.3,
    )


@cli.from_docstring(app)
def plot_fas_cli(
    realisation_ffp: Annotated[Path, typer.Argument()],
    dataset_path: Annotated[Path, typer.Argument()],
    stations: Annotated[list[str], typer.Argument()],
    title: str | None = None,
    save: Path | None = None,
    dpi: int = 300,
    width: float = 20,
    height: float = 15,
    ymin: float | None = 1e-5,
    ymax: float | None = 1,
    component: str = "geom",
) -> None:
    """Plot a station Fourier Amplitude Spectrum (FAS).

    Parameters
    ----------
    dataset_path : Path
        Path to HDF5 FAS dataset.
    station : str
        The station to plot.
    """
    dset = xr.open_dataset(dataset_path, engine="h5netcdf")
    cm = 1 / 2.54

    fig, axes = utils.balanced_subplot_grid(
        len(stations),
        subplot_size=(width * cm, height * cm),
        aspect=3 / 2,
        sharex=True,
        sharey=True,
        clear=True,
        constrained_layout=True,
    )
    for station, ax in zip(stations, axes.flatten()):
        plot_fas_estimate(
            ax,
            realisation_ffp,
            dset,
            station,
            label="EAS (BA18; μ ± σ)",
            color="blue",
        )
        plot_fas(
            ax,
            dset,
            station,
            component,
            ymin=ymin,
            ymax=ymax,
            label="EAS (Simulation)",
            color="k",
        )
        station_data = dset.sel(station=station)
        vs30 = station_data.vs30.item()
        basin = str(station_data.basin.item())
        if not basin:
            basin = 'No Basin'
        else:
            basin = re.sub('[A-Z]', r' \g<0>', basin).lstrip()
        lat = station_data.latitude.item()
        lon = station_data.longitude.item()
        pga = station_data.PGA.sel(component='rotd50').item()
        pgv = station_data.PGV.sel(component='rotd50').item()
        lon = station_data.longitude.item()
        ax.set_title(f'{station}\n({lat:.3f}, {lon:.3f}) - PGA: {pga:.2g} g - PGV: {pgv:.0f} cm/s - Vs30: {vs30:.0f} m/s - Basin: {basin}')

    if axes.size > 1:
        fig.supylabel(f"EAS [{component}]")
        fig.supxlabel("Frequency [Hz]")
    else:
        ax = axes.flatten().item()
        ax.set_ylabel(f"EAS [{component}]")
        ax.set_xlabel("Frequency [Hz]")

    ax = axes.flatten()[0]
    ax.legend()

    if title:
        fig.suptitle(title)

    if save:
        fig.savefig(save, dpi=dpi)
    else:
        fig.show()
        plt.show()


if __name__ == "__main__":
    app()
