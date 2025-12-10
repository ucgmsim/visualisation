import re
from pathlib import Path
from typing import Annotated, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import oq_wrapper as oqw
import pandas as pd
import typer
import xarray as xr
from matplotlib.axes import Axes
from rpy2.robjects import default_converter, globalenv, numpy2ri, r
from rpy2.robjects.conversion import localconverter

from visualisation import utils
from visualisation.utils import RuptureContext, SiteProperties
from workflow.realisations import (
    Magnitudes,
    Rakes,
    RupturePropagationConfig,
    SourceConfig,
)

app = typer.Typer()


def nshm2022_logic_tree_prediction(
    rupture_context: RuptureContext,
    site_properties: SiteProperties,
    period: float,
    rrup: npt.NDArray[np.floating],
) -> pd.DataFrame:
    tect_type = oqw.constants.TectType.ACTIVE_SHALLOW
    gmm_lt = oqw.constants.GMMLogicTree.NSHM2022
    rupture_df = pd.DataFrame(
        {"rrup": rrup, "vs30measured": False, **rupture_context, **site_properties}
    )
    for dist_metric in ["rjb", "rx", "ry"]:
        rupture_df[dist_metric] = rupture_df["rrup"]

    psa_results = oqw.run_gmm_logic_tree(
        gmm_lt, tect_type, rupture_df, "pSA", periods=[period]
    )
    assert isinstance(psa_results, pd.DataFrame)
    psa_results["rrup"] = rupture_df["rrup"]
    return psa_results


class LowessFit(NamedTuple):
    mean: npt.NDArray[np.floating]
    std_low: npt.NDArray[np.floating]
    std_high: npt.NDArray[np.floating]


def fit_loess_r(
    y: npt.NDArray[np.floating],
    x: npt.NDArray[np.floating],
    x_out: npt.NDArray[np.floating],
    **kwargs,
) -> LowessFit:
    """
    Fit LOESS using R and return fitted values and prediction intervals.
    """
    loess_args = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    loess_call = f"loess(y ~ x, {loess_args})" if loess_args else "loess(y ~ x)"

    with localconverter(default_converter + numpy2ri.converter):
        globalenv["x"] = x
        globalenv["y"] = y
        globalenv["x_out"] = x_out

        r(f"fit <- {loess_call}")
        r("newdat <- data.frame(x=x_out)")
        r("pred <- predict(fit, newdata=newdat, se=TRUE)")
        r("residual_se <- fit$s")

        fit_vals = r("pred$fit")
        if not isinstance(fit_vals, np.ndarray):
            raise ValueError(
                f"Residual stderr evaluation failed, expected float found: {fit_vals=}"
            )

        residual_se_eval = r("residual_se")
        if isinstance(residual_se_eval, np.ndarray):
            residual_se = residual_se_eval.item()
        else:
            raise ValueError(
                f"Residual stderr evaluation failed, expected float found: {residual_se_eval=}"
            )

    std_low = fit_vals - residual_se
    std_high = fit_vals + residual_se

    return LowessFit(fit_vals, std_low, std_high)


def plot_nshm_fit(
    ax: Axes,
    realisation_ffp: Path,
    site_ds: xr.Dataset,
    period: float,
    rrup: npt.NDArray[np.floating],
    color: str | None = None,
    label: str | None = None,
) -> None:
    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    magnitudes = Magnitudes.read_from_realisation(realisation_ffp)
    rupture_prop = RupturePropagationConfig.read_from_realisation(realisation_ffp)
    rakes = Rakes.read_from_realisation(realisation_ffp)
    rupture_context = utils.compute_rupture_context(
        source_config, magnitudes, rakes, rupture_prop
    )
    site_properties = utils.compute_site_properties(site_ds.vs30.values)
    logic_tree_results = nshm2022_logic_tree_prediction(
        rupture_context, site_properties, period, rrup
    )
    period_str = (
        f"{period:.2f}".rstrip("0") if not period.is_integer() else f"{int(period)}.0"
    )
    mean = logic_tree_results[f"pSA_{period_str}_mean"]
    std = logic_tree_results[f"pSA_{period_str}_std_Total"]

    ax.fill_between(
        rrup, np.exp(mean - std), np.exp(mean + std), alpha=0.3, color=color
    )
    ax.plot(rrup, np.exp(mean), c=color, label=label)


def plot_simulation_fit(
    ax: Axes,
    rrup: np.ndarray,
    psa: np.ndarray,
    label: str | None,
    color: str,
    span: float = 1 / 3,
) -> None:
    """Plot LOESS fit for a subset of simulation data."""
    rrup_out = np.linspace(rrup.min(), rrup.max(), num=100)
    fit, ci_low, ci_high = fit_loess_r(
        np.log(psa), np.log(rrup), np.log(rrup_out), span=span
    )
    ax.fill_between(rrup_out, np.exp(ci_low), np.exp(ci_high), alpha=0.3, color=color)
    ax.plot(rrup_out, np.exp(fit), c=color, label=label)


def human_readable_basin_name(basin_name: str) -> str:
    # NE_Otago -> NE Otago
    basin_name_no_underscore = basin_name.replace("_", " ")
    # Greate(r)(W)ellington -> Greate(r) (W)ellington
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", basin_name_no_underscore)


def _get_plotting_params(simulation_ds: xr.Dataset):
    """Calculate common plotting parameters."""
    max_rrup = min(500, simulation_ds.rrup.max().item())
    nshm_rrup = np.geomspace(1e-3, max_rrup, num=100)
    return max_rrup, nshm_rrup


def _apply_style_and_limits(
    fig,
    axes: np.ndarray,
    period: float,
    max_rrup: float,
    ymin: float | None,
    ymax: float | None,
    is_multi_plot: bool,
    ax_main: plt.Axes,
    xmin: float | None = 1e-1,
    xmax: float | None = None,
):
    """Apply final styling, labels, and limits to all axes."""
    if is_multi_plot:
        fig.supxlabel("Source to site distance, $R_{rup}$ [km]")
        fig.supylabel(f"pSA({period:.2f} s) [g]")
    else:
        ax_main.set_xlabel("Source to site distance, $R_{rup}$ [km]")
        ax_main.set_ylabel(f"pSA({period:.2f} s) [g]")

    if ymin is not None or ymax is not None:
        for ax in axes.flatten():
            ax.set_ylim(bottom=ymin, top=ymax)
    xmax = xmax or max_rrup
    for ax in axes.flatten():
        ax.set_xlim(left=xmin, right=xmax)


def _plot_nshm_fit_and_settings(
    ax: plt.Axes,
    realisation_ffp: Path,
    data_ds: xr.Dataset,
    period: float,
    nshm_rrup: np.ndarray,
    label: str | None = "NSHM logic tree prediction",
):
    """Plots the common NSHM fit and initial axis settings (log scale, grid)."""
    ax.grid(True, which="both", axis="both", lw=0.3)
    plot_nshm_fit(
        ax,
        realisation_ffp,
        data_ds,
        period,
        nshm_rrup,
        color="tab:blue",
        label=label,
    )
    ax.set_yscale("log")
    ax.set_xscale("log")


def _get_basin_stations(simulation_ds: xr.Dataset, basin: str):
    """Filter the dataset for stations belonging to a specific basin."""
    # Uses xarray's .where() for cleaner filtering
    return simulation_ds.where(simulation_ds.basin == basin, drop=True)


def plot_basin_vs_no_basin(
    realisation_ffp: Path,
    simulation_ds: xr.Dataset,
    period: float,
    component: str = "rotd50",
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    span: float = 1 / 3,
):
    """
    Creates a single plot comparing simulation data for basin stations
    vs. non-basin stations against the NSHM prediction.
    """
    max_rrup, nshm_rrup = _get_plotting_params(simulation_ds)

    fig, ax = plt.subplots(constrained_layout=True)
    all_axes = np.array([ax])  # For style helper consistency

    # 1. Plot NSHM fit and set axis scales/grid
    _plot_nshm_fit_and_settings(ax, realisation_ffp, simulation_ds, period, nshm_rrup)

    # 2. Split and plot simulation data
    basin_ds = simulation_ds.where(simulation_ds.basin != "", drop=True)
    non_basin_ds = simulation_ds.where(simulation_ds.basin == "", drop=True)

    # Basin stations
    basin_pSA = basin_ds.pSA.sel(period=period, component=component).values
    ax.scatter(basin_ds.rrup, basin_pSA, c="tab:red", alpha=0.3, s=5)
    plot_simulation_fit(
        ax,
        basin_ds.rrup.values,
        basin_pSA,
        label="Basin stations",
        color="darkred",
        span=span,
    )

    # Non-Basin stations
    non_basin_pSA = non_basin_ds.pSA.sel(period=period, component=component).values
    ax.scatter(non_basin_ds.rrup, non_basin_pSA, c="tab:purple", alpha=0.3, s=5)
    plot_simulation_fit(
        ax,
        non_basin_ds.rrup.values,
        non_basin_pSA,
        label="Non-basin stations",
        color="purple",
        span=span,
    )

    ax.legend()

    # 3. Apply final styling
    _apply_style_and_limits(
        fig, all_axes, period, max_rrup, ymin, ymax, False, ax, xmin=xmin, xmax=xmax
    )

    return fig


def plot_separate_basin_subplots(
    realisation_ffp: Path,
    simulation_ds: xr.Dataset,
    period: float,
    basins: list[str],
    component: str = "rotd50",
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    span: float = 1 / 3,
):
    """
    Creates a grid of subplots: one for all stations, and one for each basin.
    """
    if not basins:
        raise ValueError("Basins list cannot be empty for separate basin plotting.")

    max_rrup, nshm_rrup = _get_plotting_params(simulation_ds)

    # Setup figure with 1 + N_basins plots
    num_plots = 1 + len(basins)
    # utils.balanced_subplot_grid is assumed to return a 2D array of axes
    fig, axes_2d = utils.balanced_subplot_grid(
        num_plots,
        1.0,
        subplot_size=(8, 6),
        clear=True,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    all_axes = axes_2d.flatten()
    ax_all_stations = all_axes[0]

    # --- A. Plot All Stations (Primary Plot) ---
    _plot_nshm_fit_and_settings(
        ax_all_stations, realisation_ffp, simulation_ds, period, nshm_rrup
    )

    # Plot all simulation stations together
    all_pSA = simulation_ds.pSA.sel(period=period, component=component).values
    ax_all_stations.scatter(simulation_ds.rrup, all_pSA, c="k", alpha=0.3, s=10)
    plot_simulation_fit(
        ax_all_stations,
        simulation_ds.rrup.values,
        all_pSA,
        label="Simulated stations",
        color="tab:gray",
        span=span,
    )
    ax_all_stations.legend()

    # --- B. Plot Individual Basins ---
    for i, basin in enumerate(basins):
        ax_basin = all_axes[i + 1]
        subds = _get_basin_stations(simulation_ds, basin)

        if len(subds.station) == 0:
            ax_basin.set_title(f"Basin: {human_readable_basin_name(basin)} (No data)")
            continue

        _plot_nshm_fit_and_settings(
            ax_basin, realisation_ffp, subds, period, nshm_rrup, label=None
        )  # No legend for NSHM here

        basin_pSA = subds.pSA.sel(period=period, component=component).values
        ax_basin.scatter(subds.rrup, basin_pSA, c="red", alpha=0.7, s=10)
        ax_basin.set_title(f"Basin: {human_readable_basin_name(basin)}")

    # 3. Apply final styling (is_multi_plot=True)
    _apply_style_and_limits(
        fig,
        all_axes,
        period,
        max_rrup,
        ymin,
        ymax,
        True,
        ax_all_stations,
        xmin=xmin,
        xmax=xmax,
    )

    return fig


def plot_combined_basin_plot(
    realisation_ffp: Path,
    simulation_ds: xr.Dataset,
    period: float,
    basins: list[str],
    component: str = "rotd50",
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    span: float = 1 / 3,
):
    """
    Creates a single plot showing all stations and then overlays each basin.
    """
    if not basins:
        # Fall back to plotting only all stations if no basins are specified
        print("Warning: No basins specified. Plotting all stations only.")

    max_rrup, nshm_rrup = _get_plotting_params(simulation_ds)

    fig, ax = plt.subplots(constrained_layout=True)
    all_axes = np.array([ax])  # For style helper consistency

    # 1. Plot NSHM fit and set axis scales/grid
    _plot_nshm_fit_and_settings(ax, realisation_ffp, simulation_ds, period, nshm_rrup)

    # 2. Plot All Stations (Base Layer)
    all_pSA = simulation_ds.pSA.sel(period=period, component=component).values
    ax.scatter(
        simulation_ds.rrup,
        all_pSA,
        c="k",
        alpha=0.1,
        s=10,
        label="All Simulated Stations",
    )
    plot_simulation_fit(
        ax,
        simulation_ds.rrup.values,
        all_pSA,
        label="Overall Fit",
        color="tab:gray",
        span=span,
    )

    # 3. Overlay Individual Basins
    # We use a color cycle to differentiate the basin scatters
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    for i, basin in enumerate(basins):
        subds = _get_basin_stations(simulation_ds, basin)

        if len(subds.station) == 0:
            continue

        color = colors[i % len(colors)]

        basin_pSA = subds.pSA.sel(period=period, component=component).values
        ax.scatter(
            subds.rrup,
            basin_pSA,
            alpha=0.7,
            s=10,
            color=color,
            label=f"{human_readable_basin_name(basin)} Stations",
        )

    for i, basin in enumerate(basins):
        subds = _get_basin_stations(simulation_ds, basin)

        if len(subds.station) == 0:
            continue

        color = colors[i % len(colors)]

        basin_pSA = subds.pSA.sel(period=period, component=component).values

        plot_simulation_fit(
            ax,
            subds.rrup.values,
            basin_pSA,
            label=None,
            color=color,
            span=1,  # for each basin only show smooth line
        )

    ax.legend()

    # 4. Apply final styling (is_multi_plot=False)
    _apply_style_and_limits(
        fig, all_axes, period, max_rrup, ymin, ymax, False, ax, xmin=xmin, xmax=xmax
    )

    return fig


@app.command()
def plot_basin_split(
    # Arguments
    realisation_ffp: Annotated[
        Path, typer.Argument(help="Path to the NSHM fit data file.")
    ],
    simulation_dataset_path: Annotated[
        Path, typer.Argument(help="Path to the simulation xarray dataset (H5NetCDF).")
    ],
    period: Annotated[
        float, typer.Argument(help="Spectral acceleration period (T) in seconds.")
    ],
    # Options
    save: Annotated[
        Path | None,
        typer.Option("--save", "-s", help="Output path to save the figure."),
    ] = None,
    dpi: Annotated[int, typer.Option(help="DPI for saving the figure.")] = 300,
    ymin: Annotated[float | None, typer.Option(help="Minimum y-axis limit.")] = 1e-5,
    ymax: Annotated[float | None, typer.Option(help="Maximum y-axis limit.")] = 10,
    xmin: Annotated[float | None, typer.Option(help="Minimum x-axis limit.")] = None,
    xmax: Annotated[float | None, typer.Option(help="Maximum x-axis limit.")] = None,
    component: Annotated[
        str, typer.Option(help="PSA component to plot (e.g., rotd50).")
    ] = "rotd50",
    span: Annotated[
        float, typer.Option(help="Smoothing span for the simulation fit line.")
    ] = 1 / 3,
) -> None:
    """
    Creates a single plot comparing simulated pSA for Basin stations against
    Non-Basin stations, along with the NSHM prediction.
    """
    simulation_ds = xr.open_dataset(simulation_dataset_path, engine="h5netcdf")

    fig = plot_basin_vs_no_basin(
        realisation_ffp,
        simulation_ds,
        period,
        component=component,
        ymin=ymin,
        ymax=ymax,
        xmin=xmin,
        xmax=xmax,
        span=span,
    )

    if save:
        fig.savefig(save, dpi=dpi)
    else:
        plt.show()


# ----------------------------------------------------
# 2. Command: All Stations + Separate Basin Subplots
# ----------------------------------------------------
@app.command()
def plot_basins_separate(
    # Arguments
    realisation_ffp: Annotated[
        Path, typer.Argument(help="Path to the NSHM fit data file.")
    ],
    simulation_dataset_path: Annotated[
        Path, typer.Argument(help="Path to the simulation xarray dataset (H5NetCDF).")
    ],
    period: Annotated[
        float, typer.Argument(help="Spectral acceleration period (T) in seconds.")
    ],
    # Options
    basins: Annotated[
        list[str] | None,
        typer.Option(
            "--basin", "-b", help="List of basins to plot in separate subplots."
        ),
    ] = None,
    save: Annotated[
        Path | None,
        typer.Option("--save", "-s", help="Output path to save the figure."),
    ] = None,
    dpi: Annotated[int, typer.Option(help="DPI for saving the figure.")] = 300,
    ymin: Annotated[float | None, typer.Option(help="Minimum y-axis limit.")] = 1e-5,
    ymax: Annotated[float | None, typer.Option(help="Maximum y-axis limit.")] = 10,
    xmin: Annotated[float | None, typer.Option(help="Minimum x-axis limit.")] = None,
    xmax: Annotated[float | None, typer.Option(help="Maximum x-axis limit.")] = None,
    component: Annotated[
        str, typer.Option(help="PSA component to plot (e.g., rotd50).")
    ] = "rotd50",
    span: Annotated[
        float, typer.Option(help="Smoothing span for the overall simulation fit line.")
    ] = 1 / 3,
) -> None:
    """
    Creates a grid of plots: one showing all stations, and one separate subplot
    for each specified basin, comparing to NSHM.
    """
    # Note: basins will be list[str] or None. Check for empty list after parsing.
    basins_list = basins or []
    if not basins_list:
        typer.echo(
            "Error: At least one basin must be specified using --basin for this command.",
            err=True,
        )
        raise typer.Exit(code=1)

    simulation_ds = xr.open_dataset(simulation_dataset_path, engine="h5netcdf")

    fig = plot_separate_basin_subplots(
        realisation_ffp,
        simulation_ds,
        period,
        basins=basins_list,
        component=component,
        ymin=ymin,
        ymax=ymax,
        xmin=xmin,
        xmax=xmax,
        span=span,
    )

    if save:
        fig.savefig(save, dpi=dpi)
    else:
        plt.show()


# ----------------------------------------------------
# 3. Command: All Stations + Basins in Combined Plot
# ----------------------------------------------------
@app.command()
def plot_basins_combined(
    # Arguments
    realisation_ffp: Annotated[
        Path, typer.Argument(help="Path to the NSHM fit data file.")
    ],
    simulation_dataset_path: Annotated[
        Path, typer.Argument(help="Path to the simulation xarray dataset (H5NetCDF).")
    ],
    period: Annotated[
        float, typer.Argument(help="Spectral acceleration period (T) in seconds.")
    ],
    # Options
    basins: Annotated[
        list[str] | None,
        typer.Option(
            "--basin", "-b", help="List of basins to overlay on the main plot."
        ),
    ] = None,
    save: Annotated[
        Path | None,
        typer.Option("--save", "-s", help="Output path to save the figure."),
    ] = None,
    dpi: Annotated[int, typer.Option(help="DPI for saving the figure.")] = 300,
    ymin: Annotated[float | None, typer.Option(help="Minimum y-axis limit.")] = 1e-5,
    ymax: Annotated[float | None, typer.Option(help="Maximum y-axis limit.")] = 10,
    xmin: Annotated[float | None, typer.Option(help="Minimum x-axis limit.")] = None,
    xmax: Annotated[float | None, typer.Option(help="Maximum x-axis limit.")] = None,
    component: Annotated[
        str, typer.Option(help="PSA component to plot (e.g., rotd50).")
    ] = "rotd50",
    span: Annotated[
        float, typer.Option(help="Smoothing span for the overall simulation fit line.")
    ] = 1 / 3,
) -> None:
    """
    Creates a single plot showing all simulation stations (as background),
    overlaid with scatters for each specified basin, and the NSHM prediction.
    """
    basins_list = basins or []
    if not basins_list:
        typer.echo(
            "Warning: No basins specified. Plotting all stations (combined) only.",
            err=True,
        )

    simulation_ds = xr.open_dataset(simulation_dataset_path, engine="h5netcdf")

    fig = plot_combined_basin_plot(
        realisation_ffp,
        simulation_ds,
        period,
        basins=basins_list,
        component=component,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        span=span,
    )

    if save:
        fig.savefig(save, dpi=dpi)
    else:
        plt.show()
