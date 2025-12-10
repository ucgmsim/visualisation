import re
from pathlib import Path
from typing import NamedTuple

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
    label: str,
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


# ------------------------
# Compare per-basin subplots
# ------------------------
def compare_sim_to_nshm_subplots(
    realisation_ffp: Path,
    simulation_ds: xr.Dataset,
    period: float,
    basins: list[str] | None = None,
    component: str = "rotd50",
    ymin: float | None = None,
    ymax: float | None = None,
    basin_vs_no_basin: bool = False,
    all_in_one: bool = False,
    span: float = 1 / 3,
):
    """
    Create subplots: first shows all stations, then one subplot per basin.
    """
    # Determine basins to plot
    plot_basins = basins or []

    fig, axes = utils.balanced_subplot_grid(
        1 + len(plot_basins),
        1.0,
        subplot_size=(8, 6),
        clear=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    max_rrup = min(500, simulation_ds.rrup.max().item())
    nshm_rrup = np.geomspace(1e-3, max_rrup, num=100)
    # --- First subplot: all stations ---
    ax = axes[0, 0]
    ax.grid(True, which="both", axis="both", lw=0.3)
    plot_nshm_fit(
        ax,
        realisation_ffp,
        simulation_ds,
        period,
        nshm_rrup,
        color="tab:blue",
        label="NSHM logic tree prediction",
    )
    if basin_vs_no_basin:
        basin_ds = simulation_ds.where(simulation_ds.basin != "")
        non_basin_ds = simulation_ds.where(simulation_ds.basin == "")
        ax.scatter(
            basin_ds.rrup,
            basin_ds.pSA.sel(period=period, component=component).values,
            c="tab:red",
            alpha=0.3,
            s=5,
        )
        ax.scatter(
            non_basin_ds.rrup,
            non_basin_ds.pSA.sel(period=period, component=component).values,
            c="tab:purple",
            alpha=0.3,
            s=5,
        )
        plot_simulation_fit(
            ax,
            basin_ds.rrup.values,
            basin_ds.pSA.sel(period=period, component=component).values,
            label="Basin stations",
            color="darkred",
            span=span,
        )
        plot_simulation_fit(
            ax,
            non_basin_ds.rrup.values,
            non_basin_ds.pSA.sel(period=period, component=component).values,
            label="Non-basin stations",
            color="purple",
            span=span,
        )
    else:
        ax.scatter(
            simulation_ds.rrup,
            simulation_ds.pSA.sel(period=period, component=component).values,
            c="k",
            alpha=0.3,
            s=10,
        )
        plot_simulation_fit(
            ax,
            simulation_ds.rrup.values,
            simulation_ds.pSA.sel(period=period, component=component).values,
            label="Simulated stations",
            color="tab:gray",
            span=span,
        )
    # Plot NSHM
    ax.legend()

    ax.set_yscale("log")
    ax.set_xscale("log")

    if plot_basins and all_in_one:
        for basin in plot_basins:
            subds = simulation_ds.sel(
                station=[
                    s
                    for s, b in zip(
                        simulation_ds.station.values, simulation_ds.basin.values
                    )
                    if b == basin
                ]
            )
            if len(subds.station) == 0:
                continue
            ax.scatter(
                subds.rrup,
                subds.pSA.sel(period=period, component=component).values,
                alpha=0.7,
                s=10,
                label=f"{human_readable_basin(basin)}",
            )
            plot_nshm_fit(
                ax,
                realisation_ffp,
                subds,
                period,
                nshm_rrup,
                color="tab:blue",
            )
    elif plot_basins:
        for i, basin in enumerate(plot_basins):
            row, col = np.unravel_index(i + 1, axes.shape)
            ax = axes[row, col]
            subds = simulation_ds.sel(
                station=[
                    s
                    for s, b in zip(
                        simulation_ds.station.values, simulation_ds.basin.values
                    )
                    if b == basin
                ]
            )
            if len(subds.station) == 0:
                continue
            ax.grid(True, which="both", axis="both", lw=0.3)
            plot_nshm_fit(
                ax,
                realisation_ffp,
                subds,
                period,
                nshm_rrup,
                color="tab:blue",
            )
            ax.scatter(
                subds.rrup,
                subds.pSA.sel(period=period, component=component).values,
                c="red",
                alpha=0.7,
                s=10,
            )

            ax.set_title(f"Basin: {human_readable_name(basin)}")
            ax.set_yscale("log")
            ax.set_xscale("log")

    # --- Axis labels ---
    if plot_basins:
        fig.supxlabel("Source to site distance, $R_{rup}$ [km]")
        fig.supylabel(f"pSA({period:.2f} s) [g]")
    else:
        ax.set_xlabel("Source to site distance, $R_{rup}$ [km]")
        ax.set_ylabel(f"pSA({period:.2f} s) [g]")

    if ymin is not None or ymax is not None:
        for ax in axes.flatten():
            ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlim(left=1e-1, right=max_rrup)
    return fig


# ------------------------
# CLI
# ------------------------
@app.command()
def compare_sim_per_basin(
    realisation_ffp: Path,
    simulation_dataset_path: Path,
    period: float,
    basins: list[str] | None = None,
    save: Path | None = None,
    dpi: int = 300,
    ymin: float | None = 1e-5,
    ymax: float | None = 10,
    component: str = "rotd50",
    compare_basin: bool = False,
    all_in_one: bool = False,
    span: float = 1 / 3,
) -> None:
    """
    Compare simulation dataset results to NSHM with subplots per basin.
    First subplot is all stations.
    """
    simulation_ds = xr.open_dataset(simulation_dataset_path, engine="h5netcdf")

    fig = compare_sim_to_nshm_subplots(
        realisation_ffp,
        simulation_ds,
        period,
        basins=basins,
        component=component,
        ymin=ymin,
        ymax=ymax,
        basin_vs_no_basin=compare_basin,
        span=span,
    )

    if save:
        fig.savefig(save, dpi=dpi)
    else:
        plt.show()
