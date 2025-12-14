from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import oq_wrapper as oqw
import pandas as pd
import pygmt
import shapely
import typer
import xarray as xr

from pygmt_helper import plotting
from qcore import cli
from visualisation import realisation, utils
from workflow.realisations import (
    DomainParameters,
    Magnitudes,
    Rakes,
    RupturePropagationConfig,
    SourceConfig,
)

app = typer.Typer()


def plot_diff(
    fig: pygmt.Figure,
    dset: xr.Dataset,
    gmm_psa_value: pd.Series,
    period: float,
    cmap: str,
    cmap_max: float | None,
    cmap_min: float | None,
    ticks: int,
    reverse: bool,
) -> None:
    intensity = dset["pSA"].sel(period=period, component="rotd50")
    pgv_value = intensity.to_series()
    breakpoint()
    diff = np.log(pgv_value) - gmm_psa_value
    cmap_min = cmap_min or diff.min()
    cmap_max = cmap_max or diff.max()

    cmap_limits = utils.range_for(cmap_min, cmap_max, ticks)

    latitude = intensity.latitude.to_series()
    longitude = intensity.longitude.to_series()
    df = pd.DataFrame({"lat": latitude, "lon": longitude, "value": diff})

    grid: xr.DataArray = plotting.create_grid(
        df,
        "value",
        grid_spacing="1000e/1000e",
        region=tuple(fig.region),
        set_water_to_nan=True,
    )

    plotting.plot_grid(
        fig,
        grid,
        cmap,
        cmap_limits,
        ("red", "blue"),
        reverse_cmap=reverse,
        transparency=40,
        plot_contours=False,
    )


def find_region(domain: DomainParameters) -> tuple[float, float, float, float]:
    """Find an appropriate domain,"""
    nz_region = shapely.box(166.0, -48.0, 178.5, -34.0)
    region = shapely.union(
        utils.polygon_nztm_to_pygmt(domain.domain.polygon), nz_region
    )
    (min_x, min_y, max_x, max_y) = shapely.bounds(region)
    return (min_x, max_x, min_y, max_y)


def generate_basemap(region: tuple[float, float, float, float]) -> pygmt.Figure:
    fig: pygmt.Figure = plotting.gen_region_fig(
        title=None,
        region=region,
        plot_kwargs=dict(
            plot_kwargs=["af", "xaf+Longitude", "yaf+Latitude"],
            water_color="white",
            topo_cmap_min=-900,
            topo_cmap_max=3100,
        ),
        plot_highways=False,
        config_options=dict(
            MAP_FRAME_TYPE="plain",
            FORMAT_GEO_MAP="ddd.xx",
            MAP_FRAME_PEN="thinner,black",
        ),
    )
    assert isinstance(fig, pygmt.Figure)
    return fig


@cli.from_docstring(app)
def main(
    realisation_ffp: Annotated[Path, typer.Argument()],
    dataset: Annotated[
        Path,
        typer.Argument(),
    ],
    period: Annotated[
        float,
        typer.Argument(),
    ],
    output: Annotated[
        Path,
        typer.Argument(),
    ],
    cmap: Annotated[
        str,
        typer.Option(),
    ] = "polar",
    reverse: Annotated[
        bool,
        typer.Option(is_flag=True),
    ] = False,
    cmap_min: Annotated[
        float | None,
        typer.Option(),
    ] = None,
    cmap_max: Annotated[
        float | None,
        typer.Option(),
    ] = None,
    ticks: Annotated[
        int,
        typer.Option(),
    ] = 10,
) -> None:
    """Compare simulation results to predictions from the NSHM2022 logic tree.

    Parameters
    ----------
    realisation_ffp : Path
        Path to realisation.
    dataset : Path
        Path to xarray intensity measure dataset.
    period : float
        pSA period to compare against.
    output : Path
        The path to write the figure out to.
    cmap_min : float
        Colourmap minimum
    cmap_max : float
        Colourmap maximum
    ticks : int
        Number of ticks in discrete colourmap.
    output : Path
        Output path.
    cmap : str
        Colourmap to plot log residuals. Should be divering.
    reverse : bool
        If true, reverse the colourmap. Defaults to false.
    """

    dset = xr.open_dataset(dataset, engine="h5netcdf")
    domain = DomainParameters.read_from_realisation(realisation_ffp)
    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    magnitudes = Magnitudes.read_from_realisation(realisation_ffp)
    rakes = Rakes.read_from_realisation(realisation_ffp)
    rupture_propagation_config = RupturePropagationConfig.read_from_realisation(
        realisation_ffp
    )

    region = find_region(domain)

    fig = generate_basemap(region)
    gmm_psa_value = utils.get_gmm_prediction(
        dset, period, source_config, magnitudes, rakes, rupture_propagation_config
    )

    plot_diff(
        fig,
        dset,
        gmm_psa_value,
        period,
        cmap,
        cmap_max,
        cmap_min,
        ticks,
        reverse,
    )
    realisation.plot_domain(fig, domain)
    realisation.plot_sources(fig, source_config)

    fig.savefig(output)
