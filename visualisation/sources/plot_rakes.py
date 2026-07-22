"""Plot a sample of rake values across a multi-segment rupture."""

from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer

from pygmt_helper import plotting
from qcore import cli
from source_modelling import srf

app = typer.Typer()


@cli.from_docstring(app)
def plot_rakes(
    srf_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_ffp: Annotated[Path, typer.Argument(dir_okay=False)],
    dpi: Annotated[float, typer.Option()] = 300,
    title: Annotated[Optional[str], typer.Option()] = None,
    sample_size: Annotated[int, typer.Option()] = 200,
    vector_length: Annotated[float, typer.Option()] = 0.2,
    seed: Annotated[Optional[int], typer.Option()] = None,
    width: Annotated[float, typer.Option(min=0)] = 17,
) -> None:
    """Plot a sample of rake values across a multi-segment rupture.

    Parameters
    ----------
    srf_ffp : Path
        Path to the SRF file to plot.
    output_ffp : Path
        Output plot image.
    dpi : float
        Plot output DPI (higher is better).
    title : Optional[str]
        Plot title to use.
    sample_size : int
        Number of points to sample for rake.
    vector_length : float
        Length of rake vectors (cm).
    seed : Optional[int]
        Random seed to sample rakes with.
    width : float
        Width of plot (in cm).
    """
    srf_data = srf.read_srf(srf_ffp)
    region = (
        srf_data.points["lon"].min() - 0.5,
        srf_data.points["lon"].max() + 0.5,
        srf_data.points["lat"].min() - 0.25,
        srf_data.points["lat"].max() + 0.25,
    )

    fig = plotting.gen_region_fig(
        title,
        projection=f"M{width}c",
        region=region,
        # gen_region_fig dropped its `map_data` argument and now loads the NZ
        # map data itself. These reproduce what map_data=None used to give:
        # plain coastlines, no topography or roads.
        plot_topo=False,
        plot_roads=False,
        plot_highways=False,
        plot_kwargs={"land_color": "#666666", "water_color": "skyblue"},
    )
    i = 0

    np.random.seed(seed)
    vectors = srf_data.points[["lon", "lat", "rake"]].sample(sample_size)
    vectors["rake"] = (vectors["rake"] + 90) % 360
    vectors["length"] = vector_length

    fig.plot(
        data=vectors.values.tolist(), style="v0.1c+e+a30", pen="0.2p", fill="black"
    )
    for _, segment in srf_data.header.iterrows():
        nstk = int(segment["nstk"])
        ndip = int(segment["ndip"])
        point_count = nstk * ndip
        segment_points = srf_data.points.iloc[i : i + point_count]
        corners = segment_points.iloc[[0, nstk - 1, -1, (ndip - 1) * nstk]]
        fig.plot(
            x=corners["lon"].iloc[list(range(len(corners))) + [0]].to_list(),
            y=corners["lat"].iloc[list(range(len(corners))) + [0]].to_list(),
            pen="0.5p,black,-",
        )

        i += point_count

    fig.savefig(
        output_ffp,
        dpi=dpi,
        anti_alias=True,
    )


if __name__ == "__main__":
    app()
