from pathlib import Path

import pandas as pd
import pygmt
import typer

from pygmt_helper import plotting
from visualisation import realisation, utils
from workflow.realisations import DomainParameters, SourceConfig

app = typer.Typer()


def plot_towns(fig: pygmt.Figure):
    towns = {
        "Blenheim": (173.9569444, -41.5138888),
        "Christchurch": (172.6347222, -43.5313888),
        "Dunedin": (170.3794444, -45.8644444),
        "Greymouth": (171.2063889, -42.4502777),
        "Haast": (169.0405556, -43.8808333),
        "Kaikoura": (173.6802778, -42.4038888),
        "Masterton": (175.658333, -40.952778),
        "Napier": (176.916667, -39.483333),
        "New Plymouth": (174.083333, -39.066667),
        "Nelson": (173.2838889, -41.2761111),
        "Palmerston North": (175.611667, -40.355000),
        "Queenstown": (168.6680556, -45.0300000),
        "Rakaia": (172.0230556, -43.75611111),
        "Rotorua": (176.251389, -38.137778),
        "Taupo": (176.069400, -38.6875),
        "Tekapo": (170.4794444, -44.0069444),
        "Timaru": (171.2430556, -44.3958333),
        "Wellington": (174.777222, -41.288889),
        "Westport": (171.5997222, -41.7575000),
    }
    for label, (lon, lat) in towns.items():
        fig.plot(x=lon, y=lat, style="c0.1c", fill="white", pen="0.3p,black")
        fig.text(
            x=lon, y=lat, text=label, justify="BC", offset="0.15c", font="6p,black"
        )


@app.command(help="Test")
def plot_stations(
    realisation_ffp: Path,
    stations_ll: Path,
    stations_vs30: Path,
    cmap: str,
    title: str,
    width: float,
):
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    stations_vs30_df = pd.read_csv(
        stations_vs30,
        delimiter=r"\s+",
        comment="#",
        names=["name", "vs30"],
        header=None,
    ).set_index("name")
    stations_ll_df = pd.read_csv(
        stations_ll,
        delimiter=r"\s+",
        comment="#",
        names=["lon", "lat", "name"],
        header=None,
    ).set_index("name")
    stations_vs30_df = stations_vs30_df.loc[stations_ll_df.index]
    region = utils.bounding_region_for([domain_parameters.domain.polygon], 0, 0)
    fig = plotting.gen_region_fig(
        title,
        region,
        projection=f"M{width}c",
        plot_kwargs=dict(water_color="white", topo_cmap_min=-900, topo_cmap_max=3100),
        plot_highways=False,
    )

    realisation.plot_domain(fig, domain_parameters, pen="1p,black,-")
    realisation.plot_sources(fig, source_config, fill="blue")

    pygmt.makecpt(
        cmap=cmap.removesuffix("_r"),
        series=[0, 1500, 100],
        reverse=cmap.endswith("_r"),
    )
    realisation.plot_stations(
        fig,
        domain_parameters,
        stations_ll,
        fill=stations_vs30_df["vs30"],
        cmap=True,
        style="i0.3c",
    )
    fig.text(
        x=stations_ll_df["lon"],
        y=stations_ll_df["lat"],
        text=stations_ll_df.index,
        justify="BC",
        offset="0.25c",
        font="6p,black",
    )
    fig.colorbar(frame="xaf+lVs30 (m/s)")
    fig.savefig("map.png")


if __name__ == "__main__":
    app()
