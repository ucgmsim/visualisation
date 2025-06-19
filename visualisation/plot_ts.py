"""Create simulation video of surface ground motion levels."""

import functools
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shapely
import tqdm
import typer
from matplotlib.animation import FFMpegWriter, FuncAnimation

from qcore import cli, coordinates
from qcore.xyts import XYTSFile
from source_modelling import srf
from workflow.realisations import DomainParameters, SourceConfig

app = typer.Typer()

NZTM_CRS = ccrs.epsg(2193)
LATLON_CRS = ccrs.PlateCarree()


def apply_cmap_with_alpha(x: np.ndarray, vmin: float, vmax: float, cmap: str = "hot"):
    """Map the input array x into the 'hot' colormap with linear scaling on alpha.

    Parameters
    ----------
    x : np.ndarray
        Input array to be colour-mapped.
    vmin : float
        Minimum value for normalisation.
    vmax : float
        Maximum value for normalisation.
    cmap : str, optional
        The colour-map to apply to the input array. Default is hot.

    Returns
    -------
    np.ndarray
        RGBA values of the array x mapped using the `cmap` colour-map and linear alpha scaling.
    """
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap(cmap)
    rgb = cmap(norm(x))[..., :3]

    alpha = norm(x)
    rgba = np.concatenate([rgb, alpha[..., np.newaxis]], axis=-1)

    return np.clip(rgba, 0, 1)


def plot_towns(ax: plt.Axes, map_extents: tuple[float, float, float, float]) -> list:
    """Plot towns on the map.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot the towns on.
    map_extents : tuple of float
        The extents of the map in NZTM coordinates.

    Returns
    -------
    list of artists
        The list of artists created by this function, two per town.
    """
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
    x_min, x_max, y_min, y_max = map_extents
    features = []
    for town_name, (lon, lat) in towns.items():
        town_y, town_x = coordinates.wgs_depth_to_nztm(np.array([lat, lon]))
        if x_min <= town_x <= x_max and y_min <= town_y <= y_max:
            features.append(
                ax.plot(
                    town_x,
                    town_y,
                    "o",
                    markersize=4,
                    color="white",
                    markeredgecolor="black",
                    transform=NZTM_CRS,
                    zorder=4,
                )[0]
            )

            features.append(
                ax.text(
                    town_x,
                    town_y,
                    " " + town_name,
                    fontsize=8,
                    color="black",
                    ha="left",
                    va="center",
                    transform=NZTM_CRS,
                    zorder=5,
                )
            )
    return features


def plot_cartographic_features(ax: plt.Axes, scale: str) -> list:
    """Add cartographic features to the map.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot the features on.
    scale : str
        The scale for the cartographic features.

    Returns
    -------
    list of artists
            The list of artists created by this function.
    """
    features = []
    features.append(
        ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="#dcdcdc", zorder=1)
    )

    features.append(
        ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="#b0c4de", zorder=0)
    )
    features.append(
        ax.add_feature(
            cfeature.COASTLINE.with_scale(scale),
            linewidth=0.5,
            edgecolor="black",
            zorder=2,
        )
    )
    features.append(
        ax.add_feature(
            cfeature.BORDERS.with_scale(scale),
            linestyle=":",
            edgecolor="grey",
            zorder=1,
        )
    )
    features.append(
        ax.add_feature(
            cfeature.LAKES.with_scale(scale),
            alpha=0.5,
            facecolor="#b0c4de",
            edgecolor="black",
            linewidth=0.2,
            zorder=1,
        )
    )

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, alpha=0.3, color="gray", linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8, "rotation": 45}
    gl.ylabel_style = {"size": 8}
    features.append(gl)
    return features


def xyts_nztm_corners(xyts_file: XYTSFile) -> np.ndarray:
    """Get the corners of the XYTS file in NZTM coordinates.

    Parameters
    ----------
    xyts_file : XYTSFile
            The XYTS file to get the corners from.

    Returns
    -------
    np.ndarray
            The corners of the XYTS file in NZTM coordinates.
    """
    corners_geo = np.array(xyts_file.corners())
    return coordinates.wgs_depth_to_nztm(corners_geo[:, ::-1])[:, ::-1]


def map_extents(
    nztm_corners: np.ndarray, padding: float
) -> tuple[float, float, float, float]:
    """Compute map extents from XYTS file.

    Parameters
    ----------
    nztm_corners : np.ndarray
        The corners of the XYTS domain.
    padding : float
        A padding around the domain (in metres).

    Returns
    -------
    tuple[float, float, float, float]
        The map extents for the figure (x_min, x_max, y_min, y_max).
    """
    x_min, x_max = nztm_corners[:, 0].min(), nztm_corners[:, 0].max()
    y_min, y_max = nztm_corners[:, 1].min(), nztm_corners[:, 1].max()

    padding_m = padding * 1000

    map_extent_nztm = [
        x_min - padding_m,
        x_max + padding_m,
        y_min - padding_m,
        y_max + padding_m,
    ]

    return map_extent_nztm


def zoom_extents(
    map_extents: tuple[float, float, float, float],
    zoom_centre: tuple[float, float],
    zoom_factor: float,
):
    """Zoom the map extents around a given centre point.

    Parameters
    ----------
    map_extents : tuple[float, float, float, float]
        The original map extents (x_min, x_max, y_min, y_max).
    zoom_centre : tuple[float, float]
        The centre point for zooming (x, y).
    zoom_factor : float
        The zoom factor (1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out, logarithmic scale).

    Returns
    -------
    tuple[float, float, float, float]
        The new map extents after applying the zoom.
    """

    x_min, x_max, y_min, y_max = map_extents
    x_centre, y_centre = zoom_centre
    zoom_coefficient = 2 ** (1 - zoom_factor)
    x_range = (x_max - x_min) * zoom_coefficient
    y_range = (y_max - y_min) * zoom_coefficient

    new_x_min = x_centre - x_range / 2
    new_x_max = x_centre + x_range / 2
    new_y_min = y_centre - y_range / 2
    new_y_max = y_centre + y_range / 2

    return new_x_min, new_x_max, new_y_min, new_y_max


def waveform_coordinates(nztm_corners: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Compute gridpoint coordinates for XYTS waveform.

    Parameters
    ----------
    nztm_corners : np.ndarray
        The corners of the waveform grid.
    nx : int
        The number of x-points in the output grid.
    ny : int
        The number of y-points in the output grid.

    Returns
    -------
    np.ndarray
        A numpy array of shape (2 x ny x nx) containing the x and y
        coordinates of gridpoints in the NZTM coordinate system.
    """
    norm_xi, norm_eta = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    origin = nztm_corners[0]  # Bottom-left corner (x0, y0) in NZTM
    vec_x = nztm_corners[1] - origin  # Vector along xi axis (bottom edge) in NZTM
    vec_y = nztm_corners[3] - origin  # Vector along eta axis (left edge) in NZTM

    coords_nztm = (
        origin[:, np.newaxis, np.newaxis]
        + vec_x[:, np.newaxis, np.newaxis] * norm_xi[np.newaxis, :, :]
        + vec_y[:, np.newaxis, np.newaxis] * norm_eta[np.newaxis, :, :]
    )
    return coords_nztm[::-1, :, :]  # Reverse order to (x, y) for NZTM

def tslice_get(xyts_file: XYTSFile, index: int, downsample: int = 1) -> np.ndarray:
    if downsample > 1:
        frame_data = xyts_file.data[index, :, ::downsample, ::downsample]
    else:
        frame_data = xyts_file.data[index]  # shape: (3, ny, nx)
    accum = np.zeros(frame_data.shape[1:], dtype=np.float32)
    for i in range(3):
        np.add(accum, frame_data[i] ** 2, out=accum)
    np.sqrt(accum, out=accum)
    return accum

@cli.from_docstring(app, name="xyts")
def animate_low_frequency_mpl_nztm(
    realisation_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    xyts_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_mp4: Annotated[
        Path, typer.Argument(writable=True, dir_okay=False, resolve_path=True)
    ],
    max_motion: Annotated[float, typer.Option()] = 10.0,
    padding: Annotated[float, typer.Option()] = 5.0,
    cmap: Annotated[str, typer.Option()] = "hot",
    scale: Annotated[str, typer.Option()] = "10m",
    shading: Annotated[str, typer.Option()] = "gouraud",
    frame_start: Annotated[int | None, typer.Option()] = 0,
    frame_count: Annotated[int | None, typer.Option()] = None,
    width: Annotated[float, typer.Option()] = 30.0,
    height: Annotated[float, typer.Option()] = 30.0,
    dpi: Annotated[int, typer.Option()] = 150,
    fps: Annotated[int, typer.Option()] = 15,
    title: Annotated[str | None, typer.Option()] = None,
    zoom: Annotated[float, typer.Option()] = 1,
    simple_map: Annotated[bool, typer.Option()] = False,
    map_quality: Annotated[int, typer.Option()] = 4,
    downsample: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Render low-frequency output as a 2D video of ground motions.

    Parameters
    ----------
    realisation_ffp : Path
        The input realisation file.
    xyts_ffp : Path
        The input xyts file containing the simulation data.
    output_mp4 : Path
        The output file path for the generated animation.
    max_motion : float, optional
        The maximum ground motion value for color scaling, by default 10.0.
    padding : float, optional
        The padding in km for the map extent, by default 5.0.
    cmap : str, optional
        The colormap to use for the animation, by default "hot".
    scale : str, optional
        The scale for cartopy features, by default "10m".
    shading : str, optional
        The shading method for `plt.pcolormesh`, by default "gouraud".
    frame_start : int, optional
        Set the start frame for the animation.
    frame_count : int | None, optional
        The number of frames to display in the animation, by default None (uses all frames).
    width : float, optional
        The width of the figure in cm, by default 30.
    height : float, optional
        The height of the figure in cm, by default 30.
    dpi : int, optional
        The DPI for the figure, by default 150.0.
    fps : int, optional
        The frames per second for the animation, by default 15.
    title : str | None, optional
        The title for the animation, by default None (no title).
    zoom : float, optional
        Zoom factor for the map, by default 1.0, on a log-scale. Zoom
        centres on centre of source geometry.
    simple_map : bool, optional
        If True, disable OpenStreetMap background and use a simple map.
    map_quality : int, optional
        The quality of the map, by default 4. Has no effect if using a
        simple map. Lower values have lower quality but render faster.
    downsample : int, optional
        The factor by which to downsample the spatial grid, by default 1 (no downsampling).
        Higher values increase performance at the cost of resolution.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(
            "You must have ffmpeg installed. See https://ffmpeg.org/download.html.",
        )
        raise typer.Exit(code=1)

    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    xyts_file = XYTSFile(xyts_ffp)

    nztm_corners = xyts_nztm_corners(xyts_file)
    map_extent_nztm = map_extents(nztm_corners, padding)

    if zoom != 1:
        centre = shapely.centroid(
            shapely.union_all(
                [fault.geometry for fault in source_config.source_geometries.values()]
            )
        )
        map_extent_nztm = zoom_extents(
            map_extent_nztm,
            (centre.y, centre.x),
            zoom,
        )

    frame_count = frame_count or xyts_file.nt
    xr, yr = waveform_coordinates(nztm_corners, xyts_file.nx, xyts_file.ny)

    # Downsample coordinate arrays if requested
    if downsample > 1:
        xr = xr[::downsample, ::downsample]
        yr = yr[::downsample, ::downsample]

    # Create a new figure for this frame
    cm = 1 / 2.54
    fig = plt.figure(figsize=(width * cm, height * cm))
    ax = fig.add_subplot(1, 1, 1, projection=NZTM_CRS)
    ax.set_extent(map_extent_nztm, crs=NZTM_CRS)

    # Add all static elements
    if simple_map:
        plot_cartographic_features(ax, scale)
        plot_towns(ax, map_extent_nztm)
    else:
        request = cimgt.OSM(cache=True)
        ax.add_image(
            request,
            10,
            interpolation="spline36",
            regrid_shape=map_quality * 1000,
            zorder=0,
        )

    ax.add_geometries(
        [shapely.Polygon(nztm_corners)],
        facecolor="none",
        edgecolor="black",
        linestyle="--",
        zorder=1,
        crs=NZTM_CRS,
    )

    ax.add_geometries(
        [
            shapely.transform(fault.geometry, lambda coords: coords[:, ::-1])
            for fault in sorted(
                source_config.source_geometries.values(),
                key=lambda fault: -fault.centroid[-1],
            )
        ],
        facecolor="red",
        edgecolor="black",
        zorder=2,
        crs=NZTM_CRS,
    )


    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    # Add time text
    time_text = ax.text(
        0.98,
        0.02,
        f"Time: {0.0:.2f} s",
        transform=ax.transAxes,
        fontsize=12,
        color="black",
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    if title:
        fig.suptitle(title, fontsize=16)
    initial_data = np.zeros((xr.shape[0], xr.shape[1]))
    pcm = ax.pcolormesh(
        xr,
        yr,
        initial_data,
        cmap=cmap,
        vmin=0,
        vmax=max_motion,
        shading=shading,
        zorder=3,
        rasterized=True,
    )
    time_text = ax.text(
        0.98,
        0.02,
        f"Time: {0.0:.2f} s",
        transform=ax.transAxes,
        fontsize=12,
        color="black",
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )


    cbar = fig.colorbar(
        pcm, ax=ax, orientation="vertical", pad=0.02, aspect=30, shrink=0.8
    )
    cbar.set_label("Ground Motion (cm/s)")
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    def update(frame_index):
        # Remove the old pcolormesh
        nonlocal pcm

        pcm.remove()

        # Get current data and create new pcolormesh
        current_data = tslice_get(xyts_file, frame_index, downsample)
        new_pcm = ax.pcolormesh(
            xr,
            yr,
            current_data,
            cmap=cmap,
            vmin=0,
            vmax=max_motion,
            shading=shading,
            zorder=3,
            rasterized=True,
        )

        # Update the global pcm reference for next iteration
        pcm = new_pcm

        current_time = frame_index * xyts_file.dt
        time_text.set_text(f"Time: {current_time:.2f} s")

    anim = FuncAnimation(
        fig,
        update,
        frames=tqdm.tqdm(range(frame_start, frame_start + frame_count), desc="Rendering frames", unit="frame"),
        blit=False,
    )

    writer = FFMpegWriter(fps=fps)
    anim.save(output_mp4, writer=writer, dpi=dpi)
    plt.close(fig)

def non_zero_data_points(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the non-zero data points in the 3D array.

    Parameters
    ----------
    x : np.ndarray
            The x coordinates of the data points.
    y : np.ndarray
            The y coordinates of the data points.
    z : np.ndarray
            The z coordinates of the data points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
            The non-zero data points in the 3D array.
    """
    mask = z > 0
    return x[mask], y[mask], z[mask]


@cli.from_docstring(app, name="srf")
def animate_srf_slip_times(
    realisation_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    srf_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_mp4: Annotated[
        Path, typer.Argument(writable=True, dir_okay=False, resolve_path=True)
    ],
    max_slip: Annotated[float, typer.Option()] = 10.0,
    padding: Annotated[float, typer.Option()] = 5.0,
    cmap: Annotated[str, typer.Option()] = "hot",
    scale: Annotated[str, typer.Option()] = "10m",
    frame_count: Annotated[int | None, typer.Option()] = None,
    width: Annotated[float, typer.Option()] = 30.0,
    height: Annotated[float, typer.Option()] = 30.0,
    dpi: Annotated[int, typer.Option()] = 150.0,
    fps: Annotated[int, typer.Option()] = 15,
    title: Annotated[str | None, typer.Option()] = None,
    zoom: Annotated[float, typer.Option()] = 1,
    simple_map: Annotated[bool, typer.Option()] = False,
    map_quality: Annotated[int, typer.Option()] = 4,
    frame_dt: Annotated[int, typer.Option(min=0)] = 20,
    downsample: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Render SRF slip times as a 2D video.

    Parameters
    ----------
    realisation_ffp : Path
        The input realisation file.
    srf_ffp : Path
        The input srf file containing the simulation data.
    output_mp4 : Path
        The output file path for the generated animation.
    max_slip : float, optional
        The slip (not ground motion) for color scaling, by default 10.0 cm.
    padding : float, optional
        The padding in km for the map extent, by default 5.0.
    cmap : str, optional
        The colormap to use for the animation, by default "hot".
    scale : str, optional
        The scale for cartopy features, by default "10m".
    frame_count : int | None, optional
        The number of frames to display in the animation, by default None (uses all frames).
    width : float, optional
        The width of the figure in cm, by default 30.
    height : float, optional
        The height of the figure in cm, by default 30.
    dpi : int, optional
        The DPI for the figure, by default 150.0.
    fps : int, optional
        The frames per second for the animation, by default 15.0.
    title : str | None, optional
        The title for the animation, by default None (no title).
    zoom : float, optional
        Zoom factor for the map, by default 1.0, on a log-scale. Zoom
        centres on centre of source geometry.
    simple_map : bool, optional
        If True, disable OpenStreetMap background and use a simple map.
    map_quality : int, optional
        The quality of the map, by default 4. Has no effect if using a
        simple map. Lower values have lower quality but render faster.
    frame_dt : int, optional
        The number of timeslices per dt-step, default is 20.
    downsample : int, optional
        The factor by which to downsample the spatial points, by default 1 (no downsampling).
        Higher values increase performance at the cost of resolution.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(
            "You must have ffmpeg installed. See https://ffmpeg.org/download.html.",
        )
        raise typer.Exit(code=1)

    if dpi % 2:
        dpi += 1

    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    domain_config = DomainParameters.read_from_realisation(realisation_ffp)
    srf_file = srf.read_srf(srf_ffp)

    nztm_corners = coordinates.wgs_depth_to_nztm(domain_config.domain.corners)[:, ::-1]
    slip = srf_file.slipt1_array.tocsc()
    coords = coordinates.wgs_depth_to_nztm(srf_file.points[["lat", "lon"]].values)[
        :, ::-1
    ]

    # Downsample coordinate and slip arrays if requested
    if downsample > 1:
        num_points = slip.shape[0]
        indices_to_keep = np.arange(0, num_points, downsample)
        coords = coords[indices_to_keep]
        slip = slip[indices_to_keep, :]

    map_extent_nztm = map_extents(nztm_corners, padding)

    if zoom != 1:
        centre = shapely.centroid(
            shapely.union_all(
                [fault.geometry for fault in source_config.source_geometries.values()]
            )
        )
        map_extent_nztm = zoom_extents(
            map_extent_nztm,
            (centre.y, centre.x),
            zoom,
        )

    frame_count = frame_count or srf_file.nt

    # Create figure and initial setup
    cm = 1 / 2.54
    fig = plt.figure(figsize=(width * cm, height * cm))
    ax = fig.add_subplot(1, 1, 1, projection=NZTM_CRS)
    ax.set_extent(map_extent_nztm, crs=NZTM_CRS)

    # Add time text

    time_text = ax.text(
        0.98,
        0.02,
        "Time: 0s",
        transform=ax.transAxes,
        fontsize=12,
        color="black",
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    if simple_map:
        plot_cartographic_features(ax, scale)
        plot_towns(ax, map_extent_nztm)
    else:
        request = cimgt.OSM(cache=True)
        request._MAX_THREADS = (
            1  # Limit to one thread because it is in a multiprocess pool.
        )
        ax.add_image(
            request,
            10,
            interpolation="spline36",
            regrid_shape=map_quality * 1000,
            zorder=0,
        )

    ax.add_geometries(
        [shapely.Polygon(nztm_corners)],
        facecolor="none",
        edgecolor="black",
        linestyle="--",
        zorder=1,
        crs=NZTM_CRS,
    )

    ax.add_geometries(
        [
            shapely.transform(fault.geometry, lambda coords: coords[:, ::-1])
            for fault in sorted(
                source_config.source_geometries.values(),
                key=lambda fault: -fault.centroid[-1],
            )
        ],
        facecolor="red",
        edgecolor="black",
        zorder=2,
        crs=NZTM_CRS,
    )

    if title:
        fig.suptitle(title, fontsize=16)

    x, y = coords[:, 0], coords[:, 1]
    init_x, init_y, init_z = non_zero_data_points(x, y, slip[:, 0].todense())
    scat = ax.scatter(
        init_x,
        init_y,
        c=init_z,
        cmap=cmap,
        vmin=0,
        vmax=max_slip,
        transform=NZTM_CRS,
        zorder=100,
    )
    fig.colorbar(
        scat,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        aspect=30,
        shrink=0.8,
        label="Slip (cm)",
    )

    def initial_frame():  # numpydoc ignore=GL08
        time_text.set_text("Time: 0s")
        return [scat, time_text]

    # Setup the animation function
    def render_single_frame(
        frame_index: int,
    ):  # numpydoc ignore=GL08
        # Create a new figure for this frame
        slip_index = frame_index * frame_dt
        slip_end = min(slip_index + frame_dt, srf_file.nt)
        interval_slip_mean = slip[:, list(range(slip_index, slip_end))].mean(axis=1)
        # Add the actual data for this frame
        cur_x, cur_y, z = non_zero_data_points(
            x,
            y,
            interval_slip_mean,
        )
        scat.set_offsets(np.c_[cur_x, cur_y])
        scat.set_array(z)
        time_text.set_text(f"Time: {slip_index * srf_file.dt:.2f} s")
        return [scat, time_text]

    # Create the animation
    anim = FuncAnimation(
        fig,
        render_single_frame,
        init_func=initial_frame,
        frames=tqdm.trange(
            frame_count // frame_dt, desc="Rendering frames", unit="frame"
        ),
        blit=True,
    )

    # Save the animation
    writer = FFMpegWriter(fps=fps)
    anim.save(output_mp4, writer=writer)
    plt.close(fig)
