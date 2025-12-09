"""Utility functions common to many plotting scripts."""

from typing import Any, Literal, Optional, TypedDict, Unpack

import numpy as np
import numpy.typing as npt
import oq_wrapper as oqw
import pygmt
import scipy as sp
import shapely
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qcore import coordinates
from source_modelling import moment
from workflow.realisations import (
    Magnitudes,
    Rakes,
    RupturePropagationConfig,
    SourceConfig,
)


def format_description(
    arr: np.ndarray, dp: float = 0, compact: bool = False, units: Optional[str] = None
) -> str:
    """Format a statistical description of an array.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    dp : float, optional
        Decimal places to round to, by default 0.
    compact : bool, optional
        Whether to return a compact string (i.e. on one line), by default False.
    units : str, optional
        The units of the values.

    Returns
    -------
    str
        Formatted string containing min, mean, max, and standard deviation.
    """
    min = arr.min()
    mean = np.mean(arr)
    max = arr.max()
    std = np.std(arr)
    if units:
        units = " " + units
    else:
        units = ""
    min_label = f"min = {min:.{dp}f}{units}"
    mean_label = f"μ = {mean:.{dp}f}{units}"
    max_label = f"max = {max:.{dp}f}{units}"
    std_label = f"σ = {std:.{dp}f}{units}"
    if compact:
        return f"{min_label} / {mean_label} / {std_label} / {max_label}"
    return f"{min_label}\n{mean_label} ({std_label})\n{max_label}"


def nztm_to_wgs_wraparound(coords: np.ndarray) -> np.ndarray:
    """Convert NZTM coordinates to WGS84, wrapping around the international date line for PyGMT.


    Parameters
    ----------
    coords : np.ndarray
        NZTM coordinates to convert.

    Returns
    -------
    np.ndarray
        WGS84 coordinates, wrapped around the international date line.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.array([5238700.07489416, 1518491.35216903])
    >>> nztm_to_wgs_wraparound(coords)
    array([172.0, -43.0])
    >>> coords = np.array(coordinates.wgs_depth_to_nztm(np.array([-43, 181])))
    >>> nztm_to_wgs_wraparound(coords)
    array([181.0, -43.0])
    """
    coords = coordinates.nztm_to_wgs_depth(coords)[:, ::-1]
    coords[coords[:, 0] < 0, 0] += 360
    return coords


def polygon_nztm_to_pygmt(polygon: shapely.Polygon) -> shapely.Polygon:
    """Convert a polygon from NZTM to WGS84, wrapping around the international date line for PyGMT.

    Parameters
    ----------
    polygon : shapely.Polygon
        Polygon to convert.

    Returns
    -------
    shapely.Polygon
        Converted polygon.

    Examples
    --------
    >>> import shapely
    >>> p = shapely.Point(5238700.07489416, 1518491.35216903)
    >>> polygon_nztm_to_pygmt(p)
    <POINT (172 -43)>
    >>> q = shapely.Point(*coordinates.wgs_depth_to_nztm(np.array([-43, 181])))
    >>> polygon_nztm_to_pygmt(q)
    <POINT (181 -43)>
    >>> # Note that the coordinates would be negative if coordinates
    >>> # were not wrapped around the international date line.
    """
    return shapely.transform(
        polygon,
        lambda x: nztm_to_wgs_wraparound(x),
    )


def _point_on_polygon(t: float, polygon: shapely.Polygon) -> shapely.Point:
    """Maps t between 0 and 1 (inclusive) to a point on the polygon boundary.

    Parameters
    ----------
    t : float
        Value between 0 and 1 (inclusive).
    polygon : shapely.Polygon
        Polygon to find point on.

    Returns
    -------
    shapely.Point
        Point on the polygon boundary.
    """
    boundary = polygon.exterior
    length = boundary.length
    target_length = t * length
    accumulated_length = 0

    for i in range(len(boundary.coords) - 1):
        p1 = shapely.Point(boundary.coords[i])
        p2 = shapely.Point(boundary.coords[i + 1])
        segment_length = p1.distance(p2)

        if accumulated_length + segment_length >= target_length:
            # Interpolate along this segment
            ratio = (target_length - accumulated_length) / segment_length
            x = p1.x + ratio * (p2.x - p1.x)
            y = p1.y + ratio * (p2.y - p1.y)
            return shapely.Point(x, y)

        accumulated_length += segment_length

    return shapely.Point(
        boundary.coords[-1]
    )  # Should not happen but acts as a failsafe


def _hausdorff_maximisation(
    polygon: shapely.Polygon, other_geom: shapely.Polygon
) -> shapely.Point:
    """Finds the point on polygon maximizing the distance to other_geom.

    Parameters
    ----------
    polygon : shapely.Polygon
        Polygon to find point on.
    other_geom : shapely.Polygon
        Other geometry to maximize distance to.

    Returns
    -------
    shapely.Point
        Point on the polygon boundary maximizing the distance to other_geom.

    See Also
    --------
    shapely.hausdorff_distance : Computes the Hausdorff distance between two geometries.
    """

    def objective(t: float) -> float:  # numpydoc ignore=GL08
        point = _point_on_polygon(t, polygon)
        return -point.distance(other_geom.exterior)  # Negative because we maximize

    result = sp.optimize.minimize_scalar(objective, bounds=(0, 1), method="bounded")
    if result.success:
        return _point_on_polygon(result.x, polygon), -result.fun
    else:
        raise RuntimeError("Optimisation failed")


Region = tuple[float, float, float, float]


def label_polygon(
    fig: pygmt.Figure, region: Region, polygon: shapely.Polygon, label: str, **kwargs
) -> None:
    """Label a polygon on a pygmt figure.

    Will label the boundary of the polygon with the given label. The
    point chosen on the boundary is the point farthest from the region
    boundaries.

    Parameters
    ----------
    fig : pygmt.Figure
        Figure to plot on.
    region : Region
        Region to plot.
    polygon : shapely.Polygon
        Polygon to label.
    label : str
        Label to add.
    **kwargs
        Additional arguments to pass to `pygmt.Figure.text`.
    """
    region_polygon = shapely.box(region[0], region[2], region[1], region[3])
    point, _ = _hausdorff_maximisation(polygon, region_polygon)

    fig.text(x=point.x, y=point.y, text=label, **kwargs)


def plot_polygon(
    fig: pygmt.Figure,
    polygon: shapely.LineString
    | shapely.MultiLineString
    | shapely.Polygon
    | shapely.MultiPolygon,
    **kwargs,
) -> None:
    """Plot a polygon on a pygmt figure.

    Parameters
    ----------
    fig : pygmt.Figure
        Figure to plot on.
    polygon : polygon, linestring, or collection of polygons or linestrings
        Polygon to plot.
    **kwargs
        Additional arguments to pass to `pygmt.Figure.plot`.

    Examples
    --------
    >>> import pygmt
    >>> import shapely.geometry
    >>> fig = pygmt.Figure()
    >>> polygon = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> plot_polygon(fig, polygon, pen="1p,blue,-")
    """

    if isinstance(polygon, shapely.MultiPolygon | shapely.MultiLineString):
        for part in polygon.geoms:
            plot_polygon(fig, part, **kwargs)
    elif isinstance(polygon, shapely.LineString):
        coords = np.array(polygon.coords)
        fig.plot(
            x=coords[:, 0],
            y=coords[:, 1],
            **kwargs,
        )
    else:
        polygon_coords = np.array(polygon.exterior.coords)
        fig.plot(
            x=polygon_coords[:, 0],
            y=polygon_coords[:, 1],
            **kwargs,
        )


def bounding_region_for(
    polygon: shapely.Polygon | list[shapely.Polygon],
    latitude_pad: float,
    longitude_pad: float,
) -> Region:
    """Get the bounding region for a polygon or list of polygons.

    Parameters
    ----------
    polygon : shapely.Polygon | list[shapely.Polygon]
        The polygon(s) to bound.
    latitude_pad : float
        A latitude padding around the region.
    longitude_pad : float
        A longitude padding around the region.


    Returns
    -------
    utils.Region
        The pygmt region bounding the polygons + padding.
    """
    if isinstance(polygon, list):
        polygon = shapely.union_all(polygon)

    min_longitude, min_latitude, max_longitude, max_latitude = shapely.bounds(
        polygon_nztm_to_pygmt(polygon)
    )
    return (
        min_longitude - longitude_pad,
        max_longitude + longitude_pad,
        min_latitude - latitude_pad,
        max_latitude + latitude_pad,
    )


def grid_scale_for_region(region: tuple[float, float, float, float]) -> int:
    """Compute a suitable grid scale for a pygmt region.

    Parameters
    ----------
    region : tuple[float, float, float, float]
        The pygmt region you will plot a grid in.

    Returns
    -------
    int
        A value (in metres) represent for `plotting.create_grid` to
        use when plotting the lat-lon grid. Scale is based on the
        maximum extent in the lat or lon direction for the figure in
        kilometres. Works out that 10km = 25m, 100km = 250m, with a
        minimum resolution of 5m.
    """
    min_lon, max_lon, min_lat, max_lat = region
    lat_km = (max_lat - min_lat) * 111
    lon_km = (max_lon - min_lon) * 111 * np.cos(np.radians((min_lat + max_lat) / 2))
    maximum_extent = max(lat_km, lon_km)
    return int(round(max(5, 2.5 * maximum_extent)))


class SubplotsKwargs(TypedDict, total=False):
    sharex: bool | Literal["none", "all", "row", "col"]
    sharey: bool | Literal["none", "all", "row", "col"]
    subplot_kw: dict[str, Any] | None
    gridspec_kw: dict[str, Any] | None
    figsize: tuple[float, float]
    constrained_layout: bool | dict[str, Any]
    layout: Literal["constrained", "compressed", "tight"] | None


def balanced_subplot_grid(
    n_subplots: int,
    aspect: float,
    subplot_size: tuple[float, float] | None = None,
    squeeze: bool = False,
    clear: bool = False,
    **kwargs: Unpack[SubplotsKwargs],
) -> tuple[Figure, npt.NDArray[Axes]]:
    # This has more columns than rows, i.e. wide
    height = np.sqrt(n_subplots / aspect)
    rows = int(np.ceil(height))
    columns = int(np.ceil(height * aspect))
    # Ensures there are no blank rows.
    rows -= max(0, (rows * columns - n_subplots) // columns)

    if subplot_size:
        width, height = subplot_size
        kwargs["figsize"] = (width * columns, height * rows)

    fig, axes = plt.subplots(rows, columns, **kwargs)
    if n_subplots == 1:
        axes = np.atleast_2d([axes])
    if squeeze:
        axes = axes.squeeze()
    if clear:
        for ax in axes.flatten()[n_subplots:]:
            ax.remove()

    return fig, axes


class RuptureContext(TypedDict):
    mag: float
    rake: float
    dip: float
    hypo_depth: float
    ztor: float
    zbot: float


class SiteProperties(TypedDict):
    vs30: float
    z1pt0: float
    z2pt5: float


def circmean(
    samples: npt.NDArray[np.floating], weights: npt.NDArray[np.floating]
) -> float:
    x = np.cos(samples)
    y = np.sin(samples)
    z = weights * np.array([x, y])
    mean_resultant_vector = np.mean(z, axis=1)
    argument = np.arctan2(mean_resultant_vector[1], mean_resultant_vector[0])
    return float(argument)


def compute_rupture_context(
    source_config: SourceConfig,
    magnitudes_config: Magnitudes,
    rakes_config: Rakes,
    rupture_propagation: RupturePropagationConfig,
) -> RuptureContext:
    moments = []
    dips = []
    rakes = []

    for name, source in source_config.source_geometries.items():
        dips.append(source.dip)
        moments.append(moment.magnitude_to_moment(magnitudes_config[name]))
        rakes.append(rakes_config[name])

    ztor = (
        min(
            source_config.source_geometries.values(), key=lambda source: source.top_m
        ).top_m
        / 1000
    )
    zbot = (
        max(
            source_config.source_geometries.values(), key=lambda source: source.bottom_m
        ).bottom_m
        / 1000
    )
    avg_rake = np.degrees(circmean(np.radians(rakes), np.array(moments)))
    avg_dip = float(np.average(dips, weights=moments))
    avg_moment = float(np.mean(moments))
    total_moment = avg_moment * len(moments)
    magnitude = moment.moment_to_magnitude(total_moment)
    initial_source = source_config.source_geometries[rupture_propagation.initial_fault]
    hypocentre = initial_source.fault_coordinates_to_wgs_depth_coordinates(
        rupture_propagation.hypocentre
    )
    hypo_depth = float(hypocentre[2])
    hypo_depth /= 1000.0
    return RuptureContext(
        mag=magnitude,
        rake=avg_rake,
        dip=avg_dip,
        hypo_depth=hypo_depth,
        ztor=ztor,
        zbot=zbot,
    )


def compute_site_properties(
    site_vs30: npt.NDArray[np.floating] | np.floating,
) -> SiteProperties:
    # Calculate geometric mean of site vs30 using the exponential-log form:
    # exp(1/n sum vs30)
    # This is as opposed to straight-forward calculation
    # product(vs30) ^ (1/n)
    # Which is numerically unstable for a large number of stations due to
    # floating-point arithmetic overflow and inprecision at large values
    # obtained by multiplication.
    if isinstance(site_vs30, np.ndarray):
        vs30 = np.exp(1 / len(site_vs30) * np.sum(np.log(site_vs30)))
    else:
        vs30 = site_vs30
    z1pt0 = oqw.estimations.chiou_young_14_calc_z1p0(vs30)
    z2pt5 = oqw.estimations.campbell_bozorgina_14_calc_z2p5(vs30)
    return SiteProperties(vs30=vs30, z1pt0=z1pt0, z2pt5=z2pt5)
