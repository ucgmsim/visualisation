"""Utility functions common to many plotting scripts."""

from collections.abc import Sequence
from typing import Any, Literal, NamedTuple, Optional, TypedDict, Unpack

import numpy as np
import numpy.typing as npt
import oq_wrapper as oqw
import pandas as pd
import pygmt
import scipy as sp
import shapely
import xarray as xr
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rpy2.robjects import default_converter, globalenv, numpy2ri, r
from rpy2.robjects.conversion import localconverter

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
) -> tuple[shapely.Point, float]:
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
    float
        The distance from the point to other_geom

    See Also
    --------
    shapely.hausdorff_distance : Computes the Hausdorff distance between two geometries.
    """

    def objective(t: float) -> float:  # numpydoc ignore=GL08
        point = _point_on_polygon(t, polygon)
        return -point.distance(other_geom.exterior)  # Negative because we maximise

    result = sp.optimize.minimize_scalar(objective, bounds=(0, 1), method="bounded")
    if result.success:
        return _point_on_polygon(float(result.x), polygon), -result.fun
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
) -> tuple[
    Figure,
    Sequence[Axes],  # Although we are really returning a numpy array, numpy does
    # not support generic Axes objects in NDArray.
    # numpy/numpy#24738
]:
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
    vs30measured: bool | npt.NDArray[np.bool_]
    vs30: float | npt.NDArray[np.floating]
    z1pt0: float | npt.NDArray[np.floating]
    z2pt5: float | npt.NDArray[np.floating]
    rrup: float | npt.NDArray[np.floating]
    rjb: float | npt.NDArray[np.floating]
    rx: float | npt.NDArray[np.floating]
    ry: float | npt.NDArray[np.floating]


def circmean(
    samples: npt.NDArray[np.floating], weights: npt.NDArray[np.floating]
) -> float:
    x = np.cos(samples)
    y = np.sin(samples)
    z = weights * np.array([x, y])
    mean_resultant_vector = np.mean(z, axis=1)
    argument = np.arctan2(mean_resultant_vector[1], mean_resultant_vector[0])
    return float(argument)


def compute_site_properties(sites: xr.Dataset) -> SiteProperties:
    vs30 = sites.vs30.values
    z1pt0 = oqw.estimations.chiou_young_08_calc_z1p0(vs30)
    z2pt5 = oqw.estimations.chiou_young_08_calc_z2p5(vs30)
    rrup = sites.rrup.values
    rjb = sites.rjb.values
    rx = rjb
    ry = rjb
    vs30measured = False
    return SiteProperties(
        vs30measured=vs30measured,
        vs30=vs30,
        z1pt0=z1pt0,
        z2pt5=z2pt5,
        rrup=rrup,
        rjb=rjb,
        rx=rx,
        ry=ry,
    )


def get_gmm_prediction(
    sites: xr.Dataset,
    period: float,
    source_config: SourceConfig,
    magnitudes: Magnitudes,
    rakes: Rakes,
    rupture_propagation: RupturePropagationConfig,
) -> pd.Series:
    site_properties = compute_site_properties(sites)
    rupture_context = compute_rupture_context(
        source_config, magnitudes, rakes, rupture_propagation
    )
    breakpoint()
    gmm_df = nshm2022_logic_tree_prediction(rupture_context, site_properties, period)
    gmm_psa_value = gmm_df.loc[:, gmm_df.columns.str.endswith("_mean")].squeeze()
    return gmm_psa_value


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


def mean_vs30(site_vs30: npt.NDArray[np.floating]) -> float:
    # Calculate geometric mean of site vs30 using the exponential-log form:
    # exp(1/n sum vs30)
    # This is as opposed to straight-forward calculation
    # product(vs30) ^ (1/n)
    # Which is numerically unstable for a large number of stations due to
    # floating-point arithmetic overflow and imprecision at large values
    # obtained by multiplication.

    return np.exp(1 / len(site_vs30) * np.sum(np.log(site_vs30)))


def adjust_value(colour: npt.ArrayLike, gamma: float) -> npt.NDArray[np.float64]:
    """Adjust the brightness of an RGB colour.

    Parameters
    ----------
    colour : npt.ArrayLike
        Colour to transform (in RGB format).
    gamma : float
        The brightness to adjust by. Adjustment is multiplicative, so
        ``gamma=1`` is equivalent to no change.

    Returns
    -------
    npt.NDArray[np.float64]
        A brightness adjusted equivalent of `colour` with no change in
        hue or saturation.
    """
    colour = np.asarray(colour)
    # Naive colour brightness adjustment would simply multiply colour
    # by gamma. However, this also changes the hue of the colour,
    # resulting in visually incorrect results. HSV is designed for
    # this manipulation.
    hsv = colors.rgb_to_hsv(colour)
    # HSV colour scale represents every colour as a combination of three components:
    # 1. (H)ue, the quality of the colour (blue, red, green, magenta, etc).
    # 2. (S)aturation, how intense that colour is at a fixed
    #    brightness (e.g. black has zero saturation and looks the same
    #    regardless of brightness).
    # 3. (V)alue, the brightness of the colour
    #
    # We just want to adjust the brightness. To do this we adjust the value component.
    hsv[-1] *= gamma
    return colors.hsv_to_rgb(hsv)


def nice_num(x: float, round: bool) -> float:
    """Find an equivalent "nice number" for `x`.

    A nice number is a number that a power-of-ten multiple of 1, 2, or
    5. See: https://stackoverflow.com/a/16363437

    Parameters
    ----------
    x : float
        The number to find a nice number for.
    round : bool
        If true, round toward the nearest nice number. Otherwise,
        find the next largest.


    Returns
    -------
    float
        The nearest or the next largest nice number.
    """
    exponent = np.floor(np.log10(x))
    fraction = x / (10**exponent)
    if round:
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 3:
            nice_fraction = 2
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10
    else:
        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10
    return nice_fraction * 10**exponent


class PlotDomain(NamedTuple):
    """Named tuple representing plot domain with ticks."""

    low: float
    high: float
    spacing: float


def range_for(low: float, high: float, max_ticks: int) -> tuple[float, float, float]:
    """Given a plotting range and a fixed number of ticks, return its "nicest" representation.

    See: https://stackoverflow.com/a/16363437

    Parameters
    ----------
    low : float
        The lower bound of the plotting range.
    high : float
        The upper bound of the plotting range.
    max_ticks : int
        An upper bound on the number of ticks desired. The number of
        ticks returned is usually *less* than this value.

    Returns
    -------
    PlotDomain
        The lower bound, upper bound and tick spacing corresponding to
        a "nice" representation of this domain.
    """
    range = nice_num(high - low, False)
    tick_spacing = nice_num(range / (max_ticks - 1), True)
    nice_min = np.floor(low / tick_spacing) * tick_spacing
    nice_max = np.ceil(high / tick_spacing) * tick_spacing
    return PlotDomain(nice_min, nice_max, tick_spacing)


def nshm2022_logic_tree_prediction(
    rupture_context: RuptureContext,
    site_properties: SiteProperties,
    period: float,
) -> pd.DataFrame:
    tect_type = oqw.constants.TectType.ACTIVE_SHALLOW
    gmm_lt = oqw.constants.GMMLogicTree.NSHM2022
    rupture_df = pd.DataFrame(
        {"vs30measured": False, **rupture_context, **site_properties}
    )
    psa_results = oqw.run_gmm_logic_tree(
        gmm_lt, tect_type, rupture_df, "pSA", periods=[period]
    )
    assert isinstance(psa_results, pd.DataFrame)
    for site_property in site_properties:
        psa_results[site_property] = rupture_df[site_property]
    return psa_results


class ConfidenceInterval(NamedTuple):
    mean: npt.NDArray[np.floating]
    std_low: npt.NDArray[np.floating]
    std_high: npt.NDArray[np.floating]


def fit_loess_r(
    y: npt.NDArray[np.floating],
    x: npt.NDArray[np.floating],
    x_out: npt.NDArray[np.floating],
    **kwargs,
) -> ConfidenceInterval:
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

        fit_vals = np.asarray(r("pred$fit"))

        residual_se_eval = r("residual_se")
        if isinstance(residual_se_eval, np.ndarray) and len(residual_se_eval) == 1:
            residual_se = float(residual_se_eval.item())  # type: ignore[invalid-argument-type]
        else:
            raise ValueError(
                f"Residual stderr evaluation failed, expected float found: {residual_se_eval=}"
            )

    std_low = fit_vals - residual_se
    std_high = fit_vals + residual_se

    return ConfidenceInterval(fit_vals, std_low, std_high)
