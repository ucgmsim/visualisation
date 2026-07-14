"""Three ways of looking at the same velocity model.

The three sheets answer three different questions, and are deliberately not three
skins on one figure:

``qa``
    "Is this model broken?" Built to be flicked through a hundred at a time.
    Graded checks sit across the top, and every panel below earns its place by
    being somewhere a defect would show.

``poster``
    "What does this model look like?" A cut-away block carries the figure and
    everything else supports it.

``explorer``
    "What is actually in here?" The full contact sheet: every field at every
    sampled depth, over the distributions the maps cannot show.

All three render from one
:class:`~visualisation.velocity_model.reader.VelocityModelSummary`, so the file
is read once however many sheets are asked for.
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec

from visualisation.velocity_model import checks, reader, style

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    _PROJECTION = ccrs.PlateCarree()
except ImportError:  # pragma: no cover - cartopy is a declared dependency
    ccrs = None
    cfeature = None
    _PROJECTION = None

#: Vp/Vs is dimensionless and its physical centre -- the Poisson-solid sqrt(3) --
#: is the same in every model. So the ratio maps use fixed scales rather than
#: adaptive ones, which buys comparability between files as well as between
#: depths. Two are needed: rock hugs sqrt(3) so tightly that a scale wide enough
#: for saturated sediment would render the entire crust a single flat tone.
VPVS_ROCK_HALF_RANGE = 0.30
VPVS_SEDIMENT_HALF_RANGE = 0.90

#: Degrees of padding around the domain on the maps.
MAP_PAD = 0.05

#: How much the block diagram stretches depth. A crustal model is a hundred times
#: wider than it is deep, so drawn true to scale it is a sheet of paper.
BLOCK_DEPTH_EXAGGERATION = 3.0

#: How much of its axes the block fills. A 3D axes shrinks itself to its box
#: aspect and leaves the rest empty, so this claims the space back.
BLOCK_ZOOM = 1.75

#: A colour bar attached to a map steals this share of the axes height.
_BAR_SHARE = 0.09

_COLUMN_WIDTH = 3.6
_GAP_W = 0.42
_GAP_H = 0.72
_MARGIN = (0.58, 0.30, 0.34, 0.72)  # left, right, top, bottom

#: Set once, if coastlines cannot be drawn (no cached Natural Earth data and no
#: network). Everything else still works without them.
_COASTLINES_UNAVAILABLE = False


def _palette(name: str) -> plt.matplotlib.colors.Colormap:
    """A colour map that renders water as water rather than as a value.

    Parameters
    ----------
    name : str
        Name of the colour map.

    Returns
    -------
    matplotlib.colors.Colormap
        The colour map, with masked cells set to the water colour.
    """
    return plt.get_cmap(name).with_extremes(bad=style.WATER)


def _map_aspect(summary: reader.VelocityModelSummary) -> float:
    """Height over width of the domain as it will be drawn.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    float
        The aspect ratio. Sizing the figure from this is what stops the maps
        from floating in a sea of whitespace.
    """
    across = (summary.lon.max() - summary.lon.min()) + 2 * MAP_PAD
    down = (summary.lat.max() - summary.lat.min()) + 2 * MAP_PAD
    return float(down / across)


def _map_row_height(summary: reader.VelocityModelSummary) -> float:
    """Height, in inches, of a row of maps with colour bars under them.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    float
        The row height.
    """
    return _COLUMN_WIDTH * _map_aspect(summary) / (1.0 - _BAR_SHARE)


def _sheet(n_columns: int, row_heights: list[float]) -> tuple[Figure, GridSpec]:
    """Lay out a figure whose rows are the requested heights in inches.

    Parameters
    ----------
    n_columns : int
        Number of columns.
    row_heights : list of float
        Height of each row, in inches.

    Returns
    -------
    tuple
        The figure and its grid.
    """
    left, right, top, bottom = _MARGIN
    width = left + n_columns * _COLUMN_WIDTH + (n_columns - 1) * _GAP_W + right
    height = top + sum(row_heights) + (len(row_heights) - 1) * _GAP_H + bottom

    figure = Figure(figsize=(width, height))
    grid = GridSpec(
        len(row_heights),
        n_columns,
        figure=figure,
        height_ratios=row_heights,
        hspace=_GAP_H / (sum(row_heights) / len(row_heights)),
        wspace=_GAP_W / _COLUMN_WIDTH,
        left=left / width,
        right=1.0 - right / width,
        top=1.0 - top / height,
        bottom=bottom / height,
    )
    return figure, grid


def _add_coastlines(ax: plt.Axes) -> None:
    """Draw coastlines, if the Natural Earth data can be reached.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        A cartopy map axes.
    """
    global _COASTLINES_UNAVAILABLE  # noqa: PLW0603
    if cfeature is None or _COASTLINES_UNAVAILABLE:
        return
    try:
        ax.add_feature(
            cfeature.COASTLINE.with_scale("50m"),
            linewidth=0.5,
            edgecolor=style.INK_SECONDARY,
        )
    except (OSError, RuntimeError, ValueError):
        # Offline with nothing cached. A map without a coastline is still a map.
        _COASTLINES_UNAVAILABLE = True


def _map_axes(
    figure: Figure, cell: SubplotSpec, summary: reader.VelocityModelSummary
) -> plt.Axes:
    """Create a geographic axes covering the model domain.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to add the axes to.
    cell : matplotlib.gridspec.SubplotSpec
        The grid cell to place it in.
    summary : reader.VelocityModelSummary
        The model summary, for the extent.

    Returns
    -------
    matplotlib.pyplot.Axes
        The new axes.
    """
    ax = figure.add_subplot(cell, projection=_PROJECTION)
    ax.set_extent(
        [
            summary.lon.min() - MAP_PAD,
            summary.lon.max() + MAP_PAD,
            summary.lat.min() - MAP_PAD,
            summary.lat.max() + MAP_PAD,
        ],
        crs=_PROJECTION,
    )
    ax.spines["geo"].set_edgecolor(style.AXIS)
    ax.spines["geo"].set_linewidth(0.8)
    return ax


def _draw_domain(ax: plt.Axes, summary: reader.VelocityModelSummary) -> None:
    """Outline the rotated model domain.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        A cartopy map axes.
    summary : reader.VelocityModelSummary
        The model summary.
    """
    lat, lon = summary.lat, summary.lon
    ax.plot(
        np.concatenate([lon[0, :], lon[:, -1], lon[-1, ::-1], lon[::-1, 0]]),
        np.concatenate([lat[0, :], lat[:, -1], lat[-1, ::-1], lat[::-1, 0]]),
        color=style.INK_SECONDARY,
        linewidth=0.7,
        transform=_PROJECTION,
    )


def _draw_basin_outline(ax: plt.Axes, summary: reader.VelocityModelSummary) -> None:
    """Outline the basin footprint, for context on a surface map.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        A cartopy map axes.
    summary : reader.VelocityModelSummary
        The model summary.
    """
    if summary.surface_basin is None:
        return
    ax.contour(
        summary.lon,
        summary.lat,
        (summary.surface_basin != reader.FILL).astype(float),
        levels=[0.5],
        colors=[style.CATEGORICAL[5]],
        linewidths=0.6,
        transform=_PROJECTION,
    )


def _robust_limits(
    values: np.ndarray,
    mask: np.ndarray | None = None,
    span: tuple[float, float] = (2, 98),
) -> tuple[float, float]:
    """Percentile limits that ignore masked cells and outliers.

    Parameters
    ----------
    values : numpy.ndarray
        The field.
    mask : numpy.ndarray, optional
        Cells to exclude, such as water.
    span : tuple of float, optional
        Lower and upper percentiles.

    Returns
    -------
    tuple of float
        The limits, widened if the field turns out to be flat.
    """
    keep = values if mask is None else values[~mask]
    keep = keep[np.isfinite(keep)]
    if keep.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(keep, span)
    if high - low < 1e-9:
        return float(low) - 0.5, float(high) + 0.5
    return float(low), float(high)


def _vpvs_norm(depth_km: float, sediment_km: float) -> tuple[Normalize, str]:
    """The fixed Vp/Vs colour scale appropriate to a depth.

    Both scales are centred on sqrt(3), so the neutral midpoint always means
    "ordinary rock" and any colour at all is a signed departure from it. Which
    of the two is used depends only on depth, so it stays reproducible from file
    to file.

    Parameters
    ----------
    depth_km : float
        Depth of the layer.
    sediment_km : float
        Depth below which the model is rock rather than sediment.

    Returns
    -------
    tuple
        The colour scale, and a note naming it.
    """
    centre = style.POISSON_SOLID_VPVS
    if depth_km >= sediment_km:
        half, note = VPVS_ROCK_HALF_RANGE, "rock scale"
    else:
        half, note = VPVS_SEDIMENT_HALF_RANGE, "sediment scale"
    return Normalize(centre - half, centre + half), f"Vp/Vs ({note}, √3 at centre)"


def _draw_field_map(
    ax: plt.Axes,
    summary: reader.VelocityModelSummary,
    values: np.ndarray,
    water: np.ndarray,
    cmap: str,
    norm: Normalize,
) -> plt.cm.ScalarMappable:
    """Draw one field over the domain, with water rendered as water.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        A cartopy map axes.
    summary : reader.VelocityModelSummary
        The model summary.
    values : numpy.ndarray
        The field to draw.
    water : numpy.ndarray
        Cells clamped to the velocity floor.
    cmap : str
        Name of the colour map.
    norm : matplotlib.colors.Normalize
        The colour scale.

    Returns
    -------
    matplotlib.cm.ScalarMappable
        The mesh, for attaching a colour bar to.
    """
    palette = _palette(cmap)
    mesh = ax.pcolormesh(
        summary.lon,
        summary.lat,
        np.ma.masked_where(water | ~np.isfinite(values), values),
        cmap=palette,
        norm=norm,
        shading="auto",
        transform=_PROJECTION,
        rasterized=True,
    )
    _add_coastlines(ax)
    _draw_domain(ax, summary)
    return mesh


def _slim_colourbar(
    figure: Figure, mesh: plt.cm.ScalarMappable, ax: plt.Axes, label: str
) -> None:
    """Attach a compact horizontal colour bar beneath a map.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure.
    mesh : matplotlib.cm.ScalarMappable
        The mesh to describe.
    ax : matplotlib.pyplot.Axes
        The axes to attach beneath.
    label : str
        Colour bar label.
    """
    bar = figure.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        pad=0.04,
        fraction=_BAR_SHARE - 0.04,
        aspect=30,
    )
    bar.set_label(label, fontsize=7.5, color=style.INK_SECONDARY)
    bar.ax.tick_params(labelsize=7, length=2, color=style.AXIS)
    bar.outline.set_visible(False)


def _legend(ax: plt.Axes, location: str = "lower left") -> None:
    """Add a legend that stays readable over dark data.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        The axes to add it to.
    location : str, optional
        Where to place it.
    """
    ax.legend(
        loc=location,
        fontsize=7,
        frameon=True,
        facecolor=style.SURFACE,
        edgecolor="none",
        framealpha=0.88,
    )


def _draw_section(
    ax: plt.Axes,
    figure: Figure,
    section: reader.Section,
    values: np.ndarray,
    cmap: str,
    norm: Normalize,
    title: str,
) -> plt.cm.ScalarMappable:
    """Draw a vertical cross-section, with the basin outline over it.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        The axes to draw on.
    figure : matplotlib.figure.Figure
        The figure, for working out the vertical exaggeration.
    section : reader.Section
        The section geometry.
    values : numpy.ndarray
        The field to draw, shaped ``(depth, distance)``.
    cmap : str
        Name of the colour map.
    norm : matplotlib.colors.Normalize
        The colour scale.
    title : str
        Axes title.

    Returns
    -------
    matplotlib.cm.ScalarMappable
        The mesh, for attaching a colour bar to.
    """
    palette = _palette(cmap)
    mesh = ax.pcolormesh(
        section.distance_km,
        section.depth_km,
        np.ma.masked_where(section.water, values),
        cmap=palette,
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    if section.basin is not None:
        ax.contour(
            section.distance_km,
            section.depth_km,
            (section.basin != reader.FILL).astype(float),
            levels=[0.5],
            colors=[style.CATEGORICAL[5]],
            linewidths=0.7,
        )
    ax.invert_yaxis()

    # State the vertical exaggeration rather than let the reader assume there is
    # none: a 44 km column drawn beside a 400 km one is always stretched.
    box = ax.get_position()
    across = box.width * figure.get_figwidth() / section.distance_km[-1]
    down = (
        box.height
        * figure.get_figheight()
        / (section.depth_km[-1] - section.depth_km[0])
    )
    ax.set_xlabel(
        f"Distance along section (km) — vertical exaggeration ×{down / across:.1f}"
    )
    ax.set_ylabel("Depth (km)")
    ax.set_title(title, loc="left")
    return mesh


def _draw_depth_density(
    ax: plt.Axes,
    summary: reader.VelocityModelSummary,
    field: str,
    xlabel: str,
    reference: float | None = None,
    limits: tuple[float, float] | None = None,
) -> None:
    """Draw how a field is distributed at every depth in the model.

    Each depth is normalised to its own peak, so the *shape* of the distribution
    is legible at every depth rather than only where the model happens to have
    the most cells. This is where layering, gradients and the crust-mantle
    transition show themselves, and where a discontinuity that no map would
    reveal becomes obvious.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        The axes to draw on.
    summary : reader.VelocityModelSummary
        The model summary.
    field : str
        Which field to draw.
    xlabel : str
        Axis label for the field.
    reference : float, optional
        A reference value to mark, such as the Poisson-solid Vp/Vs.
    limits : tuple of float, optional
        Explicit horizontal limits.
    """
    centres, counts = summary.profile.density[field]
    depth = summary.profile.depth_km
    ax.pcolormesh(
        centres,
        depth,
        counts / np.maximum(counts.max(axis=1, keepdims=True), 1),
        cmap=style.DENSITY_CMAP,
        shading="auto",
        rasterized=True,
    )

    quantiles = summary.profile.quantiles[field]
    ax.plot(quantiles[2], depth, color=style.INK, linewidth=1.4, label="median")
    ax.plot(
        quantiles[0],
        depth,
        color=style.INK_SECONDARY,
        linewidth=0.8,
        label="5th / 95th percentile",
    )
    ax.plot(quantiles[4], depth, color=style.INK_SECONDARY, linewidth=0.8)
    if reference is not None:
        ax.axvline(
            reference,
            color=style.CATEGORICAL[5],
            linewidth=1.0,
            label=f"$\\sqrt{{3}}$ = {reference:.3f}",
        )

    if limits is None:
        finite = quantiles[np.isfinite(quantiles)]
        if finite.size:
            pad = 0.08 * (finite.max() - finite.min() + 1e-6)
            limits = (finite.min() - pad, finite.max() + pad)
    if limits is not None:
        ax.set_xlim(*limits)
    ax.set_ylim(depth[-1], depth[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth (km)")
    _legend(ax)


def _rock_limits(
    summary: reader.VelocityModelSummary, field: str
) -> tuple[float, float]:
    """Horizontal limits for a depth-density panel, set by the rock column.

    Without this the shallow sediment -- whose Vp/Vs runs past 3 -- stretches the
    axis so far that the entire crust collapses into one line.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.
    field : str
        Which field.

    Returns
    -------
    tuple of float
        The limits.
    """
    rock = summary.profile.depth_km >= reader.sediment_depth_km(summary)
    quantiles = summary.profile.quantiles[field][:, rock]
    finite = quantiles[np.isfinite(quantiles)]
    if finite.size == 0:
        return 1.4, 2.5
    pad = 0.25 * (finite.max() - finite.min() + 1e-6)
    return float(finite.min() - pad), float(finite.max() + pad)


def _draw_brocher(ax: plt.Axes, summary: reader.VelocityModelSummary) -> None:
    """Draw density against Vp, with Brocher's empirical relation over it.

    Density in a velocity model is almost always derived from Vp. If the cloud
    does not lie along the curve, something in the model build has gone wrong,
    which makes this the cheapest deep check available.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        The axes to draw on.
    summary : reader.VelocityModelSummary
        The model summary.
    """
    sample = summary.profile.sample
    good = np.isfinite(sample["vp"]) & np.isfinite(sample["rho"])
    vp, rho = sample["vp"][good], sample["rho"][good]

    ax.hist2d(
        vp, rho, bins=120, norm=LogNorm(), cmap=style.DENSITY_CMAP, rasterized=True
    )
    grid = np.linspace(vp.min(), vp.max(), 200)
    ax.plot(
        grid,
        reader.brocher_density(grid),
        color=style.CATEGORICAL[5],
        linewidth=1.6,
        label="Brocher (2005)",
    )
    ax.set_xlabel(style.FIELD_LABELS["vp"])
    ax.set_ylabel(style.FIELD_LABELS["rho"])
    ax.set_title("Density against Vp", loc="left")
    _legend(ax, "upper left")


def _draw_fractions(ax: plt.Axes, summary: reader.VelocityModelSummary) -> None:
    """Draw what share of the model is basin and water, against depth.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        The axes to draw on.
    summary : reader.VelocityModelSummary
        The model summary.
    """
    profile = summary.profile
    depth = profile.depth_km
    ax.plot(
        profile.water_fraction * 100,
        depth,
        color=style.CATEGORICAL[1],
        label="clamped to the Vs floor (water)",
    )
    if profile.basin_fraction is not None:
        ax.plot(
            profile.basin_fraction * 100,
            depth,
            color=style.CATEGORICAL[0],
            label="carries a basin label",
        )
        # Direct-label the one number that matters here: whether the basin
        # labels stop where basins stop.
        at_base = profile.basin_fraction[-1]
        if at_base > 0:
            ax.annotate(
                f"{at_base * 100:.2f}% still labelled\nat {depth[-1]:.1f} km",
                xy=(at_base * 100, depth[-1]),
                xytext=(26, 24),
                textcoords="offset points",
                fontsize=7.5,
                color=style.CATEGORICAL[0],
                arrowprops={
                    "arrowstyle": "-",
                    "color": style.CATEGORICAL[0],
                    "linewidth": 0.7,
                },
            )

    ax.set_ylim(depth[-1], depth[0])
    ax.set_xlabel("Share of sampled cells (%)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Basin labels and water with depth", loc="left")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    _legend(ax, "center right")


def _subtitle(summary: reader.VelocityModelSummary) -> str:
    """One line describing the model's geometry and provenance.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    str
        The description.
    """
    meta = summary.meta
    nz, ny, nx = meta.shape
    parts = [f"{nx} × {ny} × {nz} cells"]
    if meta.spacing_km:
        parts.append(f"{meta.spacing_km * 1000:.0f} m spacing")
    if meta.extent_km:
        parts.append(f"{meta.extent_km[0]:.0f} × {meta.extent_km[1]:.0f} km")
    parts.append(f"{meta.depth_km[0]:.0f}–{meta.depth_km[-1]:.1f} km deep")
    if meta.origin:
        parts.append(f"origin {meta.origin[0]:.3f}, {meta.origin[1]:.3f}")
    if meta.rotation_deg is not None:
        parts.append(f"rotated {meta.rotation_deg:.1f}°")
    if meta.model_version:
        parts.append(f"NZVM v{meta.model_version}")
    if meta.topo_type:
        parts.append(meta.topo_type.lower().replace("_", " "))
    return "  ·  ".join(parts)


def _draw_header(
    ax: plt.Axes,
    summary: reader.VelocityModelSummary,
    title: str,
    graded: list[checks.Check],
) -> None:
    """Draw the title block and, if given, the strip of graded checks.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        A blank axes spanning the top of the figure.
    summary : reader.VelocityModelSummary
        The model summary.
    title : str
        What this sheet is.
    graded : list of checks.Check
        The checks to display. May be empty.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0, 0.97, summary.meta.name, fontsize=19, fontweight="bold", va="top")
    ax.text(0, 0.64, title, fontsize=10.5, color=style.INK_SECONDARY, va="top")
    ax.text(0, 0.48, _subtitle(summary), fontsize=8.5, color=style.INK_MUTED, va="top")
    if not graded:
        return

    for i, check in enumerate(graded):
        x = i / len(graded)
        ax.text(
            x,
            0.26,
            check.glyph,
            fontsize=11,
            fontweight="bold",
            color=check.colour,
            va="top",
        )
        ax.text(x + 0.013, 0.26, check.name, fontsize=8.5, color=style.INK, va="top")
        ax.text(
            x + 0.013,
            0.13,
            textwrap.fill(check.detail, 40),
            fontsize=7.5,
            color=style.INK_MUTED,
            va="top",
            linespacing=1.4,
        )


def _footer(figure: Figure, summary: reader.VelocityModelSummary) -> None:
    """Note how the figure was produced, along the bottom of the sheet.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure.
    summary : reader.VelocityModelSummary
        The model summary.
    """
    figure.text(
        0.006,
        0.006,
        f"Read in {summary.read_seconds:.0f} s  ·  statistics from "
        f"{summary.profile.n_columns:,} full-depth columns  ·  water "
        f"({summary.water_fraction * 100:.0f}% of the surface) excluded from "
        f"colour scales and rock statistics",
        fontsize=7.5,
        color=style.INK_MUTED,
    )


def plot_qa(summary: reader.VelocityModelSummary) -> Figure:
    """Build the quality-assurance screening sheet.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    matplotlib.figure.Figure
        The sheet.
    """
    graded = checks.run_checks(summary)
    layers = summary.layers
    sediment = reader.sediment_depth_km(summary)
    maps = _map_row_height(summary)

    figure, grid = _sheet(len(layers), [1.55, maps, maps, 2.9, 3.0])
    _draw_header(
        figure.add_subplot(grid[0, :]),
        summary,
        "Quality-assurance screen — every panel is here because a defect would show in it",
        graded,
    )

    # Shear-wave velocity: the field ground motion actually depends on. Scaled
    # per panel, because one scale spanning 0.5 to 4.9 km/s would flatten every
    # layer into a single tone.
    for i, layer in enumerate(layers):
        ax = _map_axes(figure, grid[1, i], summary)
        mesh = _draw_field_map(
            ax,
            summary,
            layer.fields["vs"],
            layer.water,
            style.FIELD_CMAP,
            Normalize(*_robust_limits(layer.fields["vs"], layer.water)),
        )
        if i == 0:
            _draw_basin_outline(ax, summary)
            ax.set_title("Vs at 0.0 km (basins outlined)", loc="left")
        else:
            ax.set_title(f"Vs at {layer.depth_km:.1f} km", loc="left")
        _slim_colourbar(figure, mesh, ax, style.FIELD_LABELS["vs"])

    # Vp/Vs on a fixed scale anchored to sqrt(3): ordinary rock is the neutral
    # midpoint, so anything coloured is a departure -- and because the scale
    # never moves, that stays true from one file to the next.
    for i, layer in enumerate(layers):
        ax = _map_axes(figure, grid[2, i], summary)
        norm, label = _vpvs_norm(layer.depth_km, sediment)
        mesh = _draw_field_map(
            ax, summary, layer.vpvs, layer.water, style.RATIO_CMAP, norm
        )
        ax.set_title(f"Vp/Vs at {layer.depth_km:.1f} km", loc="left")
        _slim_colourbar(figure, mesh, ax, label)

    # Two orthogonal cuts, sized in proportion to their true length.
    lengths = [section.distance_km[-1] for section in summary.sections]
    cuts = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=grid[3, :], width_ratios=lengths, wspace=0.10
    )
    for i, section in enumerate(summary.sections):
        ax = figure.add_subplot(cuts[0, i])
        mesh = _draw_section(
            ax,
            figure,
            section,
            section.fields["vs"],
            style.FIELD_CMAP,
            Normalize(*_robust_limits(section.fields["vs"], section.water)),
            f"{section.label} — Vs",
        )
        bar = figure.colorbar(mesh, ax=ax, pad=0.012, fraction=0.028, aspect=13)
        bar.set_label(style.FIELD_LABELS["vs"], fontsize=7.5, color=style.INK_SECONDARY)
        bar.ax.tick_params(labelsize=7, length=2)
        bar.outline.set_visible(False)

    # The diagnostics no map can show.
    diagnostics = GridSpecFromSubplotSpec(1, 4, subplot_spec=grid[4, :], wspace=0.26)

    ax = figure.add_subplot(diagnostics[0, 0])
    _draw_depth_density(ax, summary, "vs", style.FIELD_LABELS["vs"])
    ax.set_title("Vs against depth (whole model)", loc="left")

    ax = figure.add_subplot(diagnostics[0, 1])
    _draw_depth_density(
        ax,
        summary,
        "vpvs",
        style.FIELD_LABELS["vpvs"],
        style.POISSON_SOLID_VPVS,
        _rock_limits(summary, "vpvs"),
    )
    ax.set_title("Vp/Vs against depth (scaled to rock)", loc="left")

    _draw_brocher(figure.add_subplot(diagnostics[0, 2]), summary)
    _draw_fractions(figure.add_subplot(diagnostics[0, 3]), summary)

    _footer(figure, summary)
    return figure


def plot_explorer(summary: reader.VelocityModelSummary) -> Figure:
    """Build the exploration contact sheet: every field at every sampled depth.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    matplotlib.figure.Figure
        The sheet.
    """
    layers = summary.layers
    sediment = reader.sediment_depth_km(summary)
    maps = _map_row_height(summary)

    figure, grid = _sheet(len(layers), [1.20, maps, maps, maps, maps, 3.0])
    _draw_header(
        figure.add_subplot(grid[0, :]),
        summary,
        "Contact sheet — every field at every sampled depth, over the distributions behind them",
        [],
    )

    for row, field in enumerate(("vp", "vs", "rho", "vpvs"), start=1):
        for i, layer in enumerate(layers):
            ax = _map_axes(figure, grid[row, i], summary)
            if field == "vpvs":
                values = layer.vpvs
                cmap = style.RATIO_CMAP
                norm, label = _vpvs_norm(layer.depth_km, sediment)
            else:
                values = layer.fields[field]
                cmap = style.FIELD_CMAP
                norm = Normalize(*_robust_limits(values, layer.water))
                label = style.FIELD_LABELS[field]
            mesh = _draw_field_map(ax, summary, values, layer.water, cmap, norm)
            if i == 0:
                _draw_basin_outline(ax, summary)
            name = style.FIELD_LABELS[field].split(" (")[0]
            ax.set_title(f"{name} at {layer.depth_km:.1f} km", loc="left")
            _slim_colourbar(figure, mesh, ax, label)

    # Absolute values live down here. Each map above is scaled to its own depth,
    # which is what makes its structure legible -- and what hides the trend.
    bottom = GridSpecFromSubplotSpec(1, 5, subplot_spec=grid[5, :], wspace=0.30)
    for i, field in enumerate(("vp", "vs", "rho", "vpvs")):
        ax = figure.add_subplot(bottom[0, i])
        _draw_depth_density(
            ax,
            summary,
            field,
            style.FIELD_LABELS[field],
            style.POISSON_SOLID_VPVS if field == "vpvs" else None,
            _rock_limits(summary, "vpvs") if field == "vpvs" else None,
        )
        name = style.FIELD_LABELS[field].split(" (")[0]
        ax.set_title(f"{name} against depth", loc="left")
    _draw_fractions(figure.add_subplot(bottom[0, 4]), summary)

    _footer(figure, summary)
    return figure


def _block_diagram(
    figure: Figure, ax: plt.Axes, summary: reader.VelocityModelSummary
) -> None:
    """Draw a cut-away block of the model: a solid, with a bite taken out of it.

    A quadrant is removed from the corner *nearest the camera*, which is the only
    corner from which both exposed faces can actually be seen. Four vertical
    faces are then drawn: the two the cut exposes, and the two outer edges of the
    block. Without the outer pair there is nothing for the cut to be cut *out
    of*, and the whole thing reads as a wedge sticking out rather than a bite
    taken away.

    The result is the classic geological block diagram: the map view and the
    depth structure in one object, with neither hiding the other.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure, for the colour bar.
    ax : matplotlib.pyplot.Axes
        A 3D axes.
    summary : reader.VelocityModelSummary
        The model summary.
    """
    surface = summary.layers[0]
    vs = np.ma.masked_where(surface.water, surface.fields["vs"])
    ny, nx = vs.shape

    # Camera over (+x, -y), so the near corner -- and the notch -- is (x max, y min).
    half_y, half_x = ny // 2, nx // 2
    step = summary.meta.shape[2] / nx * (summary.meta.spacing_km or 1.0)
    x = np.arange(nx) * step
    y = np.arange(ny) * step
    depth = summary.profile.depth_km

    palette = _palette(style.FIELD_CMAP)

    grid_x, grid_y = np.meshgrid(x, y)
    colours = palette(Normalize(*_robust_limits(vs.data, surface.water))(vs))
    colours[:half_y, half_x:, 3] = 0.0  # the notch
    ax.plot_surface(
        grid_x,
        grid_y,
        np.zeros_like(grid_x),
        facecolors=colours,
        shade=False,
        rstride=1,
        cstride=1,
        antialiased=False,
    )

    faces = summary.sections + summary.edges
    walls = Normalize(
        *_robust_limits(
            np.concatenate([f.fields["vs"][~f.water].ravel() for f in faces])
        )
    )

    def draw_wall(
        section: reader.Section, columns: slice, along: str, position: float
    ) -> None:
        """Draw one vertical face of the block.

        Parameters
        ----------
        section : reader.Section
            The section the face is cut from.
        columns : slice
            The part of it this face spans.
        along : str
            Which grid axis the face runs along, "x" or "y".
        position : float
            Where the face sits on the other axis, in kilometres.
        """
        values = np.ma.masked_where(section.water, section.fields["vs"])
        span = (x if along == "x" else y)[columns]
        mesh_along, mesh_depth = np.meshgrid(span, depth)
        constant = np.full_like(mesh_along, position)
        ax.plot_surface(
            mesh_along if along == "x" else constant,
            constant if along == "x" else mesh_along,
            mesh_depth,
            facecolors=palette(walls(values[:, columns])),
            shade=False,
            rstride=2,
            cstride=1,
            antialiased=False,
        )

    # The two faces the cut exposes: they look back into the block.
    draw_wall(summary.sections[0], slice(half_x, None), "x", y[half_y])
    draw_wall(summary.sections[1], slice(None, half_y), "y", x[half_x])
    # And the two outer faces, which are what make it a solid.
    if len(summary.edges) == 2:
        draw_wall(summary.edges[0], slice(None, half_x), "x", y[0])
        draw_wall(summary.edges[1], slice(half_y, None), "y", x[-1])

    # The depth axis is exaggerated: 44 km beside 400 km would otherwise be a
    # sliver. This factor sets how much.
    exaggeration = BLOCK_DEPTH_EXAGGERATION * (depth[-1] - depth[0]) / x[-1]
    ax.set_zlim(depth[-1], depth[0])
    ax.set_box_aspect((x[-1], y[-1], exaggeration * x[-1]), zoom=BLOCK_ZOOM)
    ax.view_init(elev=26, azim=-58)
    ax.set_facecolor(style.SURFACE)
    # A 3D axes frame around a block diagram is scaffolding, not information --
    # the dimensions are already stated in the caption. Take it away.
    ax.set_axis_off()

    bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=walls, cmap=palette),
        ax=ax,
        fraction=0.016,
        pad=0.0,
        aspect=22,
        shrink=0.5,
    )
    bar.set_label(style.FIELD_LABELS["vs"], fontsize=8.5, color=style.INK_SECONDARY)
    bar.ax.tick_params(labelsize=7.5, length=2)
    bar.outline.set_visible(False)
    # Attaching a colour bar to a 3D axes re-anchors the parent, which shunts the
    # block off to one side. Put it back in the middle.
    ax.set_anchor("C")


def plot_poster(summary: reader.VelocityModelSummary) -> Figure:
    """Build the presentation poster: a cut-away block, with maps supporting it.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    matplotlib.figure.Figure
        The poster.
    """
    layers = summary.layers
    maps = _map_row_height(summary)

    figure, grid = _sheet(len(layers), [1.30, 7.2, maps])
    _draw_header(
        figure.add_subplot(grid[0, :]), summary, "Shear-wave velocity structure", []
    )

    ax = figure.add_subplot(grid[1, :], projection="3d")
    _block_diagram(figure, ax, summary)
    extent = summary.meta.extent_km
    depth = summary.profile.depth_km
    size = f"{extent[0]:.0f} × {extent[1]:.0f} km" if extent else "the model domain"
    ax.set_title(
        f"The near corner cut away, to the two sections beneath it\n"
        f"{size}, {depth[0]:.0f}–{depth[-1]:.0f} km deep, "
        f"depth exaggerated ×{BLOCK_DEPTH_EXAGGERATION:.0f}",
        loc="left",
        fontsize=11,
        color=style.INK_SECONDARY,
        pad=-30,
    )

    for i, layer in enumerate(layers):
        ax = _map_axes(figure, grid[2, i], summary)
        mesh = _draw_field_map(
            ax,
            summary,
            layer.fields["vs"],
            layer.water,
            style.FIELD_CMAP,
            Normalize(*_robust_limits(layer.fields["vs"], layer.water)),
        )
        ax.set_title(f"{layer.depth_km:.1f} km", loc="left")
        _slim_colourbar(figure, mesh, ax, style.FIELD_LABELS["vs"])

    _footer(figure, summary)
    return figure


#: The sheets, by name.
LAYOUTS = {"qa": plot_qa, "poster": plot_poster, "explorer": plot_explorer}
