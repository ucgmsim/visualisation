"""Automated sanity checks on a velocity model.

The point of these is to survive being run over hundreds of models. Each check
reduces the model to a single number with a physical meaning, and grades it, so
that a bad model announces itself instead of waiting to be noticed.

Every check is paired with the value it was decided on, and the grade is always
rendered with a glyph and a label as well as a colour, so a reader who cannot
distinguish the status colours loses nothing.
"""

import dataclasses

import numpy as np

from visualisation.velocity_model import reader, style

#: Glyphs stand in for colour, so that status is never carried by colour alone.
GLYPHS = {"good": "✓", "warning": "!", "serious": "!", "critical": "✕"}


@dataclasses.dataclass(frozen=True)
class Check:
    """The outcome of one sanity check.

    Attributes
    ----------
    name : str
        What was checked.
    status : str
        A key of the reserved status palette: good, warning, serious or critical.
    detail : str
        The measured value the grade was based on.
    """

    name: str
    status: str
    detail: str

    @property
    def colour(self) -> str:
        """Status colour for the check.

        Returns
        -------
        str
            A hex colour from the reserved status palette.
        """
        return style.STATUS[self.status]

    @property
    def glyph(self) -> str:
        """Glyph for the check, so status survives without colour.

        Returns
        -------
        str
            A single character.
        """
        return GLYPHS[self.status]


def _grade(value: float, good: float, warn: float) -> str:
    """Grade a value where smaller is better.

    Parameters
    ----------
    value : float
        The measured value.
    good : float
        At or below this, the check passes.
    warn : float
        At or below this, the check warns; above it, the check is serious.

    Returns
    -------
    str
        A status key.
    """
    if value <= good:
        return "good"
    if value <= warn:
        return "warning"
    return "serious"


def _packing_headroom(summary: reader.VelocityModelSummary) -> Check:
    """Check the packed scale has room at the top for the model's fastest cells.

    The fields are packed into a byte, with 255 reserved as the no-data
    sentinel, leaving 0-254 for data. If a field's values reach level 254, the
    scale has run out of headroom -- and any cell above it lands on 255 and is
    silently read back as no-data by any CF-aware reader. The fastest cells in
    the model are exactly the ones this destroys.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    Check
        The graded outcome.
    """
    lost = sum(summary.fill_counts.values())
    spilling = ", ".join(name for name, hit in summary.saturated.items() if hit)

    if lost == 0 and not spilling:
        return Check("Packed scale has headroom", "good", "no cells near the sentinel")
    if lost == 0:
        return Check(
            "Packed scale has headroom",
            "warning",
            f"{spilling} sit on the top level; none lost yet",
        )
    return Check(
        "Packed scale has headroom",
        "serious",
        f"{spilling} saturated: {lost:,} cells spilled onto the "
        f"sentinel and read back as no-data",
    )


def _vpvs_plausible(summary: reader.VelocityModelSummary) -> Check:
    """Check that Vp/Vs in rock stays inside the physically plausible band.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    Check
        The graded outcome.
    """
    fraction = reader.vpvs_outlier_fraction(summary)
    below = reader.sediment_depth_km(summary)
    low, high = style.VPVS_PLAUSIBLE
    return Check(
        f"Vp/Vs in {low}-{high} below {below:.1f} km",
        _grade(fraction, 0.005, 0.05),
        f"{fraction * 100:.3f}% of rock cells outside",
    )


def _velocity_increases(summary: reader.VelocityModelSummary) -> Check:
    """Check the median Vs profile does not invert sharply with depth.

    Small inversions are real geology. A large one is usually a defect.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    Check
        The graded outcome.
    """
    profile = summary.profile
    rock = profile.depth_km >= reader.sediment_depth_km(summary)
    median = profile.quantiles["vs"][2][rock]
    median = median[np.isfinite(median)]
    if median.size < 2:
        return Check("Vs increases with depth", "warning", "not enough data")

    drop = float(-np.diff(median).min())
    drop = max(drop, 0.0)
    return Check(
        "Vs increases with depth",
        _grade(drop, 0.05, 0.30),
        f"largest inversion {drop:.3f} km/s"
        if drop > 0
        else "monotonic through the rock column",
    )


def _density_consistent(summary: reader.VelocityModelSummary) -> Check:
    """Check density tracks Vp along roughly the Brocher (2005) relation.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    Check
        The graded outcome.
    """
    sample = summary.profile.sample
    residual = sample["rho"] - reader.brocher_density(sample["vp"])
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return Check("Density tracks Vp (Brocher)", "warning", "not enough data")

    spread = float(np.percentile(np.abs(residual), 95))
    return Check(
        "Density tracks Vp (Brocher)",
        _grade(spread, 0.15, 0.30),
        f"95% within {spread:.3f} g/cm³, median {np.median(residual):+.3f}",
    )


def _vs_floor(summary: reader.VelocityModelSummary) -> Check:
    """Check the model honours its own declared minimum Vs.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    Check
        The graded outcome.
    """
    declared = summary.meta.min_vs
    if declared is None:
        return Check("Minimum Vs honoured", "warning", "no minimum declared")

    observed = float(
        np.nanmin([np.nanmin(layer.fields["vs"]) for layer in summary.layers])
    )
    breach = declared - observed
    status = "good" if breach <= 1e-3 else "critical"
    detail = (
        f"declared {declared:g}, observed {observed:.3f} km/s"
        if status == "good"
        else f"{breach:.3f} km/s below the declared {declared:g}"
    )
    return Check("Minimum Vs honoured", status, detail)


def _basin_extent(summary: reader.VelocityModelSummary) -> Check:
    """Check basin labels stay in the upper crust, where basins actually are.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    Check
        The graded outcome.
    """
    fraction = summary.profile.basin_fraction
    if fraction is None:
        return Check("Basin labels confined to basins", "warning", "no basin field")

    labelled = np.flatnonzero(fraction > 0)
    if labelled.size == 0:
        return Check("Basin labels confined to basins", "good", "no cells labelled")

    depth = summary.profile.depth_km
    deepest = float(depth[labelled[-1]])
    span = float(depth[-1] - depth[0])
    status = "good" if deepest <= depth[0] + 0.25 * span else "warning"
    return Check(
        "Basin labels confined to basins",
        status,
        f"labelled to {deepest:.1f} km "
        f"({fraction[labelled[-1]] * 100:.2f}% of cells at the base)",
    )


def run_checks(summary: reader.VelocityModelSummary) -> list[Check]:
    """Run every sanity check against a model.

    Parameters
    ----------
    summary : reader.VelocityModelSummary
        The model summary.

    Returns
    -------
    list of Check
        The graded outcomes, in reading order.
    """
    return [
        _packing_headroom(summary),
        _vs_floor(summary),
        _vpvs_plausible(summary),
        _velocity_increases(summary),
        _density_consistent(summary),
        _basin_extent(summary),
    ]
