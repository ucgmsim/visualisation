"""Shared visual language for velocity model summary figures.

Colour is assigned by the job it does, not by taste:

- **Magnitude** (Vp, Vs, density) uses one perceptually uniform sequential ramp
  so that "dark is slow, bright is fast" only has to be learnt once.
- **Polarity** (the Vp/Vs ratio) uses a diverging blue-red ramp with a neutral
  midpoint pinned to the Poisson-solid value of sqrt(3). Rock sits within a
  hair of that value, so anomalies glow without any threshold being chosen.
- **Identity** (basin labels) uses a fixed categorical order, never cycled. Past
  the eighth basin the tail folds into a single "other" colour.
- **State** (the QA badges) uses a reserved status palette that is never used for
  anything else, and is always paired with a text label so that meaning is never
  carried by colour alone.
"""

import matplotlib as mpl
import numpy as np

# Chart chrome and ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# Reserved status palette. Never used for a non-status series.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Fixed categorical order. The ordering is the colourblind-safety mechanism.
CATEGORICAL = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
)
OTHER = "#898781"

# Cells clamped to the model's velocity floor (open water, in a coastal domain).
# Deliberately outside every data ramp so it can never be mistaken for a value.
WATER = "#c8d6e2"

FIELD_CMAP = "viridis"  # magnitude: perceptually uniform, monotone lightness
RATIO_CMAP = "RdBu_r"  # polarity: warm/cool poles, neutral midpoint
DENSITY_CMAP = "Blues"  # counts: single hue, light to dark

#: A Poisson solid has Vp/Vs = sqrt(3). Rock sits close to it; water does not.
POISSON_SOLID_VPVS = float(np.sqrt(3.0))

#: Physically plausible Vp/Vs band for rock. Outside this, something is wrong.
VPVS_PLAUSIBLE = (1.4, 2.5)

FIELD_LABELS = {
    "vp": "Vp (km/s)",
    "vs": "Vs (km/s)",
    "rho": "Density (g/cm$^3$)",
    "vpvs": "Vp/Vs",
}


def apply_style() -> None:
    """Apply the figure-wide matplotlib defaults: thin marks, recessive chrome.

    Returns
    -------
    None
        The global matplotlib rcParams are updated in place.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": False,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",  # never dashed: dashing reads as "threshold"
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 9,
            "text.color": INK,
            "lines.linewidth": 1.6,
        }
    )


def basin_colours(n: int) -> list[str]:
    """Return colours for ``n`` basin identities, folding the tail into "other".

    Parameters
    ----------
    n : int
        Number of distinct basin identities to colour.

    Returns
    -------
    list of str
        Hex colours. Entries past the eighth are all the neutral "other" colour,
        because a ninth generated hue would not survive a colourblindness check.
    """
    return [CATEGORICAL[i] if i < len(CATEGORICAL) else OTHER for i in range(max(n, 0))]
