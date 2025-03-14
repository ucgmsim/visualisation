"""Utility functions common to many plotting scripts."""

from typing import Optional

import numpy as np


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
