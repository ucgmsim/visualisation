"""Summarise a 3D velocity model as a single figure.

The velocity models are large -- a few billion cells per field -- but the reads
are chosen so that a model is summarised in tens of seconds, which is what makes
it practical to run this over hundreds of them.

Examples
--------
Screen a model for defects::

    plot-velocity-model velocity_model.h5 out/

Produce every sheet, reading the file only once::

    plot-velocity-model velocity_model.h5 out/ --layout all
"""

from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

import typer

from qcore import cli
from visualisation.velocity_model import checks, layouts, reader, style

app = typer.Typer()


class Layout(StrEnum):
    """The summary sheets that can be produced."""

    QA = auto()
    POSTER = auto()
    COVERAGE = auto()
    ALL = auto()


@cli.from_docstring(app)
def plot_velocity_model(
    velocity_model_file: Annotated[
        Path, typer.Argument(exists=True, readable=True, dir_okay=False)
    ],
    output_directory: Annotated[Path, typer.Argument(file_okay=False, writable=True)],
    layout: Annotated[Layout, typer.Option(case_sensitive=False)] = Layout.QA,
    dpi: Annotated[int, typer.Option(min=40)] = 110,
    depth_levels: Annotated[int, typer.Option(min=2, max=8)] = 5,
    target_samples: Annotated[int, typer.Option(min=100)] = reader.TARGET_SAMPLES,
) -> None:
    """Summarise a 3D velocity model as a single figure.

    Parameters
    ----------
    velocity_model_file : Path
        The velocity model to summarise.
    output_directory : Path
        Directory to write the figures into.
    layout : Layout
        Which sheet to produce. "all" produces every sheet from one read.
    dpi : int
        Resolution of the output figures.
    depth_levels : int
        How many depths to slice, spread through the model's depth range.
    target_samples : int
        Longest map axis, in samples. Higher is sharper and slower.
    """
    style.apply_style()

    wanted = list(layouts.LAYOUTS) if layout is Layout.ALL else [layout.value]
    summary = reader.read_summary(
        velocity_model_file,
        depth_fractions=tuple(_depth_spread(depth_levels)),
        target_samples=target_samples,
        block_faces="poster" in wanted,
    )

    # The graded checks are no longer drawn on the figure, so report them here --
    # this is the screening verdict a batch run wants to grep.
    typer.echo(f"{summary.meta.name}  (read in {summary.read_seconds:.0f} s)")
    for check in checks.run_checks(summary):
        typer.echo(f"  [{check.glyph}] {check.status:<8} {check.name}: {check.detail}")

    output_directory.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        figure = layouts.LAYOUTS[name](summary)
        destination = output_directory / f"{summary.meta.name}_{name}.png"
        figure.savefig(destination, dpi=dpi)
        typer.echo(f"wrote {destination}")


def _depth_spread(count: int) -> list[float]:
    """Fractions of the depth range to slice at, weighted towards the surface.

    The interesting structure -- basins, the weathered layer, the steepest
    gradients -- is all shallow, so an evenly spaced set of depths would spend
    most of its panels on a slowly varying mantle.

    Parameters
    ----------
    count : int
        How many depths are wanted.

    Returns
    -------
    list of float
        Fractions between 0 and 1, ascending, always including both ends.
    """
    if count == len(reader.DEFAULT_DEPTH_FRACTIONS):
        return list(reader.DEFAULT_DEPTH_FRACTIONS)
    return [(i / (count - 1)) ** 2 for i in range(count)]


if __name__ == "__main__":
    app()
