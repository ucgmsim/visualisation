from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from visualisation.velocity_model import checks, layouts, reader, style

# Deliberately unlike the real models: a different grid, a different depth range,
# a different chunking. Nothing in the reader may assume any of them.
NZ, NY, NX = 30, 48, 64
DEPTH_KM = np.arange(NZ) * 0.2

# Scales chosen with headroom: nothing in this model may reach level 254, or it
# would spill onto the sentinel and the model would not be the clean one it is
# meant to be.
PACKING = {
    "vp": (0.025, 1.8),
    "vs": (0.018, 0.5),
    "rho": (0.009, 1.81),
}


def _pack(values: np.ndarray, name: str) -> np.ndarray:
    scale, offset = PACKING[name]
    return np.clip(np.rint((values - offset) / scale), 0, 254).astype(np.uint8)


@pytest.fixture(scope="module")
def velocity_model_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small, clean, physically self-consistent synthetic velocity model."""
    path = tmp_path_factory.mktemp("vm") / "synthetic.h5"

    depth = DEPTH_KM[:, None, None]
    vp = 1.9 + 0.9 * depth + 0.05 * np.random.default_rng(0).random((NZ, NY, NX))
    vs = vp / 1.73  # a Poisson solid, near enough
    rho = reader.brocher_density(vp)  # density as a real model derives it

    raw = {
        name: _pack(values, name)
        for name, values in (("vp", vp), ("vs", vs), ("rho", rho))
    }

    # A patch of "sea": every field sitting on its own floor, at the surface only.
    for name in raw:
        raw[name][0, :12, :16] = 0

    with h5py.File(path, "w") as handle:
        for name, values in raw.items():
            dataset = handle.create_dataset(
                name, data=values, chunks=(16, 16, 16), compression="gzip"
            )
            scale, offset = PACKING[name]
            dataset.attrs["scale_factor"] = np.array([scale], dtype=np.float32)
            dataset.attrs["add_offset"] = np.array([offset], dtype=np.float32)
            dataset.attrs["_FillValue"] = np.array([255], dtype=np.uint8)

        basin = np.full((NZ, NY, NX), 255, dtype=np.uint8)
        basin[:3, 20:30, 20:40] = 7  # a shallow basin, as basins ought to be
        handle.create_dataset("inbasin", data=basin, chunks=(16, 16, 16))

        handle.create_dataset("depth", data=DEPTH_KM)
        lon, lat = np.meshgrid(
            np.linspace(172.0, 173.0, NY), np.linspace(-43.8, -43.2, NX)
        )
        handle.create_dataset("lat", data=lat)
        handle.create_dataset("lon", data=lon)

        handle.attrs["h_lat_lon"] = "0.4"
        handle.attrs["min_vs"] = "0.5"
        handle.attrs["model_version"] = "2.09"
        handle.attrs["origin_lat"] = "-43.5"
        handle.attrs["origin_lon"] = "172.5"
        handle.attrs["origin_rot"] = "0.0"
        handle.attrs["extent_x"] = "25.6"
        handle.attrs["extent_y"] = "19.2"
        handle.attrs["topo_type"] = "SQUASHED"

    return path


@pytest.fixture(scope="module")
def summary(velocity_model_file: Path) -> reader.VelocityModelSummary:
    """The synthetic model, read the way the figures read it."""
    return reader.read_summary(velocity_model_file, block_faces=True)


def test_geometry_is_read_from_the_file(summary: reader.VelocityModelSummary):
    assert summary.meta.shape == (NZ, NY, NX)
    assert summary.meta.min_vs == 0.5
    assert summary.lat.shape == summary.lon.shape == (NY, NX)
    assert summary.profile.depth_km[-1] == pytest.approx(DEPTH_KM[-1])


def test_fields_are_unpacked_to_physical_units(summary: reader.VelocityModelSummary):
    # Packed as uint8; read back as km/s. The deepest layer must be the fastest.
    surface, base = summary.layers[0], summary.layers[-1]
    assert np.nanmedian(base.fields["vs"]) > np.nanmedian(surface.fields["vs"])
    assert 1.8 <= np.nanmin(base.fields["vp"]) <= 30.0
    assert np.nanmedian(base.vpvs) == pytest.approx(1.73, abs=0.05)


def test_water_is_found_at_the_surface_and_nowhere_below(
    summary: reader.VelocityModelSummary,
):
    # The sea patch is 12x16 of a 48x64 surface.
    assert summary.layers[0].water.sum() == 12 * 16
    assert not any(layer.water.any() for layer in summary.layers[1:])
    assert summary.water_fraction == pytest.approx(12 * 16 / (NY * NX))


def test_water_is_kept_out_of_the_rock_statistics(summary: reader.VelocityModelSummary):
    # Water would otherwise pin the surface median to the Vs floor and put a
    # spike near 3.6 into the Vp/Vs distribution.
    assert summary.profile.quantiles["vs"][2][0] > 0.5
    centres, counts = summary.profile.density["vpvs"]
    assert counts[0][centres > 3.0].sum() == 0


def test_a_clean_model_passes_every_check(summary: reader.VelocityModelSummary):
    graded = checks.run_checks(summary)
    assert {check.status for check in graded} == {"good"}
    assert all(check.glyph == "✓" for check in graded)


def test_saturation_is_caught(velocity_model_file: Path, tmp_path: Path):
    # Push a handful of cells onto the sentinel, as the real model does at its
    # base. A CF-aware reader turns these into NaN, so they must not pass quietly.
    spoiled = tmp_path / "saturated.h5"
    spoiled.write_bytes(velocity_model_file.read_bytes())
    with h5py.File(spoiled, "r+") as handle:
        handle["vs"][NZ - 1, :2, :3] = 255
        handle["vs"][NZ - 1, 2:4, :3] = 254

    graded = checks.run_checks(reader.read_summary(spoiled))
    headroom = next(c for c in graded if c.name == "Packed scale has headroom")
    assert headroom.status == "serious"
    assert "6" in headroom.detail


def test_deep_basin_labels_are_flagged(velocity_model_file: Path, tmp_path: Path):
    # Basin labels running to the base of the model are not basins.
    spoiled = tmp_path / "deep_basin.h5"
    spoiled.write_bytes(velocity_model_file.read_bytes())
    with h5py.File(spoiled, "r+") as handle:
        handle["inbasin"][:, 20:30, 20:40] = 7

    graded = checks.run_checks(reader.read_summary(spoiled))
    basins = next(c for c in graded if c.name.startswith("Basin labels"))
    assert basins.status == "warning"


@pytest.mark.parametrize("name", sorted(layouts.LAYOUTS))
def test_every_layout_renders(
    summary: reader.VelocityModelSummary, name: str, tmp_path: Path
):
    style.apply_style()
    figure = layouts.LAYOUTS[name](summary)
    destination = tmp_path / f"{name}.png"
    figure.savefig(destination, dpi=50)
    assert destination.stat().st_size > 0
