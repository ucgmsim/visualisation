"""Read a 3D velocity model into a compact, plot-ready summary.

The velocity models are NetCDF-4/HDF5 files holding Vp, Vs, density and a basin
label on a rotated, regular 3D grid. They are big -- a few billion cells per
field -- so this module never loads a whole field. It exists to answer one
question: what is the cheapest set of reads that still supports an honest
overview figure?

Three properties of the on-disk layout drive the answer:

1. The fields are stored **packed into ``uint8``** with a CF-convention
   ``scale_factor``/``add_offset``. Read naively they are meaningless integers,
   so everything is unpacked on the way out.
2. Chunks span the **full depth range** of a column. A chunk-aligned block is
   therefore an extremely cheap way to obtain complete depth profiles: a
   256x256 block costs about 50 ms and yields every depth in that column.
   Whole-model statistics are built from a spread of such blocks rather than by
   reading the model.
3. Because the data is ``uint8``, a 256-bin histogram per depth is **exact**,
   not an approximation, and costs a single ``bincount``. Quantiles fall out of
   it for free, and the Vp/Vs ratio is obtained through a 256x256 lookup table
   instead of dividing billions of floats.

Nothing about the grid is assumed: extent, rotation, depth range, resolution and
even whether the domain contains any water are all derived per file.
"""

import dataclasses
import time
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np

from visualisation.velocity_model import style

#: Sentinel for "no data" in the packed uint8 fields, and for "not in a basin"
#: in the basin label field.
FILL = 255

#: Number of distinct packed values that carry data (0-254; 255 is the fill).
N_LEVELS = FILL

#: Fractions of the model's depth range sampled for the map panels. Chosen to
#: land on the surface, the basins, the upper and mid crust, and the model base.
DEFAULT_DEPTH_FRACTIONS = (0.0, 0.03, 0.15, 0.45, 1.0)

#: Quantiles reported for every depth profile.
PROFILE_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

#: Bin edges for the Vp/Vs ratio histogram. Fine enough that the quantisation of
#: the underlying uint8 packing, not the binning, is what limits the resolution.
VPVS_RANGE = (1.2, 4.0)
VPVS_BINS = 400

#: Longest map axis, in samples, after decimation. A figure panel is a few
#: hundred pixels wide, so there is nothing to gain from carrying more.
TARGET_SAMPLES = 700

FIELDS = ("vp", "vs", "rho")


@dataclasses.dataclass(frozen=True)
class ModelMetadata:
    """Provenance and geometry of a velocity model, as recorded in the file.

    Attributes
    ----------
    name : str
        Stem of the file the model was read from.
    shape : tuple of int
        Grid dimensions as ``(nz, ny, nx)``.
    depth_km : numpy.ndarray
        Depth of every grid level, in kilometres.
    spacing_km : float or None
        Horizontal grid spacing, if the file records it.
    extent_km : tuple of float or None
        Domain size as ``(x, y)`` in kilometres, if recorded.
    origin : tuple of float or None
        Domain origin as ``(latitude, longitude)``, if recorded.
    rotation_deg : float or None
        Grid rotation in degrees, if recorded.
    model_version : str or None
        Velocity model version string, if recorded.
    topo_type : str or None
        Topography handling, if recorded.
    min_vs : float or None
        Enforced minimum shear-wave velocity, if recorded.
    water_floors : tuple of float
        The Vs and Vp values at or below which a cell is taken to be water, as
        resolved by :func:`water_thresholds`. Retained so that a figure can state
        the rule it drew, which is derived per file rather than fixed.
    """

    name: str
    shape: tuple[int, int, int]
    depth_km: np.ndarray
    spacing_km: float | None
    extent_km: tuple[float, float] | None
    origin: tuple[float, float] | None
    rotation_deg: float | None
    model_version: str | None
    topo_type: str | None
    min_vs: float | None
    water_floors: tuple[float, float]


@dataclasses.dataclass
class Layer:
    """A horizontal slice through the model at one depth.

    Attributes
    ----------
    index : int
        Grid level the slice was taken from.
    depth_km : float
        Depth of the slice, in kilometres.
    fields : dict of str to numpy.ndarray
        Decimated, unpacked fields keyed by name, each shaped ``(ny, nx)``.
    water : numpy.ndarray
        Boolean mask of cells clamped to the model's velocity floor.
    """

    index: int
    depth_km: float
    fields: dict[str, np.ndarray]
    water: np.ndarray

    @property
    def vpvs(self) -> np.ndarray:
        """Vp/Vs ratio of the layer.

        Returns
        -------
        numpy.ndarray
            The ratio, shaped ``(ny, nx)``.
        """
        return self.fields["vp"] / self.fields["vs"]


@dataclasses.dataclass
class Section:
    """A vertical cross-section through the model.

    Attributes
    ----------
    label : str
        Human-readable description of where the section was cut.
    distance_km : numpy.ndarray
        Distance along the section, in kilometres.
    depth_km : numpy.ndarray
        Depth of every row, in kilometres.
    fields : dict of str to numpy.ndarray
        Decimated, unpacked fields keyed by name, each shaped
        ``(depth, distance)``.
    water : numpy.ndarray
        Cells clamped to the model's velocity floor.
    basin : numpy.ndarray or None
        Basin labels on the same grid, where the model carries them.
    """

    label: str
    distance_km: np.ndarray
    depth_km: np.ndarray
    fields: dict[str, np.ndarray]
    water: np.ndarray
    basin: np.ndarray | None

    @property
    def vpvs(self) -> np.ndarray:
        """Vp/Vs ratio of the section.

        Returns
        -------
        numpy.ndarray
            The ratio, shaped ``(depth, distance)``.
        """
        return self.fields["vp"] / self.fields["vs"]


@dataclasses.dataclass
class DepthProfile:
    """Whole-model statistics as a function of depth.

    Built from a spread of chunk-aligned sample blocks, each of which carries
    every depth. The histograms are exact over the sampled columns because the
    underlying data is ``uint8``.

    Attributes
    ----------
    depth_km : numpy.ndarray
        Depth of every grid level, in kilometres.
    quantiles : dict of str to numpy.ndarray
        Per-field quantiles, each shaped ``(len(PROFILE_QUANTILES), nz)``.
    density : dict of str to tuple
        Per-field ``(bin_centres, counts)``, where counts is ``(nz, nbins)``.
    basin_fraction : numpy.ndarray or None
        Fraction of sampled cells carrying a basin label, per depth.
    water_fraction : numpy.ndarray
        Fraction of sampled cells clamped to the velocity floor, per depth. In a
        coastal domain this traces the bathymetry.
    sample : dict of str to numpy.ndarray
        A co-located random subsample of the fields, for scatter diagnostics.
    n_columns : int
        Number of full-depth columns the statistics were built from.
    """

    depth_km: np.ndarray
    quantiles: dict[str, np.ndarray]
    density: dict[str, tuple[np.ndarray, np.ndarray]]
    basin_fraction: np.ndarray | None
    water_fraction: np.ndarray
    sample: dict[str, np.ndarray]
    n_columns: int


@dataclasses.dataclass
class VelocityModelSummary:
    """Everything the figures need, gathered in a single pass over the file.

    Attributes
    ----------
    meta : ModelMetadata
        Provenance and geometry.
    lat : numpy.ndarray
        Decimated latitudes, shaped ``(ny, nx)``.
    lon : numpy.ndarray
        Decimated longitudes, shaped ``(ny, nx)``.
    layers : list of Layer
        Horizontal slices, shallowest first.
    sections : list of Section
        Vertical cross-sections through the domain centre.
    edges : list of Section
        Vertical sections down the two domain edges nearest the block diagram's
        camera. Empty unless they were asked for. Without them a cut-away block
        has no outer faces, and the cut reads as a wedge sticking out rather
        than a bite taken out.
    profile : DepthProfile
        Whole-model statistics against depth.
    surface_basin : numpy.ndarray or None
        Basin labels at the surface, decimated.
    fill_counts : dict of str to int
        Number of cells per field sitting on the no-data sentinel, counted over
        the depth slices that were read in full.
    saturated : dict of str to bool
        Whether a field reaches the very top of its packed scale. If it does,
        and there are also cells on the sentinel, then the scale is too narrow
        and the model's fastest cells are being written out as no-data.
    water_fraction : float
        Fraction of the surface clamped to the velocity floor.
    read_seconds : float
        Wall-clock time spent reading the file.
    """

    meta: ModelMetadata
    lat: np.ndarray
    lon: np.ndarray
    layers: list[Layer]
    sections: list[Section]
    edges: list[Section]
    profile: DepthProfile
    surface_basin: np.ndarray | None
    fill_counts: dict[str, int]
    saturated: dict[str, bool]
    water_fraction: float
    read_seconds: float


def _attr(handle: h5py.File, key: str) -> str | None:
    """Read a root attribute as text, if it is present.

    Parameters
    ----------
    handle : h5py.File
        Open velocity model file.
    key : str
        Attribute name.

    Returns
    -------
    str or None
        The attribute value, or None when the file does not carry it.
    """
    value = handle.attrs.get(key)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _float_attr(handle: h5py.File, key: str) -> float | None:
    """Read a root attribute as a float, if it is present and parseable.

    Parameters
    ----------
    handle : h5py.File
        Open velocity model file.
    key : str
        Attribute name.

    Returns
    -------
    float or None
        The attribute value, or None when absent or not a number.
    """
    raw = _attr(handle, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _packing(dataset: h5py.Dataset) -> tuple[float, float]:
    """Return the ``(scale, offset)`` that unpacks a field to physical units.

    Parameters
    ----------
    dataset : h5py.Dataset
        A packed field.

    Returns
    -------
    tuple of float
        Multiplicative scale and additive offset. ``(1.0, 0.0)`` when the field
        is not packed.
    """
    scale = dataset.attrs.get("scale_factor")
    offset = dataset.attrs.get("add_offset")
    return (
        float(scale[0]) if scale is not None else 1.0,
        float(offset[0]) if offset is not None else 0.0,
    )


def _field_floors(handle: h5py.File) -> dict[str, float]:
    """The lowest value each packed field is able to represent.

    Parameters
    ----------
    handle : h5py.File
        Open velocity model file.

    Returns
    -------
    dict of str to float
        Field name to floor, in that field's own units.
    """
    return {name: _packing(handle[name])[1] for name in FIELDS}


def water_thresholds(
    floors: dict[str, float], min_vs: float | None
) -> tuple[float, float]:
    """The Vs and Vp values at or below which a cell counts as water.

    The two are rarely the same number, so both are needed to state the rule.
    Vs is held to the model's declared minimum wherever the file records one,
    because a model may enforce a floor above the one its packing could reach.
    Vp has no declared equivalent, so it is held to the packing floor.

    Parameters
    ----------
    floors : dict of str to float
        The lowest value each field can represent.
    min_vs : float or None
        The model's enforced minimum Vs, when recorded.

    Returns
    -------
    tuple of float
        The Vs and Vp thresholds, in km/s.
    """
    return (min_vs if min_vs is not None else floors["vs"]), floors["vp"]


def _unpack(dataset: h5py.Dataset, raw: np.ndarray) -> np.ndarray:
    """Convert packed integers to physical units, with no-data cells as NaN.

    Parameters
    ----------
    dataset : h5py.Dataset
        The field the values were read from, carrying the packing attributes.
    raw : numpy.ndarray
        Packed values as stored.

    Returns
    -------
    numpy.ndarray
        Physical values as float32, with fill cells replaced by NaN.
    """
    scale, offset = _packing(dataset)
    out = raw.astype(np.float32) * scale + offset
    if raw.dtype == np.uint8:
        out[raw == FILL] = np.nan
    return out


def _levels(dataset: h5py.Dataset) -> np.ndarray:
    """Physical value of every packed level of a field.

    Parameters
    ----------
    dataset : h5py.Dataset
        A packed field.

    Returns
    -------
    numpy.ndarray
        The value each of the 255 data levels decodes to.
    """
    scale, offset = _packing(dataset)
    return np.arange(N_LEVELS, dtype=np.float64) * scale + offset


def _decimation(shape: tuple[int, int], target: int) -> int:
    """Choose a stride that brings the longest map axis near ``target``.

    Parameters
    ----------
    shape : tuple of int
        Map dimensions as ``(ny, nx)``.
    target : int
        Desired number of samples along the longest axis.

    Returns
    -------
    int
        A stride of at least 1, shared by both axes so the aspect is preserved.
    """
    return max(1, int(np.ceil(max(shape) / target)))


def _depth_indices(nz: int, fractions: tuple[float, ...]) -> list[int]:
    """Grid levels to slice, given fractions of the model's depth range.

    Fractions rather than absolute depths, because every file covers a different
    domain and depth range.

    Parameters
    ----------
    nz : int
        Number of grid levels.
    fractions : tuple of float
        Positions within the depth range, from 0 (surface) to 1 (base).

    Returns
    -------
    list of int
        Ascending, de-duplicated grid levels.
    """
    raw = [int(round(f * (nz - 1))) for f in fractions]
    return sorted(set(min(max(i, 0), nz - 1) for i in raw))


def _block_origins(n: int, chunk: int, count: int) -> list[int]:
    """Chunk-aligned start indices spread evenly along an axis.

    Alignment matters: a block that straddles a chunk boundary forces every
    neighbouring chunk to be decompressed, which costs about four times as much
    for the same amount of data.

    Parameters
    ----------
    n : int
        Length of the axis.
    chunk : int
        Chunk length along the axis.
    count : int
        Desired number of blocks.

    Returns
    -------
    list of int
        Start indices, each a multiple of ``chunk``.
    """
    n_chunks = max(1, n // chunk)
    picks = np.unique(np.linspace(0, n_chunks - 1, count).round().astype(int))
    return [int(p) * chunk for p in picks]


def _per_depth_histogram(block: np.ndarray, n_bins: int) -> np.ndarray:
    """Count values at every depth of a block, in one pass.

    The trick is to offset each depth's values into its own band of a single
    flat ``bincount``, which is far faster than looping over depths.

    Parameters
    ----------
    block : numpy.ndarray
        Integer bin indices shaped ``(nz, ny, nx)``.
    n_bins : int
        Number of bins per depth.

    Returns
    -------
    numpy.ndarray
        Counts shaped ``(nz, n_bins)``.
    """
    nz = block.shape[0]
    flat = block.reshape(nz, -1).astype(np.int32)
    offsets = np.arange(nz, dtype=np.int32)[:, None] * n_bins
    counts = np.bincount((flat + offsets).ravel(), minlength=nz * n_bins)
    return counts[: nz * n_bins].reshape(nz, n_bins)


def _histogram_quantiles(
    counts: np.ndarray, centres: np.ndarray, quantiles: tuple[float, ...]
) -> np.ndarray:
    """Quantiles per depth, read straight off the per-depth histograms.

    Exact to the width of one packed level, because the histogram bins *are* the
    levels the data was stored at.

    Parameters
    ----------
    counts : numpy.ndarray
        Counts shaped ``(nz, n_bins)``.
    centres : numpy.ndarray
        Physical value of each bin.
    quantiles : tuple of float
        Quantiles to evaluate, between 0 and 1.

    Returns
    -------
    numpy.ndarray
        Values shaped ``(len(quantiles), nz)``, NaN where a depth had no data.
    """
    cumulative = np.cumsum(counts, axis=1)
    totals = cumulative[:, -1]
    out = np.full((len(quantiles), counts.shape[0]), np.nan)
    populated = totals > 0
    if not populated.any():
        return out
    fraction = cumulative[populated] / totals[populated, None]
    for i, q in enumerate(quantiles):
        out[i, populated] = centres[np.argmax(fraction >= q, axis=1)]
    return out


def _vpvs_lookup(
    vp_dataset: h5py.Dataset, vs_dataset: h5py.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """Build a packed-value lookup table from ``(vp, vs)`` to a ratio bin.

    Both fields are ``uint8``, so there are only 65536 possible pairs. Resolving
    the ratio through a table turns billions of floating point divisions into an
    integer gather.

    Parameters
    ----------
    vp_dataset : h5py.Dataset
        The packed Vp field.
    vs_dataset : h5py.Dataset
        The packed Vs field.

    Returns
    -------
    tuple of numpy.ndarray
        The ``(256, 256)`` table of ratio bin indices, and the bin centres. The
        final bin is a discard bin, holding no-data cells and open water.
    """
    vp_scale, vp_offset = _packing(vp_dataset)
    vs_scale, vs_offset = _packing(vs_dataset)
    vp_values = np.arange(256, dtype=np.float64) * vp_scale + vp_offset
    vs_values = np.arange(256, dtype=np.float64) * vs_scale + vs_offset
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = vp_values[:, None] / vs_values[None, :]

    edges = np.linspace(*VPVS_RANGE, VPVS_BINS + 1)
    table = np.clip(np.digitize(ratio, edges) - 1, 0, VPVS_BINS - 1)
    table[~np.isfinite(ratio)] = VPVS_BINS
    table[FILL, :] = VPVS_BINS  # no-data in Vp
    table[:, FILL] = VPVS_BINS  # no-data in Vs

    # Water sits on the floor of both fields at once. Left in, it would put a
    # spike near 3.6 into every shallow marine depth and swamp the rock signal
    # the statistic exists to measure. Rock never lands on both floors.
    table[0, 0] = VPVS_BINS

    centres = 0.5 * (edges[:-1] + edges[1:])
    return table.astype(np.int32), centres


def _water_mask(
    fields: dict[str, np.ndarray], floors: dict[str, float], min_vs: float | None
) -> np.ndarray:
    """Flag cells sitting on the velocity floor of every field: open water.

    In a coastal domain the sea is not modelled as a medium; its cells are simply
    clamped to the model's minimum. They occupy a large share of the surface and,
    left in, they flatten every colour scale and drag the surface Vp/Vs ratio up
    to about 3.6. They are identified by sitting at the floor of Vp *and* Vs
    simultaneously, which no real rock does.

    Parameters
    ----------
    fields : dict of str to numpy.ndarray
        Unpacked fields for one layer.
    floors : dict of str to float
        The lowest value each field can represent.
    min_vs : float or None
        The model's enforced minimum Vs, when recorded.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where the cell is water. Empty for an inland domain,
        and for every layer below the surface.
    """
    vs_floor, vp_floor = water_thresholds(floors, min_vs)
    tolerance = 1e-3
    return (fields["vs"] <= vs_floor + tolerance) & (
        fields["vp"] <= vp_floor + tolerance
    )


def _read_metadata(handle: h5py.File, path: Path) -> ModelMetadata:
    """Gather provenance and geometry from the file's root attributes.

    Parameters
    ----------
    handle : h5py.File
        Open velocity model file.
    path : pathlib.Path
        Path the model was read from.

    Returns
    -------
    ModelMetadata
        The model's provenance and geometry.
    """
    nz, ny, nx = handle["vs"].shape
    extent_x = _float_attr(handle, "extent_x")
    extent_y = _float_attr(handle, "extent_y")
    origin_lat = _float_attr(handle, "origin_lat")
    origin_lon = _float_attr(handle, "origin_lon")
    min_vs = _float_attr(handle, "min_vs")
    return ModelMetadata(
        name=path.stem,
        shape=(nz, ny, nx),
        depth_km=np.asarray(handle["depth"][:], dtype=float),
        spacing_km=_float_attr(handle, "h_lat_lon"),
        extent_km=(extent_x, extent_y)
        if extent_x is not None and extent_y is not None
        else None,
        origin=(origin_lat, origin_lon)
        if origin_lat is not None and origin_lon is not None
        else None,
        rotation_deg=_float_attr(handle, "origin_rot"),
        model_version=_attr(handle, "model_version"),
        topo_type=_attr(handle, "topo_type"),
        min_vs=min_vs,
        water_floors=water_thresholds(_field_floors(handle), min_vs),
    )


def _read_coordinates(
    handle: h5py.File, step: int, ny: int, nx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read decimated latitude and longitude, oriented to match the fields.

    The coordinate arrays are stored transposed relative to the fields, so the
    orientation is resolved from the shape rather than assumed.

    Parameters
    ----------
    handle : h5py.File
        Open velocity model file.
    step : int
        Decimation stride.
    ny : int
        Grid length along y.
    nx : int
        Grid length along x.

    Returns
    -------
    tuple of numpy.ndarray
        Latitude and longitude, each shaped like a decimated map.
    """
    lat = handle["lat"]
    lon = handle["lon"]
    if lat.shape == (nx, ny):
        return lat[::step, ::step].T, lon[::step, ::step].T
    return lat[::step, ::step], lon[::step, ::step]


def _read_layers(
    handle: h5py.File,
    indices: list[int],
    depth_km: np.ndarray,
    step: int,
    min_vs: float | None,
) -> tuple[list[Layer], np.ndarray | None]:
    """Read the horizontal slices, and the basin labels at the surface.

    Parameters
    ----------
    handle : h5py.File
        Open velocity model file.
    indices : list of int
        Grid levels to slice.
    depth_km : numpy.ndarray
        Depth of every grid level.
    step : int
        Decimation stride.
    min_vs : float or None
        The model's enforced minimum Vs, when recorded.

    Returns
    -------
    tuple
        The layers, and the decimated surface basin labels (None if the model
        carries no basin field).
    """
    floors = _field_floors(handle)
    layers = []
    for index in indices:
        fields = {
            name: _unpack(handle[name], handle[name][index, ::step, ::step])
            for name in FIELDS
        }
        layers.append(
            Layer(
                index=index,
                depth_km=float(depth_km[index]),
                fields=fields,
                water=_water_mask(fields, floors, min_vs),
            )
        )

    surface_basin = None
    if "inbasin" in handle:
        surface_basin = handle["inbasin"][indices[0], ::step, ::step]
    return layers, surface_basin


def _cut(
    handle: h5py.File,
    label: str,
    take: Callable[[str], np.ndarray],
    depth_km: np.ndarray,
    step: int,
    spacing_km: float,
    min_vs: float | None,
) -> Section:
    """Build one vertical section from a slicing rule.

    Parameters
    ----------
    handle : h5py.File
        Open velocity model file.
    label : str
        Where the cut was made.
    take : callable
        Given a field name, returns that field along the cut.
    depth_km : numpy.ndarray
        Depth of every grid level.
    step : int
        Decimation stride.
    spacing_km : float
        Horizontal grid spacing.
    min_vs : float or None
        The model's enforced minimum Vs, when recorded.

    Returns
    -------
    Section
        The section.
    """
    floors = _field_floors(handle)
    fields = {name: _unpack(handle[name], take(name)) for name in ("vp", "vs")}
    n_samples = fields["vs"].shape[1]
    return Section(
        label=label,
        distance_km=np.arange(n_samples) * step * spacing_km,
        depth_km=depth_km,
        fields=fields,
        water=_water_mask(fields, floors, min_vs),
        basin=take("inbasin") if "inbasin" in handle else None,
    )


def _read_sections(
    handle: h5py.File,
    depth_km: np.ndarray,
    step: int,
    spacing_km: float,
    min_vs: float | None,
) -> list[Section]:
    """Cut two orthogonal vertical sections through the centre of the domain.

    Parameters
    ----------
    handle : h5py.File
        Open velocity model file.
    depth_km : numpy.ndarray
        Depth of every grid level.
    step : int
        Decimation stride.
    spacing_km : float
        Horizontal grid spacing.
    min_vs : float or None
        The model's enforced minimum Vs, when recorded.

    Returns
    -------
    list of Section
        One section along each grid axis.
    """
    _, ny, nx = handle["vs"].shape
    y_mid, x_mid = ny // 2, nx // 2

    # The grid is rotated, so these axes are not east and north. Say "grid".
    return [
        _cut(
            handle,
            "Grid-X section through the domain centre",
            lambda name: handle[name][:, y_mid, ::step],
            depth_km,
            step,
            spacing_km,
            min_vs,
        ),
        _cut(
            handle,
            "Grid-Y section through the domain centre",
            lambda name: handle[name][:, ::step, x_mid],
            depth_km,
            step,
            spacing_km,
            min_vs,
        ),
    ]


def _read_edges(
    handle: h5py.File,
    depth_km: np.ndarray,
    step: int,
    spacing_km: float,
    min_vs: float | None,
) -> list[Section]:
    """Cut sections down the two domain edges the block diagram looks at.

    Parameters
    ----------
    handle : h5py.File
        Open velocity model file.
    depth_km : numpy.ndarray
        Depth of every grid level.
    step : int
        Decimation stride.
    spacing_km : float
        Horizontal grid spacing.
    min_vs : float or None
        The model's enforced minimum Vs, when recorded.

    Returns
    -------
    list of Section
        The faces at minimum y and maximum x, in that order.
    """
    _, _, nx = handle["vs"].shape
    return [
        _cut(
            handle,
            "Domain edge at minimum grid Y",
            lambda name: handle[name][:, 0, ::step],
            depth_km,
            step,
            spacing_km,
            min_vs,
        ),
        _cut(
            handle,
            "Domain edge at maximum grid X",
            lambda name: handle[name][:, ::step, nx - 1],
            depth_km,
            step,
            spacing_km,
            min_vs,
        ),
    ]


def _read_profile(
    handle: h5py.File,
    depth_km: np.ndarray,
    n_blocks: tuple[int, int],
    stat_stride: int,
) -> tuple[DepthProfile, dict[str, int]]:
    """Build whole-model depth statistics from chunk-aligned sample blocks.

    Each block spans every depth, so a couple of dozen of them -- a few percent
    of the file -- give hundreds of thousands of complete columns to summarise.

    Parameters
    ----------
    handle : h5py.File
        Open velocity model file.
    depth_km : numpy.ndarray
        Depth of every grid level.
    n_blocks : tuple of int
        Number of sample blocks along ``(y, x)``.
    stat_stride : int
        Stride applied within a block before histogramming.

    Returns
    -------
    tuple
        The depth profile, and whether each field reaches the top of its packed
        scale.
    """
    nz, ny, nx = handle["vs"].shape
    chunks = handle["vs"].chunks or (nz, min(256, ny), min(256, nx))
    _, chunk_y, chunk_x = chunks

    y_origins = _block_origins(ny, chunk_y, n_blocks[0])
    x_origins = _block_origins(nx, chunk_x, n_blocks[1])

    has_basin = "inbasin" in handle
    counts = {name: np.zeros((nz, 256), dtype=np.int64) for name in FIELDS}
    vpvs_counts = np.zeros((nz, VPVS_BINS + 1), dtype=np.int64)
    basin_counts = np.zeros((nz, 256), dtype=np.int64)
    water_counts = np.zeros(nz, dtype=np.int64)
    sampled = np.zeros(nz, dtype=np.int64)
    table, vpvs_centres = _vpvs_lookup(handle["vp"], handle["vs"])
    samples: dict[str, list[np.ndarray]] = {name: [] for name in (*FIELDS, "depth")}
    n_columns = 0

    for y0 in y_origins:
        for x0 in x_origins:
            ys = slice(y0, min(y0 + chunk_y, ny))
            xs = slice(x0, min(x0 + chunk_x, nx))
            raw = {name: handle[name][:, ys, xs] for name in FIELDS}
            n_columns += raw["vs"].shape[1] * raw["vs"].shape[2]
            thin = {n: v[:, ::stat_stride, ::stat_stride] for n, v in raw.items()}

            for name in FIELDS:
                counts[name] += _per_depth_histogram(thin[name], 256)
            vpvs_counts += _per_depth_histogram(
                table[thin["vp"], thin["vs"]], VPVS_BINS + 1
            )
            if has_basin:
                basin_counts += _per_depth_histogram(
                    handle["inbasin"][:, ys, xs][:, ::stat_stride, ::stat_stride], 256
                )

            # Water sits on the floor of every field, so it lands in bin 0 of
            # each. Counting it once lets it be subtracted back out below,
            # leaving profiles that describe rock instead of the sea.
            water = (thin["vp"] == 0) & (thin["vs"] == 0)
            water_counts += water.sum(axis=(1, 2))
            sampled += water[0].size

            # A co-located subsample, for the density-velocity scatter.
            keep = (slice(None, None, 8), slice(None, None, 16), slice(None, None, 16))
            for name in FIELDS:
                samples[name].append(_unpack(handle[name], raw[name][keep]).ravel())
            depths = np.broadcast_to(
                depth_km[:: keep[0].step, None, None], raw["vs"][keep].shape
            )
            samples["depth"].append(depths.ravel())

    quantiles = {}
    density = {}
    saturated = {}
    for name in FIELDS:
        centres = _levels(handle[name])
        valid = counts[name][:, :N_LEVELS].copy()
        # Does the data actually reach the highest value the packing can hold?
        # If it does, the scale has run out of room at the top.
        saturated[name] = bool(valid[:, N_LEVELS - 1].sum() > 0)
        # Water is a subset of bin 0 by construction, so this can never go
        # negative, and what is left is rock.
        valid[:, 0] -= water_counts
        quantiles[name] = _histogram_quantiles(valid, centres, PROFILE_QUANTILES)
        density[name] = (centres, valid)

    valid_vpvs = vpvs_counts[:, :VPVS_BINS]
    quantiles["vpvs"] = _histogram_quantiles(
        valid_vpvs, vpvs_centres, PROFILE_QUANTILES
    )
    density["vpvs"] = (vpvs_centres, valid_vpvs)

    basin_fraction = None
    if has_basin:
        labelled = basin_counts[:, :FILL].sum(axis=1)
        total = basin_counts.sum(axis=1)
        basin_fraction = np.divide(labelled, total, out=np.zeros(nz), where=total > 0)

    profile = DepthProfile(
        depth_km=depth_km,
        quantiles=quantiles,
        density=density,
        basin_fraction=basin_fraction,
        water_fraction=np.divide(
            water_counts, sampled, out=np.zeros(nz), where=sampled > 0
        ),
        sample={k: np.concatenate(v) for k, v in samples.items()},
        n_columns=n_columns,
    )
    return profile, saturated


def read_summary(
    path: Path,
    depth_fractions: tuple[float, ...] = DEFAULT_DEPTH_FRACTIONS,
    target_samples: int = TARGET_SAMPLES,
    n_blocks: tuple[int, int] = (4, 5),
    stat_stride: int = 2,
    block_faces: bool = False,
) -> VelocityModelSummary:
    """Read a velocity model into everything the summary figures need.

    One pass over the file, sized so that a model of a few billion cells is
    summarised in tens of seconds rather than tens of minutes.

    Parameters
    ----------
    path : pathlib.Path
        The velocity model to read.
    depth_fractions : tuple of float, optional
        Where to slice, as fractions of the model's depth range.
    target_samples : int, optional
        Longest map axis, in samples, after decimation.
    n_blocks : tuple of int, optional
        Number of statistical sample blocks along ``(y, x)``.
    stat_stride : int, optional
        Stride applied within a sample block before histogramming.
    block_faces : bool, optional
        Also read the two domain edges the cut-away block diagram needs. Only
        the poster uses them, so this is off by default.

    Returns
    -------
    VelocityModelSummary
        The layers, sections, statistics and metadata the figures draw from.
    """
    started = time.perf_counter()
    with h5py.File(path, "r") as handle:
        meta = _read_metadata(handle, path)
        nz, ny, nx = meta.shape
        step = _decimation((ny, nx), target_samples)
        spacing = meta.spacing_km or 1.0
        tops = {name: _levels(handle[name])[-1] for name in FIELDS}

        lat, lon = _read_coordinates(handle, step, ny, nx)
        layers, surface_basin = _read_layers(
            handle,
            _depth_indices(nz, depth_fractions),
            meta.depth_km,
            step,
            meta.min_vs,
        )
        sections = _read_sections(handle, meta.depth_km, step, spacing, meta.min_vs)
        edges = (
            _read_edges(handle, meta.depth_km, step, spacing, meta.min_vs)
            if block_faces
            else []
        )
        profile, saturated = _read_profile(handle, meta.depth_km, n_blocks, stat_stride)

    # Count the no-data cells over the full depth slices alone. They have complete
    # spatial coverage, so the count is exact for the depths it covers -- whereas
    # adding the statistical sample on top would double-count every cell that
    # appears in both. Saturation, being rarer still, is taken from either.
    fill_counts = {name: 0 for name in FIELDS}
    for layer in layers:
        for name, values in layer.fields.items():
            fill_counts[name] += int(np.isnan(values).sum())
            if np.isfinite(values).any():
                saturated[name] |= bool(np.nanmax(values) >= tops[name] - 1e-6)

    return VelocityModelSummary(
        meta=meta,
        lat=lat,
        lon=lon,
        layers=layers,
        sections=sections,
        edges=edges,
        profile=profile,
        surface_basin=surface_basin,
        fill_counts=fill_counts,
        saturated=saturated,
        water_fraction=float(layers[0].water.mean()),
        read_seconds=time.perf_counter() - started,
    )


def brocher_density(vp: np.ndarray) -> np.ndarray:
    """Density predicted from Vp by Brocher's (2005) empirical fit.

    A velocity model whose density is not a function of Vp along roughly this
    curve has almost certainly gone wrong somewhere, which makes the residual a
    cheap and sensitive quality check.

    Parameters
    ----------
    vp : numpy.ndarray
        P-wave velocity, in km/s.

    Returns
    -------
    numpy.ndarray
        Predicted density, in g/cm^3.
    """
    return (
        1.6612 * vp
        - 0.4721 * vp**2
        + 0.0671 * vp**3
        - 0.0043 * vp**4
        + 0.000106 * vp**5
    )


def poisson_ratio(vpvs: np.ndarray) -> np.ndarray:
    """Poisson's ratio implied by a Vp/Vs ratio.

    Parameters
    ----------
    vpvs : numpy.ndarray
        The Vp/Vs ratio.

    Returns
    -------
    numpy.ndarray
        Poisson's ratio, which is physically confined to below 0.5.
    """
    squared = vpvs**2
    return (squared - 2.0) / (2.0 * (squared - 1.0))


def sediment_depth_km(summary: VelocityModelSummary) -> float:
    """Depth below which the model is rock rather than sediment or water.

    Taken as 5% of the model's depth range, because the domains vary and there is
    no single absolute depth that means "below the sediments" for all of them.
    Saturated sediment genuinely reaches a Vp/Vs of 4 or more, so a rock check
    applied above this depth would flag sound physics as an error.

    Parameters
    ----------
    summary : VelocityModelSummary
        The model summary.

    Returns
    -------
    float
        Depth, in kilometres.
    """
    depth = summary.profile.depth_km
    return float(depth[0] + 0.05 * (depth[-1] - depth[0]))


def vpvs_outlier_fraction(summary: VelocityModelSummary) -> float:
    """Fraction of rock cells whose Vp/Vs falls outside the plausible band.

    Water is already excluded by the lookup table, and the sediment column is
    excluded by depth, so what remains should be rock sitting close to the
    Poisson-solid value. Anything else is a defect.

    Parameters
    ----------
    summary : VelocityModelSummary
        The model summary.

    Returns
    -------
    float
        Fraction of sampled rock cells outside the plausible band.
    """
    centres, counts = summary.profile.density["vpvs"]
    rock = counts[summary.profile.depth_km >= sediment_depth_km(summary)]
    total = rock.sum()
    if not total:
        return 0.0
    low, high = style.VPVS_PLAUSIBLE
    outside = rock[:, (centres < low) | (centres > high)].sum()
    return float(outside / total)
