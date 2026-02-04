from collections.abc import Callable
from pathlib import Path

import diffimg
import pytest

from visualisation import plot_1d_velocity_model, plot_rupture_path, realisation
from visualisation.sources import (
    plot_mw_contributions,
    plot_rakes,
    plot_rise,
    plot_slip_rise_rake,
    plot_srf,
    plot_srf_cumulative_moment,
    plot_srf_distribution,
    plot_srf_moment,
    plot_stoch,
)

TEST_DATA_DIR = Path(__file__).parent
PLOT_IMAGE_DIRECTORY = Path("wiki/images")
STATIONS_FFP = TEST_DATA_DIR / "realisation" / "stations.ll"
DEFAULT_IMAGE_DIFF_TOLERANCE = 0.05


@pytest.fixture(scope="module")
def plot_image_dir() -> Path:
    """Provides the path to the directory containing expected plot images."""
    base_dir = Path.cwd()
    path = base_dir / PLOT_IMAGE_DIRECTORY
    if not path.is_dir():
        pytest.skip(f"Expected image directory not found: {path}")
    return path


@pytest.fixture(scope="module")
def srf_ffp() -> Path:
    """Path to the primary SRF file for testing."""
    return TEST_DATA_DIR / "srfs" / "rupture_1.srf"


@pytest.fixture(scope="module")
def stoch_ffp() -> Path:
    """Path to the stochastic file used for testing."""
    return TEST_DATA_DIR / "stoch" / "realisation.stoch"


@pytest.fixture(scope="module")
def domain_realisation_ffp() -> Path:
    """Path to the domain realisation file used for testing."""
    return TEST_DATA_DIR / "realisation" / "realisation.json"


@pytest.fixture(scope="module")
def multi_summary_srf_ffp() -> Path:
    """Path to the SRF file used for multi-summary plots."""
    return TEST_DATA_DIR / "srfs" / "nevis.srf"


@pytest.fixture(scope="module")
def realisation_ffp() -> Path:
    """Path to the realisation JSON file."""
    return TEST_DATA_DIR / "srfs" / "realisation.json"


@pytest.fixture(scope="module")
def velocity_model_plot_file() -> Path:
    """Path to the velocity model JSON file."""
    return TEST_DATA_DIR / "velocity_models" / "alpine_hope_1.json"


@pytest.fixture
def output_image_path(tmp_path: Path) -> Path:
    """Provides a unique temporary path for generated images in each test."""
    return tmp_path / "output.png"


@pytest.fixture(scope="module")
def realisation_base_file() -> Path:
    """Path to the realisation JSON file."""
    return TEST_DATA_DIR / "realisation" / "realisation.json"


def assert_images_match(
    generated_path: Path,
    expected_path: Path,
    tolerance: float = DEFAULT_IMAGE_DIFF_TOLERANCE,
):
    """
    Compares two images using diffimg and asserts the difference is within tolerance.
    Provides informative assertion messages.
    """
    assert generated_path.exists(), (
        f"Generated image file does not exist: {generated_path}"
    )
    assert expected_path.exists(), (
        f"Expected image file does not exist: {expected_path}"
    )

    diff_ratio = diffimg.diff(expected_path, generated_path)

    assert diff_ratio <= tolerance, (
        f"Image difference ({diff_ratio:.4f}) exceeds tolerance ({tolerance}) "
        f"between generated '{generated_path.name}' and expected '{expected_path.name}'."
        f"\nGenerated: {generated_path}"
        f"\nExpected:  {expected_path}"
    )


@pytest.mark.parametrize(
    "plot_function, expected_image_name, plot_kwargs",
    [
        (plot_srf.plot_srf, "srf_plot_example.png", {}),
        (
            plot_srf.plot_srf,
            "srf_plot_example_inset.png",
            {"show_inset": True, "latitude_pad": 0.1, "longitude_pad": 0.1},
        ),
        (plot_srf_moment.plot_srf_moment, "srf_moment_rate_example.png", {}),
        (
            plot_srf_cumulative_moment.plot_srf_cumulative_moment,
            "srf_cumulative_moment_rate_example.png",
            {},
        ),
        (plot_rise.plot_rise, "rise_example.png", {}),
        # Special case for plot_rakes needing a seed for reproducibility
        (plot_rakes.plot_rakes, "rakes_example.png", {"seed": 1}),
        (
            plot_srf_distribution.plot_srf_distribution,
            "srf_distribution_example.png",
            {},
        ),
    ],
    ids=[  # Optional: Provide clearer test IDs
        "srf_plot",
        "srf_plot_inset",
        "srf_moment",
        "srf_cumulative_moment",
        "rise_plot",
        "rakes_plot",
        "srf_distribution",
    ],
)
def test_standard_srf_plots(
    srf_ffp: Path,
    output_image_path: Path,
    plot_image_dir: Path,
    plot_function: Callable,
    expected_image_name: str,
    plot_kwargs: dict,
):
    expected_image_file = plot_image_dir / expected_image_name

    # Call the specific plot function with its arguments
    plot_function(srf_ffp, output_image_path, **plot_kwargs)

    assert_images_match(output_image_path, expected_image_file)


def test_plot_mw_contributions(
    srf_ffp: Path,
    realisation_ffp: Path,
    output_image_path: Path,
    plot_image_dir: Path,
):
    expected_image_file = plot_image_dir / "example_mw_contributions.png"

    plot_mw_contributions.plot_mw_contributions(
        srf_ffp, realisation_ffp, output_image_path, width=15, height=15
    )

    assert_images_match(output_image_path, expected_image_file)


@pytest.mark.parametrize(
    "plot_args, expected_image_name",
    [
        # Test cases using plot_type
        ({"plot_type": plot_slip_rise_rake.PlotType.slip}, "summary_slip.png"),
        (
            {"plot_type": plot_slip_rise_rake.PlotType.rise},
            "summary_rise.png",
        ),
        ({"plot_type": plot_slip_rise_rake.PlotType.rake}, "summary_rake.png"),
        # Test case using segment (note different dimensions might be needed)
        ({"segment": 1, "width": 15, "height": 30}, "summary_segment_1.png"),
    ],
    ids=["slip_type", "rise_time_type", "rake_type", "segment_1"],
)
def test_plot_slip_rise_rake(
    realisation_ffp: Path,
    multi_summary_srf_ffp: Path,
    output_image_path: Path,
    plot_image_dir: Path,
    plot_args: dict,
    expected_image_name: str,
):
    expected_image_file = plot_image_dir / expected_image_name

    # Default arguments that might be overridden by plot_args
    default_args = {"width": 30, "height": 15}
    call_args = {**default_args, **plot_args}  # Merge args, plot_args take precedence

    plot_slip_rise_rake.plot_slip_rise_rake(
        realisation_ffp,
        multi_summary_srf_ffp,
        output_image_path,
        **call_args,
    )

    assert_images_match(output_image_path, expected_image_file)


@pytest.mark.parametrize(
    "plot_kwargs, expected_image_name",
    [
        # Default case
        ({}, "alpine_hope_default_vmod.png"),
        # Vs Density case
        (
            {
                "panels": [
                    plot_1d_velocity_model.Panel.VS,
                    plot_1d_velocity_model.Panel.DENSITY,
                ]
            },
            "alpine_hope_vmod_vs_density.png",
        ),
        # Qp and Qs case
        (
            {
                "panels": [
                    plot_1d_velocity_model.Panel.QP,
                    plot_1d_velocity_model.Panel.QS,
                ],
                "title": "Qp and Qs",
            },
            "alpine_hope_vmod_qp_qs.png",
        ),
        # Custom options case
        (
            {
                "panels": [
                    plot_1d_velocity_model.Panel.VP,
                    plot_1d_velocity_model.Panel.VS,
                ],
                "title": "P and S Wave Velocity",
                "subplot_height": 12,
                "subplot_width": 12,
                "dpi": 400,
            },
            "custom_vmod_plot.png",
        ),
    ],
    ids=["default", "vs_density", "qp_qs", "custom_options"],
)
def test_plot_velocity_model(
    velocity_model_plot_file: Path,
    output_image_path: Path,
    plot_image_dir: Path,
    plot_kwargs: dict,
    expected_image_name: str,
):
    expected_image_file = plot_image_dir / expected_image_name

    plot_1d_velocity_model.plot_1d_velocity_model_to_file(
        velocity_model_plot_file,
        output_image_path,
        **plot_kwargs,
    )

    assert_images_match(output_image_path, expected_image_file)


@pytest.mark.parametrize(
    "plot_kwargs, expected_image_name",
    [
        # Default case
        ({}, "alpine_base_1.png"),
        (
            {
                "latitude_pad": 0.5,
                "longitude_pad": 0.5,
                "width": 15,
                "title": "Padded Domain",
            },
            "alpine_base_1_padded.png",
        ),
        (
            {"show_geometry": False},
            "alpine_base_1_no_geometry.png",
        ),
        (
            {"stations": STATIONS_FFP},
            "alpine_base_1_stations.png",
        ),
    ],
)
def test_realisation_plot(
    plot_image_dir: Path,
    output_image_path: Path,
    domain_realisation_ffp: Path,
    expected_image_name: str,
    plot_kwargs: dict,
):
    expected_image_file = plot_image_dir / expected_image_name
    realisation.plot_realisation_to_file(
        domain_realisation_ffp,
        output_image_path,
        **plot_kwargs,
    )

    assert_images_match(output_image_path, expected_image_file)


@pytest.mark.parametrize(
    "plot_kwargs, expected_image_name",
    [
        # Default case
        (
            {
                "width": 60,
                "height": 20,
                "dpi": 300,
                "title": "Stoch file",
            },
            "stoch_example.png",
        ),
    ],
    ids=["custom_options"],
)
def test_plot_stoch(
    tmp_path: Path,
    plot_image_dir: Path,
    output_image_path: Path,
    stoch_ffp: Path,
    expected_image_name: str,
    plot_kwargs: dict,
):
    expected_image_file = plot_image_dir / expected_image_name

    plot_stoch.plot_stoch_to_file(
        stoch_ffp,
        output_image_path,
        **plot_kwargs,
    )

    assert_images_match(output_image_path, expected_image_file)


def test_plot_rupture_path(
    velocity_model_plot_file: Path,
    output_image_path: Path,
    plot_image_dir: Path,
):
    expected_image_file = plot_image_dir / "alpine_hope_1_path.png"
    plot_rupture_path.plot_rupture_path_to_file(
        velocity_model_plot_file, output_image_path
    )
    assert_images_match(output_image_path, expected_image_file)
