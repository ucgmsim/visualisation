"""Shared pytest configuration for the visualisation test suite."""

import pygmt
import pytest


@pytest.fixture(autouse=True)
def clean_gmt_session():
    """Start every test from GMT's default configuration.

    `pygmt.config` writes to GMT's process-global defaults. It is meant to be
    used as a context manager, but `pygmt_helper.plotting.gen_region_fig`
    applies its `config_options` with a bare `pygmt.config(...)` call, so
    settings from one figure persist into every figure drawn afterwards in the
    same process. That makes image comparisons depend on test execution order:
    `plot_srf` sets `MAP_FRAME_TYPE="plain"` and `FORMAT_GEO_MAP="ddd.xx"`,
    which then leak into the plots that follow it.

    Restarting the GMT session restores the defaults, so each test renders the
    same image whether it runs alone or after the rest of the suite.
    """
    pygmt.session_management.end()
    pygmt.session_management.begin()
    yield
