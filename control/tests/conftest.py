"""conftest.py - pytest local plugins and fixtures"""

import os
from contextlib import contextmanager

import matplotlib as mpl
import numpy as np
import pytest

import control

# some common pytest marks. These can be used as test decorators or in
# pytest.param(marks=)
slycotonly = pytest.mark.skipif(
    not control.exception.slycot_check(), reason="slycot not installed")
cvxoptonly = pytest.mark.skipif(
    not control.exception.cvxopt_check(), reason="cvxopt not installed")


@pytest.fixture(scope="session", autouse=True)
def control_defaults():
    """Make sure the testing session always starts with the defaults.

    This should be the first fixture initialized, so that all other
    fixtures see the general defaults (unless they set them themselves)
    even before importing control/__init__. Enforce this by adding it as an
    argument to all other session scoped fixtures.

    """
    control.reset_defaults()
    the_defaults = control.config.defaults.copy()
    yield
    # assert that nothing changed it without reverting
    assert control.config.defaults == the_defaults


@pytest.fixture(scope="function")
def editsdefaults():
    """Make sure any changes to the defaults only last during a test."""
    restore = control.config.defaults.copy()
    yield
    control.config.defaults.clear()
    control.config.defaults.update(restore)


@pytest.fixture(scope="function")
def mplcleanup():
    """Clean up any plots and changes a test may have made to matplotlib.

    compare matplotlib.testing.decorators.cleanup() but as a fixture instead
    of a decorator.
    """
    save = mpl.units.registry.copy()
    try:
        yield
    finally:
        mpl.units.registry.clear()
        mpl.units.registry.update(save)
        mpl.pyplot.close("all")


@pytest.fixture(scope="function")
def legacy_plot_signature():
    """Turn off warnings for calls to plotting functions with old signatures"""
    import warnings
    warnings.filterwarnings(
        'ignore', message='passing systems .* is deprecated',
        category=FutureWarning)
    warnings.filterwarnings(
        'ignore', message='.* return value of .* is deprecated',
        category=FutureWarning)
    yield
    warnings.resetwarnings()


@pytest.fixture(scope="function")
def ignore_future_warning():
    """Turn off warnings for functions that generate FutureWarning"""
    import warnings
    warnings.filterwarnings(
        'ignore', message='.*deprecated', category=FutureWarning)
    yield
    warnings.resetwarnings()
    

# Allow pytest.mark.slow to mark slow tests (skip with pytest -m "not slow")
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
