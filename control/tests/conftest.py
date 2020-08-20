# contest.py - pytest local plugins and fixtures

import os

import matplotlib as mpl
import pytest

import control


@pytest.fixture(scope="session", autouse=True)
def use_numpy_ndarray():
    """Switch the config to use ndarray instead of matrix"""
    if os.getenv("PYTHON_CONTROL_STATESPACE_ARRAY") == "1":
        control.config.defaults['statesp.use_numpy_matrix'] = False


@pytest.fixture(scope="function")
def editsdefaults():
    """Make sure any changes to the defaults only last during a test"""
    restore = control.config.defaults.copy()
    yield
    control.config.defaults.update(restore)


@pytest.fixture(scope="function")
def mplcleanup():
    """Workaround for python2

    python 2 does not like to mix the original mpl decorator with pytest
    fixtures. So we roll our own.
    """
    save = mpl.units.registry.copy()
    try:
        yield
    finally:
        mpl.units.registry.clear()
        mpl.units.registry.update(save)
        mpl.pyplot.close("all")
