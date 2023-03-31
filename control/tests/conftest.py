"""conftest.py - pytest local plugins and fixtures"""

import os
from contextlib import contextmanager

import matplotlib as mpl
import numpy as np
import pytest

import control

TEST_MATRIX_AND_ARRAY = os.getenv("PYTHON_CONTROL_ARRAY_AND_MATRIX") == "1"

# some common pytest marks. These can be used as test decorators or in
# pytest.param(marks=)
slycotonly = pytest.mark.skipif(not control.exception.slycot_check(),
                                reason="slycot not installed")
cvxoptonly = pytest.mark.skipif(not control.exception.cvxopt_check(),
                                reason="cvxopt not installed")
matrixfilter = pytest.mark.filterwarnings("ignore:.*matrix subclass:"
                                          "PendingDeprecationWarning")
matrixerrorfilter = pytest.mark.filterwarnings("error:.*matrix subclass:"
                                               "PendingDeprecationWarning")


@pytest.fixture(scope="session", autouse=True)
def control_defaults():
    """Make sure the testing session always starts with the defaults.

    This should be the first fixture initialized,
    so that all other fixtures see the general defaults (unless they set them
    themselves) even before importing control/__init__. Enforce this by adding
    it as an argument to all other session scoped fixtures.
    """
    control.reset_defaults()
    the_defaults = control.config.defaults.copy()
    yield
    # assert that nothing changed it without reverting
    assert control.config.defaults == the_defaults


@pytest.fixture(scope="function", autouse=TEST_MATRIX_AND_ARRAY,
                params=[pytest.param("arrayout", marks=matrixerrorfilter),
                        pytest.param("matrixout", marks=matrixfilter)])
def matarrayout(request):
    """Switch the config to use np.ndarray and np.matrix as returns."""
    restore = control.config.defaults['statesp.use_numpy_matrix']
    control.use_numpy_matrix(request.param == "matrixout", warn=False)
    yield
    control.use_numpy_matrix(restore, warn=False)


def ismatarrayout(obj):
    """Test if the returned object has the correct type as configured.

    note that isinstance(np.matrix(obj), np.ndarray) is True
    """
    use_matrix = control.config.defaults['statesp.use_numpy_matrix']
    return (isinstance(obj, np.ndarray)
            and isinstance(obj, np.matrix) == use_matrix)


def asmatarrayout(obj):
    """Return a object according to the configured default."""
    use_matrix = control.config.defaults['statesp.use_numpy_matrix']
    matarray = np.asmatrix if use_matrix else np.asarray
    return matarray(obj)


@contextmanager
def check_deprecated_matrix():
    """Check that a call produces a deprecation warning because of np.matrix."""
    use_matrix = control.config.defaults['statesp.use_numpy_matrix']
    if use_matrix:
        with pytest.deprecated_call():
            try:
                yield
            finally:
                pass
    else:
        yield


@pytest.fixture(scope="function",
                params=[p for p, usebydefault in
                        [(pytest.param(np.array,
                                       id="arrayin"),
                          True),
                         (pytest.param(np.matrix,
                                       id="matrixin",
                                       marks=matrixfilter),
                          False)]
                        if usebydefault or TEST_MATRIX_AND_ARRAY])
def matarrayin(request):
    """Use array and matrix to construct input data in tests."""
    return request.param


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


# Allow pytest.mark.slow to mark slow tests (skip with pytest -m "not slow")
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
