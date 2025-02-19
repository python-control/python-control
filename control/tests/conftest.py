"""conftest.py - pytest local plugins, fixtures, marks and functions."""

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
    """Turn off warnings for calls to plotting functions with old signatures."""
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
    """Turn off warnings for functions that generate FutureWarning."""
    import warnings
    warnings.filterwarnings(
        'ignore', message='.*deprecated', category=FutureWarning)
    yield
    warnings.resetwarnings()


def pytest_configure(config):
    """Allow pytest.mark.slow to mark slow tests.

    skip with pytest -m "not slow"
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def assert_tf_close_coeff(actual, desired, rtol=1e-5, atol=1e-8):
    """Check if two transfer functions have close coefficients.

    Parameters
    ----------
    actual, desired : TransferFunction
        Transfer functions to compare.
    rtol : float
        Relative tolerance for ``np.testing.assert_allclose``.
    atol : float
        Absolute tolerance for ``np.testing.assert_allclose``.

    Raises
    ------
    AssertionError
    """
    # Check number of outputs and inputs
    assert actual.noutputs == desired.noutputs
    assert actual.ninputs == desired.ninputs
    # Check timestep
    assert  actual.dt == desired.dt
    # Check coefficient arrays
    for i in range(actual.noutputs):
        for j in range(actual.ninputs):
            np.testing.assert_allclose(
                actual.num[i][j],
                desired.num[i][j],
                rtol=rtol, atol=atol)
            np.testing.assert_allclose(
                actual.den[i][j],
                desired.den[i][j],
                rtol=rtol, atol=atol)
