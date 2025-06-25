"""descfcn_test.py - test describing functions and related capabilities

RMM, 23 Jan 2021

This set of unit tests covers the various operatons of the descfcn module, as
well as some of the support functions associated with static nonlinearities.

"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

import control as ct
from control.descfcn import friction_backlash_nonlinearity, \
    relay_hysteresis_nonlinearity, saturation_nonlinearity


# Static function via a class
class saturation_class:
    # Static nonlinear saturation function
    def __call__(self, x, lb=-1, ub=1):
        return np.clip(x, lb, ub)

    # Describing function for a saturation function
    def describing_function(self, a):
        if -1 <= a and a <= 1:
            return 1.
        else:
            b = 1/a
            return 2/math.pi * (math.asin(b) + b * math.sqrt(1 - b**2))


# Static function without a class
def saturation(x):
    return np.clip(x, -1, 1)


# Static nonlinear system implementing saturation
@pytest.fixture
def satsys():
    satfcn = saturation_class()
    def _satfcn(t, x, u, params):
        return satfcn(u)
    return ct.NonlinearIOSystem(None, outfcn=_satfcn, input=1, output=1)


def test_static_nonlinear_call(satsys):
    # Make sure that the saturation system is a static nonlinearity
    assert satsys._isstatic()

    # Make sure the saturation function is doing the right computation
    input = [-2, -1, -0.5, 0, 0.5, 1, 2]
    desired = [-1, -1, -0.5, 0, 0.5, 1, 1]
    for x, y in zip(input, desired):
        np.testing.assert_allclose(satsys(x), y)

    # Test squeeze properties
    assert satsys(0.) == 0.
    assert satsys([0.], squeeze=True) == 0.
    np.testing.assert_allclose(satsys([0.]), [0.])

    # Test SIMO nonlinearity
    def _simofcn(t, x, u, params):
        return np.array([np.cos(u), np.sin(u)])
    simo_sys = ct.NonlinearIOSystem(None, outfcn=_simofcn, input=1, output=2)
    np.testing.assert_allclose(simo_sys([0.]), [1, 0])
    np.testing.assert_allclose(simo_sys([0.], squeeze=True), [1, 0])

    # Test MISO nonlinearity
    def _misofcn(t, x, u, params={}):
        return np.array([np.sin(u[0]) * np.cos(u[1])])
    miso_sys = ct.NonlinearIOSystem(None, outfcn=_misofcn, input=2, output=1)
    np.testing.assert_allclose(miso_sys([0, 0]), [0])
    np.testing.assert_allclose(miso_sys([0, 0], squeeze=True), [0])


# Test saturation describing function in multiple ways
def test_saturation_describing_function(satsys):
    satfcn = saturation_class()

    # Store the analytic describing function for comparison
    amprange = np.linspace(0, 10, 100)
    df_anal = [satfcn.describing_function(a) for a in amprange]

    # Compute describing function for a static function
    df_fcn = ct.describing_function(saturation, amprange)
    np.testing.assert_almost_equal(df_fcn, df_anal, decimal=3)

    # Compute describing function for a describing function nonlinearity
    df_fcn = ct.describing_function(satfcn, amprange)
    np.testing.assert_almost_equal(df_fcn, df_anal, decimal=3)

    # Compute describing function for a static I/O system
    df_sys = ct.describing_function(satsys, amprange)
    np.testing.assert_almost_equal(df_sys, df_anal, decimal=3)

    # Compute describing function on an array of values
    df_arr = ct.describing_function(satsys, amprange)
    np.testing.assert_almost_equal(df_arr, df_anal, decimal=3)

    # Evaluate static function at a negative amplitude
    with pytest.raises(ValueError, match="cannot evaluate"):
        ct.describing_function(saturation, -1)

    # Create describing function nonlinearity w/out describing_function method
    # and make sure it drops through to the underlying computation
    class my_saturation(ct.DescribingFunctionNonlinearity):
        def __call__(self, x):
            return saturation(x)
    satfcn_nometh = my_saturation()
    df_nometh = ct.describing_function(satfcn_nometh, amprange)
    np.testing.assert_almost_equal(df_nometh, df_anal, decimal=3)


@pytest.mark.parametrize("fcn, amin, amax", [
    [saturation_nonlinearity(1), 0, 10],
    [friction_backlash_nonlinearity(2), 1, 10],
    [relay_hysteresis_nonlinearity(1, 1), 3, 10],
    ])
def test_describing_function(fcn, amin, amax):
    # Store the analytic describing function for comparison
    amprange = np.linspace(amin, amax, 100)
    df_anal = [fcn.describing_function(a) for a in amprange]

    # Compute describing function on an array of values
    df_arr = ct.describing_function(
        fcn, amprange, zero_check=False, try_method=False)
    np.testing.assert_almost_equal(df_arr, df_anal, decimal=1)

    # Make sure the describing function method also works
    df_meth = ct.describing_function(fcn, amprange, zero_check=False)
    np.testing.assert_almost_equal(df_meth, df_anal)

    # Make sure that evaluation at negative amplitude generates an exception
    with pytest.raises(ValueError, match="cannot evaluate"):
        ct.describing_function(fcn, -1)


def test_describing_function_response():
    # Simple linear system with at most 1 intersection
    H_simple = ct.tf([1], [1, 2, 2, 1])
    omega = np.logspace(-1, 2, 100)

    # Saturation nonlinearity
    F_saturation = ct.descfcn.saturation_nonlinearity(1)
    amp = np.linspace(1, 4, 10)

    # No intersection
    xsects = ct.describing_function_response(H_simple, F_saturation, amp, omega)
    assert len(xsects) == 0

    # One intersection
    H_larger = H_simple * 8
    xsects = ct.describing_function_response(H_larger, F_saturation, amp, omega)
    for a, w in xsects:
        np.testing.assert_almost_equal(
            H_larger(1j*w),
            -1/ct.describing_function(F_saturation, a), decimal=5)

    # Multiple intersections
    H_multiple = H_simple * ct.tf(*ct.pade(5, 4)) * 4
    omega = np.logspace(-1, 3, 50)
    F_backlash = ct.descfcn.friction_backlash_nonlinearity(1)
    amp = np.linspace(0.6, 5, 50)
    xsects = ct.describing_function_response(H_multiple, F_backlash, amp, omega)
    for a, w in xsects:
        np.testing.assert_almost_equal(
            -1/ct.describing_function(F_backlash, a),
            H_multiple(1j*w), decimal=5)


def test_describing_function_plot():
    # Simple linear system with at most 1 intersection
    H_larger = ct.tf([1], [1, 2, 2, 1]) * 8
    omega = np.logspace(-1, 2, 100)

    # Saturation nonlinearity
    F_saturation = ct.descfcn.saturation_nonlinearity(1)
    amp = np.linspace(1, 4, 10)

    # Plot via response
    plt.clf()                                   # clear axes
    response = ct.describing_function_response(
        H_larger, F_saturation, amp, omega)
    assert len(response.intersections) == 1
    assert len(plt.gcf().get_axes()) == 0       # make sure there is no plot

    cplt = response.plot()
    assert len(plt.gcf().get_axes()) == 1       # make sure there is a plot
    assert len(cplt.lines[0]) == 5 and len(cplt.lines[1]) == 1

    # Call plot directly
    cplt = ct.describing_function_plot(H_larger, F_saturation, amp, omega)
    assert len(cplt.lines[0]) == 5 and len(cplt.lines[1]) == 1


def test_describing_function_exceptions():
    # Describing function with non-zero bias
    with pytest.warns(UserWarning, match="asymmetric"):
        saturation = ct.descfcn.saturation_nonlinearity(lb=-1, ub=2)
        assert saturation(-3) == -1
        assert saturation(3) == 2

    # Turn off the bias check
    ct.describing_function(saturation, 0, zero_check=False)

    # Function should evaluate to zero at zero amplitude
    f = lambda x: x + 0.5
    with pytest.raises(ValueError, match="must evaluate to zero"):
        ct.describing_function(f, 0, zero_check=True)

    # Evaluate at a negative amplitude
    with pytest.raises(ValueError, match="cannot evaluate"):
        ct.describing_function(saturation, -1)

    # Describing function with bad label
    H_simple = ct.tf([8], [1, 2, 2, 1])
    F_saturation = ct.descfcn.saturation_nonlinearity(1)
    amp = np.linspace(1, 4, 10)
    with pytest.raises(ValueError, match="formatting string"):
        ct.describing_function_plot(H_simple, F_saturation, amp, label=1)

    # Unrecognized keyword
    with pytest.raises(TypeError, match="unrecognized keyword"):
        ct.describing_function_response(
            H_simple, F_saturation, amp, None, unknown=None)

    # Unrecognized keyword
    with pytest.raises(AttributeError, match="no property|unexpected keyword"):
        response = ct.describing_function_response(H_simple, F_saturation, amp)
        response.plot(unknown=None)

    # Describing function plot for non-describing function object
    resp = ct.frequency_response(H_simple)
    with pytest.raises(TypeError, match="data must be DescribingFunction"):
        ct.describing_function_plot(resp)
