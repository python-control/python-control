"""nltools_test.py - test static nonlinear feedback functionality

RMM, 23 Jan 2021

This set of unit tests covers the various operatons of the nltools module, as
well as some of the support functions associated with static nonlinearities.

"""

import pytest

import numpy as np
import control as ct
import math

class saturation():
    # Static nonlinear saturation function
    def __call__(self, x, lb=-1, ub=1):
        return np.maximum(lb, np.minimum(x, ub))

    # Describing function for a saturation function
    def describing_function(self, a):
        if -1 <= a and a <= 1:
            return 1.
        else:
            b = 1/a
            return 2/math.pi * (math.asin(b) + b * math.sqrt(1 - b**2))


# Static nonlinear system implementing saturation
@pytest.fixture
def satsys():
    satfcn = saturation()
    def _satfcn(t, x, u, params):
        return satfcn(u)
    return ct.NonlinearIOSystem(None, outfcn=_satfcn, input=1, output=1)


def test_static_nonlinear_call(satsys):
    # Make sure that the saturation system is a static nonlinearity
    assert satsys.isstatic()

    # Make sure the saturation function is doing the right computation
    input = [-2, -1, -0.5, 0, 0.5, 1, 2]
    desired = [-1, -1, -0.5, 0, 0.5, 1, 1]
    for x, y in zip(input, desired):
        assert satsys(x) == y

    # Test squeeze properties
    assert satsys(0.) == 0.
    assert satsys([0.], squeeze=True) == 0.
    np.testing.assert_array_equal(satsys([0.]), [0.])

    # Test SIMO nonlinearity
    def _simofcn(t, x, u, params={}):
        return np.array([np.cos(u), np.sin(u)])
    simo_sys = ct.NonlinearIOSystem(None, outfcn=_simofcn, input=1, output=2)
    np.testing.assert_array_equal(simo_sys([0.]), [1, 0])
    np.testing.assert_array_equal(simo_sys([0.], squeeze=True), [1, 0])

    # Test MISO nonlinearity
    def _misofcn(t, x, u, params={}):
        return np.array([np.sin(u[0]) * np.cos(u[1])])
    miso_sys = ct.NonlinearIOSystem(None, outfcn=_misofcn, input=2, output=1)
    np.testing.assert_array_equal(miso_sys([0, 0]), [0])
    np.testing.assert_array_equal(miso_sys([0, 0]), [0])
    np.testing.assert_array_equal(miso_sys([0, 0], squeeze=True), [0])
    

# Test saturation describing function in multiple ways
def test_saturation_describing_function(satsys):
    satfcn = saturation()
    
    # Store the analytic describing function for comparison
    amprange = np.linspace(0, 10, 100)
    df_anal = [satfcn.describing_function(a) for a in amprange]
    
    # Compute describing function for a static function
    df_fcn = [ct.describing_function(satfcn, a) for a in amprange]
    np.testing.assert_almost_equal(df_fcn, df_anal, decimal=3)

    # Compute describing function for a static I/O system
    df_sys = [ct.describing_function(satsys, a) for a in amprange]
    np.testing.assert_almost_equal(df_sys, df_anal, decimal=3)

    # Compute describing function on an array of values
    df_arr = ct.describing_function(satsys, amprange)
    np.testing.assert_almost_equal(df_arr, df_anal, decimal=3)

from control.nltools import saturation_nonlinearity, backlash_nonlinearity, \
    hysteresis_deadzone_nonlinearity


@pytest.mark.parametrize("fcn, amin, amax", [
    [saturation_nonlinearity(1), 0, 10],
    [backlash_nonlinearity(2), 1, 10],
    [hysteresis_deadzone_nonlinearity(1, 1, 1), 3, 10],
    ])
def test_describing_function(fcn, amin, amax):
    # Store the analytic describing function for comparison
    amprange = np.linspace(amin, amax, 100)
    df_anal = [fcn.describing_function(a) for a in amprange]
    
    # Compute describing function on an array of values
    df_arr = ct.describing_function(fcn, amprange, zero_check=False)
    np.testing.assert_almost_equal(df_arr, df_anal, decimal=1)
