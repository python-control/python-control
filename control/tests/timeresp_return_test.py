"""timeresp_return_test.py - test return values from time response functions

RMM, 22 Aug 2021

This set of unit tests covers checks to make sure that the various time
response functions are returning the right sets of objects in the (new)
InputOutputResponse class.

"""

import pytest

import numpy as np
import control as ct


def test_ioresponse_retvals():
    # SISO, single trace
    sys = ct.rss(4, 1, 1)
    T = np.linspace(0, 1, 10)
    U = np.sin(T)
    X0 = np.ones((sys.nstates,))

    # Initial response
    res = ct.initial_response(sys, X0=X0)
    assert res.outputs.shape == (res.time.shape[0],)
    assert res.states.shape == (sys.nstates, res.time.shape[0])
    np.testing.assert_equal(res.inputs, np.zeros((res.time.shape[0],)))
    
    # Impulse response
    res = ct.impulse_response(sys)
    assert res.outputs.shape == (res.time.shape[0],)
    assert res.states.shape == (sys.nstates, res.time.shape[0])
    assert res.inputs.shape == (res.time.shape[0],)
    np.testing.assert_equal(res.inputs, None)

    # Step response
    res = ct.step_response(sys)
    assert res.outputs.shape == (res.time.shape[0],)
    assert res.states.shape == (sys.nstates, res.time.shape[0])
    assert res.inputs.shape == (res.time.shape[0],)
    
