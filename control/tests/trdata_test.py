"""trdata_test.py - test return values from time response functions

RMM, 22 Aug 2021

This set of unit tests covers checks to make sure that the various time
response functions are returning the right sets of objects in the (new)
InputOutputResponse class.

"""

import pytest

import numpy as np
import control as ct


@pytest.mark.parametrize(
    "nout, nin, squeeze", [
        [1,   1,  None],
        [1,   1,  True],
        [1,   1,  False],
        [1,   2,  None],
        [1,   2,  True],
        [1,   2,  False],
        [2,   1,  None],
        [2,   1,  True],
        [2,   1,  False],
        [2,   2,  None],
        [2,   2,  True],
        [2,   2,  False],
])
def test_trdata_shapes(nin, nout, squeeze):
    # SISO, single trace
    sys = ct.rss(4, nout, nin, strictly_proper=True)
    T = np.linspace(0, 1, 10)
    U = np.outer(np.ones(nin), np.sin(T) )
    X0 = np.ones(sys.nstates)

    #
    # Initial response
    #
    res = ct.initial_response(sys, X0=X0)
    ntimes = res.time.shape[0]

    # Check shape of class members
    assert len(res.time.shape) == 1
    assert res.y.shape == (sys.noutputs, ntimes)
    assert res.x.shape == (sys.nstates, ntimes)
    assert res.u is None

    # Check shape of class properties
    if sys.issiso():
        assert res.outputs.shape == (ntimes,)
        assert res.states.shape == (sys.nstates, ntimes)
        assert res.inputs is None
    elif res.squeeze is True:
        assert res.outputs.shape == (ntimes, )
        assert res.states.shape == (sys.nstates, ntimes)
        assert res.inputs is None
    else:
        assert res.outputs.shape == (sys.noutputs, ntimes)
        assert res.states.shape == (sys.nstates, ntimes)
        assert res.inputs is None

    #
    # Impulse and step response
    #
    for fcn in (ct.impulse_response, ct.step_response):
        res = fcn(sys, squeeze=squeeze)
        ntimes = res.time.shape[0]

        # Check shape of class members
        assert len(res.time.shape) == 1
        assert res.y.shape == (sys.noutputs, sys.ninputs, ntimes)
        assert res.x.shape == (sys.nstates, sys.ninputs, ntimes)
        assert res.u.shape == (sys.ninputs, sys.ninputs, ntimes)

        # Check shape of inputs and outputs
        if sys.issiso() and squeeze is not False:
            assert res.outputs.shape == (ntimes, )
            assert res.inputs.shape == (ntimes, )
        elif res.squeeze is True:
            assert res.outputs.shape == \
                np.empty((sys.noutputs, sys.ninputs, ntimes)).squeeze().shape
            assert res.inputs.shape == \
                np.empty((sys.ninputs, sys.ninputs, ntimes)).squeeze().shape
        else:
            assert res.outputs.shape == (sys.noutputs, sys.ninputs, ntimes)
            assert res.inputs.shape == (sys.ninputs, sys.ninputs, ntimes)

        # Check state space dimensions (not affected by squeeze)
        if sys.issiso():
            assert res.states.shape == (sys.nstates, ntimes)
        else:
            assert res.states.shape == (sys.nstates, sys.ninputs, ntimes)

    #
    # Forced response
    #
    res = ct.forced_response(sys, T, U, X0, squeeze=squeeze)
    ntimes = res.time.shape[0]

    assert len(res.time.shape) == 1
    assert res.y.shape == (sys.noutputs, ntimes)
    assert res.x.shape == (sys.nstates, ntimes)
    assert res.u.shape == (sys.ninputs, ntimes)

    if sys.issiso() and squeeze is not False:
        assert res.outputs.shape == (ntimes,)
        assert res.inputs.shape == (ntimes,)
    elif squeeze is True:
        assert res.outputs.shape == \
            np.empty((sys.noutputs, 1, ntimes)).squeeze().shape
        assert res.inputs.shape == \
            np.empty((sys.ninputs, 1, ntimes)).squeeze().shape
    else:                       # MIMO or squeeze is False
        assert res.outputs.shape == (sys.noutputs, ntimes)
        assert res.inputs.shape == (sys.ninputs, ntimes)

    # Check state space dimensions (not affected by squeeze)
    assert res.states.shape == (sys.nstates, ntimes)
