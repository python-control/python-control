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
        assert res._legacy_states.shape == (sys.nstates, ntimes)
        assert res.states.shape == (sys.nstates, ntimes)
        assert res.inputs is None
    elif res.squeeze is True:
        assert res.outputs.shape == (ntimes, )
        assert res._legacy_states.shape == (sys.nstates, ntimes)
        assert res.states.shape == (sys.nstates, ntimes)
        assert res.inputs is None
    else:
        assert res.outputs.shape == (sys.noutputs, ntimes)
        assert res._legacy_states.shape == (sys.nstates, ntimes)
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
            assert res.states.shape == (sys.nstates, ntimes)
            assert res.inputs.shape == (ntimes, )
        elif res.squeeze is True:
            assert res.outputs.shape == \
                np.empty((sys.noutputs, sys.ninputs, ntimes)).squeeze().shape
            assert res.states.shape == \
                np.empty((sys.nstates, sys.ninputs, ntimes)).squeeze().shape
            assert res.inputs.shape == \
                np.empty((sys.ninputs, sys.ninputs, ntimes)).squeeze().shape
        else:
            assert res.outputs.shape == (sys.noutputs, sys.ninputs, ntimes)
            assert res.states.shape == (sys.nstates, sys.ninputs, ntimes)
            assert res.inputs.shape == (sys.ninputs, sys.ninputs, ntimes)

        # Check legacy state space dimensions (not affected by squeeze)
        if sys.issiso():
            assert res._legacy_states.shape == (sys.nstates, ntimes)
        else:
            assert res._legacy_states.shape == \
                (sys.nstates, sys.ninputs, ntimes)

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
        assert res.states.shape == (sys.nstates, ntimes)
        assert res.inputs.shape == (ntimes,)
    elif squeeze is True:
        assert res.outputs.shape == \
            np.empty((sys.noutputs, 1, ntimes)).squeeze().shape
        assert res.states.shape == \
            np.empty((sys.nstates, 1, ntimes)).squeeze().shape
        assert res.inputs.shape == \
            np.empty((sys.ninputs, 1, ntimes)).squeeze().shape
    else:                       # MIMO or squeeze is False
        assert res.outputs.shape == (sys.noutputs, ntimes)
        assert res.states.shape == (sys.nstates, ntimes)
        assert res.inputs.shape == (sys.ninputs, ntimes)

    # Check state space dimensions (not affected by squeeze)
    assert res.states.shape == (sys.nstates, ntimes)


def test_response_copy():
    # Generate some initial data to use
    sys_siso = ct.rss(4, 1, 1)
    response_siso = ct.step_response(sys_siso)
    siso_ntimes = response_siso.time.size

    sys_mimo = ct.rss(4, 2, 1)
    response_mimo = ct.step_response(sys_mimo)
    mimo_ntimes = response_mimo.time.size

    # Transpose
    response_mimo_transpose = response_mimo(transpose=True)
    assert response_mimo.outputs.shape == (2, 1, mimo_ntimes)
    assert response_mimo_transpose.outputs.shape == (mimo_ntimes, 2, 1)
    assert response_mimo.states.shape == (4, 1, mimo_ntimes)
    assert response_mimo_transpose.states.shape == (mimo_ntimes, 4, 1)

    # Squeeze
    response_siso_as_mimo = response_siso(squeeze=False)
    assert response_siso_as_mimo.outputs.shape == (1, 1, siso_ntimes)
    assert response_siso_as_mimo.states.shape == (4, 1, siso_ntimes)
    assert response_siso_as_mimo._legacy_states.shape == (4, siso_ntimes)

    response_mimo_squeezed = response_mimo(squeeze=True)
    assert response_mimo_squeezed.outputs.shape == (2, mimo_ntimes)
    assert response_mimo_squeezed.states.shape == (4, mimo_ntimes)
    assert response_mimo_squeezed._legacy_states.shape == (4, 1, mimo_ntimes)

    # Squeeze and transpose
    response_mimo_sqtr = response_mimo(squeeze=True, transpose=True)
    assert response_mimo_sqtr.outputs.shape == (mimo_ntimes, 2)
    assert response_mimo_sqtr.states.shape == (mimo_ntimes, 4)
    assert response_mimo_sqtr._legacy_states.shape == (mimo_ntimes, 4, 1)

    # Return_x
    t, y = response_mimo
    t, y = response_mimo()
    t, y, x = response_mimo(return_x=True)
    with pytest.raises(ValueError, match="too many"):
        t, y = response_mimo(return_x=True)
    with pytest.raises(ValueError, match="not enough"):
        t, y, x = response_mimo

    # Unknown keyword
    with pytest.raises(ValueError, match="unknown"):
        response_bad_kw = response_mimo(input=0)
