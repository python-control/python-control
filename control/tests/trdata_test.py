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
        [2,   3,  None],
        [2,   3,  True],
        [2,   3,  False],
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

    # Check dimensions of the response
    assert res.ntraces == 0     # single trace
    assert res.ninputs == 0     # no input for initial response
    assert res.noutputs == sys.noutputs
    assert res.nstates == sys.nstates

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

        # Check shape of class members
        assert res.ntraces == sys.ninputs
        assert res.ninputs == sys.ninputs
        assert res.noutputs == sys.noutputs
        assert res.nstates == sys.nstates

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

    # Check shape of class members
    assert len(res.time.shape) == 1
    assert res.y.shape == (sys.noutputs, ntimes)
    assert res.x.shape == (sys.nstates, ntimes)
    assert res.u.shape == (sys.ninputs, ntimes)

    # Check dimensions of the response
    assert res.ntraces == 0     # single trace
    assert res.ninputs == sys.ninputs
    assert res.noutputs == sys.noutputs
    assert res.nstates == sys.nstates

    # Check shape of inputs and outputs
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

    # Labels
    assert response_mimo.output_labels is None
    assert response_mimo.state_labels is None
    assert response_mimo.input_labels is None
    response = response_mimo(
        output_labels=['y1', 'y2'], input_labels='u',
        state_labels=["x[%d]" % i for i in range(4)])
    assert response.output_labels == ['y1', 'y2']
    assert response.state_labels == ['x[0]', 'x[1]', 'x[2]', 'x[3]']
    assert response.input_labels == ['u']

    # Unknown keyword
    with pytest.raises(TypeError, match="unrecognized keywords"):
        response_bad_kw = response_mimo(input=0)


def test_trdata_labels():
    # Create an I/O system with labels
    sys = ct.rss(4, 3, 2)
    iosys = ct.LinearIOSystem(sys)

    T = np.linspace(1, 10, 10)
    U = [np.sin(T), np.cos(T)]

    # Create a response
    response = ct.input_output_response(iosys, T, U)

    # Make sure the labels got created
    np.testing.assert_equal(
        response.output_labels, ["y[%d]" % i for i in range(sys.noutputs)])
    np.testing.assert_equal(
        response.state_labels, ["x[%d]" % i for i in range(sys.nstates)])
    np.testing.assert_equal(
        response.input_labels, ["u[%d]" % i for i in range(sys.ninputs)])


def test_trdata_multitrace():
    #
    # Output signal processing
    #

    # Proper call of multi-trace data w/ ambiguous 2D output
    response = ct.TimeResponseData(
        np.zeros(5), np.ones((2, 5)), np.zeros((3, 2, 5)),
        np.ones((4, 2, 5)), multi_trace=True)
    assert response.ntraces == 2
    assert response.noutputs == 1
    assert response.nstates == 3
    assert response.ninputs == 4

    # Proper call of single trace w/ ambiguous 2D output
    response = ct.TimeResponseData(
        np.zeros(5), np.ones((2, 5)), np.zeros((3, 5)),
        np.ones((4, 5)), multi_trace=False)
    assert response.ntraces == 0
    assert response.noutputs == 2
    assert response.nstates == 3
    assert response.ninputs == 4

    # Proper call of multi-trace data w/ ambiguous 1D output
    response = ct.TimeResponseData(
        np.zeros(5), np.ones(5), np.zeros((3, 5)),
        np.ones((4, 5)), multi_trace=False)
    assert response.ntraces == 0
    assert response.noutputs == 1
    assert response.nstates == 3
    assert response.ninputs == 4
    assert response.y.shape == (1, 5)           # Make sure reshape occured

    # Output vector not the right shape
    with pytest.raises(ValueError, match="Output vector is the wrong shape"):
        response = ct.TimeResponseData(
            np.zeros(5), np.ones((1, 2, 3, 5)), None, None)

    # Inconsistent output vector: different number of time points
    with pytest.raises(ValueError, match="Output vector does not match time"):
        response = ct.TimeResponseData(
            np.zeros(5), np.ones(6), np.zeros(5), np.zeros(5))

    #
    # State signal processing
    #

    # For multi-trace, state must be 3D
    with pytest.raises(ValueError, match="State vector is the wrong shape"):
        response = ct.TimeResponseData(
            np.zeros(5), np.ones((1, 5)), np.zeros((3, 5)), multi_trace=True)

    # If not multi-trace, state must be 2D
    with pytest.raises(ValueError, match="State vector is the wrong shape"):
        response = ct.TimeResponseData(
            np.zeros(5), np.ones(5), np.zeros((3, 1, 5)), multi_trace=False)

    # State vector in the wrong shape
    with pytest.raises(ValueError, match="State vector is the wrong shape"):
        response = ct.TimeResponseData(
            np.zeros(5), np.ones((1, 2, 5)), np.zeros((2, 1, 5)))

    # Inconsistent state vector: different number of time points
    with pytest.raises(ValueError, match="State vector does not match time"):
        response = ct.TimeResponseData(
            np.zeros(5), np.ones(5), np.zeros((1, 6)), np.zeros(5))

    #
    # Input signal processing
    #

    # Proper call of multi-trace data with 2D input
    response = ct.TimeResponseData(
        np.zeros(5), np.ones((2, 5)), np.zeros((3, 2, 5)),
        np.ones((2, 5)), multi_trace=True)
    assert response.ntraces == 2
    assert response.noutputs == 1
    assert response.nstates == 3
    assert response.ninputs == 1

    # Input vector in the wrong shape
    with pytest.raises(ValueError, match="Input vector is the wrong shape"):
        response = ct.TimeResponseData(
            np.zeros(5), np.ones((1, 2, 5)), None, np.zeros((2, 1, 5)))

    # Inconsistent input vector: different number of time points
    with pytest.raises(ValueError, match="Input vector does not match time"):
        response = ct.TimeResponseData(
            np.zeros(5), np.ones(5), np.zeros((1, 5)), np.zeros(6))


def test_trdata_exceptions():
    # Incorrect dimension for time vector
    with pytest.raises(ValueError, match="Time vector must be 1D"):
        ct.TimeResponseData(np.zeros((2,2)), np.zeros(2), None)

    # Infer SISO system from inputs and outputs
    response = ct.TimeResponseData(
        np.zeros(5), np.ones(5), None, np.ones(5))
    assert response.issiso

    response = ct.TimeResponseData(
        np.zeros(5), np.ones((1, 5)), None, np.ones((1, 5)))
    assert response.issiso

    response = ct.TimeResponseData(
        np.zeros(5), np.ones((1, 2, 5)), None, np.ones((1, 2, 5)))
    assert response.issiso

    # Not enough input to infer whether SISO
    with pytest.raises(ValueError, match="Can't determine if system is SISO"):
        response = ct.TimeResponseData(
            np.zeros(5), np.ones((1, 2, 5)), np.ones((4, 2, 5)), None)

    # Not enough input to infer whether SISO
    with pytest.raises(ValueError, match="Keyword `issiso` does not match"):
        response = ct.TimeResponseData(
            np.zeros(5), np.ones((2, 5)), None, np.ones((1, 5)), issiso=True)

    # Unknown squeeze keyword value
    with pytest.raises(ValueError, match="Unknown squeeze value"):
        response=ct.TimeResponseData(
            np.zeros(5), np.ones(5), None, np.ones(5), squeeze=1)

    # Legacy interface index error
    response[0], response[1], response[2]
    with pytest.raises(IndexError):
        response[3]
