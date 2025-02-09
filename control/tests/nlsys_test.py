"""nlsys_test.py - test nonlinear input/output system operations

RMM, 18 Jun 2022

This test suite checks various newer functions for NonlinearIOSystems.
The main test functions are contained in iosys_test.py.

"""

import math
import re

import numpy as np
import pytest

import control as ct


# Basic test of nlsys()
def test_nlsys_basic():
    def kincar_update(t, x, u, params):
        l = params['l']              # wheelbase
        return np.array([
            np.cos(x[2]) * u[0],     # x velocity
            np.sin(x[2]) * u[0],     # y velocity
            np.tan(u[1]) * u[0] / l  # angular velocity
        ])

    def kincar_output(t, x, u, params):
        return x[0:2]  # x, y position

    kincar = ct.nlsys(
        kincar_update, kincar_output,
        states=['x', 'y', 'theta'],
        inputs=2, input_prefix='U',
        outputs=2, params={'l': 1})
    assert kincar.input_labels == ['U[0]', 'U[1]']
    assert kincar.output_labels == ['y[0]', 'y[1]']
    assert kincar.state_labels == ['x', 'y', 'theta']
    assert kincar.params == {'l': 1}


# Test nonlinear initial, step, and forced response
@pytest.mark.parametrize(
    "nin, nout, input, output", [
        ( 1,    1,  None,   None),
        ( 2,    2,  None,   None),
        ( 2,    2,     0,   None),
        ( 2,    2,  None,      1),
        ( 2,    2,     1,      0),
    ])
def test_lti_nlsys_response(nin, nout, input, output):
    sys_ss = ct.rss(4, nin, nout, strictly_proper=True)
    sys_ss.A = np.diag([-1, -2, -3, -4])        # avoid random numerical errors
    sys_nl = ct.nlsys(
        lambda t, x, u, params: sys_ss.A @ x + sys_ss.B @ u,
        lambda t, x, u, params: sys_ss.C @ x + sys_ss.D @ u,
        inputs=nin, outputs=nout, states=4)

    # Figure out the time to use from the linear impulse response
    resp_ss = ct.impulse_response(sys_ss)
    timepts = np.linspace(0, resp_ss.time[-1]/10, 100)

    # Initial response
    resp_ss = ct.initial_response(sys_ss, timepts, output=output)
    resp_nl = ct.initial_response(sys_nl, timepts, output=output)
    np.testing.assert_equal(resp_ss.time, resp_nl.time)
    np.testing.assert_allclose(resp_ss.states, resp_nl.states, atol=0.01)

    # Step response
    resp_ss = ct.step_response(sys_ss, timepts, input=input, output=output)
    resp_nl = ct.step_response(sys_nl, timepts, input=input, output=output)
    np.testing.assert_equal(resp_ss.time, resp_nl.time)
    np.testing.assert_allclose(resp_ss.states, resp_nl.states, atol=0.01)

    # Forced response
    X0 = np.linspace(0, 1, sys_ss.nstates)
    U = np.zeros((nin, timepts.size))
    for i in range(nin):
        U[i] = 0.01 * np.sin(timepts + i)
    resp_ss = ct.forced_response(sys_ss, timepts, U, X0=X0)
    resp_nl = ct.forced_response(sys_nl, timepts, U, X0=X0)
    np.testing.assert_equal(resp_ss.time, resp_nl.time)
    np.testing.assert_allclose(resp_ss.states, resp_nl.states, atol=0.05)


# Test to make sure that impulse responses are not allowed
def test_nlsys_impulse():
    sys_ss = ct.rss(4, 1, 1, strictly_proper=True)
    sys_nl = ct.nlsys(
        lambda t, x, u, params: sys_ss.A @ x + sys_ss.B @ u,
        lambda t, x, u, params: sys_ss.C @ x + sys_ss.D @ u,
        inputs=1, outputs=1, states=4)

    # Figure out the time to use from the linear impulse response
    resp_ss = ct.impulse_response(sys_ss)
    timepts = np.linspace(0, resp_ss.time[-1]/10, 100)

    # Impulse_response (not implemented)
    with pytest.raises(ValueError, match="system must be LTI"):
        ct.impulse_response(sys_nl, timepts)


# Test nonlinear systems that are missing inputs or outputs
def test_nlsys_empty_io():

    # No inputs
    sys_nl = ct.nlsys(
        lambda t, x, u, params: -x, lambda t, x, u, params: x[0:2],
        name="no inputs", states=3, inputs=0, outputs=2)
    P = sys_nl.linearize(np.zeros(sys_nl.nstates), None)
    assert P.A.shape == (3, 3)
    assert P.B.shape == (3, 0)
    assert P.C.shape == (2, 3)
    assert P.D.shape == (2, 0)

    # Check that we can compute dynamics and outputs
    x = np.array([1, 2, 3])
    np.testing.assert_equal(sys_nl.dynamics(0, x, None, {}), -x)
    np.testing.assert_equal(P.dynamics(0, x, None), -x)
    np.testing.assert_equal(sys_nl.output(0, x, None, {}), x[0:2])
    np.testing.assert_equal(P.output(0, x, None), x[0:2])

    # Make sure initial response runs OK
    resp = ct.initial_response(sys_nl, np.linspace(0, 1), x)
    np.testing.assert_allclose(
        resp.states[:, -1], x * math.exp(-1), atol=1e-3, rtol=1e-3)

    resp = ct.initial_response(P, np.linspace(0, 1), x)
    np.testing.assert_allclose(resp.states[:, -1], x * math.exp(-1))

    # No outputs
    sys_nl = ct.nlsys(
        lambda t, x, u, params: -x + np.array([1, 1, 1]) * u[0], None,
        name="no outputs", states=3, inputs=1, outputs=0)
    P = sys_nl.linearize(np.zeros(sys_nl.nstates), 0)
    assert P.A.shape == (3, 3)
    assert P.B.shape == (3, 1)
    assert P.C.shape == (0, 3)
    assert P.D.shape == (0, 1)

    # Check that we can compute dynamics
    x = np.array([1, 2, 3])
    np.testing.assert_equal(sys_nl.dynamics(0, x, 1, {}), -x + 1)
    np.testing.assert_equal(P.dynamics(0, x, 1), -x + 1)

    # Make sure initial response runs OK
    resp = ct.initial_response(sys_nl, np.linspace(0, 1), x)
    np.testing.assert_allclose(
        resp.states[:, -1], x * math.exp(-1), atol=1e-3, rtol=1e-3)

    resp = ct.initial_response(P, np.linspace(0, 1), x)
    np.testing.assert_allclose(resp.states[:, -1], x * math.exp(-1))

    # Make sure forced response runs OK
    resp = ct.forced_response(sys_nl, np.linspace(0, 1), 1)
    np.testing.assert_allclose(
        resp.states[:, -1], 1 - math.exp(-1), atol=1e-3, rtol=1e-3)

    resp = ct.forced_response(P, np.linspace(0, 1), 1)
    np.testing.assert_allclose(resp.states[:, -1], 1 - math.exp(-1))


def test_ss2io():
    sys = ct.rss(
        states=4, inputs=['u1', 'u2'], outputs=['y1', 'y2'], name='sys')

    # Standard conversion
    nlsys = ct.nlsys(sys)
    for attr in ['nstates', 'ninputs', 'noutputs']:
        assert getattr(nlsys, attr) == getattr(sys, attr)
    assert nlsys.name == 'sys$converted'
    np.testing.assert_allclose(
        nlsys.dynamics(0, [1, 2, 3, 4], [0, 0], {}),
        sys.A @ np.array([1, 2, 3, 4]))

    # Put names back to defaults
    nlsys = ct.nlsys(
        sys, inputs=sys.ninputs, outputs=sys.noutputs, states=sys.nstates)
    for attr, prefix in zip(
            ['state_labels', 'input_labels', 'output_labels'],
            ['x', 'u', 'y']):
        for i in range(len(getattr(nlsys, attr))):
            assert getattr(nlsys, attr)[i] == f"{prefix}[{i}]"
    assert re.match(r"sys\$converted", nlsys.name)

    # Override the names with something new
    nlsys = ct.nlsys(
        sys, inputs=['U1', 'U2'], outputs=['Y1', 'Y2'],
        states=['X1', 'X2', 'X3', 'X4'], name='nlsys')
    for attr, prefix in zip(
            ['state_labels', 'input_labels', 'output_labels'],
            ['X', 'U', 'Y']):
        for i in range(len(getattr(nlsys, attr))):
            assert getattr(nlsys, attr)[i] == f"{prefix}{i+1}"
    assert nlsys.name == 'nlsys'

    # Make sure dimension checking works
    for attr in ['states', 'inputs', 'outputs']:
        with pytest.raises(ValueError, match=r"new .* doesn't match"):
            kwargs = {attr: getattr(sys, 'n' + attr) - 1}
            nlsys = ct.nlsys(sys, **kwargs)


def test_ICsystem_str():
    sys1 = ct.rss(2, 2, 3, name='sys1', strictly_proper=True)
    sys2 = ct.rss(2, 3, 2, name='sys2', strictly_proper=True)

    with pytest.warns(UserWarning, match="Unused") as record:
        sys = ct.interconnect(
            [sys1, sys2], inputs=['r1', 'r2'], outputs=['y1', 'y2'],
            connections=[
                ['sys1.u[0]', '-sys2.y[0]', 'sys2.y[1]'],
                ['sys1.u[1]', 'sys2.y[0]', '-sys2.y[1]'],
                ['sys2.u[0]', 'sys2.y[0]', (0, 0, -1)],
                ['sys2.u[1]', (1, 1, -2), (0, 1, -2)],
            ],
            inplist=['sys1.u[0]', 'sys1.u[1]'],
            outlist=['sys2.y[0]', 'sys2.y[1]'])
    assert len(record) == 2
    assert str(record[0].message).startswith("Unused input")
    assert str(record[1].message).startswith("Unused output")

    ref = \
        r"<LinearICSystem>: sys\[[\d]+\]" + "\n" + \
        r"Inputs \(2\): \['r1', 'r2'\]" + "\n" + \
        r"Outputs \(2\): \['y1', 'y2'\]" + "\n" + \
        r"States \(4\): \['sys1_x\[0\].*'sys2_x\[1\]'\]" + "\n" + \
        "\n" + \
        r"Subsystems \(2\):" + "\n" + \
        r" \* <StateSpace sys1: \[.*\] -> \['y\[0\]', 'y\[1\]']>" + "\n" + \
        r" \* <StateSpace sys2: \['u\[0\]', 'u\[1\]'] -> \[.*\]>" + "\n" + \
        "\n" + \
        r"Connections:" + "\n" + \
        r" \* sys1.u\[0\] <- -sys2.y\[0\] \+ sys2.y\[1\] \+ r1" + "\n" + \
        r" \* sys1.u\[1\] <- sys2.y\[0\] - sys2.y\[1\] \+ r2" + "\n" + \
        r" \* sys1.u\[2\] <-" + "\n" + \
        r" \* sys2.u\[0\] <- -sys1.y\[0\] \+ sys2.y\[0\]" + "\n" + \
        r" \* sys2.u\[1\] <- -2.0 \* sys1.y\[1\] - 2.0 \* sys2.y\[1\]" + \
        "\n\n" + \
        r"Outputs:" + "\n" + \
        r" \* y1 <- sys2.y\[0\]" + "\n" + \
        r" \* y2 <- sys2.y\[1\]" + \
        "\n\n" + \
        r"A = \[\[.*\]\]" + "\n\n" + \
        r"B = \[\[.*\]\]" + "\n\n" + \
        r"C = \[\[.*\]\]" + "\n\n" + \
        r"D = \[\[.*\]\]"

    assert re.match(ref, str(sys), re.DOTALL)


# Make sure nlsys str() works as expected
@pytest.mark.parametrize("params, expected", [
    ({}, r"States \(1\): \['x\[0\]'\]" + "\n\n"),
    ({'a': 1}, r"States \(1\): \['x\[0\]'\]" + "\n" +
     r"Parameters: \['a'\]" + "\n\n"),
    ({'a': 1, 'b': 1}, r"States \(1\): \['x\[0\]'\]" + "\n" +
     r"Parameters: \['a', 'b'\]" + "\n\n"),
])
def test_nlsys_params_str(params, expected):
    sys = ct.nlsys(
            lambda t, x, u, params: -x, inputs=1, outputs=1, states=1,
            params=params)
    out = str(sys)

    assert re.search(expected, out) is not None
