"""nlsys_test.py - test nonlinear input/output system operations

RMM, 18 Jun 2022

This test suite checks various newer functions for NonlinearIOSystems.
The main test functions are contained in iosys_test.py.

"""

import pytest
import numpy as np
import control as ct

# Basic test of nlsys()
def test_nlsys_basic():
    def kincar_update(t, x, u, params):
        l = params.get('l', 1)  # wheelbase
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
        outputs=2)
    assert kincar.input_labels == ['U[0]', 'U[1]']
    assert kincar.output_labels == ['y[0]', 'y[1]']
    assert kincar.state_labels == ['x', 'y', 'theta']


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
