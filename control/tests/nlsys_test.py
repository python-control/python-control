"""nlsys_test.py - test nonlinear input/output system operations

RMM, 18 Jun 2022

This test suite checks various newer functions for NonlinearIOSystems.
The main test functions are contained in iosys_test.py.

"""

import pytest
import numpy as np
import control as ct

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
