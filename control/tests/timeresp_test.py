#!/usr/bin/env python
#
# timeresp_test.py - test time response functions
# RMM, 17 Jun 2011 (based on TestMatlab from v0.4c)
#
# This test suite just goes through and calls all of the MATLAB
# functions using different systems and arguments to make sure that
# nothing crashes.  It doesn't test actual functionality; the module
# specific unit tests will do that.

import unittest
import numpy as np
# import scipy as sp
from control.timeresp import *
from control.statesp import *
from control.xferfcn import TransferFunction, _convertToTransferFunction
from control.dtime import c2d

class TestTimeresp(unittest.TestCase):
    def setUp(self):
        """Set up some systems for testing out MATLAB functions"""
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5.; 7.")
        C = np.matrix("6. 8.")
        D = np.matrix("9.")
        self.siso_ss1 = StateSpace(A, B, C, D)

        # Create some transfer functions
        self.siso_tf1 = TransferFunction([1], [1, 2, 1])
        self.siso_tf2 = _convertToTransferFunction(self.siso_ss1)

        # Create MIMO system, contains ``siso_ss1`` twice
        A = np.matrix("1. -2. 0.  0.;"
                      "3. -4. 0.  0.;"
                      "0.  0. 1. -2.;"
                      "0.  0. 3. -4. ")
        B = np.matrix("5. 0.;"
                      "7. 0.;"
                      "0. 5.;"
                      "0. 7. ")
        C = np.matrix("6. 8. 0. 0.;"
                      "0. 0. 6. 8. ")
        D = np.matrix("9. 0.;"
                      "0. 9. ")
        self.mimo_ss1 = StateSpace(A, B, C, D)

    def test_step_response(self):
        # Test SISO system
        sys = self.siso_ss1
        t = np.linspace(0, 1, 10)
        youttrue = np.array([9., 17.6457, 24.7072, 30.4855, 35.2234, 39.1165,
                             42.3227, 44.9694, 47.1599, 48.9776])

        # SISO call
        tout, yout = step_response(sys, T=t)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # Play with arguments
        tout, yout = step_response(sys, T=t, X0=0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        X0 = np.array([0, 0])
        tout, yout = step_response(sys, T=t, X0=X0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        tout, yout, xout = step_response(sys, T=t, X0=0, return_x=True)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # Test MIMO system, which contains ``siso_ss1`` twice
        sys = self.mimo_ss1
        _t, y_00 = step_response(sys, T=t, input=0, output=0)
        _t, y_11 = step_response(sys, T=t, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)

        # Make sure continuous and discrete time use same return conventions
        sysc = self.mimo_ss1
        sysd = c2d(sysc, 1)           # discrete time system
        Tvec = np.linspace(0, 10, 11) # make sure to use integer times 0..10
        Tc, youtc = step_response(sysc, Tvec, input=0)
        Td, youtd = step_response(sysd, Tvec, input=0)
        np.testing.assert_array_equal(Tc.shape, Td.shape)
        np.testing.assert_array_equal(youtc.shape, youtd.shape)

    def test_impulse_response(self):
        # Test SISO system
        sys = self.siso_ss1
        t = np.linspace(0, 1, 10)
        youttrue = np.array([86., 70.1808, 57.3753, 46.9975, 38.5766, 31.7344,
                             26.1668, 21.6292, 17.9245, 14.8945])
        tout, yout = impulse_response(sys, T=t)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # Play with arguments
        tout, yout = impulse_response(sys, T=t, X0=0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        X0 = np.array([0, 0])
        tout, yout = impulse_response(sys, T=t, X0=X0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        tout, yout, xout = impulse_response(sys, T=t, X0=0, return_x=True)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # Test MIMO system, which contains ``siso_ss1`` twice
        sys = self.mimo_ss1
        _t, y_00 = impulse_response(sys, T=t, input=0, output=0)
        _t, y_11 = impulse_response(sys, T=t, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)

        # Test MIMO system, as mimo, and don't trim outputs
        sys = self.mimo_ss1
        _t, yy = impulse_response(sys, T=t, input=0)
        np.testing.assert_array_almost_equal(
            yy, np.vstack((youttrue, np.zeros_like(youttrue))), decimal=4)

    def test_initial_response(self):
        # Test SISO system
        sys = self.siso_ss1
        t = np.linspace(0, 1, 10)
        x0 = np.array([[0.5], [1]])
        youttrue = np.array([11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092,
                             1.1508, 0.5833, 0.1645, -0.1391])
        tout, yout = initial_response(sys, T=t, X0=x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # Play with arguments
        tout, yout, xout = initial_response(sys, T=t, X0=x0, return_x=True)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # Test MIMO system, which contains ``siso_ss1`` twice
        sys = self.mimo_ss1
        x0 = np.matrix(".5; 1.; .5; 1.")
        _t, y_00 = initial_response(sys, T=t, X0=x0, input=0, output=0)
        _t, y_11 = initial_response(sys, T=t, X0=x0, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)

    def test_initial_response_no_trim(self):
        # test MIMO system without trimming
        t = np.linspace(0, 1, 10)
        x0 = np.matrix(".5; 1.; .5; 1.")
        youttrue = np.array([11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092,
                             1.1508, 0.5833, 0.1645, -0.1391])
        sys = self.mimo_ss1
        _t, yy = initial_response(sys, T=t, X0=x0)
        np.testing.assert_array_almost_equal(
            yy, np.vstack((youttrue, youttrue)),
            decimal=4)

    def test_forced_response(self):
        t = np.linspace(0, 1, 10)

        # compute step response - test with state space, and transfer function
        # objects
        u = np.array([1., 1, 1, 1, 1, 1, 1, 1, 1, 1])
        youttrue = np.array([9., 17.6457, 24.7072, 30.4855, 35.2234, 39.1165,
                             42.3227, 44.9694, 47.1599, 48.9776])
        tout, yout, _xout = forced_response(self.siso_ss1, t, u)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)
        _t, yout, _xout = forced_response(self.siso_tf2, t, u)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)

        # test with initial value and special algorithm for ``U=0``
        u = 0
        x0 = np.matrix(".5; 1.")
        youttrue = np.array([11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092,
                             1.1508, 0.5833, 0.1645, -0.1391])
        _t, yout, _xout = forced_response(self.siso_ss1, t, u, x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)

        # Test MIMO system, which contains ``siso_ss1`` twice
        # first system: initial value, second system: step response
        u = np.array([[0., 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1., 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        x0 = np.matrix(".5; 1; 0; 0")
        youttrue = np.array([[11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092,
                              1.1508, 0.5833, 0.1645, -0.1391],
                             [9., 17.6457, 24.7072, 30.4855, 35.2234, 39.1165,
                              42.3227, 44.9694, 47.1599, 48.9776]])
        _t, yout, _xout = forced_response(self.mimo_ss1, t, u, x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)

    def test_lsim_double_integrator(self):
        # Note: scipy.signal.lsim fails if A is not invertible
        A = np.mat("0. 1.;0. 0.")
        B = np.mat("0.; 1.")
        C = np.mat("1. 0.")
        D = 0.
        sys = StateSpace(A, B, C, D)

        def check(u, x0, xtrue):
            _t, yout, xout = forced_response(sys, t, u, x0)
            np.testing.assert_array_almost_equal(xout, xtrue, decimal=6)
            ytrue = np.squeeze(np.asarray(C.dot(xtrue)))
            np.testing.assert_array_almost_equal(yout, ytrue, decimal=6)

        # test with zero input
        npts = 10
        t = np.linspace(0, 1, npts)
        u = np.zeros_like(t)
        x0 = np.array([2., 3.])
        xtrue = np.zeros((2, npts))
        xtrue[0, :] = x0[0] + t * x0[1]
        xtrue[1, :] = x0[1]
        check(u, x0, xtrue)

        # test with step input
        u = np.ones_like(t)
        xtrue = np.array([0.5 * t**2, t])
        x0 = np.array([0., 0.])
        check(u, x0, xtrue)

        # test with linear input
        u = t
        xtrue = np.array([1./6. * t**3, 0.5 * t**2])
        check(u, x0, xtrue)

    def test_discrete_initial(self):
        h1 = TransferFunction([1.], [1., 0.], 1.)
        t, yout = impulse_response(h1, np.arange(4))
        np.testing.assert_array_equal(yout[0], [0., 1., 0., 0.])

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestTimeresp)

if __name__ == '__main__':
    unittest.main()
