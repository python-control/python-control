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
from control.timeresp import *
from control.statesp import *
from control.xferfcn import TransferFunction, _convert_to_transfer_function
from control.dtime import c2d
from control.exception import slycot_check

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
        self.siso_tf2 = _convert_to_transfer_function(self.siso_ss1)

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

        # Create discrete time systems
        self.siso_dtf1 = TransferFunction([1], [1, 1, 0.25], True)
        self.siso_dtf2 = TransferFunction([1], [1, 1, 0.25], 0.2)
        self.siso_dss1 = tf2ss(self.siso_dtf1)
        self.siso_dss2 = tf2ss(self.siso_dtf2)
        self.mimo_dss1 = StateSpace(A, B, C, D, True)
        self.mimo_dss2 = c2d(self.mimo_ss1, 0.2)

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

    def test_step_info(self):
        # From matlab docs:
        sys = TransferFunction([1,5,5],[1,1.65,5,6.5,2])
        Strue = {
            'RiseTime': 3.8456,
            'SettlingTime': 27.9762,
            'SettlingMin': 2.0689,
            'SettlingMax': 2.6873,
            'Overshoot': 7.4915,
            'Undershoot': 0,
            'Peak': 2.6873,
            'PeakTime': 8.0530
        }

        S = step_info(sys)

        # Very arbitrary tolerance because I don't know if the
        # response from the MATLAB is really that accurate.
        # maybe it is a good idea to change the Strue to match
        # but I didn't do it because I don't know if it is
        # accurate either...
        rtol = 2e-2
        np.testing.assert_allclose(
            S.get('RiseTime'),
            Strue.get('RiseTime'),
            rtol=rtol)
        np.testing.assert_allclose(
            S.get('SettlingTime'),
            Strue.get('SettlingTime'),
            rtol=rtol)
        np.testing.assert_allclose(
            S.get('SettlingMin'),
            Strue.get('SettlingMin'),
            rtol=rtol)
        np.testing.assert_allclose(
            S.get('SettlingMax'),
            Strue.get('SettlingMax'),
            rtol=rtol)
        np.testing.assert_allclose(
            S.get('Overshoot'),
            Strue.get('Overshoot'),
            rtol=rtol)
        np.testing.assert_allclose(
            S.get('Undershoot'),
            Strue.get('Undershoot'),
            rtol=rtol)
        np.testing.assert_allclose(
            S.get('Peak'),
            Strue.get('Peak'),
            rtol=rtol)
        np.testing.assert_allclose(
            S.get('PeakTime'),
            Strue.get('PeakTime'),
            rtol=rtol)
        np.testing.assert_allclose(
            S.get('SteadyStateValue'),
            2.50,
            rtol=rtol)

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
        x0 = np.array([[.5], [1], [0], [0]])
        youttrue = np.array([[11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092,
                              1.1508, 0.5833, 0.1645, -0.1391],
                             [9., 17.6457, 24.7072, 30.4855, 35.2234, 39.1165,
                              42.3227, 44.9694, 47.1599, 48.9776]])
        _t, yout, _xout = forced_response(self.mimo_ss1, t, u, x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)

        # Test discrete MIMO system to use correct convention for input
        sysc = self.mimo_ss1
        dt=t[1]-t[0]
        sysd = c2d(sysc, dt)           # discrete time system
        Tc, youtc, _xoutc = forced_response(sysc, t, u, x0)
        Td, youtd, _xoutd = forced_response(sysd, t, u, x0)
        np.testing.assert_array_equal(Tc.shape, Td.shape)
        np.testing.assert_array_equal(youtc.shape, youtd.shape)
        np.testing.assert_array_almost_equal(youtc, youtd, decimal=4)

        # Test discrete MIMO system without default T argument
        u = np.array([[0., 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1., 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        x0 = np.array([[.5], [1], [0], [0]])
        youttrue = np.array([[11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092,
                              1.1508, 0.5833, 0.1645, -0.1391],
                             [9., 17.6457, 24.7072, 30.4855, 35.2234, 39.1165,
                              42.3227, 44.9694, 47.1599, 48.9776]])
        _t, yout, _xout = forced_response(sysd, U=u, X0=x0)
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
        np.testing.assert_array_equal(yout, [0., 1., 0., 0.])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_step_robustness(self):
        "Unit test: https://github.com/python-control/python-control/issues/240"
        # Create 2 input, 2 output system
        num =  [ [[0], [1]],           [[1],   [0]] ]
        
        den1 = [ [[1], [1,1]],         [[1,4], [1]] ]
        sys1 = TransferFunction(num, den1)

        den2 = [ [[1], [1e-10, 1, 1]], [[1,4], [1]] ]   # slight perturbation
        sys2 = TransferFunction(num, den2)

        # Compute step response from input 1 to output 1, 2
        t1, y1 = step_response(sys1, input=0)
        t2, y2 = step_response(sys2, input=0)
        np.testing.assert_array_almost_equal(y1, y2)

    def test_time_vector(self):
        "Unit test: https://github.com/python-control/python-control/issues/239"
        # Discrete time simulations with specified time vectors
        Tin1 = np.arange(0, 5, 1)       # matches dtf1, dss1; multiple of 0.2
        Tin2 = np.arange(0, 5, 0.2)     # matches dtf2, dss2
        Tin3 = np.arange(0, 5, 0.5)     # incompatible with 0.2

        # Initial conditions to use for the different systems
        siso_x0 = [1, 2]
        mimo_x0 = [1, 2, 3, 4]

        #
        # Easy cases: make sure that output sample time matches input
        #
        # No timebase in system => output should match input
        #
        # Initial response
        tout, yout = initial_response(self.siso_dtf1, Tin2, siso_x0,
                                      squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin2)

        # Impulse response
        tout, yout = impulse_response(self.siso_dtf1, Tin2,
                                      squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin2)

        # Step response
        tout, yout = step_response(self.siso_dtf1, Tin2,
                                   squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin2)

        # Forced response with specified time vector
        tout, yout, xout = forced_response(self.siso_dtf1, Tin2, np.sin(Tin2),
                                           squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin2)

        # Forced response with no time vector, no sample time (should use 1)
        tout, yout, xout = forced_response(self.siso_dtf1, None, np.sin(Tin1),
                                           squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin1)

        # MIMO forced response
        tout, yout, xout = forced_response(self.mimo_dss1, Tin1, 
                                           (np.sin(Tin1), np.cos(Tin1)),
                                           mimo_x0)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        self.assertEqual(np.shape(tout), np.shape(yout[1,:]))
        np.testing.assert_array_equal(tout, Tin1)

        # Matching timebase in system => output should match input
        #
        # Initial response
        tout, yout = initial_response(self.siso_dtf2, Tin2, siso_x0,
                                      squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin2)

        # Impulse response
        tout, yout = impulse_response(self.siso_dtf2, Tin2,
                                      squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin2)

        # Step response
        tout, yout = step_response(self.siso_dtf2, Tin2,
                                   squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin2)

        # Forced response
        tout, yout, xout = forced_response(self.siso_dtf2, Tin2, np.sin(Tin2),
                                           squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin2)

        # Forced response with no time vector, use sample time
        tout, yout, xout = forced_response(self.siso_dtf2, None, np.sin(Tin2),
                                           squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin2)

        # Compatible timebase in system => output should match input
        #
        # Initial response
        tout, yout = initial_response(self.siso_dtf2, Tin1, siso_x0,
                                      squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin1)

        # Impulse response
        tout, yout = impulse_response(self.siso_dtf2, Tin1,
                                      squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin1)

        # Step response
        tout, yout = step_response(self.siso_dtf2, Tin1,
                                   squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin1)

        # Forced response
        tout, yout, xout = forced_response(self.siso_dtf2, Tin1, np.sin(Tin1),
                                           squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        np.testing.assert_array_equal(tout, Tin1)

        #
        # Interpolation of the input (to match scipy.signal.dlsim)
        #
        # Initial response
        tout, yout, xout = forced_response(self.siso_dtf2, Tin1,
                                           np.sin(Tin1), interpolate=True,
                                           squeeze=False)
        self.assertEqual(np.shape(tout), np.shape(yout[0,:]))
        self.assertTrue(np.allclose(tout[1:] - tout[:-1],  self.siso_dtf2.dt))

        #
        # Incompatible cases: make sure an error is thrown
        #
        # System timebase and given time vector are incompatible
        #
        # Initial response
        with self.assertRaises(Exception) as context:
            tout, yout = initial_response(self.siso_dtf2, Tin3, siso_x0,
                                          squeeze=False)
        self.assertTrue(isinstance(context.exception, ValueError))

    def test_discrete_time_steps(self):
        """Make sure rounding errors in sample time are handled properly"""
        # See https://github.com/python-control/python-control/issues/332)
        #
        # These tests play around with the input time vector to make sure that
        # small rounding errors don't generate spurious errors.

        # Discrete time system to use for simulation
        # self.siso_dtf2 = TransferFunction([1], [1, 1, 0.25], 0.2)

        # Set up a time range and simulate
        T = np.arange(0, 100, 0.2)
        tout1, yout1 = step_response(self.siso_dtf2, T)

        # Simulate every other time step
        T = np.arange(0, 100, 0.4)
        tout2, yout2 = step_response(self.siso_dtf2, T)
        np.testing.assert_array_almost_equal(tout1[::2], tout2)
        np.testing.assert_array_almost_equal(yout1[::2], yout2)

        # Add a small error into some of the time steps
        T = np.arange(0, 100, 0.2)
        T[1:-2:2] -= 1e-12      # tweak second value and a few others
        tout3, yout3 = step_response(self.siso_dtf2, T)
        np.testing.assert_array_almost_equal(tout1, tout3)
        np.testing.assert_array_almost_equal(yout1, yout3)

        # Add a small error into some of the time steps (w/ skipping)
        T = np.arange(0, 100, 0.4)
        T[1:-2:2] -= 1e-12      # tweak second value and a few others
        tout4, yout4 = step_response(self.siso_dtf2, T)
        np.testing.assert_array_almost_equal(tout2, tout4)
        np.testing.assert_array_almost_equal(yout2, yout4)

        # Make sure larger errors *do* generate an error
        T = np.arange(0, 100, 0.2)
        T[1:-2:2] -= 1e-3      # change second value and a few others
        self.assertRaises(ValueError, step_response, self.siso_dtf2, T)

    def test_time_series_data_convention(self):
        """Make sure time series data matches documentation conventions"""
        # SISO continuous time
        t, y = step_response(self.siso_ss1)
        self.assertTrue(isinstance(t, np.ndarray)
                        and not isinstance(t, np.matrix))
        self.assertTrue(len(t.shape) == 1)
        self.assertTrue(len(y.shape) == 1) # SISO returns "scalar" output
        self.assertTrue(len(t) == len(y))  # Allows direct plotting of output

        # SISO discrete time
        t, y = step_response(self.siso_dss1)
        self.assertTrue(isinstance(t, np.ndarray)
                        and not isinstance(t, np.matrix))
        self.assertTrue(len(t.shape) == 1)
        self.assertTrue(len(y.shape) == 1) # SISO returns "scalar" output
        self.assertTrue(len(t) == len(y))  # Allows direct plotting of output

        # MIMO continuous time
        tin = np.linspace(0, 10, 100)
        uin = [np.sin(tin), np.cos(tin)]
        t, y, x = forced_response(self.mimo_ss1, tin, uin)
        self.assertTrue(isinstance(t, np.ndarray)
                        and not isinstance(t, np.matrix))
        self.assertTrue(len(t.shape) == 1)
        self.assertTrue(len(y[0].shape) == 1)
        self.assertTrue(len(y[1].shape) == 1)
        self.assertTrue(len(t) == len(y[0]))
        self.assertTrue(len(t) == len(y[1]))

        # MIMO discrete time
        tin = np.linspace(0, 10, 100)
        uin = [np.sin(tin), np.cos(tin)]
        t, y, x = forced_response(self.mimo_dss1, tin, uin)
        self.assertTrue(isinstance(t, np.ndarray)
                        and not isinstance(t, np.matrix))
        self.assertTrue(len(t.shape) == 1)
        self.assertTrue(len(y[0].shape) == 1)
        self.assertTrue(len(y[1].shape) == 1)
        self.assertTrue(len(t) == len(y[0]))
        self.assertTrue(len(t) == len(y[1]))

        # Allow input time as 2D array (output should be 1D)
        tin = np.array(np.linspace(0, 10, 100), ndmin=2)
        t, y = step_response(self.siso_ss1, tin)
        self.assertTrue(isinstance(t, np.ndarray)
                        and not isinstance(t, np.matrix))
        self.assertTrue(len(t.shape) == 1)
        self.assertTrue(len(y.shape) == 1) # SISO returns "scalar" output
        self.assertTrue(len(t) == len(y))  # Allows direct plotting of output


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestTimeresp)

if __name__ == '__main__':
    unittest.main()
