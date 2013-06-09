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
import scipy as sp
from control.timeresp import *
from control.statesp import *
from control.xferfcn import TransferFunction, _convertToTransferFunction

class TestTimeresp(unittest.TestCase):
    def setUp(self):
        """Set up some systems for testing out MATLAB functions"""
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5.; 7.")
        C = np.matrix("6. 8.")
        D = np.matrix("9.")
        self.siso_ss1 = StateSpace(A,B,C,D)

        # Create some transfer functions
        self.siso_tf1 = TransferFunction([1], [1, 2, 1]);
        self.siso_tf2 = _convertToTransferFunction(self.siso_ss1);

        #Create MIMO system, contains ``siso_ss1`` twice
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
        #Test SISO system
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

        X0 = np.array([0, 0]);
        tout, yout = step_response(sys, T=t, X0=X0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # Test MIMO system, which contains ``siso_ss1`` twice
        sys = self.mimo_ss1
        _t, y_00 = step_response(sys, T=t, input=0, output=0)
        _t, y_11 = step_response(sys, T=t, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)

    def test_impulse_response(self):
        #Test SISO system
        sys = self.siso_ss1
        t = np.linspace(0, 1, 10)
        youttrue = np.array([86., 70.1808, 57.3753, 46.9975, 38.5766, 31.7344, 
                             26.1668, 21.6292, 17.9245, 14.8945]) 
        tout, yout = impulse_response(sys, T=t)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        #Test MIMO system, which contains ``siso_ss1`` twice
        sys = self.mimo_ss1
        _t, y_00 = impulse_response(sys, T=t, input=0, output=0)
        _t, y_11 = impulse_response(sys, T=t, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)

        #Test MIMO system, as mimo, and don't trim outputs
        sys = self.mimo_ss1
        _t, yy = impulse_response(sys, T=t, input=0)
        np.testing.assert_array_almost_equal(
            yy, np.vstack((youttrue, np.zeros_like(youttrue))), decimal=4)

    def test_initial_response(self):
        #Test SISO system
        sys = self.siso_ss1
        t = np.linspace(0, 1, 10)
        x0 = np.array([[0.5], [1]]);
        youttrue = np.array([11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092, 
                             1.1508, 0.5833, 0.1645, -0.1391]) 
        tout, yout = initial_response(sys, T=t, X0=x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        #Test MIMO system, which contains ``siso_ss1`` twice
        sys = self.mimo_ss1
        x0 = np.matrix(".5; 1.; .5; 1.")
        _t, y_00 = initial_response(sys, T=t, X0=x0, input=0, output=0)
        _t, y_11 = initial_response(sys, T=t, X0=x0, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)
        
        # test MIMO system without trimming
        _t, yy = initial_response(sys, T=t, X0=x0)
        np.testing.assert_array_almost_equal(yy, np.vstack((y_00, y_11)), 
                                             decimal=4)

    def test_forced_response(self):
        t = np.linspace(0, 1, 10)
        
        #compute step response - test with state space, and transfer function
        #objects
        u = np.array([1., 1, 1, 1, 1, 1, 1, 1, 1, 1])
        youttrue = np.array([9., 17.6457, 24.7072, 30.4855, 35.2234, 39.1165, 
                             42.3227, 44.9694, 47.1599, 48.9776]) 
        tout, yout, _xout = forced_response(self.siso_ss1, t, u)   
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)
        _t, yout, _xout = forced_response(self.siso_tf2, t, u)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        
        #test with initial value and special algorithm for ``U=0``
        u=0
        x0 = np.matrix(".5; 1.")
        youttrue = np.array([11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092, 
                             1.1508, 0.5833, 0.1645, -0.1391]) 
        _t, yout, _xout = forced_response(self.siso_ss1, t, u, x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        
        #Test MIMO system, which contains ``siso_ss1`` twice
        #first system: initial value, second system: step response
        u = np.array([[0., 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1., 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        x0 = np.matrix(".5; 1; 0; 0")
        youttrue = np.array([[11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092, 
                              1.1508, 0.5833, 0.1645, -0.1391],
                             [9., 17.6457, 24.7072, 30.4855, 35.2234, 39.1165, 
                              42.3227, 44.9694, 47.1599, 48.9776]]) 
        _t, yout, _xout = forced_response(self.mimo_ss1, t, u, x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        
def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestTimeresp)

if __name__ == '__main__':
    unittest.main()
