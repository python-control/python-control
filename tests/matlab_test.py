#!/usr/bin/env python
#
# matlab_test.py - test MATLAB compatibility
# RMM, 30 Mar 2011 (based on TestMatlab from v0.4a)
#
# This test suite just goes through and calls all of the MATLAB
# functions using different systems and arguments to make sure that
# nothing crashes.  It doesn't test actual functionality; the module
# specific unit tests will do that.

import unittest
import numpy as np
import scipy as sp
from control.matlab import *

class TestMatlab(unittest.TestCase):
    def setUp(self):
        """Set up some systems for testing out MATLAB functions"""
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5.; 7.")
        C = np.matrix("6. 8.")
        D = np.matrix("9.")
        self.siso_ss1 = ss(A,B,C,D)

        # Create some transfer functions
        self.siso_tf1 = tf([1], [1, 2, 1]);
        self.siso_tf2 = tf([1, 1], [1, 2, 3, 1]);

        # Conversions
        self.siso_tf3 = tf(self.siso_ss1);
        self.siso_ss2 = ss(self.siso_tf2);
        self.siso_ss3 = tf2ss(self.siso_tf3);
        self.siso_tf4 = ss2tf(self.siso_ss2);
        
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
        self.mimo_ss1 = ss(A, B, C, D)
        
    def testParallel(self):
        sys1 = parallel(self.siso_ss1, self.siso_ss2)
        sys1 = parallel(self.siso_ss1, self.siso_tf2)
        sys1 = parallel(self.siso_tf1, self.siso_ss2)
        sys1 = parallel(1, self.siso_ss2)
        sys1 = parallel(1, self.siso_tf2)
        sys1 = parallel(self.siso_ss1, 1)
        sys1 = parallel(self.siso_tf1, 1)

    def testSeries(self):
        sys1 = series(self.siso_ss1, self.siso_ss2)
        sys1 = series(self.siso_ss1, self.siso_tf2)
        sys1 = series(self.siso_tf1, self.siso_ss2)
        sys1 = series(1, self.siso_ss2)
        sys1 = series(1, self.siso_tf2)
        sys1 = series(self.siso_ss1, 1)
        sys1 = series(self.siso_tf1, 1)

    def testFeedback(self):
        sys1 = feedback(self.siso_ss1, self.siso_ss2)
        sys1 = feedback(self.siso_ss1, self.siso_tf2)
        sys1 = feedback(self.siso_tf1, self.siso_ss2)
        sys1 = feedback(1, self.siso_ss2)
        sys1 = feedback(1, self.siso_tf2)
        sys1 = feedback(self.siso_ss1, 1)
        sys1 = feedback(self.siso_tf1, 1)

    def testPoleZero(self):
        pole(self.siso_ss1);
        pole(self.siso_tf1);
        pole(self.siso_tf2);
        zero(self.siso_ss1);
        zero(self.siso_tf1);
        zero(self.siso_tf2);

    def testPZmap(self):
        # pzmap(self.siso_ss1);         not implemented
        # pzmap(self.siso_ss2);         not implemented
        pzmap(self.siso_tf1);
        pzmap(self.siso_tf2);
        pzmap(self.siso_tf2, Plot=False);

    def testStep(self):
        #Test SISO system
        sys = self.siso_ss1
        t = np.linspace(0, 1, 10)
        youttrue = np.array([9., 17.6457, 24.7072, 30.4855, 35.2234, 39.1165, 
                             42.3227, 44.9694, 47.1599, 48.9776]) 
        yout, tout = step(sys, T=t)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # Play with arguments
        yout, tout = step(sys, T=t, X0=0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        X0 = np.array([0, 0]);
        yout, tout = step(sys, T=t, X0=X0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        #Test MIMO system, which contains ``siso_ss1`` twice
        sys = self.mimo_ss1
        y_00, _t = step(sys, T=t, input=0, output=0)
        y_11, _t = step(sys, T=t, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)

    def testImpulse(self):
        #Test SISO system
        sys = self.siso_ss1
        t = np.linspace(0, 1, 10)
        youttrue = np.array([86., 70.1808, 57.3753, 46.9975, 38.5766, 31.7344, 
                             26.1668, 21.6292, 17.9245, 14.8945]) 
        yout, tout = impulse(sys, T=t)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        #Test MIMO system, which contains ``siso_ss1`` twice
        sys = self.mimo_ss1
        y_00, _t = impulse(sys, T=t, input=0, output=0)
        y_11, _t = impulse(sys, T=t, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)

    def testInitial(self):
        #Test SISO system
        sys = self.siso_ss1
        t = np.linspace(0, 1, 10)
        x0 = np.matrix(".5; 1.")
        youttrue = np.array([11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092, 
                             1.1508, 0.5833, 0.1645, -0.1391]) 
        yout, tout = initial(sys, T=t, X0=x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        #Test MIMO system, which contains ``siso_ss1`` twice
        sys = self.mimo_ss1
        x0 = np.matrix(".5; 1.; .5; 1.")
        y_00, _t = initial(sys, T=t, X0=x0, input=0, output=0)
        y_11, _t = initial(sys, T=t, X0=x0, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)

    def testLsim(self):
        t = np.linspace(0, 1, 10)
        
        #compute step response - test with state space, and transfer function
        #objects
        u = np.array([1., 1, 1, 1, 1, 1, 1, 1, 1, 1])
        youttrue = np.array([9., 17.6457, 24.7072, 30.4855, 35.2234, 39.1165, 
                             42.3227, 44.9694, 47.1599, 48.9776]) 
        yout, tout, _xout = lsim(self.siso_ss1, u, t)   
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)
        yout, _t, _xout = lsim(self.siso_tf3, u, t)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        
        #test with initial value and special algorithm for ``U=0``
        u=0
        x0 = np.matrix(".5; 1.")
        youttrue = np.array([11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092, 
                             1.1508, 0.5833, 0.1645, -0.1391]) 
        yout, _t, _xout = lsim(self.siso_ss1, u, t, x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        
        #Test MIMO system, which contains ``siso_ss1`` twice
        #first system: initial value, second system: step response
        u = np.array([[0., 1.], [0, 1], [0, 1], [0, 1], [0, 1],
                      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        x0 = np.matrix(".5; 1; 0; 0")
        youttrue = np.array([[11., 9.], [8.1494, 17.6457], [5.9361, 24.7072],
                             [4.2258, 30.4855], [2.9118, 35.2234],
                             [1.9092, 39.1165], [1.1508, 42.3227], 
                             [0.5833, 44.9694], [0.1645, 47.1599], 
                             [-0.1391, 48.9776]])
        yout, _t, _xout = lsim(self.mimo_ss1, u, t, x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        
    def testDcgain(self):
        #Create different forms of a SISO system
        A, B, C, D = self.siso_ss1.A, self.siso_ss1.B, self.siso_ss1.C, \
                     self.siso_ss1.D
        Z, P, k = sp.signal.ss2zpk(A, B, C, D)
        num, den = sp.signal.ss2tf(A, B, C, D)
        sys_ss = self.siso_ss1
        
        #Compute the gain with ``dcgain``
        gain_abcd = dcgain(A, B, C, D)
        gain_zpk = dcgain(Z, P, k)
        gain_numden = dcgain(np.squeeze(num), den)
        gain_sys_ss = dcgain(sys_ss)
        print
        print 'gain_abcd:', gain_abcd, 'gain_zpk:', gain_zpk
        print 'gain_numden:', gain_numden, 'gain_sys_ss:', gain_sys_ss
        
        #Compute the gain with a long simulation
        t = linspace(0, 1000, 1000)
        y, _t = step(sys_ss, t)
        gain_sim = y[-1]
        print 'gain_sim:', gain_sim
        
        #All gain values must be approximately equal to the known gain
        np.testing.assert_array_almost_equal(
            [gain_abcd[0,0], gain_zpk[0,0], gain_numden[0,0], gain_sys_ss[0,0], 
             gain_sim],
            [59, 59, 59, 59, 59])
        
        #Test with MIMO system, which contains ``siso_ss1`` twice
        gain_mimo = dcgain(self.mimo_ss1)
        print 'gain_mimo: \n', gain_mimo
        np.testing.assert_array_almost_equal(gain_mimo, [[59., 0 ], 
                                                         [0,  59.]])

    def testBode(self):
        bode(self.siso_ss1)
        bode(self.siso_tf1)
        bode(self.siso_tf2)
        (mag, phase, freq) = bode(self.siso_tf2, Plot=False)
        bode(self.siso_tf1, self.siso_tf2)
        w = logspace(-3, 3);
        bode(self.siso_ss1, w)
        bode(self.siso_ss1, self.siso_tf2, w)
        bode(self.siso_ss1, '-', self.siso_tf1, 'b--', self.siso_tf2, 'k.')

    def testRlocus(self):
        rlocus(self.siso_ss1)
        rlocus(self.siso_tf1)
        rlocus(self.siso_tf2)
        rlist, klist = rlocus(self.siso_tf2, klist=[1, 10, 100], Plot=False)

    def testNyquist(self):
        nyquist(self.siso_ss1)
        nyquist(self.siso_tf1)
        nyquist(self.siso_tf2)
        w = logspace(-3, 3);
        nyquist(self.siso_tf2, w)
        (real, imag, freq) = nyquist(self.siso_tf2, w, Plot=False)

    def testNichols(self):
        nichols(self.siso_ss1)
        nichols(self.siso_tf1)
        nichols(self.siso_tf2)
        w = logspace(-3, 3);
        nichols(self.siso_tf2, w)
        nichols(self.siso_tf2, grid=False)

    def testFreqresp(self):
        w = logspace(-3, 3)
        freqresp(self.siso_ss1, w)
        freqresp(self.siso_ss2, w)
        freqresp(self.siso_ss3, w)
        freqresp(self.siso_tf1, w)
        freqresp(self.siso_tf2, w)
        freqresp(self.siso_tf3, w)

    def testEvalfr(self):
        w = 1
        evalfr(self.siso_ss1, w)
        evalfr(self.siso_ss2, w)
        evalfr(self.siso_ss3, w)
        evalfr(self.siso_tf1, w)
        evalfr(self.siso_tf2, w)
        evalfr(self.siso_tf3, w)

    def testHsvd(self):
        hsvd(self.siso_ss1)
        hsvd(self.siso_ss2)
        hsvd(self.siso_ss3)

    def testBalred(self):
        balred(self.siso_ss1, 1)
        balred(self.siso_ss2, 2)
        balred(self.siso_ss3, [2, 2])

    def testModred(self):
        modred(self.siso_ss1, [1])
        modred(self.siso_ss2 * self.siso_ss3, [1, 2])
        modred(self.siso_ss3, [1], 'matchdc')
        modred(self.siso_ss3, [1], 'truncate')

    def testPlace(self):
        place(self.siso_ss1.A, self.siso_ss1.B, [-2, -2])

    def testLQR(self):
        (K, S, E) = lqr(self.siso_ss1.A, self.siso_ss1.B, np.eye(2), np.eye(1))
        (K, S, E) = lqr(self.siso_ss2.A, self.siso_ss2.B, np.eye(3), \
                            np.eye(1), [[1], [1], [2]])

    def testRss(self):
        rss(1)
        rss(2)
        rss(2, 3, 1)

    def testDrss(self):
        drss(1)
        drss(2)
        drss(2, 3, 1)

    def testCtrb(self):
        ctrb(self.siso_ss1.A, self.siso_ss1.B)
        ctrb(self.siso_ss2.A, self.siso_ss2.B)

    def testObsv(self):
        obsv(self.siso_ss1.A, self.siso_ss1.C)
        obsv(self.siso_ss2.A, self.siso_ss2.C)

    def testGram(self):
        gram(self.siso_ss1, 'c')
        gram(self.siso_ss2, 'c')
        gram(self.siso_ss1, 'o')
        gram(self.siso_ss2, 'o')

    def testPade(self):
        pade(1, 1)
        pade(1, 2)
        pade(5, 4)

    def testOpers(self):
        self.siso_ss1 + self.siso_ss2
        self.siso_tf1 + self.siso_tf2
        self.siso_ss1 + self.siso_tf2
        self.siso_tf1 + self.siso_ss2
        self.siso_ss1 * self.siso_ss2
        self.siso_tf1 * self.siso_tf2
        self.siso_ss1 * self.siso_tf2
        self.siso_tf1 * self.siso_ss2
        # self.siso_ss1 / self.siso_ss2         not implemented yet
        # self.siso_tf1 / self.siso_tf2
        # self.siso_ss1 / self.siso_tf2
        # self.siso_tf1 / self.siso_ss2

    def testUnwrap(self):
        phase = np.array(range(1, 100)) / 10.;
        wrapped = phase % (2 * np.pi)
        unwrapped = unwrap(wrapped)

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestMatlab)

if __name__ == '__main__':
    unittest.main()
