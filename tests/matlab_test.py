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
        sys = self.siso_ss1
        t = np.linspace(0, 1, 10)
        t, yout = step(sys, T=t)
        youttrue = np.matrix("9. 17.6457 24.7072 30.4855 35.2234 39.1165 42.3227 44.9694 47.1599 48.9776") 
        np.testing.assert_array_almost_equal(yout, youttrue,decimal=4)

    def testImpulse(self):
        sys = self.siso_ss1
        t = np.linspace(0, 1, 10)
        t, yout = impulse(sys, T=t)
        youttrue = np.matrix("86. 70.1808 57.3753 46.9975 38.5766 31.7344 26.1668 21.6292 17.9245 14.8945") 
        np.testing.assert_array_almost_equal(yout, youttrue,decimal=4)

#    def testInitial(self):
        sys = self.siso_ss1
#        t = np.linspace(0, 1, 10)
#        x0 = np.matrix(".5; 1.")
#        t, yout = initial(sys, T=t, X0=x0)
#        youttrue = np.matrix("11. 8.1494 5.9361 4.2258 2.9118 1.9092 1.1508 0.5833 0.1645 -0.1391") 
#        np.testing.assert_array_almost_equal(yout, youttrue,decimal=4)

    def testLsim(self):
        T = range(1, 100)
        u = np.sin(T)
        lsim(self.siso_tf1, u, T)
        # lsim(self.siso_ss1, u, T)                     # generates error??
        # lsim(self.siso_ss1, u, T, self.siso_ss1.B)

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
