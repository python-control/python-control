#!/usr/bin/env python
#
# matlab_test.py - test MATLAB compatibility
# RMM, 30 Mar 2011 (based on TestMatlab from v0.4a)
#
# This test suite just goes through and calls all of the MATLAB
# functions using different systems and arguments to make sure that
# nothing crashes.  It doesn't test actual functionality; the module
# specific unit tests will do that.

from __future__ import print_function
import unittest
import numpy as np
from scipy.linalg import eigvals
import scipy as sp
from control.matlab import *
from control.frdata import FRD
from control.exception import slycot_check
import warnings

# for running these through Matlab or Octave
'''
siso_ss1 = ss([1. -2.; 3. -4.], [5.; 7.], [6. 8.], [0])

siso_tf1 = tf([1], [1, 2, 1])
siso_tf2 = tf([1, 1], [1, 2, 3, 1])

siso_tf3 = tf(siso_ss1)
siso_ss2 = ss(siso_tf2)
siso_ss3 = ss(siso_tf3)
siso_tf4 = tf(siso_ss2)

A =[ 1. -2. 0.  0.;
     3. -4. 0.  0.;
     0.  0. 1. -2.;
     0.  0. 3. -4. ]
B = [ 5. 0.;
      7. 0.;
      0. 5.;
      0. 7. ]
C = [ 6. 8. 0. 0.;
      0. 0. 6. 8. ]
D = [ 9. 0.;
      0. 9. ]
mimo_ss1 = ss(A, B, C, D)

% all boring, since no cross-over
margin(siso_tf1)
margin(siso_tf2)
margin(siso_ss1)
margin(siso_ss2)

% make a bit better
[gm, pm, gmc, pmc] = margin(siso_ss2*siso_ss2*2)

'''

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

        # get consistent test results
        np.random.seed(0)

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
        pzmap(self.siso_tf2, plot=False);

    def testStep(self):
        t = np.linspace(0, 1, 10)
        # Test transfer function
        yout, tout = step(self.siso_tf1, T=t)
        youttrue = np.array([0, 0.0057, 0.0213, 0.0446, 0.0739,
                             0.1075, 0.1443, 0.1832, 0.2235, 0.2642])
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # Test SISO system with direct feedthrough
        sys = self.siso_ss1
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

        yout, tout, xout = step(sys, T=t, X0=0, return_x=True)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        if slycot_check():
            # Test MIMO system, which contains ``siso_ss1`` twice
            sys = self.mimo_ss1
            y_00, _t = step(sys, T=t, input=0, output=0)
            y_11, _t = step(sys, T=t, input=1, output=1)
            np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
            np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)

    def testImpulse(self):
        t = np.linspace(0, 1, 10)
        # test transfer function
        yout, tout = impulse(self.siso_tf1, T=t)
        youttrue = np.array([0., 0.0994, 0.1779, 0.2388, 0.2850, 0.3188,
                             0.3423, 0.3573, 0.3654, 0.3679])
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # produce a warning for a system with direct feedthrough
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #Test SISO system
            sys = self.siso_ss1
            youttrue = np.array([86., 70.1808, 57.3753, 46.9975, 38.5766, 31.7344,
                                 26.1668, 21.6292, 17.9245, 14.8945])
            yout, tout = impulse(sys, T=t)
            np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
            np.testing.assert_array_almost_equal(tout, t)

            # Play with arguments
            yout, tout = impulse(sys, T=t, X0=0)
            np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
            np.testing.assert_array_almost_equal(tout, t)

            X0 = np.array([0, 0]);
            yout, tout = impulse(sys, T=t, X0=X0)
            np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
            np.testing.assert_array_almost_equal(tout, t)

            yout, tout, xout = impulse(sys, T=t, X0=0, return_x=True)
            np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
            np.testing.assert_array_almost_equal(tout, t)

            if slycot_check():
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

        # Play with arguments
        yout, tout, xout = initial(sys, T=t, X0=x0, return_x=True)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        if slycot_check():
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

        if slycot_check():
            #Test MIMO system, which contains ``siso_ss1`` twice
            #first system: initial value, second system: step response
            u = np.array([[0., 1.], [0, 1], [0, 1], [0, 1], [0, 1],
                          [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
            x0 = np.matrix(".5; 1; 0; 0")
            youttrue = np.array([[11., 9.], [8.1494, 17.6457],
                                 [5.9361, 24.7072], [4.2258, 30.4855],
                                 [2.9118, 35.2234], [1.9092, 39.1165],
                                 [1.1508, 42.3227], [0.5833, 44.9694],
                                 [0.1645, 47.1599], [-0.1391, 48.9776]])
            yout, _t, _xout = lsim(self.mimo_ss1, u, t, x0)
            np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)

    def testMargin(self):
        #! TODO: check results to make sure they are OK
        gm, pm, wg, wp = margin(self.siso_tf1);
        gm, pm, wg, wp = margin(self.siso_tf2);
        gm, pm, wg, wp = margin(self.siso_ss1);
        gm, pm, wg, wp = margin(self.siso_ss2);
        gm, pm, wg, wp = margin(self.siso_ss2*self.siso_ss2*2);
        np.testing.assert_array_almost_equal(
            [gm, pm, wg, wp], [1.5451, 75.9933, 1.2720, 0.6559], decimal=3)

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
        # print('\ngain_abcd:', gain_abcd, 'gain_zpk:', gain_zpk)
        # print('gain_numden:', gain_numden, 'gain_sys_ss:', gain_sys_ss)

        #Compute the gain with a long simulation
        t = linspace(0, 1000, 1000)
        y, _t = step(sys_ss, t)
        gain_sim = y[-1]
        # print('gain_sim:', gain_sim)

        #All gain values must be approximately equal to the known gain
        np.testing.assert_array_almost_equal(
            [gain_abcd, gain_zpk, gain_numden, gain_sys_ss,
             gain_sim],
            [59, 59, 59, 59, 59])

        if slycot_check():
            # Test with MIMO system, which contains ``siso_ss1`` twice
            gain_mimo = dcgain(self.mimo_ss1)
            # print('gain_mimo: \n', gain_mimo)
            np.testing.assert_array_almost_equal(gain_mimo, [[59., 0 ],
                                                             [0,  59.]])

    def testBode(self):
        bode(self.siso_ss1)
        bode(self.siso_tf1)
        bode(self.siso_tf2)
        (mag, phase, freq) = bode(self.siso_tf2, plot=False)
        bode(self.siso_tf1, self.siso_tf2)
        w = logspace(-3, 3);
        bode(self.siso_ss1, w)
        bode(self.siso_ss1, self.siso_tf2, w)
#       Not yet implemented
#       bode(self.siso_ss1, '-', self.siso_tf1, 'b--', self.siso_tf2, 'k.')

    def testRlocus(self):
        rlocus(self.siso_ss1)
        rlocus(self.siso_tf1)
        rlocus(self.siso_tf2)
        klist = [1, 10, 100]
        rlist, klist_out = rlocus(self.siso_tf2, klist, plot=False)
        np.testing.assert_equal(len(rlist), len(klist))
        np.testing.assert_array_equal(klist, klist_out)

    def testNyquist(self):
        nyquist(self.siso_ss1)
        nyquist(self.siso_tf1)
        nyquist(self.siso_tf2)
        w = logspace(-3, 3);
        nyquist(self.siso_tf2, w)
        (real, imag, freq) = nyquist(self.siso_tf2, w, plot=False)

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
        w = 1j
        np.testing.assert_almost_equal(evalfr(self.siso_ss1, w), 44.8-21.4j)
        evalfr(self.siso_ss2, w)
        evalfr(self.siso_ss3, w)
        evalfr(self.siso_tf1, w)
        evalfr(self.siso_tf2, w)
        evalfr(self.siso_tf3, w)
        if slycot_check():
            np.testing.assert_array_almost_equal(
                evalfr(self.mimo_ss1, w),
                np.array( [[44.8-21.4j, 0.], [0., 44.8-21.4j]]))

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testHsvd(self):
        hsvd(self.siso_ss1)
        hsvd(self.siso_ss2)
        hsvd(self.siso_ss3)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testBalred(self):
        balred(self.siso_ss1, 1)
        balred(self.siso_ss2, 2)
        balred(self.siso_ss3, [2, 2])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testModred(self):
        modred(self.siso_ss1, [1])
        modred(self.siso_ss2 * self.siso_ss1, [0, 1])
        modred(self.siso_ss1, [1], 'matchdc')
        modred(self.siso_ss1, [1], 'truncate')

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testPlace_varga(self):
        place_varga(self.siso_ss1.A, self.siso_ss1.B, [-2, -2])

    def testPlace(self):
        place(self.siso_ss1.A, self.siso_ss1.B, [-2, -2.5])

    def testAcker(self):
        acker(self.siso_ss1.A, self.siso_ss1.B, [-2, -2.5])


    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testLQR(self):
        (K, S, E) = lqr(self.siso_ss1.A, self.siso_ss1.B, np.eye(2), np.eye(1))

        # Should work if [Q N;N' R] is positive semi-definite
        (K, S, E) = lqr(self.siso_ss2.A, self.siso_ss2.B, 10*np.eye(3), \
                            np.eye(1), [[1], [1], [2]])

    @unittest.skip("check not yet implemented")
    def testLQR_checks(self):
        # Make sure we get a warning if [Q N;N' R] is not positive semi-definite
        (K, S, E) = lqr(self.siso_ss2.A, self.siso_ss2.B, np.eye(3), \
                            np.eye(1), [[1], [1], [2]])

    def testRss(self):
        rss(1)
        rss(2)
        rss(2, 1, 3)

    def testDrss(self):
        drss(1)
        drss(2)
        drss(2, 1, 3)

    def testCtrb(self):
        ctrb(self.siso_ss1.A, self.siso_ss1.B)
        ctrb(self.siso_ss2.A, self.siso_ss2.B)

    def testObsv(self):
        obsv(self.siso_ss1.A, self.siso_ss1.C)
        obsv(self.siso_ss2.A, self.siso_ss2.C)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
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

    def testSISOssdata(self):
        ssdata_1 = ssdata(self.siso_ss2);
        ssdata_2 = ssdata(self.siso_tf2);
        for i in range(len(ssdata_1)):
            np.testing.assert_array_almost_equal(ssdata_1[i], ssdata_2[i])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testMIMOssdata(self):
        m = (self.mimo_ss1.A, self.mimo_ss1.B, self.mimo_ss1.C, self.mimo_ss1.D)
        ssdata_1 = ssdata(self.mimo_ss1);
        for i in range(len(ssdata_1)):
            np.testing.assert_array_almost_equal(ssdata_1[i], m[i])

    def testSISOtfdata(self):
        tfdata_1 = tfdata(self.siso_tf2);
        tfdata_2 = tfdata(self.siso_tf2);
        for i in range(len(tfdata_1)):
            np.testing.assert_array_almost_equal(tfdata_1[i], tfdata_2[i])

    def testDamp(self):
        A = np.mat('''-0.2  0.06 0    -1;
               0    0    1     0;
             -17    0   -3.8   1;
               9.4  0   -0.4  -0.6''')
        B = np.mat('''-0.01  0.06;
               0     0;
             -32     5.4;
               2.6  -7''')
        C = np.eye(4)
        D = np.zeros((4,2))
        sys = ss(A, B, C, D)
        wn, Z, p = damp(sys, False)
        # print (wn)
        np.testing.assert_array_almost_equal(
            wn, np.array([4.07381994,   3.28874827,   3.28874827,
                          1.08937685e-03]))
        np.testing.assert_array_almost_equal(
            Z, np.array([1.0, 0.07983139,  0.07983139, 1.0]))

    def testConnect(self):
        sys1 = ss("1. -2; 3. -4", "5.; 7", "6, 8", "9.")
        sys2 = ss("-1.", "1.", "1.", "0.")
        sys = append(sys1, sys2)
        Q= np.mat([ [ 1, 2], [2, -1] ]) # basically feedback, output 2 in 1
        sysc = connect(sys, Q, [2], [1, 2])
        # print(sysc)
        np.testing.assert_array_almost_equal(
            sysc.A, np.mat('1 -2 5; 3 -4 7; -6 -8 -10'))
        np.testing.assert_array_almost_equal(
            sysc.B, np.mat('0; 0; 1'))
        np.testing.assert_array_almost_equal(
            sysc.C, np.mat('6 8 9; 0 0 1'))
        np.testing.assert_array_almost_equal(
            sysc.D, np.mat('0; 0'))

    def testConnect2(self):
        sys = append(ss([[-5, -2.25], [4, 0]], [[2], [0]],
                          [[0, 1.125]], [[0]]),
                       ss([[-1.6667, 0], [1, 0]], [[2], [0]],
                          [[0, 3.3333]], [[0]]),
                       1)
        Q = [ [ 1, 3], [2, 1], [3, -2]]
        sysc = connect(sys, Q, [3], [3, 1, 2])
        np.testing.assert_array_almost_equal(
            sysc.A, np.mat([[-5, -2.25, 0, -6.6666],
                            [4, 0, 0, 0],
                            [0, 2.25, -1.6667, 0],
                            [0, 0, 1, 0]]))
        np.testing.assert_array_almost_equal(
            sysc.B, np.mat([[2], [0], [0], [0]]))
        np.testing.assert_array_almost_equal(
            sysc.C, np.mat([[0, 0, 0, -3.3333],
                            [0, 1.125, 0, 0],
                            [0, 0, 0, 3.3333]]))
        np.testing.assert_array_almost_equal(
            sysc.D, np.mat([[1], [0], [0]]))



    def testFRD(self):
        h = tf([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 10)
        frd1 = frd(h, omega)
        assert isinstance(frd1, FRD)
        frd2 = frd(frd1.fresp[0,0,:], omega)
        assert isinstance(frd2, FRD)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testMinreal(self, verbose=False):
        """Test a minreal model reduction"""
        #A = [-2, 0.5, 0; 0.5, -0.3, 0; 0, 0, -0.1]
        A = [[-2, 0.5, 0], [0.5, -0.3, 0], [0, 0, -0.1]]
        #B = [0.3, -1.3; 0.1, 0; 1, 0]
        B = [[0.3, -1.3], [0.1, 0.], [1.0, 0.0]]
        #C = [0, 0.1, 0; -0.3, -0.2, 0]
        C = [[0., 0.1, 0.0], [-0.3, -0.2, 0.0]]
        #D = [0 -0.8; -0.3 0]
        D = [[0., -0.8], [-0.3, 0.]]
        # sys = ss(A, B, C, D)

        sys = ss(A, B, C, D)
        sysr = minreal(sys, verbose=verbose)
        self.assertEqual(sysr.states, 2)
        self.assertEqual(sysr.inputs, sys.inputs)
        self.assertEqual(sysr.outputs, sys.outputs)
        np.testing.assert_array_almost_equal(
            eigvals(sysr.A), [-2.136154, -0.1638459])

        s = tf([1, 0], [1])
        h = (s+1)*(s+2.00000000001)/(s+2)/(s**2+s+1)
        hm = minreal(h, verbose=verbose)
        hr = (s+1)/(s**2+s+1)
        np.testing.assert_array_almost_equal(hm.num[0][0], hr.num[0][0])
        np.testing.assert_array_almost_equal(hm.den[0][0], hr.den[0][0])

    def testSS2cont(self):
        sys = ss(
            np.mat("-3 4 2; -1 -3 0; 2 5 3"),
            np.mat("1 4 ; -3 -3; -2 1"),
            np.mat("4 2 -3; 1 4 3"),
            np.mat("-2 4; 0 1"))
        sysd = c2d(sys, 0.1)
        np.testing.assert_array_almost_equal(
            np.mat(
                """0.742840837331905  0.342242024293711  0.203124211149560;
                  -0.074130792143890  0.724553295044645 -0.009143771143630;
                   0.180264783290485  0.544385612448419  1.370501013067845"""),
            sysd.A)
        np.testing.assert_array_almost_equal(
            np.mat(""" 0.012362066084719   0.301932197918268;
                      -0.260952977031384  -0.274201791021713;
                      -0.304617775734327   0.075182622718853"""), sysd.B)

    def testCombi01(self):
        # test from a "real" case, combines tf, ss, connect and margin
        # this is a type 2 system, with phase starting at -180. The
        # margin command should remove the solution for w = nearly zero

        # Example is a concocted two-body satellite with flexible link
        Jb = 400
        Jp = 1000
        k = 10
        b = 5

        # can now define an "s" variable, to make TF's
        s = tf([1, 0], [1])
        hb1 = 1/(Jb*s)
        hb2 = 1/s
        hp1 = 1/(Jp*s)
        hp2 = 1/s

        # convert to ss and append
        sat0 = append(ss(hb1), ss(hb2), k, b, ss(hp1), ss(hp2))

        # connection of the elements with connect call
        Q = [[1, -3, -4],  # link moment (spring, damper), feedback to body
             [2,  1,  0],  # link integrator to body velocity
             [3,  2, -6],  # spring input, th_b - th_p
             [4,  1, -5],  # damper input
             [5,  3,  4],  # link moment, acting on payload
             [6,  5,  0]]
        inputs = [1]
        outputs = [1, 2, 5, 6]
        sat1 = connect(sat0, Q, inputs, outputs)

        # matched notch filter
        wno = 0.19
        z1 = 0.05
        z2 = 0.7
        Hno = (1+2*z1/wno*s+s**2/wno**2)/(1+2*z2/wno*s+s**2/wno**2)

        # the controller, Kp = 1 for now
        Kp = 1.64
        tau_PD = 50.
        Hc = (1 + tau_PD*s)*Kp

        # start with the basic satellite model sat1, and get the
        # payload attitude response
        Hp = tf(sp.matrix([0, 0, 0, 1])*sat1)

        # total open loop
        Hol = Hc*Hno*Hp

        gm, pm, wg, wp = margin(Hol)
        # print("%f %f %f %f" % (gm, pm, wg, wp))
        self.assertAlmostEqual(gm, 3.32065569155)
        self.assertAlmostEqual(pm, 46.9740430224)
        self.assertAlmostEqual(wg, 0.176469728448)
        self.assertAlmostEqual(wp, 0.0616288455466)

    def test_tf_string_args(self):
        # Make sure that the 's' variable is defined properly
        s = tf('s')
        G = (s + 1)/(s**2 + 2*s + 1)
        np.testing.assert_array_almost_equal(G.num, [[[1, 1]]])
        np.testing.assert_array_almost_equal(G.den, [[[1, 2, 1]]])
        self.assertTrue(isctime(G, strict=True))

        # Make sure that the 'z' variable is defined properly
        z = tf('z')
        G = (z + 1)/(z**2 + 2*z + 1)
        np.testing.assert_array_almost_equal(G.num, [[[1, 1]]])
        np.testing.assert_array_almost_equal(G.den, [[[1, 2, 1]]])
        self.assertTrue(isdtime(G, strict=True))


#! TODO: not yet implemented
#    def testMIMOtfdata(self):
#        sisotf = ss2tf(self.siso_ss1)
#        tfdata_1 = tfdata(sisotf)
#        tfdata_2 = tfdata(self.mimo_ss1, input=0, output=0)
#        for i in range(len(tfdata)):
#            np.testing.assert_array_almost_equal(tfdata_1[i], tfdata_2[i])


if __name__ == '__main__':
    unittest.main()
