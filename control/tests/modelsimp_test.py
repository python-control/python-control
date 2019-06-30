#!/usr/bin/env python
#
# modelsimp_test.py - test model reduction functions
# RMM, 30 Mar 2011 (based on TestModelSimp from v0.4a)

import unittest
import numpy as np
from control.modelsimp import *
from control.matlab import *
from control.exception import slycot_check

class TestModelsimp(unittest.TestCase):
    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testHSVD(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5.; 7.")
        C = np.matrix("6. 8.")
        D = np.matrix("9.")
        sys = ss(A,B,C,D)
        hsv = hsvd(sys)
        hsvtrue = [24.42686, 0.5731395]  # from MATLAB
        np.testing.assert_array_almost_equal(hsv, hsvtrue)

    def testMarkov(self):
        U = np.matrix("1.; 1.; 1.; 1.; 1.")
        Y = U
        M = 3
        H = markov(Y,U,M)
        Htrue = np.matrix("1.; 0.; 0.")
        np.testing.assert_array_almost_equal( H, Htrue )

    def testModredMatchDC(self):
        #balanced realization computed in matlab for the transfer function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = np.matrix('-1.958, -1.194, 1.824, -1.464; \
        -1.194, -0.8344, 2.563, -1.351; \
        -1.824, -2.563, -1.124, 2.704; \
        -1.464, -1.351, -2.704, -11.08')
        B = np.matrix('-0.9057; -0.4068; -0.3263; -0.3474')
        C = np.matrix('-0.9057, -0.4068, 0.3263, -0.3474')
        D = np.matrix('0.')
        sys = ss(A,B,C,D)
        rsys = modred(sys,[2, 3],'matchdc')
        Artrue = np.matrix('-4.431, -4.552; -4.552, -5.361')
        Brtrue = np.matrix('-1.362; -1.031')
        Crtrue = np.matrix('-1.362, -1.031')
        Drtrue = np.matrix('-0.08384')
        np.testing.assert_array_almost_equal(rsys.A, Artrue,decimal=3)
        np.testing.assert_array_almost_equal(rsys.B, Brtrue,decimal=3)
        np.testing.assert_array_almost_equal(rsys.C, Crtrue,decimal=3)
        np.testing.assert_array_almost_equal(rsys.D, Drtrue,decimal=2)

    def testModredUnstable(self):
        # Check if an error is thrown when an unstable system is given
        A = np.matrix('4.5418, 3.3999, 5.0342, 4.3808; \
        0.3890, 0.3599, 0.4195, 0.1760; \
        -4.2117, -3.2395, -4.6760, -4.2180; \
        0.0052, 0.0429, 0.0155, 0.2743')
        B = np.matrix('1.0, 1.0; 2.0, 2.0; 3.0, 3.0; 4.0, 4.0')
        C = np.matrix('1.0, 2.0, 3.0, 4.0; 1.0, 2.0, 3.0, 4.0')
        D = np.matrix('0.0, 0.0; 0.0, 0.0')
        sys = ss(A,B,C,D)
        np.testing.assert_raises(ValueError, modred, sys, [2, 3])

    def testModredTruncate(self):
        #balanced realization computed in matlab for the transfer function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = np.matrix('-1.958, -1.194, 1.824, -1.464; \
        -1.194, -0.8344, 2.563, -1.351; \
        -1.824, -2.563, -1.124, 2.704; \
        -1.464, -1.351, -2.704, -11.08')
        B = np.matrix('-0.9057; -0.4068; -0.3263; -0.3474')
        C = np.matrix('-0.9057, -0.4068, 0.3263, -0.3474')
        D = np.matrix('0.')
        sys = ss(A,B,C,D)
        rsys = modred(sys,[2, 3],'truncate')
        Artrue = np.matrix('-1.958, -1.194; -1.194, -0.8344')
        Brtrue = np.matrix('-0.9057; -0.4068')
        Crtrue = np.matrix('-0.9057, -0.4068')
        Drtrue = np.matrix('0.')
        np.testing.assert_array_almost_equal(rsys.A, Artrue)
        np.testing.assert_array_almost_equal(rsys.B, Brtrue)
        np.testing.assert_array_almost_equal(rsys.C, Crtrue)
        np.testing.assert_array_almost_equal(rsys.D, Drtrue)


    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testBalredTruncate(self):
        #controlable canonical realization computed in matlab for the transfer function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = np.matrix('-15., -7.5, -6.25, -1.875; \
        8., 0., 0., 0.; \
        0., 4., 0., 0.; \
        0., 0., 1., 0.')
        B = np.matrix('2.; 0.; 0.; 0.')
        C = np.matrix('0.5, 0.6875, 0.7031, 0.5')
        D = np.matrix('0.')
        sys = ss(A,B,C,D)
        orders = 2
        rsys = balred(sys,orders,method='truncate')
        Artrue = np.matrix('-1.958, -1.194; -1.194, -0.8344')
        Brtrue = np.matrix('0.9057; 0.4068')
        Crtrue = np.matrix('0.9057, 0.4068')
        Drtrue = np.matrix('0.')
        np.testing.assert_array_almost_equal(rsys.A, Artrue,decimal=2)
        np.testing.assert_array_almost_equal(rsys.B, Brtrue,decimal=4)
        np.testing.assert_array_almost_equal(rsys.C, Crtrue,decimal=4)
        np.testing.assert_array_almost_equal(rsys.D, Drtrue,decimal=4)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testBalredMatchDC(self):
        #controlable canonical realization computed in matlab for the transfer function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = np.matrix('-15., -7.5, -6.25, -1.875; \
        8., 0., 0., 0.; \
        0., 4., 0., 0.; \
        0., 0., 1., 0.')
        B = np.matrix('2.; 0.; 0.; 0.')
        C = np.matrix('0.5, 0.6875, 0.7031, 0.5')
        D = np.matrix('0.')
        sys = ss(A,B,C,D)
        orders = 2
        rsys = balred(sys,orders,method='matchdc')
        Artrue = np.matrix('-4.43094773, -4.55232904; -4.55232904, -5.36195206')
        Brtrue = np.matrix('1.36235673; 1.03114388')
        Crtrue = np.matrix('1.36235673, 1.03114388')
        Drtrue = np.matrix('-0.08383902')
        np.testing.assert_array_almost_equal(rsys.A, Artrue,decimal=2)
        np.testing.assert_array_almost_equal(rsys.B, Brtrue,decimal=4)
        np.testing.assert_array_almost_equal(rsys.C, Crtrue,decimal=4)
        np.testing.assert_array_almost_equal(rsys.D, Drtrue,decimal=4)

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestModelsimp)


if __name__ == '__main__':
    unittest.main()
