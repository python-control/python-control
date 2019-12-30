#!/usr/bin/env python
#
# statefbk_test.py - test state feedback functions
# RMM, 30 Mar 2011 (based on TestStatefbk from v0.4a)

from __future__ import print_function
import unittest
import numpy as np
from control.statefbk import ctrb, obsv, place, place_varga, lqr, lqe, gram, acker
from control.matlab import *
from control.exception import slycot_check, ControlDimension
from control.mateqn import care, dare

class TestStatefbk(unittest.TestCase):
    """Test state feedback functions"""

    def setUp(self):
        # Maximum number of states to test + 1
        self.maxStates = 5
        # Maximum number of inputs and outputs to test + 1
        self.maxTries = 4
        # Set to True to print systems to the output.
        self.debug = False
        # get consistent test results
        np.random.seed(0)

    def testCtrbSISO(self):
        A = np.matrix("1. 2.; 3. 4.")
        B = np.matrix("5.; 7.")
        Wctrue = np.matrix("5. 19.; 7. 43.")
        Wc = ctrb(A,B)
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    def testCtrbMIMO(self):
        A = np.matrix("1. 2.; 3. 4.")
        B = np.matrix("5. 6.; 7. 8.")
        Wctrue = np.matrix("5. 6. 19. 22.; 7. 8. 43. 50.")
        Wc = ctrb(A,B)
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    def testObsvSISO(self):
        A = np.matrix("1. 2.; 3. 4.")
        C = np.matrix("5. 7.")
        Wotrue = np.matrix("5. 7.; 26. 38.")
        Wo = obsv(A,C)
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    def testObsvMIMO(self):
        A = np.matrix("1. 2.; 3. 4.")
        C = np.matrix("5. 6.; 7. 8.")
        Wotrue = np.matrix("5. 6.; 7. 8.; 23. 34.; 31. 46.")
        Wo = obsv(A,C)
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    def testCtrbObsvDuality(self):
        A = np.matrix("1.2 -2.3; 3.4 -4.5")
        B = np.matrix("5.8 6.9; 8. 9.1")
        Wc = ctrb(A,B);
        A = np.transpose(A)
        C = np.transpose(B)
        Wo = np.transpose(obsv(A,C));
        np.testing.assert_array_almost_equal(Wc,Wo)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testGramWc(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5. 6.; 7. 8.")
        C = np.matrix("4. 5.; 6. 7.")
        D = np.matrix("13. 14.; 15. 16.")
        sys = ss(A, B, C, D)
        Wctrue = np.matrix("18.5 24.5; 24.5 32.5")
        Wc = gram(sys,'c')
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testGramRc(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5. 6.; 7. 8.")
        C = np.matrix("4. 5.; 6. 7.")
        D = np.matrix("13. 14.; 15. 16.")
        sys = ss(A, B, C, D)
        Rctrue = np.matrix("4.30116263 5.6961343; 0. 0.23249528")
        Rc = gram(sys,'cf')
        np.testing.assert_array_almost_equal(Rc, Rctrue)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testGramWo(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5. 6.; 7. 8.")
        C = np.matrix("4. 5.; 6. 7.")
        D = np.matrix("13. 14.; 15. 16.")
        sys = ss(A, B, C, D)
        Wotrue = np.matrix("257.5 -94.5; -94.5 56.5")
        Wo = gram(sys,'o')
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testGramWo2(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5.; 7.")
        C = np.matrix("6. 8.")
        D = np.matrix("9.")
        sys = ss(A,B,C,D)
        Wotrue = np.matrix("198. -72.; -72. 44.")
        Wo = gram(sys,'o')
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testGramRo(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5. 6.; 7. 8.")
        C = np.matrix("4. 5.; 6. 7.")
        D = np.matrix("13. 14.; 15. 16.")
        sys = ss(A, B, C, D)
        Rotrue = np.matrix("16.04680654 -5.8890222; 0. 4.67112593")
        Ro = gram(sys,'of')
        np.testing.assert_array_almost_equal(Ro, Rotrue)

    def testGramsys(self):
        num =[1.]
        den = [1., 1., 1.]
        sys = tf(num,den)
        self.assertRaises(ValueError, gram, sys, 'o')
        self.assertRaises(ValueError, gram, sys, 'c')

    def testAcker(self):
        for states in range(1, self.maxStates):
            for i in range(self.maxTries):
                # start with a random SS system and transform to TF then
                # back to SS, check that the matrices are the same.
                sys = rss(states, 1, 1)
                if (self.debug):
                    print(sys)

                # Make sure the system is not degenerate
                Cmat = ctrb(sys.A, sys.B)
                if np.linalg.matrix_rank(Cmat) != states:
                    if (self.debug):
                        print("  skipping (not reachable or ill conditioned)")
                        continue

                # Place the poles at random locations
                des = rss(states, 1, 1);
                poles = pole(des)

                # Now place the poles using acker
                K = acker(sys.A, sys.B, poles)
                new = ss(sys.A - sys.B * K, sys.B, sys.C, sys.D)
                placed = pole(new)

                # Debugging code
                # diff = np.sort(poles) - np.sort(placed)
                # if not all(diff < 0.001):
                #     print("Found a problem:")
                #     print(sys)
                #     print("desired = ", poles)

                np.testing.assert_array_almost_equal(np.sort(poles),
                                                     np.sort(placed), decimal=4)

    def testPlace(self):
        # Matrices shamelessly stolen from scipy example code.
        A = np.array([[1.380,  -0.2077,  6.715, -5.676],
                      [-0.5814, -4.290,   0,      0.6750],
                      [1.067,   4.273,  -6.654,  5.893],
                      [0.0480,  4.273,   1.343, -2.104]])

        B = np.array([[0,      5.679],
                      [1.136,  1.136],
                      [0,      0,],
                      [-3.146,  0]])
        P = np.array([-0.5+1j, -0.5-1j, -5.0566, -8.6659])
        K = place(A, B, P)
        P_placed = np.linalg.eigvals(A - B.dot(K))
        # No guarantee of the ordering, so sort them
        P.sort()
        P_placed.sort()
        np.testing.assert_array_almost_equal(P, P_placed)

        # Test that the dimension checks work.
        np.testing.assert_raises(ControlDimension, place, A[1:, :], B, P)
        np.testing.assert_raises(ControlDimension, place, A, B[1:, :], P)

        # Check that we get an error if we ask for too many poles in the same
        # location. Here, rank(B) = 2, so lets place three at the same spot.
        P_repeated = np.array([-0.5, -0.5, -0.5, -8.6659])
        np.testing.assert_raises(ValueError, place, A, B, P_repeated)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testPlace_varga_continuous(self):
        """
        Check that we can place eigenvalues for dtime=False
        """
        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5.], [7.]])

        P = np.array([-2., -2.])
        K = place_varga(A, B, P)
        P_placed = np.linalg.eigvals(A - B.dot(K))
        # No guarantee of the ordering, so sort them
        P.sort()
        P_placed.sort()
        np.testing.assert_array_almost_equal(P, P_placed)

        # Test that the dimension checks work.
        np.testing.assert_raises(ControlDimension, place, A[1:, :], B, P)
        np.testing.assert_raises(ControlDimension, place, A, B[1:, :], P)

        # Regression test against bug #177
        # https://github.com/python-control/python-control/issues/177
        A = np.array([[0, 1], [100, 0]])
        B = np.array([[0], [1]])
        P = np.array([-20 + 10*1j, -20 - 10*1j])
        K = place_varga(A, B, P)
        P_placed = np.linalg.eigvals(A - B.dot(K))

        # No guarantee of the ordering, so sort them
        P.sort()
        P_placed.sort()
        np.testing.assert_array_almost_equal(P, P_placed)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testPlace_varga_continuous_partial_eigs(self):
        """
        Check that we are able to use the alpha parameter to only place
        a subset of the eigenvalues, for the continous time case.
        """
        # A matrix has eigenvalues at s=-1, and s=-2. Choose alpha = -1.5
        # and check that eigenvalue at s=-2 stays put.
        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5.], [7.]])

        P = np.array([-3.])
        P_expected = np.array([-2.0, -3.0])
        alpha = -1.5
        K = place_varga(A, B, P, alpha=alpha)

        P_placed = np.linalg.eigvals(A - B.dot(K))
        # No guarantee of the ordering, so sort them
        P_expected.sort()
        P_placed.sort()
        np.testing.assert_array_almost_equal(P_expected, P_placed)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testPlace_varga_discrete(self):
        """
        Check that we can place poles using dtime=True (discrete time)
        """
        A = np.array([[1., 0], [0, 0.5]])
        B = np.array([[5.], [7.]])

        P = np.array([0.5, 0.5])
        K = place_varga(A, B, P, dtime=True)
        P_placed = np.linalg.eigvals(A - B.dot(K))
        # No guarantee of the ordering, so sort them
        P.sort()
        P_placed.sort()
        np.testing.assert_array_almost_equal(P, P_placed)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testPlace_varga_discrete_partial_eigs(self):
        """"
        Check that we can only assign a single eigenvalue in the discrete
        time case.
        """
        # A matrix has eigenvalues at 1.0 and 0.5. Set alpha = 0.51, and
        # check that the eigenvalue at 0.5 is not moved.
        A = np.array([[1., 0], [0, 0.5]])
        B = np.array([[5.], [7.]])
        P = np.array([0.2, 0.6])
        P_expected = np.array([0.5, 0.6])
        alpha = 0.51
        K = place_varga(A, B, P, dtime=True, alpha=alpha)
        P_placed = np.linalg.eigvals(A - B.dot(K))
        P_expected.sort()
        P_placed.sort()
        np.testing.assert_array_almost_equal(P_expected, P_placed)


    def check_LQR(self, K, S, poles, Q, R):
        S_expected = np.array(np.sqrt(Q * R))
        K_expected = S_expected / R
        poles_expected = np.array([-K_expected])
        np.testing.assert_array_almost_equal(S, S_expected)
        np.testing.assert_array_almost_equal(K, K_expected)
        np.testing.assert_array_almost_equal(poles, poles_expected)


    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_LQR_integrator(self):
        A, B, Q, R = 0., 1., 10., 2.
        K, S, poles = lqr(A, B, Q, R)
        self.check_LQR(K, S, poles, Q, R)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_LQR_3args(self):
        sys = ss(0., 1., 1., 0.)
        Q, R = 10., 2.
        K, S, poles = lqr(sys, Q, R)
        self.check_LQR(K, S, poles, Q, R)

    def check_LQE(self, L, P, poles, G, QN, RN):
        P_expected = np.array(np.sqrt(G*QN*G * RN))
        L_expected = P_expected / RN
        poles_expected = np.array([-L_expected], ndmin=2)
        np.testing.assert_array_almost_equal(P, P_expected)
        np.testing.assert_array_almost_equal(L, L_expected)
        np.testing.assert_array_almost_equal(poles, poles_expected)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_LQE(self):
        A, G, C, QN, RN = 0., .1, 1., 10., 2.
        L, P, poles = lqe(A, G, C, QN, RN)
        self.check_LQE(L, P, poles, G, QN, RN)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_care(self):
        #unit test for stabilizing and anti-stabilizing feedbacks
        #continuous-time

        A = np.diag([1,-1])
        B = np.identity(2)
        Q = np.identity(2)
        R = np.identity(2)
        S = 0 * B
        E = np.identity(2)
        X, L , G = care(A, B, Q, R, S, E, stabilizing=True)
        assert np.all(np.real(L) < 0)
        X, L , G = care(A, B, Q, R, S, E, stabilizing=False)
        assert np.all(np.real(L) > 0)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_dare(self):
        #discrete-time
        A = np.diag([0.5,2])
        B = np.identity(2)
        Q = np.identity(2)
        R = np.identity(2)
        S = 0 * B
        E = np.identity(2)
        X, L , G = dare(A, B, Q, R, S, E, stabilizing=True)
        assert np.all(np.abs(L) < 1)
        X, L , G = dare(A, B, Q, R, S, E, stabilizing=False)
        assert np.all(np.abs(L) > 1)


def test_suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestStatefbk)

if __name__ == '__main__':
    unittest.main()
