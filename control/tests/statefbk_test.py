"""statefbk_test.py - test state feedback functions

RMM, 30 Mar 2011 (based on TestStatefbk from v0.4a)
"""

import numpy as np
import pytest

from control import lqe, pole, rss, ss, tf
from control.exception import ControlDimension
from control.mateqn import care, dare
from control.statefbk import ctrb, obsv, place, place_varga, lqr, gram, acker
from control.tests.conftest import (slycotonly, check_deprecated_matrix,
                                    ismatarrayout, asmatarrayout)


@pytest.fixture
def fixedseed():
    """Get consistent test results"""
    np.random.seed(0)


class TestStatefbk:
    """Test state feedback functions"""

    # Maximum number of states to test + 1
    maxStates = 5
    # Maximum number of inputs and outputs to test + 1
    maxTries = 4
    # Set to True to print systems to the output.
    debug = False

    def testCtrbSISO(self, matarrayin, matarrayout):
        A = matarrayin([[1., 2.], [3., 4.]])
        B = matarrayin([[5.], [7.]])
        Wctrue = np.array([[5., 19.], [7., 43.]])

        with check_deprecated_matrix():
            Wc = ctrb(A, B)
        assert ismatarrayout(Wc)

        np.testing.assert_array_almost_equal(Wc, Wctrue)

    def testCtrbMIMO(self, matarrayin):
        A = matarrayin([[1., 2.], [3., 4.]])
        B = matarrayin([[5., 6.], [7., 8.]])
        Wctrue = np.array([[5., 6., 19., 22.], [7., 8., 43., 50.]])
        Wc = ctrb(A, B)
        np.testing.assert_array_almost_equal(Wc, Wctrue)

        # Make sure default type values are correct
        assert ismatarrayout(Wc)

    def testObsvSISO(self, matarrayin):
        A = matarrayin([[1., 2.], [3., 4.]])
        C = matarrayin([[5., 7.]])
        Wotrue = np.array([[5., 7.], [26., 38.]])
        Wo = obsv(A, C)
        np.testing.assert_array_almost_equal(Wo, Wotrue)

        # Make sure default type values are correct
        assert ismatarrayout(Wo)


    def testObsvMIMO(self, matarrayin):
        A = matarrayin([[1., 2.], [3., 4.]])
        C = matarrayin([[5., 6.], [7., 8.]])
        Wotrue = np.array([[5., 6.], [7., 8.], [23., 34.], [31., 46.]])
        Wo = obsv(A, C)
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    def testCtrbObsvDuality(self, matarrayin):
        A = matarrayin([[1.2, -2.3], [3.4, -4.5]])
        B = matarrayin([[5.8, 6.9], [8., 9.1]])
        Wc = ctrb(A, B)
        A = np.transpose(A)
        C = np.transpose(B)
        Wo = np.transpose(obsv(A, C));
        np.testing.assert_array_almost_equal(Wc,Wo)

    @slycotonly
    def testGramWc(self, matarrayin, matarrayout):
        A = matarrayin([[1., -2.], [3., -4.]])
        B = matarrayin([[5., 6.], [7., 8.]])
        C = matarrayin([[4., 5.], [6., 7.]])
        D = matarrayin([[13., 14.], [15., 16.]])
        sys = ss(A, B, C, D)
        Wctrue = np.array([[18.5, 24.5], [24.5, 32.5]])

        with check_deprecated_matrix():
            Wc = gram(sys, 'c')

        assert ismatarrayout(Wc)
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    @slycotonly
    def testGramRc(self, matarrayin):
        A = matarrayin([[1., -2.], [3., -4.]])
        B = matarrayin([[5., 6.], [7., 8.]])
        C = matarrayin([[4., 5.], [6., 7.]])
        D = matarrayin([[13., 14.], [15., 16.]])
        sys = ss(A, B, C, D)
        Rctrue = np.array([[4.30116263, 5.6961343], [0., 0.23249528]])
        Rc = gram(sys, 'cf')
        np.testing.assert_array_almost_equal(Rc, Rctrue)

    @slycotonly
    def testGramWo(self, matarrayin):
        A = matarrayin([[1., -2.], [3., -4.]])
        B = matarrayin([[5., 6.], [7., 8.]])
        C = matarrayin([[4., 5.], [6., 7.]])
        D = matarrayin([[13., 14.], [15., 16.]])
        sys = ss(A, B, C, D)
        Wotrue = np.array([[257.5, -94.5], [-94.5, 56.5]])
        Wo = gram(sys, 'o')
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    @slycotonly
    def testGramWo2(self, matarrayin):
        A = matarrayin([[1., -2.], [3., -4.]])
        B = matarrayin([[5.], [7.]])
        C = matarrayin([[6., 8.]])
        D = matarrayin([[9.]])
        sys = ss(A,B,C,D)
        Wotrue = np.array([[198., -72.], [-72., 44.]])
        Wo = gram(sys, 'o')
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    @slycotonly
    def testGramRo(self, matarrayin):
        A = matarrayin([[1., -2.], [3., -4.]])
        B = matarrayin([[5., 6.], [7., 8.]])
        C = matarrayin([[4., 5.], [6., 7.]])
        D = matarrayin([[13., 14.], [15., 16.]])
        sys = ss(A, B, C, D)
        Rotrue = np.array([[16.04680654, -5.8890222], [0., 4.67112593]])
        Ro = gram(sys, 'of')
        np.testing.assert_array_almost_equal(Ro, Rotrue)

    def testGramsys(self):
        num =[1.]
        den = [1., 1., 1.]
        sys = tf(num,den)
        with pytest.raises(ValueError):
            gram(sys, 'o')
        with pytest.raises(ValueError):
            gram(sys, 'c')

    def testAcker(self, fixedseed):
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

    def checkPlaced(self, P_expected, P_placed):
        """Check that placed poles are correct"""
        # No guarantee of the ordering, so sort them
        P_expected = np.squeeze(np.asarray(P_expected))
        P_expected.sort()
        P_placed.sort()
        np.testing.assert_array_almost_equal(P_expected, P_placed)

    def testPlace(self, matarrayin):
        # Matrices shamelessly stolen from scipy example code.
        A = matarrayin([[1.380, -0.2077, 6.715, -5.676],
                        [-0.5814, -4.290, 0, 0.6750],
                        [1.067, 4.273, -6.654, 5.893],
                        [0.0480, 4.273, 1.343, -2.104]])
        B = matarrayin([[0, 5.679],
                        [1.136, 1.136],
                        [0, 0],
                        [-3.146, 0]])
        P = matarrayin([-0.5 + 1j, -0.5 - 1j, -5.0566, -8.6659])
        K = place(A, B, P)
        assert ismatarrayout(K)
        P_placed = np.linalg.eigvals(A - B.dot(K))
        self.checkPlaced(P, P_placed)

        # Test that the dimension checks work.
        with pytest.raises(ControlDimension):
            place(A[1:, :], B, P)
        with pytest.raises(ControlDimension):
            place(A, B[1:, :], P)

        # Check that we get an error if we ask for too many poles in the same
        # location. Here, rank(B) = 2, so lets place three at the same spot.
        P_repeated = matarrayin([-0.5, -0.5, -0.5, -8.6659])
        with pytest.raises(ValueError):
            place(A, B, P_repeated)

    @slycotonly
    def testPlace_varga_continuous(self, matarrayin):
        """
        Check that we can place eigenvalues for dtime=False
        """
        A = matarrayin([[1., -2.], [3., -4.]])
        B = matarrayin([[5.], [7.]])

        P = [-2., -2.]
        K = place_varga(A, B, P)
        P_placed = np.linalg.eigvals(A - B.dot(K))
        self.checkPlaced(P, P_placed)

        # Test that the dimension checks work.
        np.testing.assert_raises(ControlDimension, place, A[1:, :], B, P)
        np.testing.assert_raises(ControlDimension, place, A, B[1:, :], P)

        # Regression test against bug #177
        # https://github.com/python-control/python-control/issues/177
        A = matarrayin([[0, 1], [100, 0]])
        B = matarrayin([[0], [1]])
        P = matarrayin([-20 + 10*1j, -20 - 10*1j])
        K = place_varga(A, B, P)
        P_placed = np.linalg.eigvals(A - B.dot(K))
        self.checkPlaced(P, P_placed)


    @slycotonly
    def testPlace_varga_continuous_partial_eigs(self, matarrayin):
        """
        Check that we are able to use the alpha parameter to only place
        a subset of the eigenvalues, for the continous time case.
        """
        # A matrix has eigenvalues at s=-1, and s=-2. Choose alpha = -1.5
        # and check that eigenvalue at s=-2 stays put.
        A = matarrayin([[1., -2.], [3., -4.]])
        B = matarrayin([[5.], [7.]])

        P = matarrayin([-3.])
        P_expected = np.array([-2.0, -3.0])
        alpha = -1.5
        K = place_varga(A, B, P, alpha=alpha)

        P_placed = np.linalg.eigvals(A - B.dot(K))
        # No guarantee of the ordering, so sort them
        self.checkPlaced(P_expected, P_placed)

    @slycotonly
    def testPlace_varga_discrete(self, matarrayin):
        """
        Check that we can place poles using dtime=True (discrete time)
        """
        A = matarrayin([[1., 0], [0, 0.5]])
        B = matarrayin([[5.], [7.]])

        P = matarrayin([0.5, 0.5])
        K = place_varga(A, B, P, dtime=True)
        P_placed = np.linalg.eigvals(A - B.dot(K))
        # No guarantee of the ordering, so sort them
        self.checkPlaced(P, P_placed)

    @slycotonly
    def testPlace_varga_discrete_partial_eigs(self, matarrayin):
        """"
        Check that we can only assign a single eigenvalue in the discrete
        time case.
        """
        # A matrix has eigenvalues at 1.0 and 0.5. Set alpha = 0.51, and
        # check that the eigenvalue at 0.5 is not moved.
        A = matarrayin([[1., 0], [0, 0.5]])
        B = matarrayin([[5.], [7.]])
        P = matarrayin([0.2, 0.6])
        P_expected = np.array([0.5, 0.6])
        alpha = 0.51
        K = place_varga(A, B, P, dtime=True, alpha=alpha)
        P_placed = np.linalg.eigvals(A - B.dot(K))
        self.checkPlaced(P_expected, P_placed)


    def check_LQR(self, K, S, poles, Q, R):
        S_expected = asmatarrayout(np.sqrt(Q.dot(R)))
        K_expected = asmatarrayout(S_expected / R)
        poles_expected = -np.squeeze(np.asarray(K_expected))
        np.testing.assert_array_almost_equal(S, S_expected)
        np.testing.assert_array_almost_equal(K, K_expected)
        np.testing.assert_array_almost_equal(poles, poles_expected)


    @slycotonly
    def test_LQR_integrator(self, matarrayin, matarrayout):
        A, B, Q, R = (matarrayin([[X]]) for X in [0., 1., 10., 2.])
        K, S, poles = lqr(A, B, Q, R)
        self.check_LQR(K, S, poles, Q, R)

    @slycotonly
    def test_LQR_3args(self, matarrayin, matarrayout):
        sys = ss(0., 1., 1., 0.)
        Q, R = (matarrayin([[X]]) for X in [10., 2.])
        K, S, poles = lqr(sys, Q, R)
        self.check_LQR(K, S, poles, Q, R)

    @slycotonly
    @pytest.mark.xfail(reason="warning not implemented")
    def testLQR_warning(self):
        """Test lqr()

        Make sure we get a warning if [Q N;N' R] is not positive semi-definite
        """
        # from matlab_test siso.ss2 (testLQR); probably not referenced before
        # not yet implemented check
        A = np.array([[-2, 3, 1],
                      [-1, 0, 0],
                      [0, 1, 0]])
        B = np.array([[-1, 0, 0]]).T
        Q = np.eye(3)
        R = np.eye(1)
        N = np.array([[1, 1, 2]]).T
        # assert any(np.linalg.eigvals(np.block([[Q, N], [N.T, R]])) < 0)
        with pytest.warns(UserWarning):
            (K, S, E) = lqr(A, B, Q, R, N)

    def check_LQE(self, L, P, poles, G, QN, RN):
        P_expected = asmatarrayout(np.sqrt(G.dot(QN.dot(G).dot(RN))))
        L_expected = asmatarrayout(P_expected / RN)
        poles_expected = -np.squeeze(np.asarray(L_expected))
        np.testing.assert_array_almost_equal(P, P_expected)
        np.testing.assert_array_almost_equal(L, L_expected)
        np.testing.assert_array_almost_equal(poles, poles_expected)

    @slycotonly
    def test_LQE(self, matarrayin):
        A, G, C, QN, RN = (matarrayin([[X]]) for X in [0., .1, 1., 10., 2.])
        L, P, poles = lqe(A, G, C, QN, RN)
        self.check_LQE(L, P, poles, G, QN, RN)

    @slycotonly
    def test_care(self, matarrayin):
        """Test stabilizing and anti-stabilizing feedbacks, continuous"""
        A = matarrayin(np.diag([1, -1]))
        B = matarrayin(np.identity(2))
        Q = matarrayin(np.identity(2))
        R = matarrayin(np.identity(2))
        S = matarrayin(np.zeros((2, 2)))
        E = matarrayin(np.identity(2))
        X, L, G = care(A, B, Q, R, S, E, stabilizing=True)
        assert np.all(np.real(L) < 0)
        X, L, G = care(A, B, Q, R, S, E, stabilizing=False)
        assert np.all(np.real(L) > 0)

    @slycotonly
    def test_dare(self, matarrayin):
        """Test stabilizing and anti-stabilizing feedbacks, discrete"""
        A = matarrayin(np.diag([0.5, 2]))
        B = matarrayin(np.identity(2))
        Q = matarrayin(np.identity(2))
        R = matarrayin(np.identity(2))
        S = matarrayin(np.zeros((2, 2)))
        E = matarrayin(np.identity(2))
        X, L, G = dare(A, B, Q, R, S, E, stabilizing=True)
        assert np.all(np.abs(L) < 1)
        X, L, G = dare(A, B, Q, R, S, E, stabilizing=False)
        assert np.all(np.abs(L) > 1)
