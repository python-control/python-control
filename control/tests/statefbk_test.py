"""statefbk_test.py - test state feedback functions

RMM, 30 Mar 2011 (based on TestStatefbk from v0.4a)
"""

import numpy as np
import pytest
import itertools
import warnings
from math import pi

import control as ct
from control import poles, rss, ss, tf
from control.exception import ControlDimension, ControlSlycot, \
    ControlArgument, slycot_check
from control.mateqn import care, dare
from control.statefbk import (ctrb, obsv, place, place_varga, lqr, dlqr,
                              gram, place_acker)
from control.tests.conftest import slycotonly


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

    def testCtrbSISO(self):
        A = np.array([[1., 2.], [3., 4.]])
        B = np.array([[5.], [7.]])
        Wctrue = np.array([[5., 19.], [7., 43.]])
        Wc = ctrb(A, B)
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    def testCtrbMIMO(self):
        A = np.array([[1., 2.], [3., 4.]])
        B = np.array([[5., 6.], [7., 8.]])
        Wctrue = np.array([[5., 6., 19., 22.], [7., 8., 43., 50.]])
        Wc = ctrb(A, B)
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    def testCtrbT(self):
        A = np.array([[1., 2.], [3., 4.]])
        B = np.array([[5., 6.], [7., 8.]])
        t = 1
        Wctrue = np.array([[5., 6.], [7., 8.]])
        Wc = ctrb(A, B, t=t)
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    def testCtrbNdim1(self):
        # gh-1097: treat 1-dim B as nx1
        A = np.array([[1., 2.], [3., 4.]])
        B = np.array([5., 7.])
        Wctrue = np.array([[5., 19.], [7., 43.]])
        Wc = ctrb(A, B)
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    def testCtrbRejectMismatch(self):
        # gh-1097: check A, B for compatible shapes
        with pytest.raises(
                ControlDimension, match='.* A must be a square matrix'):
            ctrb([[1,2]],[1])
        with pytest.raises(
                ControlDimension, match='B has the wrong number of rows'):
            ctrb([[1,2],[2,3]], 1)
        with pytest.raises(
                ControlDimension, match='B has the wrong number of rows'):
            ctrb([[1,2],[2,3]], [[1,2]])

    def testObsvSISO(self):
        A = np.array([[1., 2.], [3., 4.]])
        C = np.array([[5., 7.]])
        Wotrue = np.array([[5., 7.], [26., 38.]])
        Wo = obsv(A, C)
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    def testObsvMIMO(self):
        A = np.array([[1., 2.], [3., 4.]])
        C = np.array([[5., 6.], [7., 8.]])
        Wotrue = np.array([[5., 6.], [7., 8.], [23., 34.], [31., 46.]])
        Wo = obsv(A, C)
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    def testObsvT(self):
        A = np.array([[1., 2.], [3., 4.]])
        C = np.array([[5., 6.], [7., 8.]])
        t = 1
        Wotrue = np.array([[5., 6.], [7., 8.]])
        Wo = obsv(A, C, t=t)
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    def testObsvNdim1(self):
        # gh-1097: treat 1-dim C as 1xn
        A = np.array([[1., 2.], [3., 4.]])
        C = np.array([5., 7.])
        Wotrue = np.array([[5., 7.], [26., 38.]])
        Wo = obsv(A, C)
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    def testObsvRejectMismatch(self):
        # gh-1097: check A, C for compatible shapes
        with pytest.raises(
                ControlDimension, match='.* A must be a square matrix'):
            obsv([[1,2]],[1])
        with pytest.raises(
                ControlDimension, match='C has the wrong number of columns'):
            obsv([[1,2],[2,3]], 1)
        with pytest.raises(
                ControlDimension, match='C has the wrong number of columns'):
            obsv([[1,2],[2,3]], [[1],[2]])

    def testCtrbObsvDuality(self):
        A = np.array([[1.2, -2.3], [3.4, -4.5]])
        B = np.array([[5.8, 6.9], [8., 9.1]])
        Wc = ctrb(A, B)
        A = np.transpose(A)
        C = np.transpose(B)
        Wo = np.transpose(obsv(A, C))
        np.testing.assert_array_almost_equal(Wc,Wo)

    @slycotonly
    def testGramWc(self):
        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5., 6.], [7., 8.]])
        C = np.array([[4., 5.], [6., 7.]])
        D = np.array([[13., 14.], [15., 16.]])
        sys = ss(A, B, C, D)
        Wctrue = np.array([[18.5, 24.5], [24.5, 32.5]])
        Wc = gram(sys, 'c')
        np.testing.assert_array_almost_equal(Wc, Wctrue)
        sysd = ct.c2d(sys, 0.2)
        Wctrue = np.array([[3.666767, 4.853625],
                           [4.853625, 6.435233]])
        Wc = gram(sysd, 'c')
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    @slycotonly
    def testGramWc2(self):
        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5.], [7.]])
        C = np.array([[6., 8.]])
        D = np.array([[9.]])
        sys = ss(A,B,C,D)
        Wctrue = np.array([[ 7.166667,  9.833333],
                           [ 9.833333,  13.5]])
        Wc = gram(sys, 'c')
        np.testing.assert_array_almost_equal(Wc, Wctrue)
        sysd = ct.c2d(sys, 0.2)
        Wctrue = np.array([[1.418978, 1.946180],
                           [1.946180, 2.670758]])
        Wc = gram(sysd, 'c')
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    @slycotonly
    def testGramRc(self):
        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5., 6.], [7., 8.]])
        C = np.array([[4., 5.], [6., 7.]])
        D = np.array([[13., 14.], [15., 16.]])
        sys = ss(A, B, C, D)
        Rctrue = np.array([[4.30116263, 5.6961343], [0., 0.23249528]])
        Rc = gram(sys, 'cf')
        np.testing.assert_array_almost_equal(Rc, Rctrue)
        sysd = ct.c2d(sys, 0.2)
        Rctrue = np.array([[1.91488054, 2.53468814],
                           [0.        , 0.10290372]])
        Rc = gram(sysd, 'cf')
        np.testing.assert_array_almost_equal(Rc, Rctrue)

    @slycotonly
    def testGramWo(self):
        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5., 6.], [7., 8.]])
        C = np.array([[4., 5.], [6., 7.]])
        D = np.array([[13., 14.], [15., 16.]])
        sys = ss(A, B, C, D)
        Wotrue = np.array([[257.5, -94.5], [-94.5, 56.5]])
        Wo = gram(sys, 'o')
        np.testing.assert_array_almost_equal(Wo, Wotrue)
        sysd = ct.c2d(sys, 0.2)
        Wotrue = np.array([[ 1305.369179, -440.046414],
                           [ -440.046414,  333.034844]])
        Wo = gram(sysd, 'o')
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    @slycotonly
    def testGramWo2(self):
        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5.], [7.]])
        C = np.array([[6., 8.]])
        D = np.array([[9.]])
        sys = ss(A,B,C,D)
        Wotrue = np.array([[198., -72.], [-72., 44.]])
        Wo = gram(sys, 'o')
        np.testing.assert_array_almost_equal(Wo, Wotrue)
        sysd = ct.c2d(sys, 0.2)
        Wotrue = np.array([[ 1001.835511, -335.337663],
                           [ -335.337663,  263.355793]])
        Wo = gram(sysd, 'o')
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    @slycotonly
    def testGramRo(self):
        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5., 6.], [7., 8.]])
        C = np.array([[4., 5.], [6., 7.]])
        D = np.array([[13., 14.], [15., 16.]])
        sys = ss(A, B, C, D)
        Rotrue = np.array([[16.04680654, -5.8890222], [0., 4.67112593]])
        Ro = gram(sys, 'of')
        np.testing.assert_array_almost_equal(Ro, Rotrue)
        sysd = ct.c2d(sys, 0.2)
        Rotrue = np.array([[ 36.12989315, -12.17956588],
                           [  0.        ,  13.59018097]])
        Ro = gram(sysd, 'of')
        np.testing.assert_array_almost_equal(Ro, Rotrue)

    def testGramsys(self):
        sys = tf([1.], [1., 1., 1.])
        with pytest.raises(ValueError) as excinfo:
            gram(sys, 'o')
        assert "must be StateSpace" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            gram(sys, 'c')
        assert "must be StateSpace" in str(excinfo.value)
        sys = tf([1], [1, -1], 0.5)
        with pytest.raises(ValueError) as excinfo:
            gram(sys, 'o')
        assert "must be StateSpace" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            gram(sys, 'c')
        assert "must be StateSpace" in str(excinfo.value)
        sys = ct.ss(sys)  # this system is unstable
        with pytest.raises(ValueError) as excinfo:
            gram(sys, 'o')
        assert "is unstable" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            gram(sys, 'c')
        assert "is unstable" in str(excinfo.value)

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
                des = rss(states, 1, 1)
                desired = poles(des)

                # Now place the poles using acker
                K = place_acker(sys.A, sys.B, desired)
                new = ss(sys.A - sys.B * K, sys.B, sys.C, sys.D)
                placed = poles(new)

                # Debugging code
                # diff = np.sort(poles) - np.sort(placed)
                # if not all(diff < 0.001):
                #     print("Found a problem:")
                #     print(sys)
                #     print("desired = ", poles)

                np.testing.assert_array_almost_equal(
                    np.sort(desired), np.sort(placed), decimal=4)

    def checkPlaced(self, P_expected, P_placed):
        """Check that placed poles are correct"""
        # No guarantee of the ordering, so sort them
        P_expected = np.squeeze(np.asarray(P_expected))
        P_expected.sort()
        P_placed.sort()
        np.testing.assert_array_almost_equal(P_expected, P_placed)

    def testPlace(self):
        # Matrices shamelessly stolen from scipy example code.
        A = np.array([[1.380, -0.2077, 6.715, -5.676],
                        [-0.5814, -4.290, 0, 0.6750],
                        [1.067, 4.273, -6.654, 5.893],
                        [0.0480, 4.273, 1.343, -2.104]])
        B = np.array([[0, 5.679],
                        [1.136, 1.136],
                        [0, 0],
                        [-3.146, 0]])
        P = np.array([-0.5 + 1j, -0.5 - 1j, -5.0566, -8.6659])
        K = place(A, B, P)
        P_placed = np.linalg.eigvals(A - B @ K)
        self.checkPlaced(P, P_placed)

        # Test that the dimension checks work.
        with pytest.raises(ControlDimension):
            place(A[1:, :], B, P)
        with pytest.raises(ControlDimension):
            place(A, B[1:, :], P)

        # Check that we get an error if we ask for too many poles in the same
        # location. Here, rank(B) = 2, so lets place three at the same spot.
        P_repeated = np.array([-0.5, -0.5, -0.5, -8.6659])
        with pytest.raises(ValueError):
            place(A, B, P_repeated)

    @slycotonly
    def testPlace_varga_continuous(self):
        """
        Check that we can place eigenvalues for dtime=False
        """
        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5.], [7.]])

        P = [-2., -2.]
        K = place_varga(A, B, P)
        P_placed = np.linalg.eigvals(A - B @ K)
        self.checkPlaced(P, P_placed)

        # Test that the dimension checks work.
        np.testing.assert_raises(ControlDimension, place, A[1:, :], B, P)
        np.testing.assert_raises(ControlDimension, place, A, B[1:, :], P)

        # Regression test against bug #177
        # https://github.com/python-control/python-control/issues/177
        A = np.array([[0, 1], [100, 0]])
        B = np.array([[0], [1]])
        P = np.array([-20 + 10*1j, -20 - 10*1j])
        K = place_varga(A, B, P)
        P_placed = np.linalg.eigvals(A - B @ K)
        self.checkPlaced(P, P_placed)


    @slycotonly
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

        P_placed = np.linalg.eigvals(A - B @ K)
        # No guarantee of the ordering, so sort them
        self.checkPlaced(P_expected, P_placed)

    @slycotonly
    def testPlace_varga_discrete(self):
        """
        Check that we can place poles using dtime=True (discrete time)
        """
        A = np.array([[1., 0], [0, 0.5]])
        B = np.array([[5.], [7.]])

        P = np.array([0.5, 0.5])
        K = place_varga(A, B, P, dtime=True)
        P_placed = np.linalg.eigvals(A - B @ K)
        # No guarantee of the ordering, so sort them
        self.checkPlaced(P, P_placed)

    @slycotonly
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
        P_placed = np.linalg.eigvals(A - B @ K)
        self.checkPlaced(P_expected, P_placed)

    def check_LQR(self, K, S, poles, Q, R):
        S_expected = np.sqrt(Q @ R)
        K_expected = S_expected / R
        poles_expected = -np.squeeze(np.asarray(K_expected))
        np.testing.assert_array_almost_equal(S, S_expected)
        np.testing.assert_array_almost_equal(K, K_expected)
        np.testing.assert_array_almost_equal(poles, poles_expected)

    def check_DLQR(self, K, S, poles, Q, R):
        S_expected = Q
        K_expected = 0
        poles_expected = -np.squeeze(np.asarray(K_expected))
        np.testing.assert_array_almost_equal(S, S_expected)
        np.testing.assert_array_almost_equal(K, K_expected)
        np.testing.assert_array_almost_equal(poles, poles_expected)

    @pytest.mark.parametrize("method", [None, 'slycot', 'scipy'])
    def test_LQR_integrator(self, method):
        if method == 'slycot' and not slycot_check():
            return
        A, B, Q, R = (np.array([[X]]) for X in [0., 1., 10., 2.])
        K, S, poles = lqr(A, B, Q, R, method=method)
        self.check_LQR(K, S, poles, Q, R)

    @pytest.mark.parametrize("method", [None, 'slycot', 'scipy'])
    def test_LQR_3args(self, method):
        if method == 'slycot' and not slycot_check():
            return
        sys = ss(0., 1., 1., 0.)
        Q, R = (np.array([[X]]) for X in [10., 2.])
        K, S, poles = lqr(sys, Q, R, method=method)
        self.check_LQR(K, S, poles, Q, R)

    @pytest.mark.parametrize("method", [None, 'slycot', 'scipy'])
    def test_DLQR_3args(self, method):
        if method == 'slycot' and not slycot_check():
            return
        dsys = ss(0., 1., 1., 0., .1)
        Q, R = (np.array([[X]]) for X in [10., 2.])
        K, S, poles = dlqr(dsys, Q, R, method=method)
        self.check_DLQR(K, S, poles, Q, R)

    def test_DLQR_4args(self):
        A, B, Q, R = (np.array([[X]]) for X in [0., 1., 10., 2.])
        K, S, poles = dlqr(A, B, Q, R)
        self.check_DLQR(K, S, poles, Q, R)

    @pytest.mark.parametrize("cdlqr", [lqr, dlqr])
    def test_lqr_badmethod(self, cdlqr):
        A, B, Q, R = 0, 1, 10, 2
        with pytest.raises(ControlArgument, match="Unknown method"):
            K, S, poles = cdlqr(A, B, Q, R, method='nosuchmethod')

    @pytest.mark.parametrize("cdlqr", [lqr, dlqr])
    def test_lqr_slycot_not_installed(self, cdlqr):
        A, B, Q, R = 0, 1, 10, 2
        if not slycot_check():
            with pytest.raises(ControlSlycot, match="Can't find slycot"):
                K, S, poles = cdlqr(A, B, Q, R, method='slycot')

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

    @pytest.mark.parametrize("cdlqr", [lqr, dlqr])
    def test_lqr_call_format(self, cdlqr):
        # Create a random state space system for testing
        sys = rss(2, 3, 2)
        sys.dt = None           # treat as either continuous or discrete time

        # Weighting matrices
        Q = np.eye(sys.nstates)
        R = np.eye(sys.ninputs)
        N = np.zeros((sys.nstates, sys.ninputs))

        # Standard calling format
        Kref, Sref, Eref = cdlqr(sys.A, sys.B, Q, R)

        # Call with system instead of matricees
        K, S, E = cdlqr(sys, Q, R)
        np.testing.assert_array_almost_equal(Kref, K)
        np.testing.assert_array_almost_equal(Sref, S)
        np.testing.assert_array_almost_equal(Eref, E)

        # Pass a cross-weighting matrix
        K, S, E = cdlqr(sys, Q, R, N)
        np.testing.assert_array_almost_equal(Kref, K)
        np.testing.assert_array_almost_equal(Sref, S)
        np.testing.assert_array_almost_equal(Eref, E)

        # Inconsistent system dimensions
        with pytest.raises(ct.ControlDimension, match="Incompatible dimen"):
            K, S, E = cdlqr(sys.A, sys.C, Q, R)

        # Incorrect covariance matrix dimensions
        with pytest.raises(ct.ControlDimension, match="Q must be a square"):
            K, S, E = cdlqr(sys.A, sys.B, sys.C, R, Q)

        # Too few input arguments
        with pytest.raises(ct.ControlArgument, match="not enough input"):
            K, S, E = cdlqr(sys.A, sys.B)

        # First argument is the wrong type (use SISO for non-slycot tests)
        sys_tf = tf(rss(3, 1, 1))
        sys_tf.dt = None        # treat as either continuous or discrete time
        with pytest.raises(ct.ControlArgument, match="LTI system must be"):
            K, S, E = cdlqr(sys_tf, Q, R)

    @pytest.mark.xfail(reason="warning not implemented")
    def testDLQR_warning(self):
        """Test dlqr()

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
            (K, S, E) = dlqr(A, B, Q, R, N)

    def test_care(self):
        """Test stabilizing and anti-stabilizing feedback, continuous"""
        A = np.diag([1, -1])
        B = np.identity(2)
        Q = np.identity(2)
        R = np.identity(2)
        S = np.zeros((2, 2))
        E = np.identity(2)

        X, L, G = care(A, B, Q, R, S, E, stabilizing=True)
        assert np.all(np.real(L) < 0)

        if slycot_check():
            X, L, G = care(A, B, Q, R, S, E, stabilizing=False)
            assert np.all(np.real(L) > 0)
        else:
            with pytest.raises(ControlArgument, match="'scipy' not valid"):
                X, L, G = care(A, B, Q, R, S, E, stabilizing=False)

    @pytest.mark.parametrize(
        "stabilizing",
        [True, pytest.param(False, marks=slycotonly)])
    def test_dare(self, stabilizing):
        """Test stabilizing and anti-stabilizing feedback, discrete"""
        A = np.diag([0.5, 2])
        B = np.identity(2)
        Q = np.identity(2)
        R = np.identity(2)
        S = np.zeros((2, 2))
        E = np.identity(2)

        X, L, G = dare(A, B, Q, R, S, E, stabilizing=stabilizing)
        sgn = {True: -1, False: 1}[stabilizing]
        assert np.all(sgn * (np.abs(L) - 1) > 0)

    def test_lqr_discrete(self):
        """Test overloading of lqr operator for discrete-time systems"""
        csys = ct.rss(2, 1, 1)
        dsys = ct.drss(2, 1, 1)
        Q = np.eye(2)
        R = np.eye(1)

        # Calling with a system versus explicit A, B should be the sam
        K_csys, S_csys, E_csys = ct.lqr(csys, Q, R)
        K_expl, S_expl, E_expl = ct.lqr(csys.A, csys.B, Q, R)
        np.testing.assert_almost_equal(K_csys, K_expl)
        np.testing.assert_almost_equal(S_csys, S_expl)
        np.testing.assert_almost_equal(E_csys, E_expl)

        # Calling lqr() with a discrete-time system should call dlqr()
        K_lqr, S_lqr, E_lqr = ct.lqr(dsys, Q, R)
        K_dlqr, S_dlqr, E_dlqr = ct.dlqr(dsys, Q, R)
        np.testing.assert_almost_equal(K_lqr, K_dlqr)
        np.testing.assert_almost_equal(S_lqr, S_dlqr)
        np.testing.assert_almost_equal(E_lqr, E_dlqr)

        # Calling lqr() with no timebase should call lqr()
        asys = ct.ss(csys.A, csys.B, csys.C, csys.D, dt=None)
        K_asys, S_asys, E_asys = ct.lqr(asys, Q, R)
        K_expl, S_expl, E_expl = ct.lqr(csys.A, csys.B, Q, R)
        np.testing.assert_almost_equal(K_asys, K_expl)
        np.testing.assert_almost_equal(S_asys, S_expl)
        np.testing.assert_almost_equal(E_asys, E_expl)

        # Calling dlqr() with a continuous-time system should raise an error
        with pytest.raises(ControlArgument, match="dsys must be discrete"):
            K, S, E = ct.dlqr(csys, Q, R)

    @pytest.mark.parametrize(
        'nstates, noutputs, ninputs, nintegrators, type_',
        [(2,      0,        1,       0,            None),
         (2,      1,        1,       0,            None),
         (4,      0,        2,       0,            None),
         (4,      3,        2,       0,            None),
         (2,      0,        1,       1,            None),
         (4,      0,        2,       2,            None),
         (4,      3,        2,       2,            None),
         (2,      0,        1,       0,            'nonlinear'),
         (4,      0,        2,       2,            'nonlinear'),
         (4,      3,        2,       2,            'nonlinear'),
         (2,      0,        1,       0,            'iosystem'),
         (2,      0,        1,       1,            'iosystem'),
        ])
    def test_statefbk_iosys(
            self, nstates, ninputs, noutputs, nintegrators, type_):
        # Create the system to be controlled (and estimator)
        # TODO: make sure it is controllable?
        if noutputs == 0:
            # Create a system with full state output
            sys = ct.rss(nstates, nstates, ninputs, strictly_proper=True)
            sys.C = np.eye(nstates)
            est = None

        else:
            # Create a system with of the desired size
            sys = ct.rss(nstates, noutputs, ninputs, strictly_proper=True)

            # Create an estimator with different signal names
            L, _, _ = ct.lqe(
                sys.A, sys.B, sys.C, np.eye(ninputs), np.eye(noutputs))
            est = ss(
                sys.A - L @ sys.C, np.hstack([L, sys.B]), np.eye(nstates), 0,
                inputs=sys.output_labels + sys.input_labels,
                outputs=[f'xhat[{i}]' for i in range(nstates)])

        # Decide whether to include integral action
        if nintegrators:
            # Choose the first 'n' outputs as integral terms
            C_int = np.eye(nintegrators, nstates)

            # Set up an augmented system for LQR computation
            # TODO: move this computation into LQR
            A_aug = np.block([
                [sys.A, np.zeros((sys.nstates, nintegrators))],
                [C_int, np.zeros((nintegrators, nintegrators))]
            ])
            B_aug = np.vstack([sys.B, np.zeros((nintegrators, ninputs))])
            C_aug = np.hstack([sys.C, np.zeros((sys.C.shape[0], nintegrators))])
            aug = ss(A_aug, B_aug, C_aug, 0)
        else:
            C_int = np.zeros((0, nstates))
            aug = sys

        # Design an LQR controller
        K, _, _ = ct.lqr(aug, np.eye(nstates + nintegrators), np.eye(ninputs))
        Kp, Ki = K[:, :nstates], K[:, nstates:]

        if type_ == 'iosystem':
            # Create an I/O system for the controller
            A_fbk = np.zeros((nintegrators, nintegrators))
            B_fbk = np.eye(nintegrators, sys.nstates)
            fbksys = ct.ss(A_fbk, B_fbk, -Ki, -Kp)
            ctrl, clsys = ct.create_statefbk_iosystem(
                sys, fbksys, integral_action=C_int, estimator=est,
                controller_type=type_, name=type_)

        else:
            ctrl, clsys = ct.create_statefbk_iosystem(
                sys, K, integral_action=C_int, estimator=est,
                controller_type=type_, name=type_)

        # Make sure the name got set correctly
        if type_ is not None:
            assert ctrl.name == type_

        # If we used a nonlinear controller, linearize it for testing
        if type_ == 'nonlinear' or type_ == 'iosystem':
            clsys = clsys.linearize(0, 0)

        # Make sure the linear system elements are correct
        if noutputs == 0:
            # No estimator
            Ac = np.block([
                [sys.A - sys.B @ Kp, -sys.B @ Ki],
                [C_int, np.zeros((nintegrators, nintegrators))]
            ])
            Bc = np.block([
                [sys.B @ Kp, sys.B],
                [-C_int, np.zeros((nintegrators, ninputs))]
            ])
            Cc = np.block([
                [np.eye(nstates), np.zeros((nstates, nintegrators))],
                [-Kp, -Ki]
            ])
            Dc = np.block([
                [np.zeros((nstates, nstates + ninputs))],
                [Kp, np.eye(ninputs)]
            ])
        else:
            # Estimator
            Be1, Be2 = est.B[:, :noutputs], est.B[:, noutputs:]
            Ac = np.block([
                [sys.A, -sys.B @ Ki, -sys.B @ Kp],
                [np.zeros((nintegrators, nstates + nintegrators)), C_int],
                [Be1 @ sys.C, -Be2 @ Ki, est.A - Be2 @ Kp]
                ])
            Bc = np.block([
                [sys.B @ Kp, sys.B],
                [-C_int, np.zeros((nintegrators, ninputs))],
                [Be2 @ Kp, Be2]
            ])
            Cc = np.block([
                [sys.C, np.zeros((noutputs, nintegrators + nstates))],
                [np.zeros_like(Kp), -Ki, -Kp]
            ])
            Dc = np.block([
                [np.zeros((noutputs, nstates + ninputs))],
                [Kp, np.eye(ninputs)]
            ])

        # Check to make sure everything matches
        np.testing.assert_array_almost_equal(clsys.A, Ac)
        np.testing.assert_array_almost_equal(clsys.B, Bc)
        np.testing.assert_array_almost_equal(clsys.C, Cc)
        np.testing.assert_array_almost_equal(clsys.D, Dc)

    def test_statefbk_iosys_unused(self):
        # Create a base system to work with
        sys = ct.rss(2, 1, 1, strictly_proper=True)

        # Create a system with extra input
        aug = ct.rss(2, inputs=[sys.input_labels[0], 'd'],
                     outputs=sys.output_labels, strictly_proper=True,)
        aug.A = sys.A
        aug.B[:, 0:1] = sys.B

        # Create an estimator
        est = ct.create_estimator_iosystem(
            sys, np.eye(sys.ninputs), np.eye(sys.noutputs))

        # Design an LQR controller
        K, _, _ = ct.lqr(sys, np.eye(sys.nstates), np.eye(sys.ninputs))

        # Create a baseline I/O control system
        ctrl0, clsys0 = ct.create_statefbk_iosystem(sys, K, estimator=est)
        clsys0_lin = clsys0.linearize(0, 0)

        # Create an I/O system with additional inputs
        ctrl1, clsys1 = ct.create_statefbk_iosystem(
            aug, K, estimator=est, control_indices=[0])
        clsys1_lin = clsys1.linearize(0, 0)

        # Make sure the extra inputs are there
        assert aug.input_labels[1] not in clsys0.input_labels
        assert aug.input_labels[1] in clsys1.input_labels
        np.testing.assert_allclose(clsys0_lin.A, clsys1_lin.A)

        # Switch around which input we use
        aug = ct.rss(2, inputs=['d', sys.input_labels[0]],
                     outputs=sys.output_labels, strictly_proper=True,)
        aug.A = sys.A
        aug.B[:, 1:2] = sys.B

        # Create an I/O system with additional inputs
        ctrl2, clsys2 = ct.create_statefbk_iosystem(
            aug, K, estimator=est, control_indices=[1])
        clsys2_lin = clsys2.linearize(0, 0)

        # Make sure the extra inputs are there
        assert aug.input_labels[0] not in clsys0.input_labels
        assert aug.input_labels[0] in clsys1.input_labels
        np.testing.assert_allclose(clsys0_lin.A, clsys2_lin.A)


    def test_lqr_integral_continuous(self):
        # Generate a continuous-time system for testing
        sys = ct.rss(4, 4, 2, strictly_proper=True)
        sys.C = np.eye(4)       # reset output to be full state
        C_int = np.eye(2, 4)    # integrate outputs for first two states
        nintegrators = C_int.shape[0]

        # Generate a controller with integral action
        K, _, _ = ct.lqr(
            sys, np.eye(sys.nstates + nintegrators), np.eye(sys.ninputs),
            integral_action=C_int)
        Kp, Ki = K[:, :sys.nstates], K[:, sys.nstates:]

        # Create an I/O system for the controller
        ctrl, clsys = ct.create_statefbk_iosystem(
            sys, K, integral_action=C_int)

        # Construct the state space matrices for the controller
        # Controller inputs = xd, ud, x
        # Controller state = z (integral of x-xd)
        # Controller output = ud - Kp(x - xd) - Ki z
        A_ctrl = np.zeros((nintegrators, nintegrators))
        B_ctrl = np.block([
            [-C_int, np.zeros((nintegrators, sys.ninputs)), C_int]
        ])
        C_ctrl = -K[:, sys.nstates:]
        D_ctrl = np.block([[Kp, np.eye(nintegrators), -Kp]])

        # Check to make sure everything matches
        np.testing.assert_array_almost_equal(ctrl.A, A_ctrl)
        np.testing.assert_array_almost_equal(ctrl.B, B_ctrl)
        np.testing.assert_array_almost_equal(ctrl.C, C_ctrl)
        np.testing.assert_array_almost_equal(ctrl.D, D_ctrl)

        # Construct the state space matrices for the closed loop system
        A_clsys = np.block([
            [sys.A - sys.B @ Kp, -sys.B @ Ki],
            [C_int, np.zeros((nintegrators, nintegrators))]
        ])
        B_clsys = np.block([
            [sys.B @ Kp, sys.B],
            [-C_int, np.zeros((nintegrators, sys.ninputs))]
        ])
        C_clsys = np.block([
            [np.eye(sys.nstates), np.zeros((sys.nstates, nintegrators))],
            [-Kp, -Ki]
        ])
        D_clsys = np.block([
            [np.zeros((sys.nstates, sys.nstates + sys.ninputs))],
            [Kp, np.eye(sys.ninputs)]
        ])

        # Check to make sure closed loop matches
        np.testing.assert_array_almost_equal(clsys.A, A_clsys)
        np.testing.assert_array_almost_equal(clsys.B, B_clsys)
        np.testing.assert_array_almost_equal(clsys.C, C_clsys)
        np.testing.assert_array_almost_equal(clsys.D, D_clsys)

        # Check the poles of the closed loop system
        assert all(np.real(clsys.poles()) < 0)

        # Make sure controller infinite zero frequency gain
        if slycot_check():
            ctrl_tf = tf(ctrl)
            assert abs(ctrl_tf(1e-9)[0][0]) > 1e6
            assert abs(ctrl_tf(1e-9)[1][1]) > 1e6

    def test_lqr_integral_discrete(self):
        # Generate a discrete-time system for testing
        sys = ct.drss(4, 4, 2, strictly_proper=True)
        sys.C = np.eye(4)       # reset output to be full state
        C_int = np.eye(2, 4)    # integrate outputs for first two states
        nintegrators = C_int.shape[0]

        # Generate a controller with integral action
        K, _, _ = ct.lqr(
            sys, np.eye(sys.nstates + nintegrators), np.eye(sys.ninputs),
            integral_action=C_int)
        Kp, _Ki = K[:, :sys.nstates], K[:, sys.nstates:]

        # Create an I/O system for the controller
        ctrl, clsys = ct.create_statefbk_iosystem(
            sys, K, integral_action=C_int)

        # Construct the state space matrices by hand
        A_ctrl = np.eye(nintegrators)
        B_ctrl = np.block([
            [-C_int, np.zeros((nintegrators, sys.ninputs)), C_int]
        ])
        C_ctrl = -K[:, sys.nstates:]
        D_ctrl = np.block([[Kp, np.eye(nintegrators), -Kp]])

        # Check to make sure everything matches
        assert ct.isdtime(clsys)
        np.testing.assert_array_almost_equal(ctrl.A, A_ctrl)
        np.testing.assert_array_almost_equal(ctrl.B, B_ctrl)
        np.testing.assert_array_almost_equal(ctrl.C, C_ctrl)
        np.testing.assert_array_almost_equal(ctrl.D, D_ctrl)

    @pytest.mark.parametrize(
        "rss_fun, lqr_fun",
        [(ct.rss, lqr), (ct.drss, dlqr)])
    def test_lqr_errors(self, rss_fun, lqr_fun):
        # Generate a discrete-time system for testing
        sys = rss_fun(4, 4, 2, strictly_proper=True)

        with pytest.raises(ControlArgument, match="must pass an array"):
            K, _, _ = lqr_fun(
                sys, np.eye(sys.nstates), np.eye(sys.ninputs),
                integral_action="invalid argument")

        with pytest.raises(ControlArgument, match="gain size must match"):
            C_int = np.eye(2, 3)
            K, _, _ = lqr_fun(
                sys, np.eye(sys.nstates), np.eye(sys.ninputs),
                integral_action=C_int)

        with pytest.raises(TypeError, match="unrecognized keywords"):
            K, _, _ = lqr_fun(
                sys, np.eye(sys.nstates), np.eye(sys.ninputs),
                integrator=None)

    def test_statefbk_errors(self):
        sys = ct.rss(4, 4, 2, strictly_proper=True)
        K, _, _ = ct.lqr(sys, np.eye(sys.nstates), np.eye(sys.ninputs))

        with pytest.warns(UserWarning, match="cannot verify system output"):
            ctrl, clsys = ct.create_statefbk_iosystem(sys, K)

        # reset the system output
        sys.C = np.eye(sys.nstates)

        with pytest.raises(ControlArgument, match="must be I/O system"):
            sys_tf = ct.tf([1], [1, 1])
            ctrl, clsys = ct.create_statefbk_iosystem(sys_tf, K)

        with pytest.raises(ControlArgument,
                           match="estimator output must include the full"):
            est = ct.rss(3, 3, 2)
            ctrl, clsys = ct.create_statefbk_iosystem(sys, K, estimator=est)

        with pytest.raises(ControlArgument,
                           match="system output must include the full state"):
            sys_nf = ct.rss(4, 3, 2, strictly_proper=True)
            ctrl, clsys = ct.create_statefbk_iosystem(sys_nf, K)

        with pytest.raises(ControlArgument, match="gain must be an array"):
            ctrl, clsys = ct.create_statefbk_iosystem(sys, "bad argument")

        with pytest.warns(FutureWarning, match="'type' is deprecated"):
            ctrl, clsys = ct.create_statefbk_iosystem(sys, K, type='nonlinear')

        with pytest.raises(ControlArgument, match="duplicate keywords"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ctrl, clsys = ct.create_statefbk_iosystem(
                    sys, K, type='nonlinear', controller_type='nonlinear')

        with pytest.raises(TypeError, match="unrecognized keyword"):
            ctrl, clsys = ct.create_statefbk_iosystem(sys, K, typo='nonlinear')

        with pytest.raises(ControlArgument, match="unknown controller_type"):
            ctrl, clsys = ct.create_statefbk_iosystem(sys, K, controller_type=1)

        # Errors involving integral action
        C_int = np.eye(2, 4)
        K_int, _, _ = ct.lqr(
            sys, np.eye(sys.nstates + C_int.shape[0]), np.eye(sys.ninputs),
            integral_action=C_int)

        with pytest.raises(ControlArgument, match="must pass an array"):
            ctrl, clsys = ct.create_statefbk_iosystem(
                sys, K_int, integral_action="bad argument")

        with pytest.raises(ControlArgument, match="must be an array of size"):
            ctrl, clsys = ct.create_statefbk_iosystem(
                sys, K, integral_action=C_int)


# Kinematic car example for testing gain scheduling
@pytest.fixture
def unicycle():
    # Create a simple nonlinear system to check (kinematic car)
    def unicycle_update(t, x, u, params):
        return np.array([np.cos(x[2]) * u[0], np.sin(x[2]) * u[0], u[1]])

    return ct.NonlinearIOSystem(
        unicycle_update, None,
        inputs=['v', 'phi'],
        outputs=['x', 'y', 'theta'],
        states=['x_', 'y_', 'theta_'],
        params={'a': 1})        # only used for testing params


@pytest.mark.parametrize("method", ['nearest', 'linear', 'cubic'])
def test_gainsched_unicycle(unicycle, method):
    # Speeds and angles at which to compute the gains
    speeds = [1, 5, 10]
    angles = np.linspace(0, pi/2, 4)
    points = list(itertools.product(speeds, angles))

    # Gains for each speed (using LQR controller)
    Q = np.identity(unicycle.nstates)
    R = np.identity(unicycle.ninputs)
    gains = [np.array(ct.lqr(unicycle.linearize(
        [0, 0, angle], [speed, 0]), Q, R)[0]) for speed, angle in points]

    #
    # Schedule on desired speed and angle
    #

    # Create gain scheduled controller
    ctrl, clsys = ct.create_statefbk_iosystem(
        unicycle, (gains, points),
        gainsched_indices=[3, 2], gainsched_method=method)

    # Check the gain at the selected points
    for speed, angle in points:
        # Figure out the desired state and input
        xe, ue = np.array([0, 0, angle]), np.array([speed, 0])
        xd, ud = np.array([0, 0, angle]), np.array([speed, 0])

        # Check the control system at the scheduling points
        ctrl_lin = ctrl.linearize([], [xd, ud, xe*0])
        K, S, E = ct.lqr(unicycle.linearize(xd, ud), Q, R)
        np.testing.assert_allclose(
            ctrl_lin.D[-xe.size:, -xe.size:], -K, rtol=1e-2)

        # Check the closed loop system at the scheduling points
        clsys_lin = clsys.linearize(xe, [xd, ud])
        np.testing.assert_allclose(
            np.sort(clsys_lin.poles()), np.sort(E), rtol=1e-2)

    # Check the gain at an intermediate point and confirm stability
    speed, angle = 2, pi/3
    xe, ue = np.array([0, 0, angle]), np.array([speed, 0])
    xd, ud = np.array([0, 0, angle]), np.array([speed, 0])
    clsys_lin = clsys.linearize(xe, [xd, ud])
    assert np.all(np.real(clsys_lin.poles()) < 0)

    # Make sure that gains are different from 'nearest'
    if method is not None and method != 'nearest':
        ctrl_nearest, clsys_nearest = ct.create_statefbk_iosystem(
            unicycle, (gains, points),
            gainsched_indices=['ud[0]', 2], gainsched_method='nearest')
        nearest_lin = clsys_nearest.linearize(xe, [xd, ud])
        assert not np.allclose(
            np.sort(clsys_lin.poles()), np.sort(nearest_lin.poles()), rtol=1e-2)

    # Run a simulation following a curved path
    T = 10                      # length of the trajectory [sec]
    r = 10                      # radius of the circle [m]
    timepts = np.linspace(0, T, 50)
    Xd = np.vstack([
        r * np.cos(timepts/T * pi/2 + 3*pi/2),
        r * np.sin(timepts/T * pi/2 + 3*pi/2) + r,
        timepts/T * pi/2
    ])
    Ud = np.vstack([
        np.ones_like(timepts) * (r * pi/2) / T,
        np.ones_like(timepts) * (pi / 2) / T
    ])
    X0 = Xd[:, 0] + np.array([-0.1, -0.1, -0.1])

    resp = ct.input_output_response(clsys, timepts, [Xd, Ud], X0)
    # plt.plot(resp.states[0], resp.states[1])
    np.testing.assert_allclose(
        resp.states[:, -1], Xd[:, -1], atol=1e-2, rtol=1e-2)

    #
    # Schedule on actual speed
    #

    # Create gain scheduled controller
    ctrl, clsys = ct.create_statefbk_iosystem(
        unicycle, (gains, points),
        ud_labels=['vd', 'phid'], gainsched_indices=['vd', 'theta'])

    # Check the gain at the selected points
    for speed, angle in points:
        # Figure out the desired state and input
        xe, ue = np.array([0, 0, angle]), np.array([speed, 0])
        xd, ud = np.array([0, 0, angle]), np.array([speed, 0])

        # Check the control system at the scheduling points
        ctrl_lin = ctrl.linearize([], [xd*0, ud, xe])
        K, S, E = ct.lqr(unicycle.linearize(xe, ue), Q, R)
        np.testing.assert_allclose(
            ctrl_lin.D[-xe.size:, -xe.size:], -K, rtol=1e-2)

        # Check the closed loop system at the scheduling points
        clsys_lin = clsys.linearize(xe, [xd, ud])
        np.testing.assert_allclose(np.sort(
            clsys_lin.poles()), np.sort(E), rtol=1e-2)

    # Run a simulation following a curved path
    resp = ct.input_output_response(clsys, timepts, [Xd, Ud], X0)
    np.testing.assert_allclose(
        resp.states[:, -1], Xd[:, -1], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("method", ['nearest', 'linear', 'cubic'])
def test_gainsched_1d(method):
    # Define a linear system to test
    sys = ct.ss([[-1, 0.1], [0, -2]], [[0], [1]], np.eye(2), 0)

    # Define gains for the first state only
    points = [-1, 0, 1]

    # Define gain to be constant
    K, _, _ = ct.lqr(sys, np.eye(sys.nstates), np.eye(sys.ninputs))
    gains = [K for p in points]

    # Define the paramters for the simulations
    timepts = np.linspace(0, 10, 100)
    X0 = np.ones(sys.nstates) * 1.1     # Start outside defined range

    # Create a controller and simulate the initial response
    gs_ctrl, gs_clsys = ct.create_statefbk_iosystem(
        sys, (gains, points), gainsched_indices=[0])
    gs_resp = ct.input_output_response(gs_clsys, timepts, 0, X0)

    # Verify that we get the same result as a constant gain
    ck_clsys = ct.ss(sys.A - sys.B @ K, sys.B, sys.C, 0)
    ck_resp = ct.input_output_response(ck_clsys, timepts, 0, X0)

    np.testing.assert_allclose(gs_resp.states, ck_resp.states)


def test_gainsched_default_indices():
    # Define a linear system to test
    sys = ct.ss([[-1, 0.1], [0, -2]], [[0], [1]], np.eye(2), 0)

    # Define gains at origin + corners of unit cube
    points = [[0, 0]] + list(itertools.product([-1, 1], [-1, 1]))

    # Define gain to be constant
    K, _, _ = ct.lqr(sys, np.eye(sys.nstates), np.eye(sys.ninputs))
    gains = [K for p in points]

    # Define the paramters for the simulations
    timepts = np.linspace(0, 10, 100)
    X0 = np.ones(sys.nstates) * 1.1     # Start outside defined range

    # Create a controller and simulate the initial response
    gs_ctrl, gs_clsys = ct.create_statefbk_iosystem(sys, (gains, points))
    gs_resp = ct.input_output_response(gs_clsys, timepts, 0, X0)

    # Verify that we get the same result as a constant gain
    ck_clsys = ct.ss(sys.A - sys.B @ K, sys.B, sys.C, 0)
    ck_resp = ct.input_output_response(ck_clsys, timepts, 0, X0)

    np.testing.assert_allclose(gs_resp.states, ck_resp.states)


def test_gainsched_errors(unicycle):
    # Set up gain schedule (same as previous test)
    speeds = [1, 5, 10]
    angles = np.linspace(0, pi/2, 4)
    points = list(itertools.product(speeds, angles))

    Q = np.identity(unicycle.nstates)
    R = np.identity(unicycle.ninputs)
    gains = [np.array(ct.lqr(unicycle.linearize(
        [0, 0, angle], [speed, 0]), Q, R)[0]) for speed, angle in points]

    # Make sure the generic case works OK
    ctrl, clsys = ct.create_statefbk_iosystem(
        unicycle, (gains, points), gainsched_indices=[3, 2])
    xd, ud = np.array([0, 0, angles[0]]), np.array([speeds[0], 0])
    ctrl_lin = ctrl.linearize([], [xd, ud, xd*0])
    K, S, E = ct.lqr(unicycle.linearize(xd, ud), Q, R)
    np.testing.assert_allclose(
        ctrl_lin.D[-xd.size:, -xd.size:], -K, rtol=1e-2)

    # Wrong type of gain schedule argument
    with pytest.raises(ControlArgument, match="gain must be an array"):
        ctrl, clsys = ct.create_statefbk_iosystem(
            unicycle, [gains, points], gainsched_indices=[3, 2])

    # Wrong number of gain schedule argument
    with pytest.raises(ControlArgument, match="gain must be a 2-tuple"):
        ctrl, clsys = ct.create_statefbk_iosystem(
            unicycle, (gains, speeds, angles), gainsched_indices=[3, 2])

    # Mismatched dimensions for gains and points
    with pytest.raises(ControlArgument, match="length of gainsched_indices"):
        ctrl, clsys = ct.create_statefbk_iosystem(
            unicycle, (gains, [speeds]), gainsched_indices=[3, 2])

    # Unknown gain scheduling variable label
    with pytest.raises(ValueError, match=".* not in list"):
        ctrl, clsys = ct.create_statefbk_iosystem(
            unicycle, (gains, points), gainsched_indices=['stuff', 2])

    # Unknown gain scheduling method
    with pytest.raises(ControlArgument, match="unknown gain scheduling method"):
        ctrl, clsys = ct.create_statefbk_iosystem(
            unicycle, (gains, points),
            gainsched_indices=[3, 2], gainsched_method='unknown')


@pytest.mark.parametrize("ninputs, Kf", [
    (1, 1),
    (1, None),
    (2, np.diag([1, 1])),
    (2, None),
])
def test_refgain_pattern(ninputs, Kf):
    sys = ct.rss(2, 2, ninputs, strictly_proper=True)
    sys.C = np.eye(2)

    K, _, _ = ct.lqr(sys.A, sys.B, np.eye(sys.nstates), np.eye(sys.ninputs))
    if Kf is None:
        # Make sure we get an error if we don't specify Kf
        with pytest.raises(ControlArgument, match="'feedfwd_gain' required"):
            ctrl, clsys = ct.create_statefbk_iosystem(
                sys, K, Kf, feedfwd_pattern='refgain')

        # Now compute the gain to give unity zero frequency gain
        C = np.eye(ninputs, sys.nstates)
        Kf = -np.linalg.inv(
            C @ np.linalg.inv(sys.A - sys.B @ K) @ sys.B)
        ctrl, clsys = ct.create_statefbk_iosystem(
            sys, K, Kf, feedfwd_pattern='refgain')

        np.testing.assert_almost_equal(
            C @ clsys(0)[0:sys.nstates], np.eye(ninputs))

    else:
        ctrl, clsys = ct.create_statefbk_iosystem(
            sys, K, Kf, feedfwd_pattern='refgain')

    manual = ct.feedback(sys, K) * Kf
    np.testing.assert_almost_equal(clsys.A, manual.A)
    np.testing.assert_almost_equal(clsys.B, manual.B)
    np.testing.assert_almost_equal(clsys.C[:sys.nstates, :], manual.C)
    np.testing.assert_almost_equal(clsys.D[:sys.nstates, :], manual.D)


def test_create_statefbk_errors():
    sys = ct.rss(2, 2, 1, strictly_proper=True)
    sys.C = np.eye(2)
    K = -np.ones((1, 4))
    Kf = 1

    K, _, _ = ct.lqr(sys.A, sys.B, np.eye(sys.nstates), np.eye(sys.ninputs))
    with pytest.raises(NotImplementedError, match="unknown pattern"):
        ct.create_statefbk_iosystem(sys, K, feedfwd_pattern='mypattern')

    with pytest.raises(ControlArgument, match="feedfwd_pattern != 'refgain'"):
        ct.create_statefbk_iosystem(sys, K, Kf, feedfwd_pattern='trajgen')


def test_create_statefbk_params(unicycle):
    Q = np.identity(unicycle.nstates)
    R = np.identity(unicycle.ninputs)
    gain, _, _ = ct.lqr(unicycle.linearize([0, 0, 0], [5, 0]), Q, R)

    # Create a linear controller
    ctrl, clsys = ct.create_statefbk_iosystem(unicycle, gain)
    assert [k for k in ctrl.params.keys()] == []
    assert [k for k in clsys.params.keys()] == ['a']
    assert clsys.params['a'] == 1

    # Create a nonlinear controller
    ctrl, clsys = ct.create_statefbk_iosystem(
        unicycle, gain, controller_type='nonlinear')
    assert [k for k in ctrl.params.keys()] == ['K']
    assert [k for k in clsys.params.keys()] == ['K', 'a']
    assert clsys.params['a'] == 1

    # Override the default parameters
    ctrl, clsys = ct.create_statefbk_iosystem(
        unicycle, gain, controller_type='nonlinear', params={'a': 2, 'b': 1})
    assert [k for k in ctrl.params.keys()] == ['K']
    assert [k for k in clsys.params.keys()] == ['K', 'a', 'b']
    assert clsys.params['a'] == 2
    assert clsys.params['b'] == 1
