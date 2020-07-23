#!/usr/bin/env python
#
# statesp_test.py - test state space class
# RMM, 30 Mar 2011 (based on TestStateSp from v0.4a)

import unittest
import numpy as np
from numpy.linalg import solve
from scipy.linalg import eigvals, block_diag
from control import matlab
from control.statesp import StateSpace, _convertToStateSpace, tf2ss
from control.xferfcn import TransferFunction, ss2tf
from control.lti import evalfr
from control.exception import slycot_check


class TestStateSpace(unittest.TestCase):
    """Tests for the StateSpace class."""

    def setUp(self):
        """Set up a MIMO system to test operations on."""

        # sys1: 3-states square system (2 inputs x 2 outputs)
        A322 = [[-3., 4., 2.],
                [-1., -3., 0.],
                [2., 5., 3.]]
        B322 = [[1., 4.],
                [-3., -3.],
                [-2., 1.]]
        C322 = [[4., 2., -3.],
                [1., 4., 3.]]
        D322 = [[-2., 4.],
                [0., 1.]]
        self.sys322 = StateSpace(A322, B322, C322, D322)

        # sys1: 2-states square system (2 inputs x 2 outputs)
        A222 = [[4., 1.],
                [2., -3]]
        B222 = [[5., 2.],
                [-3., -3.]]
        C222 = [[2., -4],
                [0., 1.]]
        D222 = [[3., 2.],
                [1., -1.]]
        self.sys222 = StateSpace(A222, B222, C222, D222)

        # sys3: 6 states non square system (2 inputs x 3 outputs)
        A623 = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 3, 0, 0, 0],
                         [0, 0, 0, -4, 0, 0],
                         [0, 0, 0, 0, -1, 0],
                         [0, 0, 0, 0, 0, 3]])
        B623 = np.array([[0, -1],
                        [-1, 0],
                        [1, -1],
                        [0, 0],
                        [0, 1],
                        [-1, -1]])
        C623 = np.array([[1, 0, 0, 1, 0, 0],
                         [0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 0, 1]])
        D623 = np.zeros((3, 2))
        self.sys623 = StateSpace(A623, B623, C623, D623)

    def test_D_broadcast(self):
        """Test broadcast of D=0 to the right shape"""
        # Giving D as a scalar 0 should broadcast to the right shape
        sys = StateSpace(self.sys623.A, self.sys623.B, self.sys623.C, 0)
        np.testing.assert_array_equal(self.sys623.D, sys.D)

        # Giving D as a matrix of the wrong size should generate an error
        with self.assertRaises(ValueError):
            sys = StateSpace(sys.A, sys.B, sys.C, np.array([[0]]))

        # Make sure that empty systems still work
        sys = StateSpace([], [], [], 1)
        np.testing.assert_array_equal(sys.D, [[1]])

        sys = StateSpace([], [], [], [[0]])
        np.testing.assert_array_equal(sys.D, [[0]])

        sys = StateSpace([], [], [], [0])
        np.testing.assert_array_equal(sys.D, [[0]])

        sys = StateSpace([], [], [], 0)
        np.testing.assert_array_equal(sys.D, [[0]])

    def test_pole(self):
        """Evaluate the poles of a MIMO system."""

        p = np.sort(self.sys322.pole())
        true_p = np.sort([3.34747678408874,
                          -3.17373839204437 + 1.47492908003839j,
                          -3.17373839204437 - 1.47492908003839j])

        np.testing.assert_array_almost_equal(p, true_p)

    def test_zero_empty(self):
        """Test to make sure zero() works with no zeros in system."""
        sys = _convertToStateSpace(TransferFunction([1], [1, 2, 1]))
        np.testing.assert_array_equal(sys.zero(), np.array([]))

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_zero_siso(self):
        """Evaluate the zeros of a SISO system."""
        # extract only first input / first output system of sys222. This system is denoted sys111
        #  or tf111
        tf111 = ss2tf(self.sys222)
        sys111 = tf2ss(tf111[0, 0])

        # compute zeros as root of the characteristic polynomial at the numerator of tf111
        # this method is simple and assumed as valid in this test
        true_z = np.sort(tf111[0, 0].zero())
        # Compute the zeros through ab08nd, which is tested here
        z = np.sort(sys111.zero())

        np.testing.assert_almost_equal(true_z, z)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_zero_mimo_sys322_square(self):
        """Evaluate the zeros of a square MIMO system."""

        z = np.sort(self.sys322.zero())
        true_z = np.sort([44.41465, -0.490252, -5.924398])
        np.testing.assert_array_almost_equal(z, true_z)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_zero_mimo_sys222_square(self):
        """Evaluate the zeros of a square MIMO system."""

        z = np.sort(self.sys222.zero())
        true_z = np.sort([-10.568501,   3.368501])
        np.testing.assert_array_almost_equal(z, true_z)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_zero_mimo_sys623_non_square(self):
        """Evaluate the zeros of a non square MIMO system."""

        z = np.sort(self.sys623.zero())
        true_z = np.sort([2., -1.])
        np.testing.assert_array_almost_equal(z, true_z)

    def test_add_ss(self):
        """Add two MIMO systems."""

        A = [[-3., 4., 2., 0., 0.], [-1., -3., 0., 0., 0.],
             [2., 5., 3., 0., 0.], [0., 0., 0., 4., 1.], [0., 0., 0., 2., -3.]]
        B = [[1., 4.], [-3., -3.], [-2., 1.], [5., 2.], [-3., -3.]]
        C = [[4., 2., -3., 2., -4.], [1., 4., 3., 0., 1.]]
        D = [[1., 6.], [1., 0.]]

        sys = self.sys322 + self.sys222

        np.testing.assert_array_almost_equal(sys.A, A)
        np.testing.assert_array_almost_equal(sys.B, B)
        np.testing.assert_array_almost_equal(sys.C, C)
        np.testing.assert_array_almost_equal(sys.D, D)

    def test_subtract_ss(self):
        """Subtract two MIMO systems."""

        A = [[-3., 4., 2., 0., 0.], [-1., -3., 0., 0., 0.],
             [2., 5., 3., 0., 0.], [0., 0., 0., 4., 1.], [0., 0., 0., 2., -3.]]
        B = [[1., 4.], [-3., -3.], [-2., 1.], [5., 2.], [-3., -3.]]
        C = [[4., 2., -3., -2., 4.], [1., 4., 3., 0., -1.]]
        D = [[-5., 2.], [-1., 2.]]

        sys = self.sys322 - self.sys222

        np.testing.assert_array_almost_equal(sys.A, A)
        np.testing.assert_array_almost_equal(sys.B, B)
        np.testing.assert_array_almost_equal(sys.C, C)
        np.testing.assert_array_almost_equal(sys.D, D)

    def test_multiply_ss(self):
        """Multiply two MIMO systems."""

        A = [[4., 1., 0., 0., 0.], [2., -3., 0., 0., 0.], [2., 0., -3., 4., 2.],
             [-6., 9., -1., -3., 0.], [-4., 9., 2., 5., 3.]]
        B = [[5., 2.], [-3., -3.], [7., -2.], [-12., -3.], [-5., -5.]]
        C = [[-4., 12., 4., 2., -3.], [0., 1., 1., 4., 3.]]
        D = [[-2., -8.], [1., -1.]]

        sys = self.sys322 * self.sys222

        np.testing.assert_array_almost_equal(sys.A, A)
        np.testing.assert_array_almost_equal(sys.B, B)
        np.testing.assert_array_almost_equal(sys.C, C)
        np.testing.assert_array_almost_equal(sys.D, D)

    def test_evalfr(self):
        """Evaluate the frequency response at one frequency."""

        A = [[-2, 0.5], [0.5, -0.3]]
        B = [[0.3, -1.3], [0.1, 0.]]
        C = [[0., 0.1], [-0.3, -0.2]]
        D = [[0., -0.8], [-0.3, 0.]]
        sys = StateSpace(A, B, C, D)

        resp = [[4.37636761487965e-05 - 0.0152297592997812j,
                 -0.792603938730853 + 0.0261706783369803j],
                [-0.331544857768052 + 0.0576105032822757j,
                 0.128919037199125 - 0.143824945295405j]]

        # Correct versions of the call
        np.testing.assert_almost_equal(evalfr(sys, 1j), resp)
        np.testing.assert_almost_equal(sys._evalfr(1.), resp)

        # Deprecated version of the call (should generate warning)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            # Set up warnings filter to only show warnings in control module
            warnings.filterwarnings("ignore")
            warnings.filterwarnings("always", module="control")

            # Make sure that we get a pending deprecation warning
            sys.evalfr(1.)
            assert len(w) == 1
            assert issubclass(w[-1].category, PendingDeprecationWarning)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_freq_resp(self):
        """Evaluate the frequency response at multiple frequencies."""

        A = [[-2, 0.5], [0.5, -0.3]]
        B = [[0.3, -1.3], [0.1, 0.]]
        C = [[0., 0.1], [-0.3, -0.2]]
        D = [[0., -0.8], [-0.3, 0.]]
        sys = StateSpace(A, B, C, D)

        true_mag = [[[0.0852992637230322, 0.00103596611395218],
                    [0.935374692849736, 0.799380720864549]],
                   [[0.55656854563842, 0.301542699860857],
                    [0.609178071542849, 0.0382108097985257]]]
        true_phase = [[[-0.566195599644593, -1.68063565332582],
                      [3.0465958317514, 3.14141384339534]],
                     [[2.90457947657161, 3.10601268291914],
                      [-0.438157380501337, -1.40720969147217]]]
        true_omega = [0.1, 10.]

        mag, phase, omega = sys.freqresp(true_omega)

        np.testing.assert_almost_equal(mag, true_mag)
        np.testing.assert_almost_equal(phase, true_phase)
        np.testing.assert_equal(omega, true_omega)

    def test_is_static_gain(self):
        A0 = np.zeros((2,2))
        A1 = A0.copy()
        A1[0,1] = 1.1
        B0 = np.zeros((2,1))
        B1 = B0.copy()
        B1[0,0] = 1.3
        C0 = A0
        C1 = np.eye(2)
        D0 = 0
        D1 = np.ones((2,1))
        self.assertTrue(StateSpace(A0, B0, C1, D1).is_static_gain()) # True
        # fix this once remove_useless_states is false by default
        #print(StateSpace(A1, B0, C1, D1).is_static_gain()) # should be False when remove_useless is false
        self.assertFalse(StateSpace(A0, B1, C1, D1).is_static_gain()) # False
        self.assertFalse(StateSpace(A1, B1, C1, D1).is_static_gain()) # False
        self.assertTrue(StateSpace(A0, B0, C0, D0).is_static_gain()) # True
        self.assertTrue(StateSpace(A0, B0, C0, D1).is_static_gain()) # True
        self.assertTrue(StateSpace(A0, B0, C1, D0).is_static_gain()) # True
        
    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_minreal(self):
        """Test a minreal model reduction."""
        # A = [-2, 0.5, 0; 0.5, -0.3, 0; 0, 0, -0.1]
        A = [[-2, 0.5, 0], [0.5, -0.3, 0], [0, 0, -0.1]]
        # B = [0.3, -1.3; 0.1, 0; 1, 0]
        B = [[0.3, -1.3], [0.1, 0.], [1.0, 0.0]]
        # C = [0, 0.1, 0; -0.3, -0.2, 0]
        C = [[0., 0.1, 0.0], [-0.3, -0.2, 0.0]]
        # D = [0 -0.8; -0.3 0]
        D = [[0., -0.8], [-0.3, 0.]]
        # sys = ss(A, B, C, D)

        sys = StateSpace(A, B, C, D)
        sysr = sys.minreal()
        self.assertEqual(sysr.states, 2)
        self.assertEqual(sysr.inputs, sys.inputs)
        self.assertEqual(sysr.outputs, sys.outputs)
        np.testing.assert_array_almost_equal(
            eigvals(sysr.A), [-2.136154, -0.1638459])

    def test_append_ss(self):
        """Test appending two state-space systems."""
        A1 = [[-2, 0.5, 0], [0.5, -0.3, 0], [0, 0, -0.1]]
        B1 = [[0.3, -1.3], [0.1, 0.], [1.0, 0.0]]
        C1 = [[0., 0.1, 0.0], [-0.3, -0.2, 0.0]]
        D1 = [[0., -0.8], [-0.3, 0.]]
        A2 = [[-1.]]
        B2 = [[1.2]]
        C2 = [[0.5]]
        D2 = [[0.4]]
        A3 = [[-2, 0.5, 0, 0], [0.5, -0.3, 0, 0], [0, 0, -0.1, 0],
              [0, 0, 0., -1.]]
        B3 = [[0.3, -1.3, 0], [0.1, 0., 0], [1.0, 0.0, 0], [0., 0, 1.2]]
        C3 = [[0., 0.1, 0.0, 0.0], [-0.3, -0.2, 0.0, 0.0], [0., 0., 0., 0.5]]
        D3 = [[0., -0.8, 0.], [-0.3, 0., 0.], [0., 0., 0.4]]
        sys1 = StateSpace(A1, B1, C1, D1)
        sys2 = StateSpace(A2, B2, C2, D2)
        sys3 = StateSpace(A3, B3, C3, D3)
        sys3c = sys1.append(sys2)
        np.testing.assert_array_almost_equal(sys3.A, sys3c.A)
        np.testing.assert_array_almost_equal(sys3.B, sys3c.B)
        np.testing.assert_array_almost_equal(sys3.C, sys3c.C)
        np.testing.assert_array_almost_equal(sys3.D, sys3c.D)

    def test_append_tf(self):
        """Test appending a state-space system with a tf"""
        A1 = [[-2, 0.5, 0], [0.5, -0.3, 0], [0, 0, -0.1]]
        B1 = [[0.3, -1.3], [0.1, 0.], [1.0, 0.0]]
        C1 = [[0., 0.1, 0.0], [-0.3, -0.2, 0.0]]
        D1 = [[0., -0.8], [-0.3, 0.]]
        s = TransferFunction([1, 0], [1])
        h = 1 / (s + 1) / (s + 2)
        sys1 = StateSpace(A1, B1, C1, D1)
        sys2 = _convertToStateSpace(h)
        sys3c = sys1.append(sys2)
        np.testing.assert_array_almost_equal(sys1.A, sys3c.A[:3, :3])
        np.testing.assert_array_almost_equal(sys1.B, sys3c.B[:3, :2])
        np.testing.assert_array_almost_equal(sys1.C, sys3c.C[:2, :3])
        np.testing.assert_array_almost_equal(sys1.D, sys3c.D[:2, :2])
        np.testing.assert_array_almost_equal(sys2.A, sys3c.A[3:, 3:])
        np.testing.assert_array_almost_equal(sys2.B, sys3c.B[3:, 2:])
        np.testing.assert_array_almost_equal(sys2.C, sys3c.C[2:, 3:])
        np.testing.assert_array_almost_equal(sys2.D, sys3c.D[2:, 2:])
        np.testing.assert_array_almost_equal(sys3c.A[:3, 3:], np.zeros((3, 2)))
        np.testing.assert_array_almost_equal(sys3c.A[3:, :3], np.zeros((2, 3)))

    def test_array_access_ss(self):

        sys1 = StateSpace([[1., 2.], [3., 4.]],
                          [[5., 6.], [6., 8.]],
                          [[9., 10.], [11., 12.]],
                          [[13., 14.], [15., 16.]], 1)

        sys1_11 = sys1[0, 1]
        np.testing.assert_array_almost_equal(sys1_11.A,
                                             sys1.A)
        np.testing.assert_array_almost_equal(sys1_11.B,
                                             sys1.B[:, 1])
        np.testing.assert_array_almost_equal(sys1_11.C,
                                             sys1.C[0, :])
        np.testing.assert_array_almost_equal(sys1_11.D,
                                             sys1.D[0, 1])

        assert sys1.dt == sys1_11.dt

    def test_dc_gain_cont(self):
        """Test DC gain for continuous-time state-space systems."""
        sys = StateSpace(-2., 6., 5., 0)
        np.testing.assert_equal(sys.dcgain(), 15.)

        sys2 = StateSpace(-2, [6., 4.], [[5.], [7.], [11]], np.zeros((3, 2)))
        expected = np.array([[15., 10.], [21., 14.], [33., 22.]])
        np.testing.assert_array_equal(sys2.dcgain(), expected)

        sys3 = StateSpace(0., 1., 1., 0.)
        np.testing.assert_equal(sys3.dcgain(), np.nan)

    def test_dc_gain_discr(self):
        """Test DC gain for discrete-time state-space systems."""
        # static gain
        sys = StateSpace([], [], [], 2, True)
        np.testing.assert_equal(sys.dcgain(), 2)

        # averaging filter
        sys = StateSpace(0.5, 0.5, 1, 0, True)
        np.testing.assert_almost_equal(sys.dcgain(), 1)

        # differencer
        sys = StateSpace(0, 1, -1, 1, True)
        np.testing.assert_equal(sys.dcgain(), 0)

        # summer
        sys = StateSpace(1, 1, 1, 0, True)
        np.testing.assert_equal(sys.dcgain(), np.nan)

    def test_dc_gain_integrator(self):
        """DC gain when eigenvalue at DC returns appropriately sized array of nan."""
        # the SISO case is also tested in test_dc_gain_{cont,discr}
        import itertools
        # iterate over input and output sizes, and continuous (dt=None) and discrete (dt=True) time
        for inputs, outputs, dt in itertools.product(range(1, 6), range(1, 6), [None, True]):
            states = max(inputs, outputs)

            # a matrix that is singular at DC, and has no "useless" states as in
            # _remove_useless_states
            a = np.triu(np.tile(2, (states, states)))
            # eigenvalues all +2, except for ...
            a[0, 0] = 0 if dt is None else 1
            b = np.eye(max(inputs, states))[:states, :inputs]
            c = np.eye(max(outputs, states))[:outputs, :states]
            d = np.zeros((outputs, inputs))
            sys = StateSpace(a, b, c, d, dt)
            dc = np.squeeze(np.tile(np.nan, (outputs, inputs)))
            np.testing.assert_array_equal(dc, sys.dcgain())

    def test_scalar_static_gain(self):
        """Regression: can we create a scalar static gain?"""
        g1 = StateSpace([], [], [], [2])
        g2 = StateSpace([], [], [], [3])

        # make sure StateSpace internals, specifically ABC matrix
        # sizes, are OK for LTI operations
        g3 = g1 * g2
        self.assertEqual(6, g3.D[0, 0])
        g4 = g1 + g2
        self.assertEqual(5, g4.D[0, 0])
        g5 = g1.feedback(g2)
        self.assertAlmostEqual(2. / 7, g5.D[0, 0])
        g6 = g1.append(g2)
        np.testing.assert_array_equal(np.diag([2, 3]), g6.D)

    def test_matrix_static_gain(self):
        """Regression: can we create matrix static gains?"""
        d1 = np.matrix([[1, 2, 3], [4, 5, 6]])
        d2 = np.matrix([[7, 8], [9, 10], [11, 12]])
        g1 = StateSpace([], [], [], d1)

        # _remove_useless_states was making A = [[0]]
        self.assertEqual((0, 0), g1.A.shape)

        g2 = StateSpace([], [], [], d2)
        g3 = StateSpace([], [], [], d2.T)

        h1 = g1 * g2
        np.testing.assert_array_equal(d1 * d2, h1.D)
        h2 = g1 + g3
        np.testing.assert_array_equal(d1 + d2.T, h2.D)
        h3 = g1.feedback(g2)
        np.testing.assert_array_almost_equal(
            solve(np.eye(2) + d1 * d2, d1), h3.D)
        h4 = g1.append(g2)
        np.testing.assert_array_equal(block_diag(d1, d2), h4.D)

    def test_remove_useless_states(self):
        """Regression: _remove_useless_states gives correct ABC sizes."""
        g1 = StateSpace(np.zeros((3, 3)),
                        np.zeros((3, 4)),
                        np.zeros((5, 3)),
                        np.zeros((5, 4)))
        self.assertEqual((0, 0), g1.A.shape)
        self.assertEqual((0, 4), g1.B.shape)
        self.assertEqual((5, 0), g1.C.shape)
        self.assertEqual((5, 4), g1.D.shape)
        self.assertEqual(0, g1.states)

    def test_bad_empty_matrices(self):
        """Mismatched ABCD matrices when some are empty."""
        self.assertRaises(ValueError, StateSpace, [1], [], [], [1])
        self.assertRaises(ValueError, StateSpace, [1], [1], [], [1])
        self.assertRaises(ValueError, StateSpace, [1], [], [1], [1])
        self.assertRaises(ValueError, StateSpace, [], [1], [], [1])
        self.assertRaises(ValueError, StateSpace, [], [1], [1], [1])
        self.assertRaises(ValueError, StateSpace, [], [], [1], [1])
        self.assertRaises(ValueError, StateSpace, [1], [1], [1], [])

    def test_minreal_static_gain(self):
        """Regression: minreal on static gain was failing."""
        g1 = StateSpace([], [], [], [1])
        g2 = g1.minreal()
        np.testing.assert_array_equal(g1.A, g2.A)
        np.testing.assert_array_equal(g1.B, g2.B)
        np.testing.assert_array_equal(g1.C, g2.C)
        np.testing.assert_array_equal(g1.D, g2.D)

    def test_empty(self):
        """Regression: can we create an empty StateSpace object?"""
        g1 = StateSpace([], [], [], [])
        self.assertEqual(0, g1.states)
        self.assertEqual(0, g1.inputs)
        self.assertEqual(0, g1.outputs)

    def test_matrix_to_state_space(self):
        """_convertToStateSpace(matrix) gives ss([],[],[],D)"""
        D = np.matrix([[1, 2, 3], [4, 5, 6]])
        g = _convertToStateSpace(D)

        def empty(shape):
            m = np.matrix([])
            m.shape = shape
            return m
        np.testing.assert_array_equal(empty((0, 0)), g.A)
        np.testing.assert_array_equal(empty((0, D.shape[1])), g.B)
        np.testing.assert_array_equal(empty((D.shape[0], 0)), g.C)
        np.testing.assert_array_equal(D, g.D)

    def test_lft(self):
        """ test lft function with result obtained from matlab implementation"""
        # test case
        A = [[1, 2, 3],
             [1, 4, 5],
             [2, 3, 4]]
        B = [[0, 2],
             [5, 6],
             [5, 2]]
        C = [[1, 4, 5],
             [2, 3, 0]]
        D = [[0, 0],
             [3, 0]]
        P = StateSpace(A, B, C, D)
        Ak = [[0, 2, 3],
              [2, 3, 5],
              [2, 1, 9]]
        Bk = [[1, 1],
              [2, 3],
              [9, 4]]
        Ck = [[1, 4, 5],
              [2, 3, 6]]
        Dk = [[0, 2],
              [0, 0]]
        K = StateSpace(Ak, Bk, Ck, Dk)

        # case 1
        pk = P.lft(K, 2, 1)
        Amatlab = [1, 2, 3, 4, 6, 12, 1, 4, 5, 17, 38, 61, 2, 3, 4, 9, 26, 37, 2, 3, 0, 3, 14, 18, 4, 6, 0, 8, 27, 35, 18, 27, 0, 29, 109, 144]
        Bmatlab = [0, 10, 10, 7, 15, 58]
        Cmatlab = [1, 4, 5, 0, 0, 0]
        Dmatlab = [0]
        np.testing.assert_allclose(np.array(pk.A).reshape(-1), Amatlab)
        np.testing.assert_allclose(np.array(pk.B).reshape(-1), Bmatlab)
        np.testing.assert_allclose(np.array(pk.C).reshape(-1), Cmatlab)
        np.testing.assert_allclose(np.array(pk.D).reshape(-1), Dmatlab)

        # case 2
        pk = P.lft(K)
        Amatlab = [1, 2, 3, 4, 6, 12, -3, -2, 5, 11, 14, 31, -2, -3, 4, 3, 2, 7, 0.6, 3.4, 5, -0.6, -0.4, 0, 0.8, 6.2, 10, 0.2, -4.2, -4, 7.4, 33.6, 45, -0.4, -8.6, -3]
        Bmatlab = []
        Cmatlab = []
        Dmatlab = []
        np.testing.assert_allclose(np.array(pk.A).reshape(-1), Amatlab)
        np.testing.assert_allclose(np.array(pk.B).reshape(-1), Bmatlab)
        np.testing.assert_allclose(np.array(pk.C).reshape(-1), Cmatlab)
        np.testing.assert_allclose(np.array(pk.D).reshape(-1), Dmatlab)

    def test_repr(self):
        ref322 = """StateSpace(array([[-3.,  4.,  2.],
       [-1., -3.,  0.],
       [ 2.,  5.,  3.]]), array([[ 1.,  4.],
       [-3., -3.],
       [-2.,  1.]]), array([[ 4.,  2., -3.],
       [ 1.,  4.,  3.]]), array([[-2.,  4.],
       [ 0.,  1.]]){dt})"""
        self.assertEqual(repr(self.sys322), ref322.format(dt=''))
        sysd = StateSpace(self.sys322.A, self.sys322.B,
                          self.sys322.C, self.sys322.D, 0.4)
        self.assertEqual(repr(sysd), ref322.format(dt=", 0.4"))
        array = np.array
        sysd2 = eval(repr(sysd))
        np.testing.assert_allclose(sysd.A, sysd2.A)
        np.testing.assert_allclose(sysd.B, sysd2.B)
        np.testing.assert_allclose(sysd.C, sysd2.C)
        np.testing.assert_allclose(sysd.D, sysd2.D)

    def test_str(self):
        """Test that printing the system works"""
        tsys = self.sys322
        tref = ("A = [[-3.  4.  2.]\n"
                "     [-1. -3.  0.]\n"
                "     [ 2.  5.  3.]]\n"
                "\n"
                "B = [[ 1.  4.]\n"
                "     [-3. -3.]\n"
                "     [-2.  1.]]\n"
                "\n"
                "C = [[ 4.  2. -3.]\n"
                "     [ 1.  4.  3.]]\n"
                "\n"
                "D = [[-2.  4.]\n"
                "     [ 0.  1.]]\n")
        assert str(tsys) == tref
        tsysdtunspec = StateSpace(tsys.A, tsys.B, tsys.C, tsys.D, True)
        assert str(tsysdtunspec) == tref + "\ndt unspecified\n"
        sysdt1 = StateSpace(tsys.A, tsys.B, tsys.C, tsys.D, 1.)
        assert str(sysdt1) == tref + "\ndt = 1.0\n"


class TestRss(unittest.TestCase):
    """These are tests for the proper functionality of statesp.rss."""

    def setUp(self):
        # Number of times to run each of the randomized tests.
        self.numTests = 100
        # Maxmimum number of states to test + 1
        self.maxStates = 10
        # Maximum number of inputs and outputs to test + 1
        self.maxIO = 5

    def test_shape(self):
        """Test that rss outputs have the right state, input, and output size."""

        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.rss(states, outputs, inputs)
                    self.assertEqual(sys.states, states)
                    self.assertEqual(sys.inputs, inputs)
                    self.assertEqual(sys.outputs, outputs)

    def test_pole(self):
        """Test that the poles of rss outputs have a negative real part."""

        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.rss(states, outputs, inputs)
                    p = sys.pole()
                    for z in p:
                        self.assertTrue(z.real < 0)


class TestDrss(unittest.TestCase):
    """These are tests for the proper functionality of statesp.drss."""

    def setUp(self):
        # Number of times to run each of the randomized tests.
        self.numTests = 100
        # Maximum number of states to test + 1
        self.maxStates = 10
        # Maximum number of inputs and outputs to test + 1
        self.maxIO = 5

    def test_shape(self):
        """Test that drss outputs have the right state, input, and output size."""

        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.drss(states, outputs, inputs)
                    self.assertEqual(sys.states, states)
                    self.assertEqual(sys.inputs, inputs)
                    self.assertEqual(sys.outputs, outputs)

    def test_pole(self):
        """Test that the poles of drss outputs have less than unit magnitude."""

        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.drss(states, outputs, inputs)
                    p = sys.pole()
                    for z in p:
                        self.assertTrue(abs(z) < 1)

    def test_pole_static(self):
        """Regression: pole() of static gain is empty array."""
        np.testing.assert_array_equal(np.array([]),
                                      StateSpace([], [], [], [[1]]).pole())

    def test_copy_constructor(self):
        # Create a set of matrices for a simple linear system
        A = np.array([[-1]])
        B = np.array([[1]])
        C = np.array([[1]])
        D = np.array([[0]])

        # Create the first linear system and a copy
        linsys = StateSpace(A, B, C, D)
        cpysys = StateSpace(linsys)

        # Change the original A matrix
        A[0, 0] = -2
        np.testing.assert_array_equal(linsys.A, [[-1]]) # original value
        np.testing.assert_array_equal(cpysys.A, [[-1]]) # original value

        # Change the A matrix for the original system
        linsys.A[0, 0] = -3
        np.testing.assert_array_equal(cpysys.A, [[-1]]) # original value

if __name__ == "__main__":
    unittest.main()
