"""bdalg_test.py - test suite for block diagram algebra

RMM, 30 Mar 2011 (based on TestBDAlg from v0.4a)
"""

import numpy as np
from numpy import sort
import pytest

import control as ctrl
from control.xferfcn import TransferFunction
from control.statesp import StateSpace
from control.bdalg import feedback, append, connect
from control.lti import zero, pole


class TestFeedback:
    """These are tests for the feedback function in bdalg.py.  Currently, some
    of the tests are not implemented, or are not working properly.  TODO: these
    need to be fixed."""

    @pytest.fixture
    def tsys(self):
        class T:
            pass
        # Three SISO systems.
        T.sys1 = TransferFunction([1, 2], [1, 2, 3])
        T.sys2 = StateSpace([[1., 4.], [3., 2.]], [[1.], [-4.]],
                            [[1., 0.]], [[0.]])
        T.sys3 = StateSpace([[-1.]], [[1.]], [[1.]], [[0.]])  # 1 state, SISO

        # Two random scalars.
        T.x1 = 2.5
        T.x2 = -3.
        return T

    def testScalarScalar(self, tsys):
        """Scalar system with scalar feedback block."""
        ans1 = feedback(tsys.x1, tsys.x2)
        ans2 = feedback(tsys.x1, tsys.x2, 1.)

        np.testing.assert_almost_equal(
            ans1.num[0][0][0] / ans1.den[0][0][0], -2.5 / 6.5)
        np.testing.assert_almost_equal(
            ans2.num[0][0][0] / ans2.den[0][0][0], 2.5 / 8.5)

    def testScalarSS(self, tsys):
        """Scalar system with state space feedback block."""
        ans1 = feedback(tsys.x1, tsys.sys2)
        ans2 = feedback(tsys.x1, tsys.sys2, 1.)

        np.testing.assert_array_almost_equal(ans1.A, [[-1.5, 4.], [13., 2.]])
        np.testing.assert_array_almost_equal(ans1.B, [[2.5], [-10.]])
        np.testing.assert_array_almost_equal(ans1.C, [[-2.5, 0.]])
        np.testing.assert_array_almost_equal(ans1.D, [[2.5]])
        np.testing.assert_array_almost_equal(ans2.A, [[3.5, 4.], [-7., 2.]])
        np.testing.assert_array_almost_equal(ans2.B, [[2.5], [-10.]])
        np.testing.assert_array_almost_equal(ans2.C, [[2.5, 0.]])
        np.testing.assert_array_almost_equal(ans2.D, [[2.5]])

        # Make sure default arugments work as well
        ans3 = feedback(tsys.sys2, 1)
        ans4 = feedback(tsys.sys2)
        np.testing.assert_array_almost_equal(ans3.A, ans4.A)
        np.testing.assert_array_almost_equal(ans3.B, ans4.B)
        np.testing.assert_array_almost_equal(ans3.C, ans4.C)
        np.testing.assert_array_almost_equal(ans3.D, ans4.D)

    def testScalarTF(self, tsys):
        """Scalar system with transfer function feedback block."""
        ans1 = feedback(tsys.x1, tsys.sys1)
        ans2 = feedback(tsys.x1, tsys.sys1, 1.)

        np.testing.assert_array_almost_equal(ans1.num, [[[2.5, 5., 7.5]]])
        np.testing.assert_array_almost_equal(ans1.den, [[[1., 4.5, 8.]]])
        np.testing.assert_array_almost_equal(ans2.num, [[[2.5, 5., 7.5]]])
        np.testing.assert_array_almost_equal(ans2.den, [[[1., -0.5, -2.]]])

        # Make sure default arugments work as well
        ans3 = feedback(tsys.sys1, 1)
        ans4 = feedback(tsys.sys1)
        np.testing.assert_array_almost_equal(ans3.num, ans4.num)
        np.testing.assert_array_almost_equal(ans3.den, ans4.den)

    def testSSScalar(self, tsys):
        """State space system with scalar feedback block."""
        ans1 = feedback(tsys.sys2, tsys.x1)
        ans2 = feedback(tsys.sys2, tsys.x1, 1.)

        np.testing.assert_array_almost_equal(ans1.A, [[-1.5, 4.], [13., 2.]])
        np.testing.assert_array_almost_equal(ans1.B, [[1.], [-4.]])
        np.testing.assert_array_almost_equal(ans1.C, [[1., 0.]])
        np.testing.assert_array_almost_equal(ans1.D, [[0.]])
        np.testing.assert_array_almost_equal(ans2.A, [[3.5, 4.], [-7., 2.]])
        np.testing.assert_array_almost_equal(ans2.B, [[1.], [-4.]])
        np.testing.assert_array_almost_equal(ans2.C, [[1., 0.]])
        np.testing.assert_array_almost_equal(ans2.D, [[0.]])

    def testSSSS1(self, tsys):
        """State space system with state space feedback block."""
        ans1 = feedback(tsys.sys2, tsys.sys2)
        ans2 = feedback(tsys.sys2, tsys.sys2, 1.)

        np.testing.assert_array_almost_equal(ans1.A, [[1., 4., -1., 0.],
            [3., 2., 4., 0.], [1., 0., 1., 4.], [-4., 0., 3., 2]])
        np.testing.assert_array_almost_equal(ans1.B, [[1.], [-4.], [0.], [0.]])
        np.testing.assert_array_almost_equal(ans1.C, [[1., 0., 0., 0.]])
        np.testing.assert_array_almost_equal(ans1.D, [[0.]])
        np.testing.assert_array_almost_equal(ans2.A, [[1., 4., 1., 0.],
            [3., 2., -4., 0.], [1., 0., 1., 4.], [-4., 0., 3., 2.]])
        np.testing.assert_array_almost_equal(ans2.B, [[1.], [-4.], [0.], [0.]])
        np.testing.assert_array_almost_equal(ans2.C, [[1., 0., 0., 0.]])
        np.testing.assert_array_almost_equal(ans2.D, [[0.]])

    def testSSSS2(self, tsys):
        """State space system with state space feedback block, including a
        direct feedthrough term."""
        sys3 = StateSpace([[-1., 4.], [2., -3]], [[2.], [3.]], [[-3., 1.]],
            [[-2.]])
        sys4 = StateSpace([[-3., -2.], [1., 4.]], [[-2.], [-6.]], [[2., -3.]],
            [[3.]])

        ans1 = feedback(sys3, sys4)
        ans2 = feedback(sys3, sys4, 1.)

        np.testing.assert_array_almost_equal(ans1.A,
            [[-4.6, 5.2, 0.8, -1.2], [-3.4, -1.2, 1.2, -1.8],
             [-1.2, 0.4, -1.4, -4.4], [-3.6, 1.2, 5.8, -3.2]])
        np.testing.assert_array_almost_equal(ans1.B,
            [[-0.4], [-0.6], [-0.8], [-2.4]])
        np.testing.assert_array_almost_equal(ans1.C, [[0.6, -0.2, -0.8, 1.2]])
        np.testing.assert_array_almost_equal(ans1.D, [[0.4]])
        np.testing.assert_array_almost_equal(ans2.A,
            [[-3.57142857142857, 4.85714285714286, 0.571428571428571,
                -0.857142857142857],
             [-1.85714285714286, -1.71428571428571, 0.857142857142857,
                -1.28571428571429],
             [0.857142857142857, -0.285714285714286, -1.85714285714286,
                -3.71428571428571],
             [2.57142857142857, -0.857142857142857, 4.42857142857143,
                -1.14285714285714]])
        np.testing.assert_array_almost_equal(ans2.B, [[0.285714285714286],
            [0.428571428571429], [0.571428571428571], [1.71428571428571]])
        np.testing.assert_array_almost_equal(ans2.C, [[-0.428571428571429,
            0.142857142857143, -0.571428571428571, 0.857142857142857]])
        np.testing.assert_array_almost_equal(ans2.D, [[-0.285714285714286]])


    def testSSTF(self, tsys):
        """State space system with transfer function feedback block."""
        # This functionality is not implemented yet.
        pass

    def testTFScalar(self, tsys):
        """Transfer function system with scalar feedback block."""
        ans1 = feedback(tsys.sys1, tsys.x1)
        ans2 = feedback(tsys.sys1, tsys.x1, 1.)

        np.testing.assert_array_almost_equal(ans1.num, [[[1., 2.]]])
        np.testing.assert_array_almost_equal(ans1.den, [[[1., 4.5, 8.]]])
        np.testing.assert_array_almost_equal(ans2.num, [[[1., 2.]]])
        np.testing.assert_array_almost_equal(ans2.den, [[[1., -0.5, -2.]]])

    def testTFSS(self, tsys):
        """Transfer function system with state space feedback block."""
        # This functionality is not implemented yet.
        pass

    def testTFTF(self, tsys):
        """Transfer function system with transfer function feedback block."""
        ans1 = feedback(tsys.sys1, tsys.sys1)
        ans2 = feedback(tsys.sys1, tsys.sys1, 1.)

        np.testing.assert_array_almost_equal(ans1.num, [[[1., 4., 7., 6.]]])
        np.testing.assert_array_almost_equal(ans1.den,
                                             [[[1., 4., 11., 16., 13.]]])
        np.testing.assert_array_almost_equal(ans2.num, [[[1., 4., 7., 6.]]])
        np.testing.assert_array_almost_equal(ans2.den,
                                             [[[1., 4., 9., 8., 5.]]])

    def testLists(self, tsys):
        """Make sure that lists of various lengths work for operations"""
        sys1 = ctrl.tf([1, 1], [1, 2])
        sys2 = ctrl.tf([1, 3], [1, 4])
        sys3 = ctrl.tf([1, 5], [1, 6])
        sys4 = ctrl.tf([1, 7], [1, 8])
        sys5 = ctrl.tf([1, 9], [1, 0])

        # Series
        sys1_2 = ctrl.series(sys1, sys2)
        np.testing.assert_array_almost_equal(sort(pole(sys1_2)), [-4., -2.])
        np.testing.assert_array_almost_equal(sort(zero(sys1_2)), [-3., -1.])

        sys1_3 = ctrl.series(sys1, sys2, sys3)
        np.testing.assert_array_almost_equal(sort(pole(sys1_3)),
                                             [-6., -4., -2.])
        np.testing.assert_array_almost_equal(sort(zero(sys1_3)),
                                             [-5., -3., -1.])

        sys1_4 = ctrl.series(sys1, sys2, sys3, sys4)
        np.testing.assert_array_almost_equal(sort(pole(sys1_4)),
                                             [-8., -6., -4., -2.])
        np.testing.assert_array_almost_equal(sort(zero(sys1_4)),
                                             [-7., -5., -3., -1.])

        sys1_5 = ctrl.series(sys1, sys2, sys3, sys4, sys5)
        np.testing.assert_array_almost_equal(sort(pole(sys1_5)),
                                             [-8., -6., -4., -2., -0.])
        np.testing.assert_array_almost_equal(sort(zero(sys1_5)),
                                             [-9., -7., -5., -3., -1.])

        # Parallel
        sys1_2 = ctrl.parallel(sys1, sys2)
        np.testing.assert_array_almost_equal(sort(pole(sys1_2)), [-4., -2.])
        np.testing.assert_array_almost_equal(sort(zero(sys1_2)),
                                             sort(zero(sys1 + sys2)))

        sys1_3 = ctrl.parallel(sys1, sys2, sys3)
        np.testing.assert_array_almost_equal(sort(pole(sys1_3)),
                                             [-6., -4., -2.])
        np.testing.assert_array_almost_equal(sort(zero(sys1_3)),
                                             sort(zero(sys1 + sys2 + sys3)))

        sys1_4 = ctrl.parallel(sys1, sys2, sys3, sys4)
        np.testing.assert_array_almost_equal(sort(pole(sys1_4)),
                                             [-8., -6., -4., -2.])
        np.testing.assert_array_almost_equal(
            sort(zero(sys1_4)),
            sort(zero(sys1 + sys2 + sys3 + sys4)))

        sys1_5 = ctrl.parallel(sys1, sys2, sys3, sys4, sys5)
        np.testing.assert_array_almost_equal(sort(pole(sys1_5)),
                                             [-8., -6., -4., -2., -0.])
        np.testing.assert_array_almost_equal(
            sort(zero(sys1_5)),
            sort(zero(sys1 + sys2 + sys3 + sys4 + sys5)))

    def testMimoSeries(self, tsys):
        """regression: bdalg.series reverses order of arguments"""
        g1 = ctrl.ss([], [], [], [[1, 2], [0, 3]])
        g2 = ctrl.ss([], [], [], [[1, 0], [2, 3]])
        ref = g2 * g1
        tst = ctrl.series(g1, g2)

        np.testing.assert_array_equal(ref.A, tst.A)
        np.testing.assert_array_equal(ref.B, tst.B)
        np.testing.assert_array_equal(ref.C, tst.C)
        np.testing.assert_array_equal(ref.D, tst.D)

    def test_feedback_args(self, tsys):
        # Added 25 May 2019 to cover missing exception handling in feedback()
        # If first argument is not LTI or convertable, generate an exception
        args = ([1], tsys.sys2)
        with pytest.raises(TypeError):
            ctrl.feedback(*args)

        # If second argument is not LTI or convertable, generate an exception
        args = (tsys.sys1, 'hello world')
        with pytest.raises(TypeError):
            ctrl.feedback(*args)

        # Convert first argument to FRD, if needed
        h = TransferFunction([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 10)
        frd = ctrl.FRD(h, omega)
        sys = ctrl.feedback(1, frd)
        assert isinstance(sys, ctrl.FRD)

    def testConnect(self, tsys):
        sys = append(tsys.sys2, tsys.sys3)  # two siso systems

        # should not raise error
        connect(sys, [[1, 2], [2, -2]], [2], [1, 2])
        connect(sys, [[1, 2], [2, 0]], [2], [1, 2])
        connect(sys, [[1, 2, 0], [2, -2, 1]], [2], [1, 2])
        connect(sys, [[1, 2], [2, -2]], [2, 1], [1])
        sys3x3 = append(sys, tsys.sys3)  # 3x3 mimo
        connect(sys3x3, [[1, 2, 0], [2, -2, 1], [3, -3, 0]], [2], [1, 2])
        connect(sys3x3, [[1, 2, 0], [2, -2, 1], [3, -3, 0]], [1, 2, 3], [3])
        connect(sys3x3, [[1, 2, 0], [2, -2, 1], [3, -3, 0]], [2, 3], [2, 1])

        # feedback interconnection out of bounds: input too high
        Q = [[1, 3], [2, -2]]
        with pytest.raises(IndexError):
            connect(sys, Q, [2], [1, 2])
        # feedback interconnection out of bounds: input too low
        Q = [[0, 2], [2, -2]]
        with pytest.raises(IndexError):
            connect(sys, Q, [2], [1, 2])

        # feedback interconnection out of bounds: output too high
        Q = [[1, 2], [2, -3]]
        with pytest.raises(IndexError):
            connect(sys, Q, [2], [1, 2])
        Q = [[1, 2], [2, 4]]
        with pytest.raises(IndexError):
            connect(sys, Q, [2], [1, 2])

        # input/output index testing
        Q = [[1, 2], [2, -2]]  # OK interconnection

        # input index is out of bounds: too high
        with pytest.raises(IndexError):
            connect(sys, Q, [3], [1, 2])
        # input index is out of bounds: too low
        with pytest.raises(IndexError):
            connect(sys, Q, [0], [1, 2])
        with pytest.raises(IndexError):
            connect(sys, Q, [-2], [1, 2])
        # output index is out of bounds: too high
        with pytest.raises(IndexError):
            connect(sys, Q, [2], [1, 3])
        # output index is out of bounds: too low
        with pytest.raises(IndexError):
            connect(sys, Q, [2], [1, 0])
        with pytest.raises(IndexError):
            connect(sys, Q, [2], [1, -1])
