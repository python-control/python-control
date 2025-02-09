"""bdalg_test.py - test suite for block diagram algebra.

RMM, 30 Mar 2011 (based on TestBDAlg from v0.4a)
"""

import control as ctrl
import numpy as np
import pytest
from control.bdalg import _ensure_tf, append, connect, feedback
from control.lti import poles, zeros
from control.statesp import StateSpace
from control.tests.conftest import assert_tf_close_coeff
from control.xferfcn import TransferFunction
from numpy import sort


class TestFeedback:
    """Tests for the feedback function in bdalg.py."""

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
        """Make sure that lists of various lengths work for operations."""
        sys1 = ctrl.tf([1, 1], [1, 2])
        sys2 = ctrl.tf([1, 3], [1, 4])
        sys3 = ctrl.tf([1, 5], [1, 6])
        sys4 = ctrl.tf([1, 7], [1, 8])
        sys5 = ctrl.tf([1, 9], [1, 0])

        # Series
        sys1_2 = ctrl.series(sys1, sys2)
        np.testing.assert_array_almost_equal(sort(poles(sys1_2)), [-4., -2.])
        np.testing.assert_array_almost_equal(sort(zeros(sys1_2)), [-3., -1.])

        sys1_3 = ctrl.series(sys1, sys2, sys3)
        np.testing.assert_array_almost_equal(sort(poles(sys1_3)),
                                             [-6., -4., -2.])
        np.testing.assert_array_almost_equal(sort(zeros(sys1_3)),
                                             [-5., -3., -1.])

        sys1_4 = ctrl.series(sys1, sys2, sys3, sys4)
        np.testing.assert_array_almost_equal(sort(poles(sys1_4)),
                                             [-8., -6., -4., -2.])
        np.testing.assert_array_almost_equal(sort(zeros(sys1_4)),
                                             [-7., -5., -3., -1.])

        sys1_5 = ctrl.series(sys1, sys2, sys3, sys4, sys5)
        np.testing.assert_array_almost_equal(sort(poles(sys1_5)),
                                             [-8., -6., -4., -2., -0.])
        np.testing.assert_array_almost_equal(sort(zeros(sys1_5)),
                                             [-9., -7., -5., -3., -1.])

        # Parallel
        sys1_2 = ctrl.parallel(sys1, sys2)
        np.testing.assert_array_almost_equal(sort(poles(sys1_2)), [-4., -2.])
        np.testing.assert_array_almost_equal(sort(zeros(sys1_2)),
                                             sort(zeros(sys1 + sys2)))

        sys1_3 = ctrl.parallel(sys1, sys2, sys3)
        np.testing.assert_array_almost_equal(sort(poles(sys1_3)),
                                             [-6., -4., -2.])
        np.testing.assert_array_almost_equal(sort(zeros(sys1_3)),
                                             sort(zeros(sys1 + sys2 + sys3)))

        sys1_4 = ctrl.parallel(sys1, sys2, sys3, sys4)
        np.testing.assert_array_almost_equal(sort(poles(sys1_4)),
                                             [-8., -6., -4., -2.])
        np.testing.assert_array_almost_equal(
            sort(zeros(sys1_4)),
            sort(zeros(sys1 + sys2 + sys3 + sys4)))

        sys1_5 = ctrl.parallel(sys1, sys2, sys3, sys4, sys5)
        np.testing.assert_array_almost_equal(sort(poles(sys1_5)),
                                             [-8., -6., -4., -2., -0.])
        np.testing.assert_array_almost_equal(
            sort(zeros(sys1_5)),
            sort(zeros(sys1 + sys2 + sys3 + sys4 + sys5)))

    def testMimoSeries(self, tsys):
        """regression: bdalg.series reverses order of arguments."""
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

        with pytest.warns(FutureWarning, match="use interconnect()"):
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


@pytest.mark.parametrize(
    "op, nsys, ninputs, noutputs, nstates", [
        (ctrl.series, 2, 1, 1, 4),
        (ctrl.parallel, 2, 1, 1, 4),
        (ctrl.feedback, 2, 1, 1, 4),
        (ctrl.append, 2, 2, 2, 4),
        (ctrl.negate, 1, 1, 1, 2),
    ])
def test_bdalg_update_names(op, nsys, ninputs, noutputs, nstates):
    syslist = [ctrl.rss(2, 1, 1), ctrl.rss(2, 1, 1)]
    inputs = ['in1', 'in2']
    outputs = ['out1', 'out2']
    states = ['x1', 'x2', 'x3', 'x4']

    newsys = op(
        *syslist[:nsys], name='newsys', inputs=inputs[:ninputs],
        outputs=outputs[:noutputs], states=states[:nstates])
    assert newsys.name == 'newsys'
    assert newsys.ninputs == ninputs
    assert newsys.input_labels == inputs[:ninputs]
    assert newsys.noutputs == noutputs
    assert newsys.output_labels == outputs[:noutputs]
    assert newsys.nstates == nstates
    assert newsys.state_labels == states[:nstates]


def test_bdalg_udpate_names_errors():
    sys1 = ctrl.rss(2, 1, 1)
    sys2 = ctrl.rss(2, 1, 1)

    with pytest.raises(ValueError, match="number of inputs does not match"):
        ctrl.series(sys1, sys2, inputs=2)

    with pytest.raises(ValueError, match="number of outputs does not match"):
        ctrl.series(sys1, sys2, outputs=2)

    with pytest.raises(ValueError, match="number of states does not match"):
        ctrl.series(sys1, sys2, states=2)

    with pytest.raises(ValueError, match="number of states does not match"):
        ctrl.series(ctrl.tf(sys1), ctrl.tf(sys2), states=2)

    with pytest.raises(TypeError, match="unrecognized keywords"):
        ctrl.series(sys1, sys2, dt=1)


class TestEnsureTf:
    """Test `_ensure_tf`."""

    @pytest.mark.parametrize(
        "arraylike_or_tf, dt, tf",
        [
            (
                ctrl.TransferFunction([1], [1, 2, 3]),
                None,
                ctrl.TransferFunction([1], [1, 2, 3]),
            ),
            (
                ctrl.TransferFunction([1], [1, 2, 3]),
                0,
                ctrl.TransferFunction([1], [1, 2, 3]),
            ),
            (
                2,
                None,
                ctrl.TransferFunction([2], [1]),
            ),
            (
                np.array([2]),
                None,
                ctrl.TransferFunction([2], [1]),
            ),
            (
                np.array([[2]]),
                None,
                ctrl.TransferFunction([2], [1]),
            ),
            (
                np.array(
                    [
                        [2, 0, 3],
                        [1, 2, 3],
                    ]
                ),
                None,
                ctrl.TransferFunction(
                    [
                        [[2], [0], [3]],
                        [[1], [2], [3]],
                    ],
                    [
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                    ],
                ),
            ),
            (
                np.array([2, 0, 3]),
                None,
                ctrl.TransferFunction(
                    [
                        [[2], [0], [3]],
                    ],
                    [
                        [[1], [1], [1]],
                    ],
                ),
            ),
        ],
    )
    def test_ensure(self, arraylike_or_tf, dt, tf):
        """Test nominal cases."""
        ensured_tf = _ensure_tf(arraylike_or_tf, dt)
        assert_tf_close_coeff(tf, ensured_tf)

    @pytest.mark.parametrize(
        "arraylike_or_tf, dt, exception",
        [
            (
                ctrl.TransferFunction([1], [1, 2, 3]),
                0.1,
                ValueError,
            ),
            (
                ctrl.TransferFunction([1], [1, 2, 3], 0.1),
                0,
                ValueError,
            ),
            (
                np.ones((1, 1, 1)),
                None,
                ValueError,
            ),
            (
                np.ones((1, 1, 1, 1)),
                None,
                ValueError,
            ),
        ],
    )
    def test_error_ensure(self, arraylike_or_tf, dt, exception):
        """Test error cases."""
        with pytest.raises(exception):
            _ensure_tf(arraylike_or_tf, dt)


class TestTfCombineSplit:
    """Test `combine_tf` and `split_tf`."""

    @pytest.mark.parametrize(
        "tf_array, tf",
        [
            # Continuous-time
            (
                [
                    [ctrl.TransferFunction([1], [1, 1])],
                    [ctrl.TransferFunction([2], [1, 0])],
                ],
                ctrl.TransferFunction(
                    [
                        [[1]],
                        [[2]],
                    ],
                    [
                        [[1, 1]],
                        [[1, 0]],
                    ],
                ),
            ),
            # Discrete-time
            (
                [
                    [ctrl.TransferFunction([1], [1, 1], dt=1)],
                    [ctrl.TransferFunction([2], [1, 0], dt=1)],
                ],
                ctrl.TransferFunction(
                    [
                        [[1]],
                        [[2]],
                    ],
                    [
                        [[1, 1]],
                        [[1, 0]],
                    ],
                    dt=1,
                ),
            ),
            # Scalar
            (
                [
                    [2],
                    [ctrl.TransferFunction([2], [1, 0])],
                ],
                ctrl.TransferFunction(
                    [
                        [[2]],
                        [[2]],
                    ],
                    [
                        [[1]],
                        [[1, 0]],
                    ],
                ),
            ),
            # Matrix
            (
                [
                    [np.eye(3)],
                    [
                        ctrl.TransferFunction(
                            [
                                [[2], [0], [3]],
                                [[1], [2], [3]],
                            ],
                            [
                                [[1], [1], [1]],
                                [[1], [1], [1]],
                            ],
                        )
                    ],
                ],
                ctrl.TransferFunction(
                    [
                        [[1], [0], [0]],
                        [[0], [1], [0]],
                        [[0], [0], [1]],
                        [[2], [0], [3]],
                        [[1], [2], [3]],
                    ],
                    [
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                    ],
                ),
            ),
            # Inhomogeneous
            (
                [
                    [np.eye(3)],
                    [
                        ctrl.TransferFunction(
                            [
                                [[2], [0]],
                                [[1], [2]],
                            ],
                            [
                                [[1], [1]],
                                [[1], [1]],
                            ],
                        ),
                        ctrl.TransferFunction(
                            [
                                [[3]],
                                [[3]],
                            ],
                            [
                                [[1]],
                                [[1]],
                            ],
                        ),
                    ],
                ],
                ctrl.TransferFunction(
                    [
                        [[1], [0], [0]],
                        [[0], [1], [0]],
                        [[0], [0], [1]],
                        [[2], [0], [3]],
                        [[1], [2], [3]],
                    ],
                    [
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                    ],
                ),
            ),
            # Discrete-time
            (
                [
                    [2],
                    [ctrl.TransferFunction([2], [1, 0], dt=0.1)],
                ],
                ctrl.TransferFunction(
                    [
                        [[2]],
                        [[2]],
                    ],
                    [
                        [[1]],
                        [[1, 0]],
                    ],
                    dt=0.1,
                ),
            ),
        ],
    )
    def test_combine_tf(self, tf_array, tf):
        """Test combining transfer functions."""
        tf_combined = ctrl.combine_tf(tf_array)
        assert_tf_close_coeff(tf_combined, tf)

    @pytest.mark.parametrize(
        "tf_array, tf",
        [
            (
                np.array(
                    [
                        [ctrl.TransferFunction([1], [1, 1])],
                    ],
                    dtype=object,
                ),
                ctrl.TransferFunction(
                    [
                        [[1]],
                    ],
                    [
                        [[1, 1]],
                    ],
                ),
            ),
            (
                np.array(
                    [
                        [ctrl.TransferFunction([1], [1, 1])],
                        [ctrl.TransferFunction([2], [1, 0])],
                    ],
                    dtype=object,
                ),
                ctrl.TransferFunction(
                    [
                        [[1]],
                        [[2]],
                    ],
                    [
                        [[1, 1]],
                        [[1, 0]],
                    ],
                ),
            ),
            (
                np.array(
                    [
                        [ctrl.TransferFunction([1], [1, 1], dt=1)],
                        [ctrl.TransferFunction([2], [1, 0], dt=1)],
                    ],
                    dtype=object,
                ),
                ctrl.TransferFunction(
                    [
                        [[1]],
                        [[2]],
                    ],
                    [
                        [[1, 1]],
                        [[1, 0]],
                    ],
                    dt=1,
                ),
            ),
            (
                np.array(
                    [
                        [ctrl.TransferFunction([2], [1], dt=0.1)],
                        [ctrl.TransferFunction([2], [1, 0], dt=0.1)],
                    ],
                    dtype=object,
                ),
                ctrl.TransferFunction(
                    [
                        [[2]],
                        [[2]],
                    ],
                    [
                        [[1]],
                        [[1, 0]],
                    ],
                    dt=0.1,
                ),
            ),
        ],
    )
    def test_split_tf(self, tf_array, tf):
        """Test splitting transfer functions."""
        tf_split = ctrl.split_tf(tf)
        # Test entry-by-entry
        for i in range(tf_split.shape[0]):
            for j in range(tf_split.shape[1]):
                assert_tf_close_coeff(
                    tf_split[i, j],
                    tf_array[i, j],
                )
        # Test combined
        assert_tf_close_coeff(
            ctrl.combine_tf(tf_split),
            ctrl.combine_tf(tf_array),
        )

    @pytest.mark.parametrize(
        "tf_array, exception",
        [
            # Wrong timesteps
            (
                [
                    [ctrl.TransferFunction([1], [1, 1], 0.1)],
                    [ctrl.TransferFunction([2], [1, 0], 0.2)],
                ],
                ValueError,
            ),
            (
                [
                    [ctrl.TransferFunction([1], [1, 1], 0.1)],
                    [ctrl.TransferFunction([2], [1, 0], 0)],
                ],
                ValueError,
            ),
            # Too few dimensions
            (
                [
                    ctrl.TransferFunction([1], [1, 1]),
                    ctrl.TransferFunction([2], [1, 0]),
                ],
                ValueError,
            ),
            # Too many dimensions
            (
                [
                    [[ctrl.TransferFunction([1], [1, 1], 0.1)]],
                    [[ctrl.TransferFunction([2], [1, 0], 0)]],
                ],
                ValueError,
            ),
            # Incompatible dimensions
            (
                [
                    [
                        ctrl.TransferFunction(
                            [
                                [
                                    [1],
                                ]
                            ],
                            [
                                [
                                    [1, 1],
                                ]
                            ],
                        ),
                        ctrl.TransferFunction(
                            [
                                [[2], [1]],
                                [[1], [3]],
                            ],
                            [
                                [[1, 0], [1, 0]],
                                [[1, 0], [1, 0]],
                            ],
                        ),
                    ],
                ],
                ValueError,
            ),
            (
                [
                    [
                        ctrl.TransferFunction(
                            [
                                [[2], [1]],
                                [[1], [3]],
                            ],
                            [
                                [[1, 0], [1, 0]],
                                [[1, 0], [1, 0]],
                            ],
                        ),
                        ctrl.TransferFunction(
                            [
                                [
                                    [1],
                                ]
                            ],
                            [
                                [
                                    [1, 1],
                                ]
                            ],
                        ),
                    ],
                ],
                ValueError,
            ),
            (
                [
                    [
                        ctrl.TransferFunction(
                            [
                                [[2], [1]],
                                [[1], [3]],
                            ],
                            [
                                [[1, 0], [1, 0]],
                                [[1, 0], [1, 0]],
                            ],
                        ),
                        ctrl.TransferFunction(
                            [
                                [[2], [1]],
                                [[1], [3]],
                            ],
                            [
                                [[1, 0], [1, 0]],
                                [[1, 0], [1, 0]],
                            ],
                        ),
                    ],
                    [
                        ctrl.TransferFunction(
                            [
                                [[2], [1], [1]],
                                [[1], [3], [2]],
                            ],
                            [
                                [[1, 0], [1, 0], [1, 0]],
                                [[1, 0], [1, 0], [1, 0]],
                            ],
                        ),
                        ctrl.TransferFunction(
                            [
                                [[2], [1]],
                                [[1], [3]],
                            ],
                            [
                                [[1, 0], [1, 0]],
                                [[1, 0], [1, 0]],
                            ],
                        ),
                    ],
                ],
                ValueError,
            ),
        ],
    )
    def test_error_combine_tf(self, tf_array, exception):
        """Test error cases."""
        with pytest.raises(exception):
            ctrl.combine_tf(tf_array)
