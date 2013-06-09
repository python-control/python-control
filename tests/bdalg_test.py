#!/usr/bin/env python
#
# bdalg_test.py - test suit for block diagram algebra
# RMM, 30 Mar 2011 (based on TestBDAlg from v0.4a)

import unittest
import numpy as np
from control.xferfcn import TransferFunction
from control.statesp import StateSpace
from control.bdalg import feedback

class TestFeedback(unittest.TestCase):
    """These are tests for the feedback function in bdalg.py.  Currently, some
    of the tests are not implemented, or are not working properly.  TODO: these
    need to be fixed."""

    def setUp(self):
        """This contains some random LTI systems and scalars for testing."""

        # Two random SISO systems.
        self.sys1 = TransferFunction([1, 2], [1, 2, 3])
        self.sys2 = StateSpace([[1., 4.], [3., 2.]], [[1.], [-4.]],
            [[1., 0.]], [[0.]])
        # Two random scalars.
        self.x1 = 2.5
        self.x2 = -3.

    def testScalarScalar(self):
        """Scalar system with scalar feedback block."""

        ans1 = feedback(self.x1, self.x2)
        ans2 = feedback(self.x1, self.x2, 1.)

        self.assertAlmostEqual(ans1.num[0][0][0] / ans1.den[0][0][0],
            -2.5 / 6.5)
        self.assertAlmostEqual(ans2.num[0][0][0] / ans2.den[0][0][0], 2.5 / 8.5)

    def testScalarSS(self):
        """Scalar system with state space feedback block."""

        ans1 = feedback(self.x1, self.sys2)
        ans2 = feedback(self.x1, self.sys2, 1.)
        
        np.testing.assert_array_almost_equal(ans1.A, [[-1.5, 4.], [13., 2.]])
        np.testing.assert_array_almost_equal(ans1.B, [[2.5], [-10.]])
        np.testing.assert_array_almost_equal(ans1.C, [[-2.5, 0.]])
        np.testing.assert_array_almost_equal(ans1.D, [[2.5]])
        np.testing.assert_array_almost_equal(ans2.A, [[3.5, 4.], [-7., 2.]])
        np.testing.assert_array_almost_equal(ans2.B, [[2.5], [-10.]])
        np.testing.assert_array_almost_equal(ans2.C, [[2.5, 0.]])
        np.testing.assert_array_almost_equal(ans2.D, [[2.5]])

        # Make sure default arugments work as well
        ans3 = feedback(self.sys2, 1)
        ans4 = feedback(self.sys2)
        np.testing.assert_array_almost_equal(ans3.A, ans4.A)
        np.testing.assert_array_almost_equal(ans3.B, ans4.B)
        np.testing.assert_array_almost_equal(ans3.C, ans4.C)
        np.testing.assert_array_almost_equal(ans3.D, ans4.D)

    def testScalarTF(self):
        """Scalar system with transfer function feedback block."""

        ans1 = feedback(self.x1, self.sys1)
        ans2 = feedback(self.x1, self.sys1, 1.)

        np.testing.assert_array_almost_equal(ans1.num, [[[2.5, 5., 7.5]]])
        np.testing.assert_array_almost_equal(ans1.den, [[[1., 4.5, 8.]]])
        np.testing.assert_array_almost_equal(ans2.num, [[[2.5, 5., 7.5]]])
        np.testing.assert_array_almost_equal(ans2.den, [[[1., -0.5, -2.]]])

        # Make sure default arugments work as well
        ans3 = feedback(self.sys1, 1)
        ans4 = feedback(self.sys1)
        np.testing.assert_array_almost_equal(ans3.num, ans4.num)
        np.testing.assert_array_almost_equal(ans3.den, ans4.den)

    def testSSScalar(self):
        """State space system with scalar feedback block."""
        
        ans1 = feedback(self.sys2, self.x1)
        ans2 = feedback(self.sys2, self.x1, 1.)

        np.testing.assert_array_almost_equal(ans1.A, [[-1.5, 4.], [13., 2.]])
        np.testing.assert_array_almost_equal(ans1.B, [[1.], [-4.]])
        np.testing.assert_array_almost_equal(ans1.C, [[1., 0.]])
        np.testing.assert_array_almost_equal(ans1.D, [[0.]])
        np.testing.assert_array_almost_equal(ans2.A, [[3.5, 4.], [-7., 2.]])
        np.testing.assert_array_almost_equal(ans2.B, [[1.], [-4.]])
        np.testing.assert_array_almost_equal(ans2.C, [[1., 0.]])
        np.testing.assert_array_almost_equal(ans2.D, [[0.]])

    def testSSSS1(self):
        """State space system with state space feedback block."""

        ans1 = feedback(self.sys2, self.sys2)
        ans2 = feedback(self.sys2, self.sys2, 1.)

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

    def testSSSS2(self):
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


    def testSSTF(self):
        """State space system with transfer function feedback block."""
        
        # This functionality is not implemented yet.
        pass

    def testTFScalar(self):
        """Transfer function system with scalar feedback block."""

        ans1 = feedback(self.sys1, self.x1)
        ans2 = feedback(self.sys1, self.x1, 1.)

        np.testing.assert_array_almost_equal(ans1.num, [[[1., 2.]]])
        np.testing.assert_array_almost_equal(ans1.den, [[[1., 4.5, 8.]]])
        np.testing.assert_array_almost_equal(ans2.num, [[[1., 2.]]])
        np.testing.assert_array_almost_equal(ans2.den, [[[1., -0.5, -2.]]])

    def testTFSS(self):
        """Transfer function system with state space feedback block."""

        # This functionality is not implemented yet.
        pass

    def testTFTF(self):
        """Transfer function system with transfer function feedback block."""

        ans1 = feedback(self.sys1, self.sys1)
        ans2 = feedback(self.sys1, self.sys1, 1.)

        np.testing.assert_array_almost_equal(ans1.num, [[[1., 4., 7., 6.]]])
        np.testing.assert_array_almost_equal(ans1.den,
            [[[1., 4., 11., 16., 13.]]])
        np.testing.assert_array_almost_equal(ans2.num, [[[1., 4., 7., 6.]]])
        np.testing.assert_array_almost_equal(ans2.den, [[[1., 4., 9., 8., 5.]]])

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestFeedback)

if __name__ == "__main__":
    unittest.main()
