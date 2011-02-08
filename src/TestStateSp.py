#!/usr/bin/env python

import numpy as np
import matlab
from statesp import StateSpace
import unittest

class TestStateSpace(unittest.TestCase):
    """Tests for the StateSpace class."""

    def setUp(self):
        """Set up a MIMO system to test operations on."""

        A = [[-3., 4., 2.], [-1., -3., 0.], [2., 5., 3.]]
        B = [[1., 4.], [-3., -3.], [-2., 1.]]
        C = [[4., 2., -3.], [1., 4., 3.]]
        D = [[-2., 4.], [0., 1.]]
        
        a = [[4., 1.], [2., -3]]
        b = [[5., 2.], [-3., -3.]]
        c = [[2., -4], [0., 1.]]
        d = [[3., 2.], [1., -1.]]

        self.sys1 = StateSpace(A, B, C, D) 
        self.sys2 = StateSpace(a, b, c, d)

    def testPole(self):
        """Evaluate the poles of a MIMO system."""

        p = self.sys1.pole()

        np.testing.assert_array_almost_equal(p, [3.34747678408874,
            -3.17373839204437 + 1.47492908003839j,
            -3.17373839204437 - 1.47492908003839j])

    def testZero(self):
        """Evaluate the zeros of a SISO system."""

        sys = StateSpace(self.sys1.A, [[3.], [-2.], [4.]], [[-1., 3., 2.]], [[-4.]])
        z = sys.zero()

        np.testing.assert_array_almost_equal(z, [4.26864638637134,
            -3.75932319318567 + 1.10087776649554j,
            -3.75932319318567 - 1.10087776649554j])

    def testAdd(self):
        """Add two MIMO systems."""

        A = [[-3., 4., 2., 0., 0.], [-1., -3., 0., 0., 0.],
             [2., 5., 3., 0., 0.], [0., 0., 0., 4., 1.], [0., 0., 0., 2., -3.]]
        B = [[1., 4.], [-3., -3.], [-2., 1.], [5., 2.], [-3., -3.]]
        C = [[4., 2., -3., 2., -4.], [1., 4., 3., 0., 1.]]
        D = [[1., 6.], [1., 0.]]

        sys = self.sys1 + self.sys2
        
        np.testing.assert_array_almost_equal(sys.A, A)
        np.testing.assert_array_almost_equal(sys.B, B)
        np.testing.assert_array_almost_equal(sys.C, C)
        np.testing.assert_array_almost_equal(sys.D, D)

    def testSub(self):
        """Subtract two MIMO systems."""

        A = [[-3., 4., 2., 0., 0.], [-1., -3., 0., 0., 0.],
             [2., 5., 3., 0., 0.], [0., 0., 0., 4., 1.], [0., 0., 0., 2., -3.]]
        B = [[1., 4.], [-3., -3.], [-2., 1.], [5., 2.], [-3., -3.]]
        C = [[4., 2., -3., -2., 4.], [1., 4., 3., 0., -1.]]
        D = [[-5., 2.], [-1., 2.]]

        sys = self.sys1 - self.sys2

        np.testing.assert_array_almost_equal(sys.A, A)
        np.testing.assert_array_almost_equal(sys.B, B)
        np.testing.assert_array_almost_equal(sys.C, C)
        np.testing.assert_array_almost_equal(sys.D, D)

    def testMul(self):
        """Multiply two MIMO systems."""

        A = [[4., 1., 0., 0., 0.], [2., -3., 0., 0., 0.], [2., 0., -3., 4., 2.],
             [-6., 9., -1., -3., 0.], [-4., 9., 2., 5., 3.]]
        B = [[5., 2.], [-3., -3.], [7., -2.], [-12., -3.], [-5., -5.]]
        C = [[-4., 12., 4., 2., -3.], [0., 1., 1., 4., 3.]]
        D = [[-2., -8.], [1., -1.]]

        sys = self.sys1 * self.sys2
        
        np.testing.assert_array_almost_equal(sys.A, A)
        np.testing.assert_array_almost_equal(sys.B, B)
        np.testing.assert_array_almost_equal(sys.C, C)
        np.testing.assert_array_almost_equal(sys.D, D)

    def testEvalFr(self):
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
        
        np.testing.assert_almost_equal(sys.evalfr(1.), resp)

    def testFreqResp(self):
        """Evaluate the frequency response at multiple frequencies."""

        A = [[-2, 0.5], [0.5, -0.3]]
        B = [[0.3, -1.3], [0.1, 0.]]
        C = [[0., 0.1], [-0.3, -0.2]]
        D = [[0., -0.8], [-0.3, 0.]]
        sys = StateSpace(A, B, C, D)

        truemag = [[[0.0852992637230322, 0.00103596611395218], 
                    [0.935374692849736, 0.799380720864549]],
                   [[0.55656854563842, 0.301542699860857],
                    [0.609178071542849, 0.0382108097985257]]]
        truephase = [[[-0.566195599644593, -1.68063565332582],
                      [3.0465958317514, 3.14141384339534]],
                     [[2.90457947657161, 3.10601268291914],
                      [-0.438157380501337, -1.40720969147217]]]
        trueomega = [0.1, 10.]

        mag, phase, omega = sys.freqresp(trueomega)

        np.testing.assert_almost_equal(mag, truemag)
        np.testing.assert_almost_equal(phase, truephase)
        np.testing.assert_equal(omega, trueomega)

class TestRss(unittest.TestCase):
    """These are tests for the proper functionality of statesp.rss."""
    
    def setUp(self):
        # Number of times to run each of the randomized tests.
        self.numTests = 100
        # Maxmimum number of states to test + 1
        self.maxStates = 10
        # Maximum number of inputs and outputs to test + 1
        self.maxIO = 5
        
    def testShape(self):
        """Test that rss outputs have the right state, input, and output
        size."""
        
        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.rss(states, inputs, outputs)
                    self.assertEqual(sys.states, states)
                    self.assertEqual(sys.inputs, inputs)
                    self.assertEqual(sys.outputs, outputs)
    
    def testPole(self):
        """Test that the poles of rss outputs have a negative real part."""
        
        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.rss(states, inputs, outputs)
                    p = sys.pole()
                    for z in p:
                        self.assertTrue(z.real < 0)

class TestDrss(unittest.TestCase):
    """These are tests for the proper functionality of statesp.drss."""
    
    def setUp(self):
        # Number of times to run each of the randomized tests.
        self.numTests = 100
        # Maxmimum number of states to test + 1
        self.maxStates = 10
        # Maximum number of inputs and outputs to test + 1
        self.maxIO = 5
        
    def testShape(self):
        """Test that drss outputs have the right state, input, and output
        size."""
        
        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.drss(states, inputs, outputs)
                    self.assertEqual(sys.states, states)
                    self.assertEqual(sys.inputs, inputs)
                    self.assertEqual(sys.outputs, outputs)
    
    def testPole(self):
        """Test that the poles of drss outputs have less than unit magnitude."""
        
        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.drss(states, inputs, outputs)
                    p = sys.pole()
                    for z in p:
                        self.assertTrue(abs(z) < 1)
                        
if __name__ == "__main__":
    unittest.main()
