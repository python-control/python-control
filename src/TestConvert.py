#!/usr/bin/env python

import numpy as np
import matlab
import unittest

class TestConvert(unittest.TestCase):
    """Test state space and transfer function conversions."""

    def setUp(self):
        """Set up testing parameters."""

        # Number of times to run each of the randomized tests.
        self.numTests = 10
        # Maximum number of states to test + 1
        self.maxStates = 5
        # Maximum number of inputs and outputs to test + 1
        self.maxIO = 5

    def testConvert(self):
        """Test state space to transfer function conversion."""

        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.rss(states, inputs, outputs)
                    print "sys1:\n", sys
                    sys2 = matlab.tf(sys)
                    print "sys2:\n", sys2
                    sys3 = matlab.ss(sys2)
                    print "sys3:\n", sys3
                    sys4 = matlab.tf(sys3)
                    print "sys4:\n", sys4

if __name__ == "__main__":
    unittest.main()
