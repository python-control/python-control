#!/usr/bin/env python

"""TestConvert.py

Test state space and transfer function conversion.

Currently, this unit test script is not complete.  Ideally, it would convert
several random state spaces back and forth between state space and transfer
function representations, and assert that the conversion outputs are correct.
As they currently stand, the td04ad and tb04ad functions appear to be buggy from
time to time.  Therefore, this script can be used to diagnose the errors.

"""

import numpy as np
import matlab
import unittest

class TestConvert(unittest.TestCase):
    """Test state space and transfer function conversions."""

    def setUp(self):
        """Set up testing parameters."""

        # Number of times to run each of the randomized tests.
        self.numTests = 1
        # Maximum number of states to test + 1
        self.maxStates = 3
        # Maximum number of inputs and outputs to test + 1
        self.maxIO = 3
        # Set to True to print systems to the output.
        self.debug = False

    def printSys(self, sys, ind):
        """Print system to the standard output."""

        if self.debug:
            print "sys%i:\n" % ind
            print sys

    def testConvert(self):
        """Test state space to transfer function conversion."""
        
        print __doc__

        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys1 = matlab.rss(states, inputs, outputs)
                    self.printSys(sys1, 1)

                    sys2 = matlab.tf(sys1)
                    self.printSys(sys2, 2)

                    sys3 = matlab.ss(sys2)
                    self.printSys(sys3, 3)

                    sys4 = matlab.tf(sys3)
                    self.printSys(sys4, 4)

if __name__ == "__main__":
    unittest.main()
