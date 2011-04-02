#!/usr/bin/env python
#
# nichols_test.py - test Nichols plot
# RMM, 31 Mar 2011

import unittest
import numpy as np
from control.matlab import *

class TestStateSpace(unittest.TestCase):
    """Tests for the Nichols plots."""

    def setUp(self):
        """Set up a system to test operations on."""

        A = [[-3., 4., 2.], [-1., -3., 0.], [2., 5., 3.]]
        B = [[1.], [-3.], [-2.]]
        C = [[4., 2., -3.]]
        D = [[0.]]
        
        self.sys = StateSpace(A, B, C, D) 

    def testNicholsPlain(self):
        """Generate a Nichols plot."""
        nichols(self.sys)

    def testNgrid(self):
        """Generate a Nichols plot."""
        nichols(self.sys, grid=False)
        ngrid()

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestStateSpace)

        
if __name__ == "__main__":
    unittest.main()
