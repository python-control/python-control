#!/usr/bin/env python
#
# rlocus_test.py - unit test for root locus diagrams
# RMM, 1 Jul 2011

import unittest
import numpy as np
from control.rlocus import root_locus
from control.xferfcn import TransferFunction
from control.statesp import StateSpace
from control.bdalg import feedback

class TestRootLocus(unittest.TestCase):
    """These are tests for the feedback function in rlocus.py."""

    def setUp(self):
        """This contains some random LTI systems and scalars for testing."""

        # Two random SISO systems.
        self.sys1 = TransferFunction([1, 2], [1, 2, 3])
        self.sys2 = StateSpace([[1., 4.], [3., 2.]], [[1.], [-4.]],
            [[1., 0.]], [[0.]])

    def testRootLocus(self):
        """Basic root locus plot"""
        klist = [-1, 0, 1]
        rlist = root_locus(self.sys1, [-1, 0, 1], Plot=False)

        for k in klist:
            np.testing.assert_array_almost_equal(
                np.sort(rlist[k]), 
                np.sort(feedback(self.sys1, klist[k]).pole()))

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestRootLocus)

if __name__ == "__main__":
    unittest.main()
