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
        sys1 = TransferFunction([1, 2], [1, 2, 3])
        sys2 = StateSpace([[1., 4.], [3., 2.]], [[1.], [-4.]],
            [[1., 0.]], [[0.]])
        self.systems = (sys1, sys2)

    def check_cl_poles(self, sys, pole_list, k_list):
        for k, poles in zip(k_list, pole_list):
            poles_expected = np.sort(feedback(sys, k).pole())
            poles = np.sort(poles)
            np.testing.assert_array_almost_equal(poles, poles_expected)

    def testRootLocus(self):
        """Basic root locus plot"""
        klist = [-1, 0, 1]
        for sys in self.systems:
            roots, k_out = root_locus(sys, klist, Plot=False)
            np.testing.assert_equal(len(roots), len(klist))
            np.testing.assert_array_equal(klist, k_out)
            self.check_cl_poles(sys, roots, klist)

    def test_without_gains(self):
        for sys in self.systems:
            roots, kvect = root_locus(sys, Plot=False)
            self.check_cl_poles(sys, roots, kvect)

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestRootLocus)

if __name__ == "__main__":
    unittest.main()
