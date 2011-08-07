#!/usr/bin/env python
#
# margin_test.py - test suit for stability margin commands
# RMM, 15 Jul 2011

import unittest
import numpy as np
from control.xferfcn import TransferFunction
from control.statesp import StateSpace
from control.margins import *

class TestMargin(unittest.TestCase):
    """These are tests for the margin commands in margin.py."""

    def setUp(self):
        self.sys1 = TransferFunction([1, 2], [1, 2, 3])
        self.sys2 = TransferFunction([1], [1, 2, 3, 4])
        self.sys3 = StateSpace([[1., 4.], [3., 2.]], [[1.], [-4.]],
            [[1., 0.]], [[0.]])

    def test_stability_margins(self):
        gm, pm, sm, wg, wp, ws = stability_margins(self.sys1);
        gm, pm, sm, wg, wp, ws = stability_margins(self.sys2);
        gm, pm, sm, wg, wp, ws = stability_margins(self.sys3);

    def test_phase_crossover_frequencies(self):
        omega, gain = phase_crossover_frequencies(self.sys2)
        np.testing.assert_array_almost_equal(omega, [1.73205,  0.])
        np.testing.assert_array_almost_equal(gain, [-0.5,  0.25])

        tf = TransferFunction([1],[1,1])
        omega, gain = phase_crossover_frequencies(tf)
        np.testing.assert_array_almost_equal(omega, [0.])
        np.testing.assert_array_almost_equal(gain, [1.])

        # testing MIMO, only (0,0) element is considered
        tf = TransferFunction([[[1],[2]],[[3],[4]]],
                              [[[1, 2, 3, 4],[1,1]],[[1,1],[1,1]]])
        omega, gain = phase_crossover_frequencies(tf)
        np.testing.assert_array_almost_equal(omega, [1.73205081,  0.])
        np.testing.assert_array_almost_equal(gain, [-0.5,  0.25])

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestMargin)

if __name__ == "__main__":
    unittest.main()
