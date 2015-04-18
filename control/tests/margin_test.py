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
        s = TransferFunction([1, 0], [1])
        self.sys4 = (8.75*(4*s**2+0.4*s+1))/((100*s+1)*(s**2+0.22*s+1)) * \
                                      1./(s**2/(10.**2)+2*0.04*s/10.+1)

    def test_stability_margins(self):
        gm, pm, sm, wg, wp, ws = stability_margins(self.sys1);
        gm, pm, sm, wg, wp, ws = stability_margins(self.sys2);
        gm, pm, sm, wg, wp, ws = stability_margins(self.sys3);
        gm, pm, sm, wg, wp, ws = stability_margins(self.sys4);
        np.testing.assert_array_almost_equal(
            [gm, pm, sm, wg, wp, ws],
            [2.2716, 97.5941, 1.0454, 10.0053, 0.0850, 0.4973], 3) 

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

    def test_mag_phase_omega(self):
        # test for bug reported in gh-58
        sys = TransferFunction(15, [1, 6, 11, 6])
        out = stability_margins(sys)
        omega = np.logspace(-1,1,100)
        mag, phase, omega = sys.freqresp(omega)
        out2 = stability_margins((mag, phase*180/np.pi, omega))
        ind = [0,1,3,4]   # indices of gm, pm, wg, wp -- ignore sm
        marg1 = np.array(out)[ind]
        marg2 = np.array(out2)[ind]
        np.testing.assert_array_almost_equal(marg1, marg2, 4)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestMargin)

if __name__ == "__main__":
    unittest.main()
