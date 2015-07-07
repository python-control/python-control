#!/usr/bin/env python
#
# margin_test.py - test suit for stability margin commands
# RMM, 15 Jul 2011

import unittest
import numpy as np
from control.xferfcn import TransferFunction
from control.frdata import FRD
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
            [2.2716, 97.5941, 0.9565, 10.0053, 0.0850, 0.4973], 3) 

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

    def test_frd(self):
        f = np.array([0.005, 0.010, 0.020, 0.030, 0.040,
              0.050, 0.060, 0.070, 0.080, 0.090,
              0.100, 0.200, 0.300, 0.400, 0.500,
              0.750, 1.000, 1.250, 1.500, 1.750,
              2.000, 2.250, 2.500, 2.750, 3.000,
              3.250, 3.500, 3.750, 4.000, 4.250,
              4.500, 4.750, 5.000, 6.000, 7.000,
              8.000, 9.000, 10.000 ])
        gain = np.array([  0.0,   0.0,   0.0,   0.0,   0.0,
                        0.0,   0.0,   0.0,   0.0,   0.0,
                   0.0,   0.1,   0.2,   0.3,   0.5,
                   0.5,  -0.4,  -2.3,  -4.8,  -7.3,
                  -9.6, -11.7, -13.6, -15.3, -16.9,
                 -18.3, -19.6, -20.8, -22.0, -23.1,
                 -24.1, -25.0, -25.9, -29.1, -31.9,
                 -34.2, -36.2, -38.1 ])
        phase = np.array([    0,    -1,    -2,    -3,    -4,
                     -5,    -6,    -7,    -8,    -9,
                    -10,   -19,   -29,   -40,   -51,
                    -81,  -114,  -144,  -168,  -187,
                   -202,  -214,  -224,  -233,  -240,
                   -247,  -253,  -259,  -264,  -269,
                   -273,  -277,  -280,  -292,  -301,
                   -307,  -313,  -317 ])
        # calculate response as complex number
        resp = 10**(gain / 20) * np.exp(1j * phase / (180./np.pi))
        # frequency response data
        fresp = FRD(resp, f*2*np.pi, smooth=True)
        s=TransferFunction([1,0],[1])
        G=1./(s**2)
        K=1.
        C=K*(1+1.9*s)
        TFopen=fresp*C*G
        gm, pm, sm, wg, wp, ws = stability_margins(TFopen)
        print gm
        np.testing.assert_array_almost_equal(
            [pm], [44.55], 2)

    def test_nocross(self):
        # what happens when no gain/phase crossover?
        s = TransferFunction([1, 0], [1])
        h1 = 1/(1+s)
        h2 = 3*(10+s)/(2+s)
        h3 = 0.01*(10-s)/(2+s)/(1+s)
        gm, pm, sm, wg, wp, ws = stability_margins(h1)
        self.assertEqual(gm, None)
        self.assertEqual(wg, None)
        gm, pm, wm, wg, wp, ws = stability_margins(h2)
        self.assertEqual(pm, None)
        gm, pm, wm, wg, wp, ws = stability_margins(h3)
        self.assertEqual(pm, None)
        omega = np.logspace(-2,2, 100)
        out1b = stability_margins(FRD(h1, omega))
        out2b = stability_margins(FRD(h2, omega))
        out3b = stability_margins(FRD(h3, omega))
        

        
def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestMargin)

if __name__ == "__main__":
    unittest.main()
