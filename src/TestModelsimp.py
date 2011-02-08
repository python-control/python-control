#!/usr/bin/env python

from modelsimp import *
from matlab import *
import numpy as np
import unittest

class TestModelsimp(unittest.TestCase):
    def testHSVD(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5.; 7.")
        C = np.matrix("6. 8.")
        D = np.matrix("9.")
        sys = ss(A,B,C,D)
        hsv = hsvd(sys)
        hsvtrue = np.matrix("24.42686 0.5731395")
        np.testing.assert_array_almost_equal(hsv, hsvtrue)

    def testMarkov(self):
        U = np.matrix("1.; 1.; 1.; 1.; 1.")
        Y = U
        M = 3
        H = markov(Y,U,M)
        Htrue = np.matrix("1.; 0.; 0.")
        np.testing.assert_array_almost_equal( H, Htrue )

if __name__ == '__main__':
    unittest.main()
