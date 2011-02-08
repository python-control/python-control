#!/usr/bin/env python

from matlab import *
import numpy as np
import unittest

class TestMatlab(unittest.TestCase):
    def testStep(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5.; 7.")
        C = np.matrix("6. 8.")
        D = np.matrix("9.")
        sys = ss(A,B,C,D)
        t = np.linspace(0, 1, 10)
        t, yout = step(sys, t)
        youttrue = np.matrix("9. 17.6457 24.7072 30.4855 35.2234 39.1165 42.3227 44.9694 47.1599 48.9776") 
        np.testing.assert_array_almost_equal(yout, youttrue,decimal=4)

    def testImpulse(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5.; 7.")
        C = np.matrix("6. 8.")
        D = np.matrix("9.")
        sys = ss(A,B,C,D)
        t = np.linspace(0, 1, 10)
        t, yout = impulse(sys, t)
        youttrue = np.matrix("86. 70.1808 57.3753 46.9975 38.5766 31.7344 26.1668 21.6292 17.9245 14.8945") 
        np.testing.assert_array_almost_equal(yout, youttrue,decimal=4)

    def testInitial(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5.; 7.")
        C = np.matrix("6. 8.")
        D = np.matrix("9.")
        sys = ss(A,B,C,D)
        t = np.linspace(0, 1, 10)
        x0 = np.matrix(".5; 1.")
        t, yout = initial(sys, t, x0)
        youttrue = np.matrix("11. 8.1494 5.9361 4.2258 2.9118 1.9092 1.1508 0.5833 0.1645 -0.1391") 
        np.testing.assert_array_almost_equal(yout, youttrue,decimal=4)


if __name__ == '__main__':
    unittest.main()
