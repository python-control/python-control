#!/usr/bin/env python

from statefbk import ctrb, obsv, place, lqr
import numpy as N
import unittest

class TestStatefbk(unittest.TestCase):
    def testCtrbSISO(self):
        A = N.matrix("1. 2.; 3. 4.")
        B = N.matrix("5.; 7.")
        Wctrue = N.matrix("5. 19.; 7. 43.")
        Wc = ctrb(A,B)
        N.testing.assert_array_almost_equal(Wc, Wctrue)

    def testCtrbMIMO(self):
        A = N.matrix("1. 2.; 3. 4.")
        B = N.matrix("5. 6.; 7. 8.")
        Wctrue = N.matrix("5. 6. 19. 22.; 7. 8. 43. 50.")
        Wc = ctrb(A,B)
        N.testing.assert_array_almost_equal(Wc, Wctrue)

    def testObsvSISO(self):
        A = N.matrix("1. 2.; 3. 4.")
        C = N.matrix("5. 7.")
        Wotrue = N.matrix("5. 7.; 26. 38.")
        Wo = obsv(A,C)
        N.testing.assert_array_almost_equal(Wo, Wotrue)

    def testObsvMIMO(self):
        A = N.matrix("1. 2.; 3. 4.")
        C = N.matrix("5. 6.; 7. 8.")
        Wotrue = N.matrix("5. 6.; 7. 8.; 23. 34.; 31. 46.")
        Wo = obsv(A,C)
        N.testing.assert_array_almost_equal(Wo, Wotrue)

if __name__ == '__main__':
    unittest.main()
