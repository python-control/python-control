#!/usr/bin/env python

from statefbk import ctrb, obsv, place, lqr, gram
from matlab import *
import numpy as np
import unittest

class TestStatefbk(unittest.TestCase):
    def testCtrbSISO(self):
        A = np.matrix("1. 2.; 3. 4.")
        B = np.matrix("5.; 7.")
        Wctrue = np.matrix("5. 19.; 7. 43.")
        Wc = ctrb(A,B)
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    def testCtrbMIMO(self):
        A = np.matrix("1. 2.; 3. 4.")
        B = np.matrix("5. 6.; 7. 8.")
        Wctrue = np.matrix("5. 6. 19. 22.; 7. 8. 43. 50.")
        Wc = ctrb(A,B)
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    def testObsvSISO(self):
        A = np.matrix("1. 2.; 3. 4.")
        C = np.matrix("5. 7.")
        Wotrue = np.matrix("5. 7.; 26. 38.")
        Wo = obsv(A,C)
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    def testObsvMIMO(self):
        A = np.matrix("1. 2.; 3. 4.")
        C = np.matrix("5. 6.; 7. 8.")
        Wotrue = np.matrix("5. 6.; 7. 8.; 23. 34.; 31. 46.")
        Wo = obsv(A,C)
        np.testing.assert_array_almost_equal(Wo, Wotrue)
    
    def testCtrbObsvDuality(self):
        A = np.matrix("1.2 -2.3; 3.4 -4.5")
        B = np.matrix("5.8 6.9; 8. 9.1")
        Wc = ctrb(A,B);
        A = np.transpose(A)
        C = np.transpose(B)
        Wo = np.transpose(obsv(A,C));
        np.testing.assert_array_almost_equal(Wc,Wo)

    def testGramWc(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5. 6.; 7. 8.")
        C = np.matrix("4. 5.; 6. 7.")
        D = np.matrix("13. 14.; 15. 16.")
        sys = ss(A, B, C, D)
        Wctrue = np.matrix("18.5 24.5; 24.5 32.5")
        Wc = gram(sys,'c')
        np.testing.assert_array_almost_equal(Wc, Wctrue)

    def testGramWo(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5. 6.; 7. 8.")
        C = np.matrix("4. 5.; 6. 7.")
        D = np.matrix("13. 14.; 15. 16.")
        sys = ss(A, B, C, D)
        Wotrue = np.matrix("257.5 -94.5; -94.5 56.5")
        Wo = gram(sys,'o')
        np.testing.assert_array_almost_equal(Wo, Wotrue)

    def testGramWo2(self):
        A = np.matrix("1. -2.; 3. -4.")
        B = np.matrix("5.; 7.")
        C = np.matrix("6. 8.")
        D = np.matrix("9.")
        sys = ss(A,B,C,D)
        Wotrue = np.matrix("198. -72.; -72. 44.")
        Wo = gram(sys,'o')
        np.testing.assert_array_almost_equal(Wo, Wotrue)


if __name__ == '__main__':
    unittest.main()
