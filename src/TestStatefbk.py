#!/usr/bin/env python

from statefbk import ctrb, obsv, place, lqr, gram
from matlab import *
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
    
    def testCtrbObsvDuality(self):
        A = N.matrix("1.2 -2.3; 3.4 -4.5")
        B = N.matrix("5.8 6.9; 8. 9.1")
        Wc = ctrb(A,B);
        A = N.transpose(A)
        C = N.transpose(B)
        Wo = N.transpose(obsv(A,C));
        N.testing.assert_array_almost_equal(Wc,Wo)

    def testGramWc(self):
        A = N.matrix("1. -2.; 3. -4.")
        B = N.matrix("5. 6.; 7. 8.")
        C = N.matrix("4. 5.; 6. 7.")
        D = N.matrix("13. 14.; 15. 16.")
    #    sys = ss(A, B, C, D)
        sys = 1.
        Wctrue = N.matrix("18.5 24.5; 24.5 32.5")
        Wc = gram(sys,'c')
        N.testing.assert_array_almost_equal(Wc, Wctrue)

    def testGramWo(self):
        A = N.matrix("1. -2.; 3. -4.")
        B = N.matrix("5. 6.; 7. 8.")
        C = N.matrix("4. 5.; 6. 7.")
        D = N.matrix("13. 14.; 15. 16.")
        sys = ss(A, B, C, D)
        sys = 1.
        Wotrue = N.matrix("257.5 -94.5; -94.5 56.5")
        Wo = gram(sys,'o')
        N.testing.assert_array_almost_equal(Wo, Wotrue)

if __name__ == '__main__':
    unittest.main()
