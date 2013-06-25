#!/usr/bin/env python
#
# minreal_test.py - test state space class
# Rvp, 13 Jun 2013

import unittest
import numpy as np
from scipy.linalg import eigvals
import control.matlab as matlab
from control.statesp import StateSpace, _convertToStateSpace
from control.xferfcn import TransferFunction
from itertools import permutations 

class TestMinreal(unittest.TestCase):
    """Tests for the StateSpace class."""

    def setUp(self):
        np.random.seed(5)
        # depending on the seed and minreal performance, a number of
        # reductions is produced. If random gen or minreal change, this
        # will be likely to fail
        self.nreductions = 0

    def assert_numden_almost_equal(self, n1, n2, d1, d2):
        n1[np.abs(n1) < 1e-10] = 0.
        n1 = np.trim_zeros(n1)
        d1[np.abs(d1) < 1e-10] = 0.
        d1 = np.trim_zeros(d1)
        n2[np.abs(n2) < 1e-10] = 0.
        n2 = np.trim_zeros(n2)
        d2[np.abs(d2) < 1e-10] = 0.
        d2 = np.trim_zeros(d2)
        np.testing.assert_array_almost_equal(n1, n2)
        np.testing.assert_array_almost_equal(d2, d2)
            
        
    def testMinrealBrute(self):
        for n, m, p in permutations(range(1,6), 3):
            s = matlab.rss(n, p, m)
            sr = s.minreal()
            if s.states > sr.states:
                self.nreductions += 1
            else:
                np.testing.assert_array_almost_equal(
                    np.sort(eigvals(s.A)), np.sort(eigvals(sr.A)))
                for i in range(m):
                    for j in range(p):
                        ht1 = matlab.tf(
                            matlab.ss(s.A, s.B[:,i], s.C[j,:], s.D[j,i])) 
                        ht2 = matlab.tf(
                            matlab.ss(sr.A, sr.B[:,i], sr.C[j,:], sr.D[j,i]))
                        try:
                            self.assert_numden_almost_equal(
                                ht1.num[0][0], ht2.num[0][0], 
                                ht1.den[0][0], ht2.den[0][0])
                        except Exception as e:
                            # for larger systems, the tf minreal's 
                            # the original rss, but not the balanced one
                            if n < 6:
                                raise e
                        
        self.assertEqual(self.nreductions, 2)

    def testMinrealSS(self):
        """Test a minreal model reduction"""
        #A = [-2, 0.5, 0; 0.5, -0.3, 0; 0, 0, -0.1]
        A = [[-2, 0.5, 0], [0.5, -0.3, 0], [0, 0, -0.1]]
        #B = [0.3, -1.3; 0.1, 0; 1, 0]
        B = [[0.3, -1.3], [0.1, 0.], [1.0, 0.0]]
        #C = [0, 0.1, 0; -0.3, -0.2, 0]
        C = [[0., 0.1, 0.0], [-0.3, -0.2, 0.0]]
        #D = [0 -0.8; -0.3 0]
        D = [[0., -0.8], [-0.3, 0.]]
        # sys = ss(A, B, C, D)
        
        sys = StateSpace(A, B, C, D)
        sysr = sys.minreal()
        self.assertEqual(sysr.states, 2)
        self.assertEqual(sysr.inputs, sys.inputs)
        self.assertEqual(sysr.outputs, sys.outputs)
        np.testing.assert_array_almost_equal(
            eigvals(sysr.A), [-2.136154, -0.1638459])

    def testMinrealtf(self):
        """Try the minreal function, and also test easy entry by creation
        of a Laplace variable s"""
        s = TransferFunction([1, 0], [1])
        h = (s+1)*(s+2.00000000001)/(s+2)/(s**2+s+1)
        hm = h.minreal()
        hr = (s+1)/(s**2+s+1)
        np.testing.assert_array_almost_equal(hm.num[0][0], hr.num[0][0])
        np.testing.assert_array_almost_equal(hm.den[0][0], hr.den[0][0])

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestMinreal)
      

if __name__ == "__main__":
    unittest.main()
