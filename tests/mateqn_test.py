#!/usr/bin/env python
#
# mateqn_test.py - test wuit for matrix equation solvers
#
#! Currently uses numpy.testing framework; will dump you out of unittest
#! if an error occurs.  Should figure out the right way to fix this.

""" Test cases for lyap, dlyap, care and dare functions in the file
pyctrl_lin_alg.py. """ 

"""Copyright (c) 2011, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the project author nor the names of its 
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

Author: Bjorn Olofsson
"""

import unittest
from numpy import array
from numpy.testing import assert_array_almost_equal
from numpy.linalg import inv
from scipy import zeros,dot
from control.mateqn import lyap,dlyap,care,dare
from control.exception import slycot_check

@unittest.skipIf(not slycot_check(), "slycot not installed")
class TestMatrixEquations(unittest.TestCase):
    """These are tests for the matrix equation solvers in mateqn.py"""

    def test_lyap(self):
        A = array([[-1, 1],[-1, 0]])
        Q = array([[1,0],[0,1]])
        X = lyap(A,Q)
        # print "The solution obtained is ", X
        assert_array_almost_equal(dot(A,X)+dot(X,A.T)+Q,zeros((2,2)))

        A = array([[1, 2],[-3, -4]])  
        Q = array([[3, 1],[1, 1]])
        X = lyap(A,Q)
        # print "The solution obtained is ", X
        assert_array_almost_equal(dot(A,X)+dot(X,A.T)+Q,zeros((2,2)))

    def test_lyap_sylvester(self):
        A = 5
        B = array([[4, 3], [4, 3]])
        C = array([2, 1])
        X = lyap(A,B,C)
        # print "The solution obtained is ", X
        assert_array_almost_equal(dot(A,X)+dot(X,B)+C,zeros((1,2)))

        A = array([[2,1],[1,2]])
        B = array([[1,2],[0.5,0.1]])
        C = array([[1,0],[0,1]])
        X = lyap(A,B,C)
        # print "The solution obtained is ", X
        assert_array_almost_equal(dot(A,X)+dot(X,B)+C,zeros((2,2)))

    def test_lyap_g(self):
        A = array([[-1, 2],[-3, -4]])
        Q = array([[3, 1],[1, 1]])
        E = array([[1,2],[2,1]])
        X = lyap(A,Q,None,E)
        # print "The solution obtained is ", X
        assert_array_almost_equal(dot(A,dot(X,E.T)) + dot(E,dot(X,A.T)) + Q, \
                                      zeros((2,2)))

    def test_dlyap(self):
        A = array([[-0.6, 0],[-0.1, -0.4]])
        Q = array([[1,0],[0,1]])
        X = dlyap(A,Q)
        # print "The solution obtained is ", X
        assert_array_almost_equal(dot(A,dot(X,A.T))-X+Q,zeros((2,2)))

        A = array([[-0.6, 0],[-0.1, -0.4]])
        Q = array([[3, 1],[1, 1]])
        X = dlyap(A,Q)
        # print "The solution obtained is ", X
        assert_array_almost_equal(dot(A,dot(X,A.T))-X+Q,zeros((2,2)))

    def test_dlyap_g(self):
        A = array([[-0.6, 0],[-0.1, -0.4]])
        Q = array([[3, 1],[1, 1]])
        E = array([[1, 1],[2, 1]])
        X = dlyap(A,Q,None,E)
        # print "The solution obtained is ", X
        assert_array_almost_equal(dot(A,dot(X,A.T))-dot(E,dot(X,E.T))+Q, \
                                      zeros((2,2)))

    def test_dlyap_sylvester(self):
        A = 5
        B = array([[4, 3], [4, 3]])
        C = array([2, 1])
        X = dlyap(A,B,C)
        # print "The solution obtained is ", X
        assert_array_almost_equal(dot(A,dot(X,B.T))-X+C,zeros((1,2)))

        A = array([[2,1],[1,2]])
        B = array([[1,2],[0.5,0.1]])
        C = array([[1,0],[0,1]])
        X = dlyap(A,B,C)
        # print "The solution obtained is ", X
        assert_array_almost_equal(dot(A,dot(X,B.T))-X+C,zeros((2,2)))

    def test_care(self):
        A = array([[-2, -1],[-1, -1]])
        Q = array([[0, 0],[0, 1]])
        B = array([[1, 0],[0, 4]])

        X,L,G = care(A,B,Q)
        # print "The solution obtained is", X
        assert_array_almost_equal(dot(A.T,X) + dot(X,A) - \
            dot(X,dot(B,dot(B.T,X))) + Q , zeros((2,2)))
        assert_array_almost_equal(dot(B.T,X) , G)

    def test_care_g(self):
        A = array([[-2, -1],[-1, -1]])
        Q = array([[0, 0],[0, 1]])
        B = array([[1, 0],[0, 4]])
        R = array([[2, 0],[0, 1]])
        S = array([[0, 0],[0, 0]])
        E = array([[2, 1],[1, 2]])

        X,L,G = care(A,B,Q,R,S,E)
        # print "The solution obtained is", X
        assert_array_almost_equal(dot(A.T,dot(X,E)) + dot(E.T,dot(X,A)) - \
            dot(dot(dot(E.T,dot(X,B))+S,inv(R) ) ,
            dot(B.T,dot(X,E))+S.T ) + Q , \
            zeros((2,2)))
        assert_array_almost_equal(dot( inv(R) , dot(B.T,dot(X,E)) + S.T) , G)

        A = array([[-2, -1],[-1, -1]])
        Q = array([[0, 0],[0, 1]])
        B = array([[1],[0]])
        R = 1
        S = array([[1],[0]])
        E = array([[2, 1],[1, 2]])

        X,L,G = care(A,B,Q,R,S,E)
        # print "The solution obtained is", X
        assert_array_almost_equal(dot(A.T,dot(X,E)) + dot(E.T,dot(X,A)) - \
            dot( dot( dot(E.T,dot(X,B))+S,1/R ) , dot(B.T,dot(X,E))+S.T ) \
            + Q , zeros((2,2)))
        assert_array_almost_equal(dot( 1/R , dot(B.T,dot(X,E)) + S.T) , G)

    def test_dare(self):
        A = array([[-0.6, 0],[-0.1, -0.4]])
        Q = array([[2, 1],[1, 0]])
        B = array([[2, 1],[0, 1]])
        R = array([[1, 0],[0, 1]])

        X,L,G = dare(A,B,Q,R)
        # print "The solution obtained is", X
        assert_array_almost_equal(dot(A.T,dot(X,A))-X-dot(dot(dot(A.T,dot(X,B)) , \
            inv(dot(B.T,dot(X,B))+R)) , dot(B.T,dot(X,A))) + Q , zeros((2,2)) )
        assert_array_almost_equal( dot( inv( dot(B.T,dot(X,B)) + R) , \
            dot(B.T,dot(X,A)) ) , G)

        A = array([[1, 0],[-1, 1]])
        Q = array([[0, 1],[1, 1]])
        B = array([[1],[0]])
        R = 2

        X,L,G = dare(A,B,Q,R)
        # print "The solution obtained is", X
        assert_array_almost_equal(dot(A.T,dot(X,A))-X-dot(dot(dot(A.T,dot(X,B)) , \
            inv(dot(B.T,dot(X,B))+R)) , dot(B.T,dot(X,A))) + Q , zeros((2,2)) )
        assert_array_almost_equal( dot( 1 / ( dot(B.T,dot(X,B)) + R) , \
            dot(B.T,dot(X,A)) ) , G)

    def test_dare_g(self):
        A = array([[-0.6, 0],[-0.1, -0.4]])
        Q = array([[2, 1],[1, 3]])
        B = array([[1, 5],[2, 4]])
        R = array([[1, 0],[0, 1]])
        S = array([[1, 0],[2, 0]])
        E = array([[2, 1],[1, 2]])

        X,L,G = dare(A,B,Q,R,S,E)
        # print "The solution obtained is", X
        assert_array_almost_equal(dot(A.T,dot(X,A))-dot(E.T,dot(X,E)) - \
            dot( dot(A.T,dot(X,B))+S , dot( inv(dot(B.T,dot(X,B)) + R) ,
            dot(B.T,dot(X,A))+S.T)) + Q , zeros((2,2)) )
        assert_array_almost_equal( dot( inv( dot(B.T,dot(X,B)) + R) , \
            dot(B.T,dot(X,A)) + S.T ) , G)

        A = array([[-0.6, 0],[-0.1, -0.4]])
        Q = array([[2, 1],[1, 3]])
        B = array([[1],[2]])
        R = 1
        S = array([[1],[2]])
        E = array([[2, 1],[1, 2]])

        X,L,G = dare(A,B,Q,R,S,E)
        # print "The solution obtained is", X
        assert_array_almost_equal(dot(A.T,dot(X,A))-dot(E.T,dot(X,E)) - \
            dot( dot(A.T,dot(X,B))+S , dot( inv(dot(B.T,dot(X,B)) + R) ,
            dot(B.T,dot(X,A))+S.T)) + Q , zeros((2,2)) )
        assert_array_almost_equal( dot( 1 / ( dot(B.T,dot(X,B)) + R) , \
            dot(B.T,dot(X,A)) + S.T ) , G)

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestMatrixEquations)

if __name__ == "__main__":
    unittest.main()
