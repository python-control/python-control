"""mateqn_test.py - test suite for matrix equation solvers

Copyright (c) 2020, All rights reserved.

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

import numpy as np
from numpy import array, zeros
from numpy.testing import assert_array_almost_equal, assert_array_less
import pytest
from scipy.linalg import eigvals, solve

import control as ct
from control.mateqn import lyap, dlyap, care, dare
from control.exception import ControlArgument, ControlDimension, slycot_check
from control.tests.conftest import slycotonly


class TestMatrixEquations:
    """These are tests for the matrix equation solvers in mateqn.py"""

    def test_lyap(self):
        A = array([[-1, 1], [-1, 0]])
        Q = array([[1, 0], [0, 1]])
        X = lyap(A, Q)
        # print("The solution obtained is ", X)
        assert_array_almost_equal(A @ X + X @ A.T + Q, zeros((2,2)))

        A = array([[1, 2], [-3, -4]])
        Q = array([[3, 1], [1, 1]])
        X = lyap(A,Q)
        # print("The solution obtained is ", X)
        assert_array_almost_equal(A @ X + X @ A.T + Q, zeros((2,2)))

        # Compare methods
        if slycot_check():
            X_scipy = lyap(A, Q, method='scipy')
            X_slycot = lyap(A, Q, method='slycot')
            assert_array_almost_equal(X_scipy, X_slycot)

    def test_lyap_sylvester(self):
        A = 5
        B = array([[4, 3], [4, 3]])
        C = array([2, 1])
        X = lyap(A, B, C)
        # print("The solution obtained is ", X)
        assert_array_almost_equal(A * X + X @ B + C, zeros((1,2)))

        A = array([[2, 1], [1, 2]])
        B = array([[1, 2], [0.5, 0.1]])
        C = array([[1, 0], [0, 1]])
        X = lyap(A, B, C)
        # print("The solution obtained is ", X)
        assert_array_almost_equal(A @ X + X @ B + C, zeros((2,2)))

        # Compare methods
        if slycot_check():
            X_scipy = lyap(A, B, C, method='scipy')
            X_slycot = lyap(A, B, C, method='slycot')
            assert_array_almost_equal(X_scipy, X_slycot)

    @slycotonly
    def test_lyap_g(self):
        A = array([[-1, 2], [-3, -4]])
        Q = array([[3, 1], [1, 1]])
        E = array([[1, 2], [2, 1]])
        X = lyap(A, Q, None, E)
        # print("The solution obtained is ", X)
        assert_array_almost_equal(A @ X @ E.T + E @ X @ A.T + Q,
                                  zeros((2,2)))

        # Make sure that trying to solve with SciPy generates an error
        with pytest.raises(ControlArgument, match="'scipy' not valid"):
            X = lyap(A, Q, None, E, method='scipy')

    def test_dlyap(self):
        A = array([[-0.6, 0],[-0.1, -0.4]])
        Q = array([[1,0],[0,1]])
        X = dlyap(A,Q)
        # print("The solution obtained is ", X)
        assert_array_almost_equal(A @ X @ A.T - X + Q, zeros((2,2)))

        A = array([[-0.6, 0],[-0.1, -0.4]])
        Q = array([[3, 1],[1, 1]])
        X = dlyap(A,Q)
        # print("The solution obtained is ", X)
        assert_array_almost_equal(A @ X @ A.T - X + Q, zeros((2,2)))

    @slycotonly
    def test_dlyap_g(self):
        A = array([[-0.6, 0],[-0.1, -0.4]])
        Q = array([[3, 1],[1, 1]])
        E = array([[1, 1],[2, 1]])
        X = dlyap(A, Q, None, E)
        # print("The solution obtained is ", X)
        assert_array_almost_equal(A @ X @ A.T - E @ X @ E.T + Q,
                                  zeros((2,2)))

        # Make sure that trying to solve with SciPy generates an error
        with pytest.raises(ControlArgument, match="'scipy' not valid"):
            X = dlyap(A, Q, None, E, method='scipy')

    @slycotonly
    def test_dlyap_sylvester(self):
        A = 5
        B = array([[4, 3], [4, 3]])
        C = array([2, 1])
        X = dlyap(A,B,C)
        # print("The solution obtained is ", X)
        assert_array_almost_equal(A * X @ B.T - X + C, zeros((1,2)))

        A = array([[2, 1], [1, 2]])
        B = array([[1, 2], [0.5, 0.1]])
        C = array([[1, 0], [0, 1]])
        X = dlyap(A, B, C)
        # print("The solution obtained is ", X)
        assert_array_almost_equal(A @ X @ B.T - X + C, zeros((2,2)))

        # Make sure that trying to solve with SciPy generates an error
        with pytest.raises(ControlArgument, match="'scipy' not valid"):
            X = dlyap(A, B, C, method='scipy')

    def test_care(self):
        A = array([[-2, -1],[-1, -1]])
        Q = array([[0, 0],[0, 1]])
        B = array([[1, 0],[0, 4]])

        X, L, G = care(A, B, Q)
        # print("The solution obtained is", X)
        M = A.T @ X + X @ A - X @ B @ B.T @ X + Q
        assert_array_almost_equal(M,
                                  zeros((2,2)))
        assert_array_almost_equal(B.T @ X, G)

        # Compare methods
        if slycot_check():
            X_scipy, L_scipy, G_scipy = care(A, B, Q, method='scipy')
            X_slycot, L_slycot, G_slycot = care(A, B, Q, method='slycot')
            assert_array_almost_equal(X_scipy, X_slycot)
            assert_array_almost_equal(np.sort(L_scipy), np.sort(L_slycot))
            assert_array_almost_equal(G_scipy, G_slycot)

    def test_care_g(self):
        A = array([[-2, -1],[-1, -1]])
        Q = array([[0, 0],[0, 1]])
        B = array([[1, 0],[0, 4]])
        R = array([[2, 0],[0, 1]])
        S = array([[0, 0],[0, 0]])
        E = array([[2, 1],[1, 2]])

        X,L,G = care(A,B,Q,R,S,E)
        # print("The solution obtained is", X)
        Gref = solve(R, B.T @ X @ E + S.T)
        assert_array_almost_equal(Gref, G)
        assert_array_almost_equal(
            A.T @ X @ E + E.T @ X @ A
            - (E.T @ X @ B + S) @ Gref + Q,
            zeros((2,2)))

        # Compare methods
        if slycot_check():
            X_scipy, L_scipy, G_scipy = care(
                A, B, Q, R, S, E, method='scipy')
            X_slycot, L_slycot, G_slycot = care(
                A, B, Q, R, S, E, method='slycot')
            assert_array_almost_equal(X_scipy, X_slycot)
            assert_array_almost_equal(np.sort(L_scipy), np.sort(L_slycot))
            assert_array_almost_equal(G_scipy, G_slycot)

    def test_care_g2(self):
        A = array([[-2, -1],[-1, -1]])
        Q = array([[0, 0],[0, 1]])
        B = array([[1],[0]])
        R = 1
        S = array([[1],[0]])
        E = array([[2, 1],[1, 2]])

        X,L,G = care(A,B,Q,R,S,E)
        # print("The solution obtained is", X)
        Gref = 1/R * (B.T @ X @ E + S.T)
        assert_array_almost_equal(
            A.T @ X @ E + E.T @ X @ A
            - (E.T @ X @ B + S) @ Gref + Q ,
            zeros((2,2)))
        assert_array_almost_equal(Gref , G)

        # Compare methods
        if slycot_check():
            X_scipy, L_scipy, G_scipy = care(
                A, B, Q, R, S, E, method='scipy')
            X_slycot, L_slycot, G_slycot = care(
                A, B, Q, R, S, E, method='slycot')
            assert_array_almost_equal(X_scipy, X_slycot)
            assert_array_almost_equal(L_scipy, L_slycot)
            assert_array_almost_equal(G_scipy, G_slycot)

    def test_dare(self):
        A = array([[-0.6, 0],[-0.1, -0.4]])
        Q = array([[2, 1],[1, 0]])
        B = array([[2, 1],[0, 1]])
        R = array([[1, 0],[0, 1]])

        X, L, G = dare(A, B, Q, R)
        # print("The solution obtained is", X)
        Gref = solve(B.T @ X @ B + R, B.T @ X @ A)
        assert_array_almost_equal(Gref, G)
        assert_array_almost_equal(
            X, A.T @ X @ A - A.T @ X @ B @ Gref + Q)
        # check for stable closed loop
        lam = eigvals(A - B @ G)
        assert_array_less(abs(lam), 1.0)

        A = array([[1, 0],[-1, 1]])
        Q = array([[0, 1],[1, 1]])
        B = array([[1],[0]])
        R = 2

        X, L, G = dare(A, B, Q, R)
        # print("The solution obtained is", X)
        AtXA = A.T @ X @ A
        AtXB = A.T @ X @ B
        BtXA = B.T @ X @ A
        BtXB = B.T @ X @ B
        assert_array_almost_equal(
            X, AtXA - AtXB @ solve(BtXB + R, BtXA) + Q)
        assert_array_almost_equal(BtXA / (BtXB + R), G)
        # check for stable closed loop
        lam = eigvals(A - B @ G)
        assert_array_less(abs(lam), 1.0)

    def test_dare_compare(self):
        A = np.array([[-0.6, 0], [-0.1, -0.4]])
        Q = np.array([[2, 1], [1, 0]])
        B = np.array([[2, 1], [0, 1]])
        R = np.array([[1, 0], [0, 1]])
        S = np.zeros((A.shape[0], B.shape[1]))
        E = np.eye(A.shape[0])

        # Solve via scipy
        X_scipy, L_scipy, G_scipy = dare(A, B, Q, R, method='scipy')

        # Solve via slycot
        if ct.slycot_check():
            X_slicot, L_slicot, G_slicot = dare(
                A, B, Q, R, S, E, method='scipy')
            np.testing.assert_almost_equal(X_scipy, X_slicot)
            np.testing.assert_almost_equal(L_scipy, L_slicot)
            np.testing.assert_almost_equal(G_scipy, G_slicot)

    def test_dare_g(self):
        A = array([[-0.6, 0],[-0.1, -0.4]])
        Q = array([[2, 1],[1, 3]])
        B = array([[1, 5],[2, 4]])
        R = array([[1, 0],[0, 1]])
        S = array([[1, 0],[2, 0]])
        E = array([[2, 1],[1, 2]])

        X, L, G = dare(A, B, Q, R, S, E)
        # print("The solution obtained is", X)
        Gref = solve(B.T @ X @ B + R, B.T @ X @ A + S.T)
        assert_array_almost_equal(Gref, G)
        assert_array_almost_equal(
            E.T @ X @ E,
            A.T @ X @ A - (A.T @ X @ B + S) @ Gref + Q)
        # check for stable closed loop
        lam = eigvals(A - B @ G, E)
        assert_array_less(abs(lam), 1.0)

    def test_dare_g2(self):
        A = array([[-0.6, 0], [-0.1, -0.4]])
        Q = array([[2, 1], [1, 3]])
        B = array([[1], [2]])
        R = 1
        S = array([[1], [2]])
        E = array([[2, 1], [1, 2]])

        X, L, G = dare(A, B, Q, R, S, E)
        # print("The solution obtained is", X)
        AtXA = A.T @ X @ A
        AtXB = A.T @ X @ B
        BtXA = B.T @ X @ A
        BtXB = B.T @ X @ B
        EtXE = E.T @ X @ E
        assert_array_almost_equal(
            EtXE, AtXA - (AtXB + S)  @ solve(BtXB + R, BtXA + S.T) + Q)
        assert_array_almost_equal((BtXA + S.T) / (BtXB + R), G)
        # check for stable closed loop
        lam = eigvals(A - B @ G, E)
        assert_array_less(abs(lam), 1.0)

    def test_raise(self):
        """ Test exception raise for invalid inputs """

        # correct shapes and forms
        A = array([[1, 0], [-1, -1]])
        Q = array([[2, 1], [1, 2]])
        C = array([[1, 0], [0, 1]])
        E = array([[2, 1], [1, 2]])

        # these fail
        Afq = array([[1, 0, 0], [-1, -1, 0]])
        Qfq = array([[2, 1, 0], [1, 2, 0]])
        Qfs = array([[2, 1], [-1, 2]])
        Cfd = array([[1, 0, 0], [0, 1, 0]])
        Efq = array([[2, 1, 0], [1, 2, 0]])

        for cdlyap in [lyap, dlyap]:
            with pytest.raises(ControlDimension):
                cdlyap(Afq, Q)
            with pytest.raises(ControlDimension):
                cdlyap(A, Qfq)
            with pytest.raises(ControlArgument):
                cdlyap(A, Qfs)
            with pytest.raises(ControlDimension):
                cdlyap(Afq, Q, C)
            with pytest.raises(ControlDimension):
                cdlyap(A, Qfq, C)
            with pytest.raises(ControlDimension):
                cdlyap(A, Q, Cfd)
            with pytest.raises(ControlDimension):
                cdlyap(A, Qfq, None, E)
            with pytest.raises(ControlDimension):
                cdlyap(A, Q, None, Efq)
            with pytest.raises(ControlArgument):
                cdlyap(A, Qfs, None, E)
            with pytest.raises(ControlArgument):
                cdlyap(A, Q, C, E)

        B = array([[1, 0], [0, 1]])
        Bf = array([[1, 0], [0, 1], [1, 1]])
        R = Q
        Rfs = Qfs
        Rfq = Qfq
        S = array([[0, 0], [0, 0]])
        Sf = array([[0, 0, 0], [0, 0, 0]])
        E = array([[2, 1], [1, 2]])
        Ef = array([[2, 1], [1, 2], [1, 2]])

        with pytest.raises(ControlDimension):
            care(Afq, B, Q)
        with pytest.raises(ControlDimension):
            care(A, B, Qfq)
        with pytest.raises(ControlDimension):
            care(A, Bf, Q)
        with pytest.raises(ControlDimension):
            care(1, B, 1)
        with pytest.raises(ControlArgument):
            care(A, B, Qfs)
        with pytest.raises(ControlArgument):
            dare(A, B, Q, Rfs)
        for cdare in [care, dare]:
            with pytest.raises(ControlDimension):
                cdare(Afq, B, Q, R, S, E)
            with pytest.raises(ControlDimension):
                cdare(A, B, Qfq, R, S, E)
            with pytest.raises(ControlDimension):
                cdare(A, Bf, Q, R, S, E)
            with pytest.raises(ControlDimension):
                cdare(A, B, Q, R, S, Ef)
            with pytest.raises(ControlDimension):
                cdare(A, B, Q, Rfq, S, E)
            with pytest.raises(ControlDimension):
                cdare(A, B, Q, R, Sf, E)
            with pytest.raises(ControlArgument):
                cdare(A, B, Qfs, R, S, E)
            with pytest.raises(ControlArgument):
                cdare(A, B, Q, Rfs, S, E)
