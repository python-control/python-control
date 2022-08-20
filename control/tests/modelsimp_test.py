"""modelsimp_array_test.py - test model reduction functions

RMM, 30 Mar 2011 (based on TestModelSimp from v0.4a)
"""

import numpy as np
import pytest


from control import StateSpace, forced_response, tf, rss, c2d
from control.exception import ControlMIMONotImplemented
from control.tests.conftest import slycotonly, matarrayin
from control.modelsimp import balred, hsvd, markov, modred


class TestModelsimp:
    """Test model reduction functions"""

    @slycotonly
    def testHSVD(self, matarrayout, matarrayin):
        A = matarrayin([[1., -2.], [3., -4.]])
        B = matarrayin([[5.], [7.]])
        C = matarrayin([[6., 8.]])
        D = matarrayin([[9.]])
        sys = StateSpace(A, B, C, D)
        hsv = hsvd(sys)
        hsvtrue = np.array([24.42686, 0.5731395])  # from MATLAB
        np.testing.assert_array_almost_equal(hsv, hsvtrue)

        # test for correct return type: ALWAYS return ndarray, even when
        # use_numpy_matrix(True) was used
        assert isinstance(hsv, np.ndarray)
        assert not isinstance(hsv, np.matrix)

    def testMarkovSignature(self, matarrayout, matarrayin):
        U = matarrayin([[1., 1., 1., 1., 1.]])
        Y = U
        m = 3
        H = markov(Y, U, m, transpose=False)
        Htrue = np.array([[1., 0., 0.]])
        np.testing.assert_array_almost_equal(H, Htrue)

        # Make sure that transposed data also works
        H = markov(np.transpose(Y), np.transpose(U), m, transpose=True)
        np.testing.assert_array_almost_equal(H, np.transpose(Htrue))

        # Generate Markov parameters without any arguments
        H = markov(Y, U, m)
        np.testing.assert_array_almost_equal(H, Htrue)

        # Test example from docstring
        T = np.linspace(0, 10, 100)
        U = np.ones((1, 100))
        T, Y = forced_response(tf([1], [1, 0.5], True), T, U)
        H = markov(Y, U, 3, transpose=False)

        # Test example from issue #395
        inp = np.array([1, 2])
        outp = np.array([2, 4])
        mrk = markov(outp, inp, 1, transpose=False)

        # Make sure MIMO generates an error
        U = np.ones((2, 100))   # 2 inputs (Y unchanged, with 1 output)
        with pytest.raises(ControlMIMONotImplemented):
            markov(Y, U, m)

    # Make sure markov() returns the right answer
    @pytest.mark.parametrize("k, m, n",
                             [(2, 2, 2),
                              (2, 5, 5),
                              (5, 2, 2),
                              (5, 5, 5),
                              (5, 10, 10)])
    def testMarkovResults(self, k, m, n):
        #
        # Test over a range of parameters
        #
        # k = order of the system
        # m = number of Markov parameters
        # n = size of the data vector
        #
        # Values *should* match exactly for n = m, otherewise you get a
        # close match but errors due to the assumption that C A^k B =
        # 0 for k > m-2 (see modelsimp.py).
        #

        # Generate stable continuous time system
        Hc = rss(k, 1, 1)

        # Choose sampling time based on fastest time constant / 10
        w, _ = np.linalg.eig(Hc.A)
        Ts = np.min(-np.real(w)) / 10.

        # Convert to a discrete time system via sampling
        Hd = c2d(Hc, Ts, 'zoh')

        # Compute the Markov parameters from state space
        Mtrue = np.hstack([Hd.D] + [
            Hd.C @ np.linalg.matrix_power(Hd.A, i) @ Hd.B
            for i in range(m-1)])

        # Generate input/output data
        T = np.array(range(n)) * Ts
        U = np.cos(T) + np.sin(T/np.pi)
        _, Y = forced_response(Hd, T, U, squeeze=True)
        Mcomp = markov(Y, U, m)

        # Compare to results from markov()
        # experimentally determined probability to get non matching results
        # with rtot=1e-6 and atol=1e-8 due to numerical errors
        # for k=5, m=n=10: 0.015 %
        np.testing.assert_allclose(Mtrue, Mcomp, rtol=1e-6, atol=1e-8)

    def testModredMatchDC(self, matarrayin):
        #balanced realization computed in matlab for the transfer function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = matarrayin(
            [[-1.958, -1.194, 1.824, -1.464],
             [-1.194, -0.8344, 2.563, -1.351],
             [-1.824, -2.563, -1.124, 2.704],
             [-1.464, -1.351, -2.704, -11.08]])
        B = matarrayin([[-0.9057], [-0.4068], [-0.3263], [-0.3474]])
        C = matarrayin([[-0.9057, -0.4068, 0.3263, -0.3474]])
        D = matarrayin([[0.]])
        sys = StateSpace(A, B, C, D)
        rsys = modred(sys,[2, 3],'matchdc')
        Artrue = np.array([[-4.431, -4.552], [-4.552, -5.361]])
        Brtrue = np.array([[-1.362], [-1.031]])
        Crtrue = np.array([[-1.362, -1.031]])
        Drtrue = np.array([[-0.08384]])
        np.testing.assert_array_almost_equal(rsys.A, Artrue, decimal=3)
        np.testing.assert_array_almost_equal(rsys.B, Brtrue, decimal=3)
        np.testing.assert_array_almost_equal(rsys.C, Crtrue, decimal=3)
        np.testing.assert_array_almost_equal(rsys.D, Drtrue, decimal=2)

    def testModredUnstable(self, matarrayin):
        """Check if an error is thrown when an unstable system is given"""
        A = matarrayin(
            [[4.5418, 3.3999, 5.0342, 4.3808],
             [0.3890, 0.3599, 0.4195, 0.1760],
             [-4.2117, -3.2395, -4.6760, -4.2180],
             [0.0052, 0.0429, 0.0155, 0.2743]])
        B = matarrayin([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        C = matarrayin([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        D = matarrayin([[0.0, 0.0], [0.0, 0.0]])
        sys = StateSpace(A, B, C, D)
        np.testing.assert_raises(ValueError, modred, sys, [2, 3])

    def testModredTruncate(self, matarrayin):
        #balanced realization computed in matlab for the transfer function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = matarrayin(
            [[-1.958, -1.194, 1.824, -1.464],
             [-1.194, -0.8344, 2.563, -1.351],
             [-1.824, -2.563, -1.124, 2.704],
             [-1.464, -1.351, -2.704, -11.08]])
        B = matarrayin([[-0.9057], [-0.4068], [-0.3263], [-0.3474]])
        C = matarrayin([[-0.9057, -0.4068, 0.3263, -0.3474]])
        D = matarrayin([[0.]])
        sys = StateSpace(A, B, C, D)
        rsys = modred(sys,[2, 3],'truncate')
        Artrue = np.array([[-1.958, -1.194], [-1.194, -0.8344]])
        Brtrue = np.array([[-0.9057], [-0.4068]])
        Crtrue = np.array([[-0.9057, -0.4068]])
        Drtrue = np.array([[0.]])
        np.testing.assert_array_almost_equal(rsys.A, Artrue)
        np.testing.assert_array_almost_equal(rsys.B, Brtrue)
        np.testing.assert_array_almost_equal(rsys.C, Crtrue)
        np.testing.assert_array_almost_equal(rsys.D, Drtrue)


    @slycotonly
    def testBalredTruncate(self, matarrayin):
        # controlable canonical realization computed in matlab for the transfer
        # function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = matarrayin(
            [[-15., -7.5, -6.25, -1.875],
             [8., 0., 0., 0.],
             [0., 4., 0., 0.],
             [0., 0., 1., 0.]])
        B = matarrayin([[2.], [0.], [0.], [0.]])
        C = matarrayin([[0.5, 0.6875, 0.7031, 0.5]])
        D = matarrayin([[0.]])
        
        sys = StateSpace(A, B, C, D)
        orders = 2
        rsys = balred(sys, orders, method='truncate')
        Ar, Br, Cr, Dr = rsys.A, rsys.B, rsys.C, rsys.D

        # Result from MATLAB
        Artrue = np.array([[-1.958, -1.194], [-1.194, -0.8344]])
        Brtrue = np.array([[0.9057], [0.4068]])
        Crtrue = np.array([[0.9057, 0.4068]])
        Drtrue = np.array([[0.]])

        # Look for possible changes in state in slycot
        T1 = np.array([[1, 0], [0, -1]])
        T2 = np.array([[-1, 0], [0, 1]])
        T3 = np.array([[0, 1], [1, 0]])
        for T in (T1, T2, T3):
            if np.allclose(T @ Ar @ T, Artrue, atol=1e-2, rtol=1e-2):
                # Apply a similarity transformation
                Ar, Br, Cr = T @ Ar @ T, T @ Br, Cr @ T
                break
            
        # Make sure we got the correct answer
        np.testing.assert_array_almost_equal(Ar, Artrue, decimal=2)
        np.testing.assert_array_almost_equal(Br, Brtrue, decimal=4)
        np.testing.assert_array_almost_equal(Cr, Crtrue, decimal=4)
        np.testing.assert_array_almost_equal(Dr, Drtrue, decimal=4)

    @slycotonly
    def testBalredMatchDC(self, matarrayin):
        # controlable canonical realization computed in matlab for the transfer
        # function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = matarrayin(
            [[-15., -7.5, -6.25, -1.875],
             [8., 0., 0., 0.],
             [0., 4., 0., 0.],
             [0., 0., 1., 0.]])
        B = matarrayin([[2.], [0.], [0.], [0.]])
        C = matarrayin([[0.5, 0.6875, 0.7031, 0.5]])
        D = matarrayin([[0.]])
        
        sys = StateSpace(A, B, C, D)
        orders = 2
        rsys = balred(sys,orders,method='matchdc')
        Ar, Br, Cr, Dr = rsys.A, rsys.B, rsys.C, rsys.D
        
        # Result from MATLAB
        Artrue = np.array(
            [[-4.43094773, -4.55232904],
             [-4.55232904, -5.36195206]])
        Brtrue = np.array([[1.36235673], [1.03114388]])
        Crtrue = np.array([[1.36235673, 1.03114388]])
        Drtrue = np.array([[-0.08383902]])
        
        # Look for possible changes in state in slycot
        T1 = np.array([[1, 0], [0, -1]])
        T2 = np.array([[-1, 0], [0, 1]])
        T3 = np.array([[0, 1], [1, 0]])
        for T in (T1, T2, T3):
            if np.allclose(T @ Ar @ T, Artrue, atol=1e-2, rtol=1e-2):
                # Apply a similarity transformation
                Ar, Br, Cr = T @ Ar @ T, T @ Br, Cr @ T
                break
            
        # Make sure we got the correct answer
        np.testing.assert_array_almost_equal(Ar, Artrue, decimal=2)
        np.testing.assert_array_almost_equal(Br, Brtrue, decimal=4)
        np.testing.assert_array_almost_equal(Cr, Crtrue, decimal=4)
        np.testing.assert_array_almost_equal(Dr, Drtrue, decimal=4)
