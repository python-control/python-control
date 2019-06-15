#!/usr/bin/env python
#
# modelsimp_test.py - test model reduction functions
# RMM, 30 Mar 2011 (based on TestModelSimp from v0.4a)

import unittest
import numpy as np
import warnings
import control
from control.modelsimp import *
from control.matlab import *
from control.exception import slycot_check

class TestModelsimp(unittest.TestCase):
    def setUp(self):
        # Use array instead of matrix (and save old value to restore at end)
        control.use_numpy_matrix(False)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testHSVD(self):
        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5.], [7.]])
        C = np.array([[6., 8.]])
        D = np.array([[9.]])
        sys = ss(A,B,C,D)
        hsv = hsvd(sys)
        hsvtrue = np.array([24.42686, 0.5731395]) # from MATLAB
        np.testing.assert_array_almost_equal(hsv, hsvtrue)

        # Make sure default type values are correct
        self.assertTrue(isinstance(hsv, np.ndarray))
        self.assertFalse(isinstance(hsv, np.matrix))

        # Check that using numpy.matrix does *not* affect answer
        with warnings.catch_warnings(record=True) as w:
            control.use_numpy_matrix(True)
            self.assertTrue(issubclass(w[-1].category, UserWarning))

            # Redefine the system (using np.matrix for storage)
            sys = ss(A, B, C, D)

            # Compute the Hankel singular value decomposition
            hsv = hsvd(sys)

            # Make sure that return type is correct
            self.assertTrue(isinstance(hsv, np.ndarray))
            self.assertFalse(isinstance(hsv, np.matrix))

            # Go back to using the normal np.array representation
            control.use_numpy_matrix(False)

    def testMarkov(self):
        U = np.array([[1.], [1.], [1.], [1.], [1.]])
        Y = U
        M = 3
        H = markov(Y,U,M)
        Htrue = np.array([[1.], [0.], [0.]])
        np.testing.assert_array_almost_equal( H, Htrue )

    def testModredMatchDC(self):
        #balanced realization computed in matlab for the transfer function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = np.array(
            [[-1.958, -1.194, 1.824, -1.464],
             [-1.194, -0.8344, 2.563, -1.351],
             [-1.824, -2.563, -1.124, 2.704],
             [-1.464, -1.351, -2.704, -11.08]])
        B = np.array([[-0.9057], [-0.4068], [-0.3263], [-0.3474]])
        C = np.array([[-0.9057, -0.4068, 0.3263, -0.3474]])
        D = np.array([[0.]])
        sys = ss(A,B,C,D)
        rsys = modred(sys,[2, 3],'matchdc')
        Artrue = np.array([[-4.431, -4.552], [-4.552, -5.361]])
        Brtrue = np.array([[-1.362], [-1.031]])
        Crtrue = np.array([[-1.362, -1.031]])
        Drtrue = np.array([[-0.08384]])
        np.testing.assert_array_almost_equal(rsys.A, Artrue,decimal=3)
        np.testing.assert_array_almost_equal(rsys.B, Brtrue,decimal=3)
        np.testing.assert_array_almost_equal(rsys.C, Crtrue,decimal=3)
        np.testing.assert_array_almost_equal(rsys.D, Drtrue,decimal=2)

    def testModredUnstable(self):
        # Check if an error is thrown when an unstable system is given
        A = np.array(
            [[4.5418, 3.3999, 5.0342, 4.3808],
             [0.3890, 0.3599, 0.4195, 0.1760],
             [-4.2117, -3.2395, -4.6760, -4.2180],
             [0.0052, 0.0429, 0.0155, 0.2743]])
        B = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        C = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        D = np.array([[0.0, 0.0], [0.0, 0.0]])
        sys = ss(A,B,C,D)
        np.testing.assert_raises(ValueError, modred, sys, [2, 3])

    def testModredTruncate(self):
        #balanced realization computed in matlab for the transfer function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = np.array(
            [[-1.958, -1.194, 1.824, -1.464],
             [-1.194, -0.8344, 2.563, -1.351],
             [-1.824, -2.563, -1.124, 2.704],
             [-1.464, -1.351, -2.704, -11.08]])
        B = np.array([[-0.9057], [-0.4068], [-0.3263], [-0.3474]])
        C = np.array([[-0.9057, -0.4068, 0.3263, -0.3474]])
        D = np.array([[0.]])
        sys = ss(A,B,C,D)
        rsys = modred(sys,[2, 3],'truncate')
        Artrue = np.array([[-1.958, -1.194], [-1.194, -0.8344]])
        Brtrue = np.array([[-0.9057], [-0.4068]])
        Crtrue = np.array([[-0.9057, -0.4068]])
        Drtrue = np.array([[0.]])
        np.testing.assert_array_almost_equal(rsys.A, Artrue)
        np.testing.assert_array_almost_equal(rsys.B, Brtrue)
        np.testing.assert_array_almost_equal(rsys.C, Crtrue)
        np.testing.assert_array_almost_equal(rsys.D, Drtrue)


    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testBalredTruncate(self):
        #controlable canonical realization computed in matlab for the transfer function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = np.array(
            [[-15., -7.5, -6.25, -1.875],
             [8., 0., 0., 0.],
             [0., 4., 0., 0.],
             [0., 0., 1., 0.]])
        B = np.array([[2.], [0.], [0.], [0.]])
        C = np.array([[0.5, 0.6875, 0.7031, 0.5]])
        D = np.array([[0.]])
        sys = ss(A,B,C,D)
        orders = 2
        rsys = balred(sys,orders,method='truncate')
        Artrue = np.array([[-1.958, -1.194], [-1.194, -0.8344]])
        Brtrue = np.array([[0.9057], [0.4068]])
        Crtrue = np.array([[0.9057, 0.4068]])
        Drtrue = np.array([[0.]])
        np.testing.assert_array_almost_equal(rsys.A, Artrue,decimal=2)
        np.testing.assert_array_almost_equal(rsys.B, Brtrue,decimal=4)
        np.testing.assert_array_almost_equal(rsys.C, Crtrue,decimal=4)
        np.testing.assert_array_almost_equal(rsys.D, Drtrue,decimal=4)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testBalredMatchDC(self):
        #controlable canonical realization computed in matlab for the transfer function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = np.array(
            [[-15., -7.5, -6.25, -1.875],
             [8., 0., 0., 0.],
             [0., 4., 0., 0.],
             [0., 0., 1., 0.]])
        B = np.array([[2.], [0.], [0.], [0.]])
        C = np.array([[0.5, 0.6875, 0.7031, 0.5]])
        D = np.array([[0.]])
        sys = ss(A,B,C,D)
        orders = 2
        rsys = balred(sys,orders,method='matchdc')
        Artrue = np.array(
            [[-4.43094773, -4.55232904],
             [-4.55232904, -5.36195206]])
        Brtrue = np.array([[1.36235673], [1.03114388]])
        Crtrue = np.array([[1.36235673, 1.03114388]])
        Drtrue = np.array([[-0.08383902]])
        np.testing.assert_array_almost_equal(rsys.A, Artrue,decimal=2)
        np.testing.assert_array_almost_equal(rsys.B, Brtrue,decimal=4)
        np.testing.assert_array_almost_equal(rsys.C, Crtrue,decimal=4)
        np.testing.assert_array_almost_equal(rsys.D, Drtrue,decimal=4)

    def tearDown(self):
        # Reset configuration variables to their original settings
        control.config.reset_defaults()
        
def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestModelsimp)


if __name__ == '__main__':
    unittest.main()
