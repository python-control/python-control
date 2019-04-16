#!/usr/bin/env python
#
# ssmatrix_test.py - test state-space matrix class
# RMM, 13 Apr 2019

import unittest
import numpy as np
import control as ct

class TestStateSpaceMatrix(unittest.TestCase):
    """Tests for the StateSpaceMatrix class."""

    def setUp(self):
        pass                    # Do nothing for now

    def test_constructor(self):
        # Create a matrix and make sure we get back a 2D array
        M = ct.StateSpaceMatrix([[1, 1], [1, 1]])
        self.assertEqual(M.shape, (2, 2))
        np.testing.assert_array_equal(M, np.array([[1, 1], [1, 1]]))

        # Passing a vector should give back a row vector (as 2D matrix)
        M = ct.StateSpaceMatrix([1, 1])
        self.assertEqual(M.shape, (1, 2))
        np.testing.assert_array_equal(M, np.array([[1, 1]]))

        # Use axis to switch to a column vector
        #! TODO: not yet implemented
        M = ct.StateSpaceMatrix([1, 1], axis=0)
        self.assertEqual(M.shape, (2, 1))
        np.testing.assert_array_equal(M, np.array([[1], [1]]))

        # Scalars should get converted to 1x1 matrix
        M = ct.StateSpaceMatrix(1)
        self.assertEqual(M.shape, (1, 1))
        np.testing.assert_array_equal(M, np.array([[1]]))
        
        # Empty matrix should have shape (0, 0)
        M = ct.StateSpaceMatrix([[]])
        self.assertEqual(M.shape, (0, 0))

        # Use (deprecated?) matrix-style construction string (w/ warnings off)
        import warnings
        warnings.filterwarnings("ignore")
        M = ct.StateSpaceMatrix("1, 1; 1, 1")
        warnings.filterwarnings("default")
        self.assertEqual(M.shape, (2,2))
        self.assertTrue(isinstance(M, ct.StateSpaceMatrix))

    def test_mul(self):
        # Make sure that multiplying two matrices gives a matrix
        M1 = ct.StateSpaceMatrix([[1, 1], [1, -1]])
        M2 = ct.StateSpaceMatrix([[1, 2], [2, 1]])
        Mprod = M1 * M2
        self.assertTrue(isinstance(Mprod, ct.StateSpaceMatrix))
        self.assertEqual(Mprod.shape, (2, 2))
        np.testing.assert_array_equal(Mprod, np.dot(M1, M2))

        # Matrix times a (state-space) column vector gives a column vector
        Cm = ct.StateSpaceMatrix([[1], [2]])
        MCm = M1 * Cm
        self.assertTrue(isinstance(MCm, ct.StateSpaceMatrix))
        self.assertEqual(MCm.shape, (2, 1))
        np.testing.assert_array_equal(MCm, np.dot(M1, Cm))

        # Matrix times a (ndarray) column vector gives a column vector
        Ca = np.array([[1], [2]])
        MCa = M1 * Ca
        self.assertTrue(isinstance(MCa, ct.StateSpaceMatrix))
        self.assertEqual(MCa.shape, (2, 1))
        np.testing.assert_array_equal(MCa, np.dot(M1, Ca))

        # (State-space) row vector time SSMatrix gives SSMatrix row vector
        Rm = ct.StateSpaceMatrix([[1, 2]])
        MRm = Rm * M1
        self.assertTrue(isinstance(MRm, ct.StateSpaceMatrix))
        self.assertEqual(MRm.shape, (1, 2))
        np.testing.assert_array_equal(MRm, np.dot(Rm, M1))

        # (ndarray) row vector time SSMatrix gives SSMatrix row vector
        Rm = np.array([[1, 2]])
        MRm = Rm * M1
        self.assertTrue(isinstance(MRm, ct.StateSpaceMatrix))
        self.assertEqual(MRm.shape, (1, 2))
        np.testing.assert_array_equal(MRm, np.dot(Rm, M1))

        # Row vector times column vector gives 1x1 matrix
        RmCm = Rm * Cm
        self.assertTrue(isinstance(RmCm, ct.StateSpaceMatrix))
        self.assertEqual(RmCm.shape, (1, 1))
        np.testing.assert_array_equal(RmCm, np.dot(Rm, Cm))

        # Column vector times row vector gives nxn matrix
        CmRm = Cm * Rm
        self.assertTrue(isinstance(CmRm, ct.StateSpaceMatrix))
        self.assertEqual(CmRm.shape, (2, 2))
        np.testing.assert_array_equal(CmRm, np.dot(Cm, Rm))

    def test_multiple_other(self):
        """Make sure that certain operations preserve StateSpaceMatrix type"""
        M = ct.StateSpaceMatrix([[1, 1], [1, 1]])
        Mint = M * 5
        self.assertTrue(isinstance(Mint, ct.StateSpaceMatrix))

        Mint = 5 * M
        self.assertTrue(isinstance(Mint, ct.StateSpaceMatrix))
        
        Mreal = M * 5.5
        self.assertTrue(isinstance(Mreal, ct.StateSpaceMatrix))
        
        Mreal = 5.5 * M
        self.assertTrue(isinstance(Mreal, ct.StateSpaceMatrix))
        
        Mcomplex = M * 1j
        self.assertFalse(isinstance(Mcomplex, ct.StateSpaceMatrix))

        Mcomplex = 1j * M
        self.assertFalse(isinstance(Mcomplex, ct.StateSpaceMatrix))

        Mreal = M * np.array([[5.5, 0], [0, 5.5]])
        self.assertTrue(isinstance(Mreal, ct.StateSpaceMatrix))
        
        Mreal = np.array([[5.5, 0], [0, 5.5]]) * M
        self.assertTrue(isinstance(Mreal, ct.StateSpaceMatrix))
        
        Mcomplex = M * np.array([[1j, 0], [0, 1j]])
        self.assertFalse(isinstance(Mcomplex, ct.StateSpaceMatrix))

        Mcomplex = np.array([[1j, 0], [0, 1j]]) * M
        self.assertFalse(isinstance(Mcomplex, ct.StateSpaceMatrix))
        
    def test_getitem(self):
        M = ct.StateSpaceMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Extracting full slice gives back what we started with
        S1 = M[:,:]
        self.assertTrue(isinstance(S1, ct.StateSpaceMatrix))
        self.assertEqual(S1.shape, M.shape)
        np.testing.assert_array_equal(S1, M)

        # Extracting multiple columns using slice
        S2 = M[:,0:2]
        self.assertTrue(isinstance(S2, ct.StateSpaceMatrix))
        self.assertEqual(S2.shape, (3, 2))
        np.testing.assert_array_equal(S2, [[1, 2], [4, 5], [7, 8]])

        # Extracting multiple columns using array
        S3 = M[:,[0,1]]
        self.assertTrue(isinstance(S3, ct.StateSpaceMatrix))
        self.assertEqual(S3.shape, (3, 2))
        np.testing.assert_array_equal(S3, [[1, 2], [4, 5], [7, 8]])
        
        # Extracting single column returns column matrix
        S4 = M[:,1]
        self.assertTrue(isinstance(S4, ct.StateSpaceMatrix))
        self.assertEqual(S4.shape, (3, 1))
        np.testing.assert_array_equal(S4, [[2], [5], [8]])

        # Extracting multiple rows using slice
        S5 = M[0:2,:]
        self.assertTrue(isinstance(S5, ct.StateSpaceMatrix))
        self.assertEqual(S5.shape, (2, 3))
        np.testing.assert_array_equal(S5, [[1, 2, 3], [4, 5, 6]])

        # Extracting multiple rows using array
        S6 = M[[0,1],:]
        self.assertTrue(isinstance(S6, ct.StateSpaceMatrix))
        self.assertEqual(S6.shape, (2, 3))
        np.testing.assert_array_equal(S6, [[1, 2, 3], [4, 5, 6]])
        
        # Extracting single row returns row matrix
        S6 = M[1,:]
        self.assertTrue(isinstance(S6, ct.StateSpaceMatrix))
        self.assertEqual(S6.shape, (1, 3))
        np.testing.assert_array_equal(S6, [[4, 5, 6]])

        # Extracting row and column slices returns matrix
        S7 = M[0:2,0:2]
        self.assertTrue(isinstance(S7, ct.StateSpaceMatrix))
        self.assertEqual(S7.shape, (2, 2))
        np.testing.assert_array_equal(S7, [[1, 2], [4, 5]])

        # Extracting single row and column returns matrix
        #! TODO: uncomment (not true for original matrix class)
        S8 = M[1, 1]
        # self.assertTrue(isinstance(S8, ct.StateSpaceMatrix))
        # self.assertEqual(S8.shape, (1, 1))
        np.testing.assert_array_equal(S8, [[5]])
        

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestStateSpaceMatrix)

if __name__ == "__main__":
    unittest.main()
