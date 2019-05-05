#!/usr/bin/env python

import unittest
import numpy as np
from control import ss
from control.canonical import canonical_form


class TestCanonical(unittest.TestCase):
    """Tests for the canonical forms class"""

    def test_reachable_form(self):
        """Test the reachable canonical form"""

        # Create a system in the reachable canonical form
        coeffs = [1.0, 2.0, 3.0, 4.0, 1.0]
        A_true = np.polynomial.polynomial.polycompanion(coeffs)
        A_true = np.fliplr(np.rot90(A_true))
        B_true = np.matrix("1.0 0.0 0.0 0.0").T
        C_true = np.matrix("1.0 1.0 1.0 1.0")
        D_true = 42.0

        # Perform a coordinate transform with a random invertible matrix
        T_true = np.matrix([[-0.27144004, -0.39933167,  0.75634684,  0.44135471],
                            [-0.74855725, -0.39136285, -0.18142339, -0.50356997],
                            [-0.40688007,  0.81416369,  0.38002113, -0.16483334],
                            [-0.44769516,  0.15654653, -0.50060858,  0.72419146]])
        A = np.linalg.solve(T_true, A_true)*T_true
        B = np.linalg.solve(T_true, B_true)
        C = C_true*T_true
        D = D_true

        # Create a state space system and convert it to the reachable canonical form
        sys_check, T_check = canonical_form(ss(A, B, C, D), "reachable")

        # Check against the true values
        np.testing.assert_array_almost_equal(sys_check.A, A_true)
        np.testing.assert_array_almost_equal(sys_check.B, B_true)
        np.testing.assert_array_almost_equal(sys_check.C, C_true)
        np.testing.assert_array_almost_equal(sys_check.D, D_true)
        np.testing.assert_array_almost_equal(T_check, T_true)

    def test_unreachable_system(self):
        """Test reachable canonical form with an unreachable system"""

        # Create an unreachable system
        A = np.matrix("1.0 2.0 2.0; 4.0 5.0 5.0; 7.0 8.0 8.0")
        B = np.matrix("1.0 1.0 1.0").T
        C = np.matrix("1.0 1.0 1.0")
        D = 42.0
        sys = ss(A, B, C, D)

        # Check if an exception is raised
        np.testing.assert_raises(ValueError, canonical_form, sys, "reachable")

    def test_modal_form(self):
        """Test the modal canonical form"""

        # Create a system in the modal canonical form
        A_true = np.diag([4.0, 3.0, 2.0, 1.0]) # order from the largest to the smallest
        B_true = np.matrix("1.1 2.2 3.3 4.4").T
        C_true = np.matrix("1.3 1.4 1.5 1.6")
        D_true = 42.0

        # Perform a coordinate transform with a random invertible matrix
        T_true = np.matrix([[-0.27144004, -0.39933167,  0.75634684,  0.44135471],
                            [-0.74855725, -0.39136285, -0.18142339, -0.50356997],
                            [-0.40688007,  0.81416369,  0.38002113, -0.16483334],
                            [-0.44769516,  0.15654653, -0.50060858,  0.72419146]])
        A = np.linalg.solve(T_true, A_true)*T_true
        B = np.linalg.solve(T_true, B_true)
        C = C_true*T_true
        D = D_true

        # Create a state space system and convert it to the modal canonical form
        sys_check, T_check = canonical_form(ss(A, B, C, D), "modal")

        # Check against the true values
        #TODO: Test in respect to ambiguous transformation (system characteristics?)
        np.testing.assert_array_almost_equal(sys_check.A, A_true)
        #np.testing.assert_array_almost_equal(sys_check.B, B_true)
        #np.testing.assert_array_almost_equal(sys_check.C, C_true)
        np.testing.assert_array_almost_equal(sys_check.D, D_true)
        #np.testing.assert_array_almost_equal(T_check, T_true)
        
    def test_observable_form(self):
        """Test the observable canonical form"""

        # Create a system in the observable canonical form
        coeffs = [1.0, 2.0, 3.0, 4.0, 1.0]
        A_true = np.polynomial.polynomial.polycompanion(coeffs)
        A_true = np.fliplr(np.flipud(A_true))
        B_true = np.matrix("1.0 1.0 1.0 1.0").T
        C_true = np.matrix("1.0 0.0 0.0 0.0")
        D_true = 42.0

        # Perform a coordinate transform with a random invertible matrix
        T_true = np.matrix([[-0.27144004, -0.39933167,  0.75634684,  0.44135471],
                            [-0.74855725, -0.39136285, -0.18142339, -0.50356997],
                            [-0.40688007,  0.81416369,  0.38002113, -0.16483334],
                            [-0.44769516,  0.15654653, -0.50060858,  0.72419146]])
        A = np.linalg.solve(T_true, A_true)*T_true
        B = np.linalg.solve(T_true, B_true)
        C = C_true*T_true
        D = D_true

        # Create a state space system and convert it to the observable canonical form
        sys_check, T_check = canonical_form(ss(A, B, C, D), "observable")

        # Check against the true values
        np.testing.assert_array_almost_equal(sys_check.A, A_true)
        np.testing.assert_array_almost_equal(sys_check.B, B_true)
        np.testing.assert_array_almost_equal(sys_check.C, C_true)
        np.testing.assert_array_almost_equal(sys_check.D, D_true)
        np.testing.assert_array_almost_equal(T_check, T_true)

    def test_unobservable_system(self):
        """Test observable canonical form with an unobservable system"""

        # Create an unobservable system
        A = np.matrix("1.0 2.0 2.0; 4.0 5.0 5.0; 7.0 8.0 8.0")
        B = np.matrix("1.0 1.0 1.0").T
        C = np.matrix("1.0 1.0 1.0")
        D = 42.0
        sys = ss(A, B, C, D)

        # Check if an exception is raised
        np.testing.assert_raises(ValueError, canonical_form, sys, "observable")
