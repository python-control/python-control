"""canonical_test.py"""

import numpy as np
import pytest

from control import ss, tf, tf2ss
from control.canonical import canonical_form, reachable_form, \
    observable_form, modal_form, similarity_transform
from control.exception import ControlNotImplemented

class TestCanonical:
    """Tests for the canonical forms class"""

    def test_reachable_form(self):
        """Test the reachable canonical form"""
        # Create a system in the reachable canonical form
        coeffs = [1.0, 2.0, 3.0, 4.0, 1.0]
        A_true = np.polynomial.polynomial.polycompanion(coeffs)
        A_true = np.fliplr(np.rot90(A_true))
        B_true = np.array([[1.0, 0.0, 0.0, 0.0]]).T
        C_true = np.array([[1.0, 1.0, 1.0, 1.0]])
        D_true = 42.0

        # Perform a coordinate transform with a random invertible matrix
        T_true =  np.array([[-0.27144004, -0.39933167,  0.75634684,  0.44135471],
                            [-0.74855725, -0.39136285, -0.18142339, -0.50356997],
                            [-0.40688007,  0.81416369,  0.38002113, -0.16483334],
                            [-0.44769516,  0.15654653, -0.50060858,  0.72419146]])
        A = np.linalg.solve(T_true, A_true).dot(T_true)
        B = np.linalg.solve(T_true, B_true)
        C = C_true.dot(T_true)
        D = D_true

        # Create a state space system and convert it to the reachable canonical form
        sys_check, T_check = canonical_form(ss(A, B, C, D), "reachable")

        # Check against the true values
        np.testing.assert_array_almost_equal(sys_check.A, A_true)
        np.testing.assert_array_almost_equal(sys_check.B, B_true)
        np.testing.assert_array_almost_equal(sys_check.C, C_true)
        np.testing.assert_array_almost_equal(sys_check.D, D_true)
        np.testing.assert_array_almost_equal(T_check, T_true)

        # Reachable form only supports SISO
        sys = tf([[ [1], [1] ]], [[ [1, 2, 1], [1, 2, 1] ]])
        np.testing.assert_raises(ControlNotImplemented, reachable_form, sys)

    def test_unreachable_system(self):
        """Test reachable canonical form with an unreachable system"""
        # Create an unreachable system
        A = np.array([[1., 2., 2.],
                      [4., 5., 5.],
                      [7., 8., 8.]])
        B = np.array([[1.], [1.],[1.]])
        C = np.array([[1., 1.,1.]])
        D = np.array([[42.0]])
        sys = ss(A, B, C, D)

        # Check if an exception is raised
        np.testing.assert_raises(ValueError, canonical_form, sys, "reachable")

    @pytest.mark.parametrize(
        "A_true, B_true, C_true, D_true",
        [(np.diag([4.0, 3.0, 2.0, 1.0]),  # order from largest to smallest
          np.array([[1.1, 2.2, 3.3, 4.4]]).T,
          np.array([[1.3, 1.4, 1.5, 1.6]]),
          np.array([[42.0]])),
         (np.array([[-1,  1,  0,  0],
                    [-1, -1,  0,  0],
                    [ 0,  0, -2,  0],
                    [ 0,  0,  0, -3]]),
          np.array([[0, 1, 0, 1]]).T,
          np.array([[1, 0, 0, 1]]),
          np.array([[0]])),
         # Reorder rows to get complete coverage (real eigenvalue cxrtvfirst)
         (np.array([[-1,  0, 0,  0],
                    [ 0, -2,  1,  0],
                    [ 0, -1, -2,  0],
                    [ 0,  0,  0, -3]]),
          np.array([[0, 0, 1, 1]]).T,
          np.array([[0, 1, 0, 1]]),
          np.array([[0]])),
         ],
        ids=["sys1", "sys2", "sys3"])
    def test_modal_form(self, A_true, B_true, C_true, D_true):
        """Test the modal canonical form"""
        # Perform a coordinate transform with a random invertible matrix
        T_true =  np.array([[-0.27144004, -0.39933167,  0.75634684,  0.44135471],
                            [-0.74855725, -0.39136285, -0.18142339, -0.50356997],
                            [-0.40688007,  0.81416369,  0.38002113, -0.16483334],
                            [-0.44769516,  0.15654653, -0.50060858,  0.72419146]])
        A = np.linalg.solve(T_true, A_true).dot(T_true)
        B = np.linalg.solve(T_true, B_true)
        C = C_true.dot(T_true)
        D = D_true

        # Create a state space system and convert it to modal canonical form
        sys_check, T_check = canonical_form(ss(A, B, C, D), "modal")

        # Check against the true values
        # TODO: Test in respect to ambiguous transformation
        # (system characteristics?)
        np.testing.assert_array_almost_equal(sys_check.A, A_true)
        #np.testing.assert_array_almost_equal(sys_check.B, B_true)
        #np.testing.assert_array_almost_equal(sys_check.C, C_true)
        np.testing.assert_array_almost_equal(sys_check.D, D_true)
        #np.testing.assert_array_almost_equal(T_check, T_true)

        # Create state space system and convert to modal canonical form
        sys_check, T_check = canonical_form(ss(A, B, C, D), 'modal')

        # B matrix should be all ones (or zero if not controllable)
        # TODO: need to update modal_form() to implement this
        if np.allclose(T_check, T_true):
            np.testing.assert_array_almost_equal(sys_check.B, B_true)
            np.testing.assert_array_almost_equal(sys_check.C, C_true)

        # Make sure Hankel coefficients are OK
        for i in range(A.shape[0]):
            np.testing.assert_almost_equal(
                np.dot(np.dot(C_true, np.linalg.matrix_power(A_true, i)),
                       B_true),
                np.dot(np.dot(C, np.linalg.matrix_power(A, i)), B))

    def test_modal_form_MIMO(self):
        """Test error because modal form only supports SISO"""
        sys = tf([[[1], [1]]], [[[1, 2, 1], [1, 2, 1]]])
        with pytest.raises(ControlNotImplemented):
            modal_form(sys)

    def test_observable_form(self):
        """Test the observable canonical form"""
        # Create a system in the observable canonical form
        coeffs = [1.0, 2.0, 3.0, 4.0, 1.0]
        A_true = np.polynomial.polynomial.polycompanion(coeffs)
        A_true = np.fliplr(np.flipud(A_true))
        B_true = np.array([[1.0, 1.0, 1.0, 1.0]]).T
        C_true = np.array([[1.0, 0.0, 0.0, 0.0]])
        D_true = 42.0

        # Perform a coordinate transform with a random invertible matrix
        T_true =  np.array([[-0.27144004, -0.39933167,  0.75634684,  0.44135471],
                            [-0.74855725, -0.39136285, -0.18142339, -0.50356997],
                            [-0.40688007,  0.81416369,  0.38002113, -0.16483334],
                            [-0.44769516,  0.15654653, -0.50060858,  0.72419146]])
        A = np.linalg.solve(T_true, A_true).dot(T_true)
        B = np.linalg.solve(T_true, B_true)
        C = C_true.dot(T_true)
        D = D_true

        # Create a state space system and convert it to the observable canonical form
        sys_check, T_check = canonical_form(ss(A, B, C, D), "observable")

        # Check against the true values
        np.testing.assert_array_almost_equal(sys_check.A, A_true)
        np.testing.assert_array_almost_equal(sys_check.B, B_true)
        np.testing.assert_array_almost_equal(sys_check.C, C_true)
        np.testing.assert_array_almost_equal(sys_check.D, D_true)
        np.testing.assert_array_almost_equal(T_check, T_true)

    def test_observable_form_MIMO(self):
        """Test error as Observable form only supports SISO"""
        sys = tf([[[1], [1] ]], [[[1, 2, 1], [1, 2, 1]]])
        with pytest.raises(ControlNotImplemented):
            observable_form(sys)

    def test_unobservable_system(self):
        """Test observable canonical form with an unobservable system"""
        # Create an unobservable system
        A = np.array([[1., 2., 2.],
                      [4., 5., 5.],
                      [7., 8., 8.]])

        B = np.array([[1.], [1.], [1.]])
        C = np.array([[1., 1., 1.]])
        D = 42.0
        sys = ss(A, B, C, D)

        # Check if an exception is raised
        with pytest.raises(ValueError):
            canonical_form(sys, "observable")

    def test_arguments(self):
        # Additional unit tests added on 25 May 2019 to increase coverage

        # Unknown canonical forms should generate exception
        sys = tf([1], [1, 2, 1])
        with pytest.raises(ControlNotImplemented):
            canonical_form(sys, 'unknown')

    def test_similarity(self):
        """Test similarty transform"""

        # Single input, single output systems
        siso_ini = tf2ss(tf([1, 1], [1, 1, 1]))
        for form in 'reachable', 'observable':
            # Convert the system to one of the canonical forms
            siso_can, T_can = canonical_form(siso_ini, form)

            # Use a similarity transformation to transform it back
            siso_sim = similarity_transform(siso_can, np.linalg.inv(T_can))

            # Make sure everything goes back to the original form
            np.testing.assert_array_almost_equal(siso_sim.A, siso_ini.A)
            np.testing.assert_array_almost_equal(siso_sim.B, siso_ini.B)
            np.testing.assert_array_almost_equal(siso_sim.C, siso_ini.C)
            np.testing.assert_array_almost_equal(siso_sim.D, siso_ini.D)

        # Multi-input, multi-output systems
        mimo_ini = ss(
            [[-1, 1, 0, 0], [0, -2, 1, 0], [0, 0, -3, 1], [0, 0, 0, -4]],
            [[1, 0], [0, 0], [0, 1], [1, 1]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            np.zeros((3, 2)))

        # Simple transformation: row/col flips + scaling
        mimo_txf = np.array(
            [[0, 1, 0, 0], [2, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Transform the system and transform it back
        mimo_sim = similarity_transform(mimo_ini, mimo_txf)
        mimo_new = similarity_transform(mimo_sim, np.linalg.inv(mimo_txf))
        np.testing.assert_array_almost_equal(mimo_new.A, mimo_ini.A)
        np.testing.assert_array_almost_equal(mimo_new.B, mimo_ini.B)
        np.testing.assert_array_almost_equal(mimo_new.C, mimo_ini.C)
        np.testing.assert_array_almost_equal(mimo_new.D, mimo_ini.D)

        # Make sure rescaling by identify does nothing
        mimo_new = similarity_transform(mimo_ini, np.eye(4))
        np.testing.assert_array_almost_equal(mimo_new.A, mimo_ini.A)
        np.testing.assert_array_almost_equal(mimo_new.B, mimo_ini.B)
        np.testing.assert_array_almost_equal(mimo_new.C, mimo_ini.C)
        np.testing.assert_array_almost_equal(mimo_new.D, mimo_ini.D)

        # Time rescaling
        mimo_tim = similarity_transform(mimo_ini, np.eye(4), timescale=0.3)
        mimo_new = similarity_transform(mimo_tim, np.eye(4), timescale=1/0.3)
        np.testing.assert_array_almost_equal(mimo_new.A, mimo_ini.A)
        np.testing.assert_array_almost_equal(mimo_new.B, mimo_ini.B)
        np.testing.assert_array_almost_equal(mimo_new.C, mimo_ini.C)
        np.testing.assert_array_almost_equal(mimo_new.D, mimo_ini.D)

        # Time + transformation, in one step
        mimo_sim = similarity_transform(mimo_ini, mimo_txf, timescale=0.3)
        mimo_new = similarity_transform(mimo_sim, np.linalg.inv(mimo_txf),
                                        timescale=1/0.3)
        np.testing.assert_array_almost_equal(mimo_new.A, mimo_ini.A)
        np.testing.assert_array_almost_equal(mimo_new.B, mimo_ini.B)
        np.testing.assert_array_almost_equal(mimo_new.C, mimo_ini.C)
        np.testing.assert_array_almost_equal(mimo_new.D, mimo_ini.D)

        # Time + transformation, in two steps
        mimo_sim = similarity_transform(mimo_ini, mimo_txf, timescale=0.3)
        mimo_tim = similarity_transform(mimo_sim, np.eye(4), timescale=1/0.3)
        mimo_new = similarity_transform(mimo_tim, np.linalg.inv(mimo_txf))
        np.testing.assert_array_almost_equal(mimo_new.A, mimo_ini.A)
        np.testing.assert_array_almost_equal(mimo_new.B, mimo_ini.B)
        np.testing.assert_array_almost_equal(mimo_new.C, mimo_ini.C)
        np.testing.assert_array_almost_equal(mimo_new.D, mimo_ini.D)

