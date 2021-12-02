"""canonical_test.py"""

import numpy as np
import pytest
import scipy.linalg

from control.tests.conftest import slycotonly

from control import ss, tf, tf2ss
from control.canonical import canonical_form, reachable_form, \
    observable_form, modal_form, similarity_transform, bdschur
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
        A = np.linalg.solve(T_true, A_true) @ T_true
        B = np.linalg.solve(T_true, B_true)
        C = C_true @ T_true
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
        A = np.linalg.solve(T_true, A_true) @ T_true
        B = np.linalg.solve(T_true, B_true)
        C = C_true @ T_true
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


def extract_bdiag(a, blksizes):
    """
    Extract block diagonals

    Parameters
    ----------
    a - matrix to get blocks from
    blksizes - sequence of block diagonal sizes

    Returns
    -------
    Block diagonals

    Notes
    -----
    Conceptually, inverse of scipy.linalg.block_diag
    """
    idx0s = np.hstack([0, np.cumsum(blksizes[:-1], dtype=int)])
    return tuple(a[idx0:idx0+blksize,idx0:idx0+blksize]
                 for idx0, blksize in zip(idx0s, blksizes))


def companion_from_eig(eigvals):
    """
    Find companion matrix for given eigenvalue sequence.
    """
    from numpy.polynomial.polynomial import polyfromroots, polycompanion
    return polycompanion(polyfromroots(eigvals)).real


def block_diag_from_eig(eigvals):
    """
    Find block-diagonal matrix for given eigenvalue sequence

    Returns ideal, non-defective, schur block-diagonal form.
    """
    blocks = []
    i = 0
    while i < len(eigvals):
        e = eigvals[i]
        if e.imag == 0:
            blocks.append(e.real)
            i += 1
        else:
            assert e == eigvals[i+1].conjugate()
            blocks.append([[e.real, e.imag],
                           [-e.imag, e.real]])
            i += 2
    return scipy.linalg.block_diag(*blocks)


@slycotonly
@pytest.mark.parametrize(
    "eigvals, condmax, blksizes",
    [
        ([-1,-2,-3,-4,-5], None, [1,1,1,1,1]),
        ([-1,-2,-3,-4,-5], 1.01, [5]),
        ([-1,-1,-2,-2,-2], None, [2,3]),
        ([-1+1j,-1-1j,-2+2j,-2-2j,-2], None, [2,2,1]),
    ])
def test_bdschur_ref(eigvals, condmax, blksizes):
    # "reference" check
    # uses companion form to introduce numerical complications
    from numpy.linalg import solve

    a = companion_from_eig(eigvals)
    b, t, test_blksizes = bdschur(a, condmax=condmax)

    np.testing.assert_array_equal(np.sort(test_blksizes), np.sort(blksizes))

    bdiag_b = scipy.linalg.block_diag(*extract_bdiag(b, test_blksizes))
    np.testing.assert_array_almost_equal(bdiag_b, b)

    np.testing.assert_array_almost_equal(solve(t, a) @ t, b)


@slycotonly
@pytest.mark.parametrize(
    "eigvals, sorted_blk_eigvals, sort",
    [
        ([-2,-1,0,1,2], [2,1,0,-1,-2], 'continuous'),
        ([-2,-2+2j,-2-2j,-2-3j,-2+3j], [-2+3j,-2+2j,-2], 'continuous'),
        (np.exp([-0.2,-0.1,0,0.1,0.2]), np.exp([0.2,0.1,0,-0.1,-0.2]), 'discrete'),
        (np.exp([-0.2+0.2j,-0.2-0.2j, -0.01, -0.03-0.3j,-0.03+0.3j,]),
         np.exp([-0.01, -0.03+0.3j, -0.2+0.2j]),
         'discrete'),
    ])
def test_bdschur_sort(eigvals, sorted_blk_eigvals, sort):
    # use block diagonal form to prevent numerical complications
    # for discrete case, exp and log introduce round-off, can't test as compeletely
    a = block_diag_from_eig(eigvals)

    b, t, blksizes = bdschur(a, sort=sort)
    assert len(blksizes) == len(sorted_blk_eigvals)

    blocks = extract_bdiag(b, blksizes)
    for block, blk_eigval in zip(blocks, sorted_blk_eigvals):
        test_eigvals = np.linalg.eigvals(block)
        np.testing.assert_allclose(test_eigvals.real,
                                   blk_eigval.real)

        np.testing.assert_allclose(abs(test_eigvals.imag),
                                   blk_eigval.imag)


@slycotonly
def test_bdschur_defective():
    # the eigenvalues of this simple defective matrix cannot be separated
    # a previous version of the bdschur would fail on this
    a = companion_from_eig([-1, -1])
    amodal, tmodal, blksizes = bdschur(a, condmax=1e200)


def test_bdschur_empty():
    # empty matrix in gives empty matrix out
    a = np.empty(shape=(0,0))
    b, t, blksizes = bdschur(a)
    np.testing.assert_array_equal(b, a)
    np.testing.assert_array_equal(t, a)
    np.testing.assert_array_equal(blksizes, np.array([]))


def test_bdschur_condmax_lt_1():
    # require condmax >= 1.0
    with pytest.raises(ValueError):
        bdschur(1, condmax=np.nextafter(1, 0))


@slycotonly
def test_bdschur_invalid_sort():
    # sort must be in ('continuous', 'discrete')
    with pytest.raises(ValueError):
        bdschur(1, sort='no-such-sort')


@slycotonly
@pytest.mark.parametrize(
    "A_true, B_true, C_true, D_true",
    [(np.diag([4.0, 3.0, 2.0, 1.0]),  # order from largest to smallest
      np.array([[1.1, 2.2, 3.3, 4.4]]).T,
      np.array([[1.3, 1.4, 1.5, 1.6]]),
      np.array([[42.0]])),

     (np.array([[-1,  1,  0,  0],
                [-1, -1,  0,  0],
                [ 0,  0, -2,  1],
                [ 0,  0,  0, -3]]),
      np.array([[0, 1, 0, 0],
                [0, 0, 0, 1]]).T,
      np.array([[1, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]]),
      np.array([[0, 1],
                [1, 0],
                [0, 0]])),
     ],
     ids=["sys1", "sys2"])
def test_modal_form(A_true, B_true, C_true, D_true):
    # Check modal_canonical corresponds to bdschur
    # Perform a coordinate transform with a random invertible matrix
    T_true =  np.array([[-0.27144004, -0.39933167,  0.75634684,  0.44135471],
                        [-0.74855725, -0.39136285, -0.18142339, -0.50356997],
                        [-0.40688007,  0.81416369,  0.38002113, -0.16483334],
                        [-0.44769516,  0.15654653, -0.50060858,  0.72419146]])
    A = np.linalg.solve(T_true, A_true) @ T_true
    B = np.linalg.solve(T_true, B_true)
    C = C_true @ T_true
    D = D_true

    # Create a state space system and convert it to modal canonical form
    sys_check, T_check = modal_form(ss(A, B, C, D))

    a_bds, t_bds, _ = bdschur(A)

    np.testing.assert_array_almost_equal(sys_check.A, a_bds)
    np.testing.assert_array_almost_equal(T_check, t_bds)
    np.testing.assert_array_almost_equal(sys_check.B, np.linalg.solve(t_bds, B))
    np.testing.assert_array_almost_equal(sys_check.C, C @ t_bds)
    np.testing.assert_array_almost_equal(sys_check.D, D)

    # canonical_form(...,'modal') is the same as modal_form with default parameters
    cf_sys, T_cf = canonical_form(ss(A, B, C, D), 'modal')
    np.testing.assert_array_almost_equal(cf_sys.A, sys_check.A)
    np.testing.assert_array_almost_equal(cf_sys.B, sys_check.B)
    np.testing.assert_array_almost_equal(cf_sys.C, sys_check.C)
    np.testing.assert_array_almost_equal(cf_sys.D, sys_check.D)
    np.testing.assert_array_almost_equal(T_check, T_cf)

    # Make sure Hankel coefficients are OK
    for i in range(A.shape[0]):
        np.testing.assert_almost_equal(
            C_true @ np.linalg.matrix_power(A_true, i) @  B_true,
            C @ np.linalg.matrix_power(A, i) @ B)


@slycotonly
@pytest.mark.parametrize(
    "condmax, len_blksizes",
    [(1.1, 1),
     (None, 5)])
def test_modal_form_condmax(condmax, len_blksizes):
    # condmax passed through as expected
    a = companion_from_eig([-1, -2, -3, -4, -5])
    amodal, tmodal, blksizes = bdschur(a, condmax=condmax)
    assert len(blksizes) == len_blksizes
    xsys = ss(a, [[1],[0],[0],[0],[0]], [0,0,0,0,1], 0)
    zsys, t = modal_form(xsys, condmax=condmax)
    np.testing.assert_array_almost_equal(zsys.A, amodal)
    np.testing.assert_array_almost_equal(t, tmodal)
    np.testing.assert_array_almost_equal(zsys.B, np.linalg.solve(tmodal, xsys.B))
    np.testing.assert_array_almost_equal(zsys.C, xsys.C @ tmodal)
    np.testing.assert_array_almost_equal(zsys.D, xsys.D)


@slycotonly
@pytest.mark.parametrize(
    "sys_type",
    ['continuous',
     'discrete'])
def test_modal_form_sort(sys_type):
    a = companion_from_eig([0.1+0.9j,0.1-0.9j, 0.2+0.8j, 0.2-0.8j])
    amodal, tmodal, blksizes = bdschur(a, sort=sys_type)

    dt = 0 if sys_type == 'continuous' else True

    xsys = ss(a, [[1],[0],[0],[0],], [0,0,0,1], 0, dt)
    zsys, t = modal_form(xsys, sort=True)

    my_amodal = np.linalg.solve(tmodal, a) @ tmodal
    np.testing.assert_array_almost_equal(amodal, my_amodal)

    np.testing.assert_array_almost_equal(t, tmodal)
    np.testing.assert_array_almost_equal(zsys.A, amodal)
    np.testing.assert_array_almost_equal(zsys.B, np.linalg.solve(tmodal, xsys.B))
    np.testing.assert_array_almost_equal(zsys.C, xsys.C @ tmodal)
    np.testing.assert_array_almost_equal(zsys.D, xsys.D)


def test_modal_form_empty():
    # empty system should be returned as-is
    # t empty matrix
    insys = ss([], [], [], 123)
    outsys, t = modal_form(insys)
    np.testing.assert_array_equal(outsys.A, insys.A)
    np.testing.assert_array_equal(outsys.B, insys.B)
    np.testing.assert_array_equal(outsys.C, insys.C)
    np.testing.assert_array_equal(outsys.D, insys.D)
    assert t.shape == (0,0)
