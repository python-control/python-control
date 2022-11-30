# mateqn.py - Matrix equation solvers (Lyapunov, Riccati)
#
# Implementation of the functions lyap, dlyap, care and dare
# for solution of Lyapunov and Riccati equations.
#
# Original author: Bjorn Olofsson

# Copyright (c) 2011, All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the name of the project author nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

import warnings
import numpy as np
from numpy import copy, eye, dot, finfo, inexact, atleast_2d

import scipy as sp
from scipy.linalg import eigvals, solve

from .exception import ControlSlycot, ControlArgument, ControlDimension, \
    slycot_check
from .statesp import _ssmatrix

# Make sure we have access to the right slycot routines
try:
    from slycot.exceptions import SlycotResultWarning
except ImportError:
    SlycotResultWarning = UserWarning

try:
    from slycot import sb03md57

    # wrap without the deprecation warning
    def sb03md(n, C, A, U, dico, job='X', fact='N', trana='N', ldwork=None):
        ret = sb03md57(A, U, C, dico, job, fact, trana, ldwork)
        return ret[2:]
except ImportError:
    try:
        from slycot import sb03md
    except ImportError:
        sb03md = None

try:
    from slycot import sb04md
except ImportError:
    sb04md = None

try:
    from slycot import sb04qd
except ImportError:
    sb0qmd = None

try:
    from slycot import sg03ad
except ImportError:
    sb04ad = None

__all__ = ['lyap', 'dlyap', 'dare', 'care']

#
# Lyapunov equation solvers lyap and dlyap
#


def lyap(A, Q, C=None, E=None, method=None):
    """Solves the continuous-time Lyapunov equation

    X = lyap(A, Q) solves

        :math:`A X + X A^T + Q = 0`

    where A and Q are square matrices of the same dimension.  Q must be
    symmetric.

    X = lyap(A, Q, C) solves the Sylvester equation

        :math:`A X + X Q + C = 0`

    where A and Q are square matrices.

    X = lyap(A, Q, None, E) solves the generalized continuous-time
    Lyapunov equation

        :math:`A X E^T + E X A^T + Q = 0`

    where Q is a symmetric matrix and A, Q and E are square matrices of the
    same dimension.

    Parameters
    ----------
    A, Q : 2D array_like
        Input matrices for the Lyapunov or Sylvestor equation
    C : 2D array_like, optional
        If present, solve the Sylvester equation
    E : 2D array_like, optional
        If present, solve the generalized Lyapunov equation
    method : str, optional
        Set the method used for computing the result.  Current methods are
        'slycot' and 'scipy'.  If set to None (default), try 'slycot' first
        and then 'scipy'.

    Returns
    -------
    X : 2D array (or matrix)
        Solution to the Lyapunov or Sylvester equation

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    """
    # Decide what method to use
    method = _slycot_or_scipy(method)
    if method == 'slycot':
        if sb03md is None:
            raise ControlSlycot("Can't find slycot module 'sb03md'")
        if sb04md is None:
            raise ControlSlycot("Can't find slycot module 'sb04md'")

    # Reshape input arrays
    A = np.array(A, ndmin=2)
    Q = np.array(Q, ndmin=2)
    if C is not None:
        C = np.array(C, ndmin=2)
    if E is not None:
        E = np.array(E, ndmin=2)

    # Determine main dimensions
    n = A.shape[0]
    m = Q.shape[0]

    # Check to make sure input matrices are the right shape and type
    _check_shape("A", A, n, n, square=True)

    # Solve standard Lyapunov equation
    if C is None and E is None:
        # Check to make sure input matrices are the right shape and type
        _check_shape("Q", Q, n, n, square=True, symmetric=True)

        if method == 'scipy':
            # Solve the Lyapunov equation using SciPy
            return sp.linalg.solve_continuous_lyapunov(A, -Q)

        # Solve the Lyapunov equation by calling Slycot function sb03md
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=SlycotResultWarning)
            X, scale, sep, ferr, w = \
                sb03md(n, -Q, A, eye(n, n), 'C', trana='T')

    # Solve the Sylvester equation
    elif C is not None and E is None:
        # Check to make sure input matrices are the right shape and type
        _check_shape("Q", Q, m, m, square=True)
        _check_shape("C", C, n, m)

        if method == 'scipy':
            # Solve the Sylvester equation using SciPy
            return sp.linalg.solve_sylvester(A, Q, -C)

        # Solve the Sylvester equation by calling the Slycot function sb04md
        X = sb04md(n, m, A, Q, -C)

    # Solve the generalized Lyapunov equation
    elif C is None and E is not None:
        # Check to make sure input matrices are the right shape and type
        _check_shape("Q", Q, n, n, square=True, symmetric=True)
        _check_shape("E", E, n, n, square=True)

        if method == 'scipy':
            raise ControlArgument(
                "method='scipy' not valid for generalized Lyapunov equation")

        # Make sure we have access to the write slicot routine
        try:
            from slycot import sg03ad

        except ImportError:
            raise ControlSlycot("Can't find slycot module 'sg03ad'")

        # Solve the generalized Lyapunov equation by calling Slycot
        # function sg03ad
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=SlycotResultWarning)
            A, E, Q, Z, X, scale, sep, ferr, alphar, alphai, beta = \
                sg03ad('C', 'B', 'N', 'T', 'L', n,
                       A, E, eye(n, n), eye(n, n), -Q)

    # Invalid set of input parameters (C and E specified)
    else:
        raise ControlArgument("Invalid set of input parameters")

    return _ssmatrix(X)


def dlyap(A, Q, C=None, E=None, method=None):
    """Solves the discrete-time Lyapunov equation

    X = dlyap(A, Q) solves

        :math:`A X A^T - X + Q = 0`

    where A and Q are square matrices of the same dimension. Further
    Q must be symmetric.

    dlyap(A, Q, C) solves the Sylvester equation

        :math:`A X Q^T - X + C = 0`

    where A and Q are square matrices.

    dlyap(A, Q, None, E) solves the generalized discrete-time Lyapunov
    equation

        :math:`A X A^T - E X E^T + Q = 0`

    where Q is a symmetric matrix and A, Q and E are square matrices of the
    same dimension.

    Parameters
    ----------
    A, Q : 2D array_like
        Input matrices for the Lyapunov or Sylvestor equation
    C : 2D array_like, optional
        If present, solve the Sylvester equation
    E : 2D array_like, optional
        If present, solve the generalized Lyapunov equation
    method : str, optional
        Set the method used for computing the result.  Current methods are
        'slycot' and 'scipy'.  If set to None (default), try 'slycot' first
        and then 'scipy'.

    Returns
    -------
    X : 2D array (or matrix)
        Solution to the Lyapunov or Sylvester equation

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    """
    # Decide what method to use
    method = _slycot_or_scipy(method)

    if method == 'slycot':
        # Make sure we have access to the right slycot routines
        if sb03md is None:
            raise ControlSlycot("Can't find slycot module 'sb03md'")
        if sb04qd is None:
            raise ControlSlycot("Can't find slycot module 'sb04qd'")
        if sg03ad is None:
            raise ControlSlycot("Can't find slycot module 'sg03ad'")

    # Reshape input arrays
    A = np.array(A, ndmin=2)
    Q = np.array(Q, ndmin=2)
    if C is not None:
        C = np.array(C, ndmin=2)
    if E is not None:
        E = np.array(E, ndmin=2)

    # Determine main dimensions
    n = A.shape[0]
    m = Q.shape[0]

    # Check to make sure input matrices are the right shape and type
    _check_shape("A", A, n, n, square=True)

    # Solve standard Lyapunov equation
    if C is None and E is None:
        # Check to make sure input matrices are the right shape and type
        _check_shape("Q", Q, n, n, square=True, symmetric=True)

        if method == 'scipy':
            # Solve the Lyapunov equation using SciPy
            return sp.linalg.solve_discrete_lyapunov(A, Q)

        # Solve the Lyapunov equation by calling the Slycot function sb03md
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=SlycotResultWarning)
            X, scale, sep, ferr, w = \
                sb03md(n, -Q, A, eye(n, n), 'D', trana='T')

    # Solve the Sylvester equation
    elif C is not None and E is None:
        # Check to make sure input matrices are the right shape and type
        _check_shape("Q", Q, m, m, square=True)
        _check_shape("C", C, n, m)

        if method == 'scipy':
            raise ControlArgument(
                "method='scipy' not valid for Sylvester equation")

        # Solve the Sylvester equation by calling Slycot function sb04qd
        X = sb04qd(n, m, -A, Q.T, C)

    # Solve the generalized Lyapunov equation
    elif C is None and E is not None:
        # Check to make sure input matrices are the right shape and type
        _check_shape("Q", Q, n, n, square=True, symmetric=True)
        _check_shape("E", E, n, n, square=True)

        if method == 'scipy':
            raise ControlArgument(
                "method='scipy' not valid for generalized Lyapunov equation")

        # Solve the generalized Lyapunov equation by calling Slycot
        # function sg03ad
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=SlycotResultWarning)
            A, E, Q, Z, X, scale, sep, ferr, alphar, alphai, beta = \
                sg03ad('D', 'B', 'N', 'T', 'L', n,
                       A, E, eye(n, n), eye(n, n), -Q)

    # Invalid set of input parameters (C and E specified)
    else:
        raise ControlArgument("Invalid set of input parameters")

    return _ssmatrix(X)


#
# Riccati equation solvers care and dare
#

def care(A, B, Q, R=None, S=None, E=None, stabilizing=True, method=None,
         A_s="A", B_s="B", Q_s="Q", R_s="R", S_s="S", E_s="E"):
    """Solves the continuous-time algebraic Riccati equation

    X, L, G = care(A, B, Q, R=None) solves

        :math:`A^T X + X A - X B R^{-1} B^T X + Q = 0`

    where A and Q are square matrices of the same dimension. Further,
    Q and R are a symmetric matrices. If R is None, it is set to the
    identity matrix. The function returns the solution X, the gain
    matrix G = B^T X and the closed loop eigenvalues L, i.e., the
    eigenvalues of A - B G.

    X, L, G = care(A, B, Q, R, S, E) solves the generalized
    continuous-time algebraic Riccati equation

        :math:`A^T X E + E^T X A - (E^T X B + S) R^{-1} (B^T X E + S^T) + Q = 0`

    where A, Q and E are square matrices of the same dimension. Further, Q
    and R are symmetric matrices. If R is None, it is set to the identity
    matrix. The function returns the solution X, the gain matrix G = R^-1
    (B^T X E + S^T) and the closed loop eigenvalues L, i.e., the eigenvalues
    of A - B G , E.

    Parameters
    ----------
    A, B, Q : 2D array_like
        Input matrices for the Riccati equation
    R, S, E : 2D array_like, optional
        Input matrices for generalized Riccati equation
    method : str, optional
        Set the method used for computing the result.  Current methods are
        'slycot' and 'scipy'.  If set to None (default), try 'slycot' first
        and then 'scipy'.

    Returns
    -------
    X : 2D array (or matrix)
        Solution to the Ricatti equation
    L : 1D array
        Closed loop eigenvalues
    G : 2D array (or matrix)
        Gain matrix

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    """
    # Decide what method to use
    method = _slycot_or_scipy(method)

    # Reshape input arrays
    A = np.array(A, ndmin=2)
    B = np.array(B, ndmin=2)
    Q = np.array(Q, ndmin=2)
    R = np.eye(B.shape[1]) if R is None else np.array(R, ndmin=2)
    if S is not None:
        S = np.array(S, ndmin=2)
    if E is not None:
        E = np.array(E, ndmin=2)

    # Determine main dimensions
    n = A.shape[0]
    m = B.shape[1]

    # Check to make sure input matrices are the right shape and type
    _check_shape(A_s, A, n, n, square=True)
    _check_shape(B_s, B, n, m)
    _check_shape(Q_s, Q, n, n, square=True, symmetric=True)
    _check_shape(R_s, R, m, m, square=True, symmetric=True)

    # Solve the standard algebraic Riccati equation
    if S is None and E is None:
        # See if we should solve this using SciPy
        if method == 'scipy':
            if not stabilizing:
                raise ControlArgument(
                    "method='scipy' not valid when stabilizing is not True")

            X = sp.linalg.solve_continuous_are(A, B, Q, R)
            K = np.linalg.solve(R, B.T @ X)
            E, _ = np.linalg.eig(A - B @ K)
            return _ssmatrix(X), E, _ssmatrix(K)

        # Make sure we can import required slycot routines
        try:
            from slycot import sb02md
        except ImportError:
            raise ControlSlycot("Can't find slycot module 'sb02md'")

        try:
            from slycot import sb02mt
        except ImportError:
            raise ControlSlycot("Can't find slycot module 'sb02mt'")

        # Solve the standard algebraic Riccati equation by calling Slycot
        # functions sb02mt and sb02md
        A_b, B_b, Q_b, R_b, L_b, ipiv, oufact, G = sb02mt(n, m, B, R)

        sort = 'S' if stabilizing else 'U'
        X, rcond, w, S_o, U, A_inv = sb02md(n, A, G, Q, 'C', sort=sort)

        # Calculate the gain matrix G
        G = solve(R, B.T) @ X

        # Return the solution X, the closed-loop eigenvalues L and
        # the gain matrix G
        return _ssmatrix(X), w[:n], _ssmatrix(G)

    # Solve the generalized algebraic Riccati equation
    else:
        # Initialize optional matrices
        S = np.zeros((n, m)) if S is None else np.array(S, ndmin=2)
        E = np.eye(A.shape[0]) if E is None else np.array(E, ndmin=2)

        # Check to make sure input matrices are the right shape and type
        _check_shape(E_s, E, n, n, square=True)
        _check_shape(S_s, S, n, m)

        # See if we should solve this using SciPy
        if method == 'scipy':
            if not stabilizing:
                raise ControlArgument(
                    "method='scipy' not valid when stabilizing is not True")

            X = sp.linalg.solve_continuous_are(A, B, Q, R, s=S, e=E)
            K = np.linalg.solve(R, B.T @ X @ E + S.T)
            eigs, _ = sp.linalg.eig(A - B @ K, E)
            return _ssmatrix(X), eigs, _ssmatrix(K)

        # Make sure we can find the required slycot routine
        try:
            from slycot import sg02ad
        except ImportError:
            raise ControlSlycot("Can't find slycot module 'sg02ad'")

        # Solve the generalized algebraic Riccati equation by calling the
        # Slycot function sg02ad
        with warnings.catch_warnings():
            sort = 'S' if stabilizing else 'U'
            warnings.simplefilter("error", category=SlycotResultWarning)
            rcondu, X, alfar, alfai, beta, S_o, T, U, iwarn = \
                sg02ad('C', 'B', 'N', 'U', 'N', 'N', sort,
                       'R', n, m, 0, A, E, B, Q, R, S)

        # Calculate the closed-loop eigenvalues L
        L = np.array([(alfar[i] + alfai[i]*1j) / beta[i] for i in range(n)])

        # Calculate the gain matrix G
        G = solve(R, B.T @ X @ E + S.T)

        # Return the solution X, the closed-loop eigenvalues L and
        # the gain matrix G
        return _ssmatrix(X), L, _ssmatrix(G)

def dare(A, B, Q, R, S=None, E=None, stabilizing=True, method=None,
         A_s="A", B_s="B", Q_s="Q", R_s="R", S_s="S", E_s="E"):
    """Solves the discrete-time algebraic Riccati
    equation

    X, L, G = dare(A, B, Q, R) solves

        :math:`A^T X A - X - A^T X B (B^T X B + R)^{-1} B^T X A + Q = 0`

    where A and Q are square matrices of the same dimension. Further, Q
    is a symmetric matrix. The function returns the solution X, the gain
    matrix G = (B^T X B + R)^-1 B^T X A and the closed loop eigenvalues L,
    i.e., the eigenvalues of A - B G.

    X, L, G = dare(A, B, Q, R, S, E) solves the generalized discrete-time
    algebraic Riccati equation

        :math:`A^T X A - E^T X E - (A^T X B + S) (B^T X B + R)^{-1} (B^T X A + S^T) + Q = 0`

    where A, Q and E are square matrices of the same dimension. Further, Q
    and R are symmetric matrices. If R is None, it is set to the identity
    matrix.  The function returns the solution X, the gain matrix :math:`G =
    (B^T X B + R)^{-1} (B^T X A + S^T)` and the closed loop eigenvalues L,
    i.e., the (generalized) eigenvalues of A - B G (with respect to E, if
    specified).

    Parameters
    ----------
    A, B, Q : 2D arrays
        Input matrices for the Riccati equation
    R, S, E : 2D arrays, optional
        Input matrices for generalized Riccati equation
    method : str, optional
        Set the method used for computing the result.  Current methods are
        'slycot' and 'scipy'.  If set to None (default), try 'slycot' first
        and then 'scipy'.

    Returns
    -------
    X : 2D array (or matrix)
        Solution to the Ricatti equation
    L : 1D array
        Closed loop eigenvalues
    G : 2D array (or matrix)
        Gain matrix

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    """
    # Decide what method to use
    method = _slycot_or_scipy(method)

    # Reshape input arrays
    A = np.array(A, ndmin=2)
    B = np.array(B, ndmin=2)
    Q = np.array(Q, ndmin=2)
    R = np.eye(B.shape[1]) if R is None else np.array(R, ndmin=2)
    if S is not None:
        S = np.array(S, ndmin=2)
    if E is not None:
        E = np.array(E, ndmin=2)

    # Determine main dimensions
    n = A.shape[0]
    m = B.shape[1]

    # Check to make sure input matrices are the right shape and type
    _check_shape(A_s, A, n, n, square=True)
    _check_shape(B_s, B, n, m)
    _check_shape(Q_s, Q, n, n, square=True, symmetric=True)
    _check_shape(R_s, R, m, m, square=True, symmetric=True)
    if E is not None:
        _check_shape(E_s, E, n, n, square=True)
    if S is not None:
        _check_shape(S_s, S, n, m)

    # Figure out how to solve the problem
    if method == 'scipy':
        if not stabilizing:
            raise ControlArgument(
                "method='scipy' not valid when stabilizing is not True")

        X = sp.linalg.solve_discrete_are(A, B, Q, R, e=E, s=S)
        if S is None:
            G = solve(B.T @ X @ B + R, B.T @ X @ A)
        else:
            G = solve(B.T @ X @ B + R, B.T @ X @ A + S.T)
        if E is None:
            L = eigvals(A - B @ G)
        else:
            L, _ = sp.linalg.eig(A - B @ G, E)

        return _ssmatrix(X), L, _ssmatrix(G)

    # Make sure we can import required slycot routine
    try:
        from slycot import sg02ad
    except ImportError:
        raise ControlSlycot("Can't find slycot module 'sg02ad'")

    # Initialize optional matrices
    S = np.zeros((n, m)) if S is None else np.array(S, ndmin=2)
    E = np.eye(A.shape[0]) if E is None else np.array(E, ndmin=2)

    # Solve the generalized algebraic Riccati equation by calling the
    # Slycot function sg02ad
    sort = 'S' if stabilizing else 'U'
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=SlycotResultWarning)
        rcondu, X, alfar, alfai, beta, S_o, T, U, iwarn = \
            sg02ad('D', 'B', 'N', 'U', 'N', 'N', sort,
                   'R', n, m, 0, A, E, B, Q, R, S)

    # Calculate the closed-loop eigenvalues L
    L = np.array([(alfar[i] + alfai[i]*1j) / beta[i] for i in range(n)])

    # Calculate the gain matrix G
    G = solve(B.T @ X @ B + R, B.T @ X @ A + S.T)

    # Return the solution X, the closed-loop eigenvalues L and
    # the gain matrix G
    return _ssmatrix(X), L, _ssmatrix(G)


# Utility function to decide on method to use
def _slycot_or_scipy(method):
    if method == 'slycot' or (method is None and slycot_check()):
        return 'slycot'
    elif method == 'scipy' or (method is None and not slycot_check()):
        return 'scipy'
    else:
        raise ControlArgument("Unknown method %s" % method)


# Utility function to check matrix dimensions
def _check_shape(name, M, n, m, square=False, symmetric=False):
    if square and M.shape[0] != M.shape[1]:
        raise ControlDimension("%s must be a square matrix" % name)

    if symmetric and not _is_symmetric(M):
        raise ControlArgument("%s must be a symmetric matrix" % name)

    if M.shape[0] != n or M.shape[1] != m:
        raise ControlDimension("Incompatible dimensions of %s matrix" % name)


# Utility function to check if a matrix is symmetric
def _is_symmetric(M):
    M = np.atleast_2d(M)
    if isinstance(M[0, 0], inexact):
        eps = finfo(M.dtype).eps
        return ((M - M.T) < eps).all()
    else:
        return (M == M.T).all()
