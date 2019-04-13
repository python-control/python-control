""" statefbk.py

This file contains routines for designing state space controllers through state feedback control.
"""

__author__ = "Richard M. Murray"
__contributor__ = "Roberto Bucher"

# Copyright (c) 2010 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
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
#
# $Id$

# External packages and modules
import numpy as np
import scipy as sp
from . import statesp
from .exception import ControlSlycot, ControlArgument, ControlDimension
import warnings

__all__ = ['ctrb', 'obsv', 'gram', 'place', 'lqr', 'place_varga', 'acker']


# Pole placement
def place(A, B, p, method="YT", dtime=False, alpha=None):
    """Place closed loop eigenvalues
    K = place(A, B, p)

    Parameters
    ----------
    A : 2-d array
        Dynamics matrix
    B : 2-d array
        Input matrix
    p : 1-d list
        Desired eigenvalue locations
    method : string - optional
        Method used for pole placement: {"YT", "KNV0", "varga", "acker"}
    dtime: optional (useful only if method=="varga")
        False for continuous time pole placement or True for discrete time.
        The default is dtime=False.
        If dtime is not null, place_varga will leave the eigenvalues with modulus
        less than alpha untouched. Otherwise, place_varga will leave the eigenvalues with real
        real part less than alpha untouched.
    alpha: double scalar (useful only if method=="varga")

       By default (alpha=None), place_varga computes alpha such that all
       poles will be placed.

    Returns
    -------
    K : 2-d array
        Gain such that A - B K has eigenvalues given in p

    Algorithm
    ---------
    This is a wrapper function for scipy.signal.place_poles, which
    implements the Tits and Yang algorithm [1]. It will handle SISO,
    MISO, and MIMO systems. If you want more control over the algorithm,
    use scipy.signal.place_poles directly.

    [1] A.L. Tits and Y. Yang, "Globally convergent algorithms for robust
    pole assignment by state feedback, IEEE Transactions on Automatic
    Control, Vol. 41, pp. 1432-1452, 1996.

    Limitations
    -----------
    The algorithm will not place poles at the same location more
    than rank(B) times.

    Examples
    --------
    >>> A = [[-1, -1], [0, 1]]
    >>> B = [[0], [1]]
    >>> K = place(A, B, [-2, -5])

    See Also
    --------
    acker, place_varga
    """
    if method == "acker":
        return acker(A, B, p)
    if method == "varga":
        try:
            return place_varga(A, B, p, dtime=dtime, alpha=alpha)
        except ControlSlycot as error:
            warnings.warn("[Pole placement] Fallback strategy: using Tits-Yang method from scipy.")
            return place(A, B, p)
    else:
        try:
            from scipy.signal import place_poles
        except ImportError as error:
            warnings.warn(error)
            warnings.warn("[Pole placement] Fallback strategy: using Varga method from slycot.")
            try:
                K = place_varga(A, B, p, dtime=dtime, alpha=alpha)
                return K
            except ControlSlycot as error:
                warnings.warn("[Pole placement] Fallback strategy: using Ackermanm method.")
                return acker(A, B, p)

        # Convert the system inputs to NumPy arrays
        A_mat = np.array(A)
        B_mat = np.array(B)
        if A_mat.shape[0] != A_mat.shape[1]:
            raise ControlDimension("A must be a square matrix. "
                                   "rows: {} != columns: {}".format(A_mat.shape[0],
                                                                    A_mat.shape[1]))

        if A_mat.shape[0] != B_mat.shape[0]:
            raise ControlDimension("The number of rows of A ({}) must equal "
                                   "the number of rows in B ({})".format(A_mat.shape[0],
                                                                         B_mat.shape[0]))

        # Convert desired poles to numpy array
        placed_eigs = np.array(p)

        try:
            result = place_poles(A_mat, B_mat, placed_eigs, method=method)
        except ValueError as error:
            warnings.warn("[Pole placement] Redundant pole location. "
                  "Fallback strategy: using Ackermann method.")
            try:
                return acker(A, B, p)
            except ValueError as error:
                warnings.warn("Pole placement failed.")
                return None
        K = result.gain_matrix
        return K


def place_varga(A, B, p, dtime=False, alpha=None):
    """Place closed loop eigenvalues
    K = place_varga(A, B, p, dtime=False, alpha=None)

    Required Parameters
    ----------
    A : 2-d array
        Dynamics matrix
    B : 2-d array
        Input matrix
    p : 1-d list
        Desired eigenvalue locations

    Optional Parameters
    ---------------
    dtime: False for continuous time pole placement or True for discrete time.
            The default is dtime=False.
            If dtime is not null, place_varga will leave the eigenvalues with modulus
            less than alpha untouched. Otherwise, place_varga will leave the eigenvalues with real
            real part less than alpha untouched.
    alpha: double scalar

           By default (alpha=None), place_varga computes alpha such that all
           poles will be placed.

    Returns
    -------
    K : 2-d array
        Gain such that A - B K has eigenvalues given in p.


    Algorithm
    ---------
        This function is a wrapper for the slycot function sb01bd, which
        implements the pole placement algorithm of Varga [1]. In contrast to
        the algorithm used by place(), the Varga algorithm can place
        multiple poles at the same location. The placement, however, may not
        be as robust.

        [1] Varga A. "A Schur method for pole assignment."
            IEEE Trans. Automatic Control, Vol. AC-26, pp. 517-519, 1981.

    Examples
    --------
    >>> A = [[-1, -1], [0, 1]]
    >>> B = [[0], [1]]
    >>> K = place_varga(A, B, [-2, -5])

    See Also:
    --------
    place, acker
    """

    # Make sure that SLICOT is installed
    try:
        from slycot import sb01bd
    except ImportError as error:
        raise ControlSlycot(error)

    # Convert the system inputs to NumPy arrays
    A_mat = np.array(A)
    B_mat = np.array(B)
    if A_mat.shape[0] != A_mat.shape[1] or A_mat.shape[0] != B_mat.shape[0]:
        raise ControlDimension("Matrix dimensions are incorrect.")

    # Compute the system eigenvalues and convert poles to numpy array
    system_eigs = np.linalg.eig(A_mat)[0]
    placed_eigs = np.array(p)

    # Need a character parameter for SB01BD
    if dtime:
        dico = 'D'
    else:
        dico = 'C'

    if alpha is None:
        # SB01BD ignores eigenvalues with real part less than alpha
        # (if dico = 'C') or with modulus less than alpha
        # (if dico = 'D').
        if dtime:
            # For discrete time, slycot only cares about modulus, so just make
            # alpha the smallest it can be.
            alpha = 0.0
        else:
            # Choosing alpha=min_eig is insufficient and can lead to an
            # error or not having all the eigenvalues placed that we wanted.
            # Evidently, what python thinks are the eigs is not precisely
            # the same as what slicot thinks are the eigs. So we need some
            # numerical breathing room. The following is pretty heuristic,
            # but does the trick
            alpha = -2 * abs(min(system_eigs.real))
    elif dtime and alpha < 0.0:
        raise ValueError("Need alpha > 0 when dico='D'.")

    # Call SLICOT routine to place the eigenvalues
    A_z, w, nfp, nap, nup, F, Z = \
        sb01bd(B_mat.shape[0], B_mat.shape[1], len(placed_eigs), alpha,
               A_mat, B_mat, placed_eigs, dico)

    # Return the gain matrix, with MATLAB gain convention
    return -F


# Contributed by Roberto Bucher <roberto.bucher@supsi.ch>
def acker(A, B, poles):
    """Pole placement using Ackermann method

    Call:
    K = acker(A, B, poles)

    Parameters
    ----------
    A, B : 2-d arrays
        State and input matrix of the system
    poles: 1-d list
        Desired eigenvalue locations

    Returns
    -------
    K: matrix
        Gains such that A - B K has given eigenvalues

    See Also:
    --------
    place, place_varga
    """

    # Convert the inputs to matrices
    a = np.mat(A)
    b = np.mat(B)

    # Make sure the system is controllable
    ct = ctrb(A, B)
    if np.linalg.matrix_rank(ct) != a.shape[0]:
        raise ValueError("System is not reachable, pole placement is invalid.")

    # Compute the desired characteristic polynomial
    p = np.real(np.poly(poles))

    # Place the poles using Ackermann's method
    n = np.size(p)
    pmat = p[n - 1] * a ** 0
    for i in np.arange(1, n):
        pmat = pmat + p[n - i - 1] * a ** i
    K = np.linalg.solve(ct, pmat)

    K = K[-1][:]  # Extract the last row
    return K


def lqr(*args):
    """lqr(A, B, Q, R[, N])

    Linear quadratic regulator design

    The lqr() function computes the optimal state feedback controller
    that minimizes the quadratic cost

    .. math:: J = \int_0^\infty (x' Q x + u' R u + 2 x' N u) dt

    The function can be called with either 3, 4, or 5 arguments:

    * ``lqr(sys, Q, R)``
    * ``lqr(sys, Q, R, N)``
    * ``lqr(A, B, Q, R)``
    * ``lqr(A, B, Q, R, N)``

    where `sys` is an `LTI` object, and `A`, `B`, `Q`, `R`, and `N` are
    2d arrays or matrices of appropriate dimension.

    Parameters
    ----------
    A, B: 2-d array
        Dynamics and input matrices
    sys: LTI (StateSpace or TransferFunction)
        Linear I/O system
    Q, R: 2-d array
        State and input weight matrices
    N: 2-d array, optional
        Cross weight matrix

    Returns
    -------
    K: 2-d array
        State feedback gains
    S: 2-d array
        Solution to Riccati equation
    E: 1-d array
        Eigenvalues of the closed loop system

    Examples
    --------
    >>> K, S, E = lqr(sys, Q, R, [N])
    >>> K, S, E = lqr(A, B, Q, R, [N])

    """

    # Make sure that SLICOT is installed
    try:
        from slycot import sb02md
        from slycot import sb02mt
    except ImportError:
        raise ControlSlycot("Can't find slycot module 'sb02md' or 'sb02nt'.")

    #
    # Process the arguments and figure out what inputs we received
    #

    # Get the system description
    if len(args) < 3:
        raise ControlArgument("Not enough input arguments.")

    try:
        # If this works, we were (probably) passed a system as the
        # first argument; extract A and B
        A = np.array(args[0].A, ndmin=2, dtype=float)
        B = np.array(args[0].B, ndmin=2, dtype=float)
        index = 1
    except AttributeError:
        # Arguments should be A and B matrices
        A = np.array(args[0], ndmin=2, dtype=float)
        B = np.array(args[1], ndmin=2, dtype=float)
        index = 2

    # Get the weighting matrices (converting to matrices, if needed)
    Q = np.array(args[index], ndmin=2, dtype=float)
    R = np.array(args[index + 1], ndmin=2, dtype=float)
    if len(args) > index + 2:
        N = np.array(args[index + 2], ndmin=2, dtype=float)
    else:
        N = np.zeros((Q.shape[0], R.shape[1]))

    # Check dimensions for consistency
    nstates = B.shape[0]
    ninputs = B.shape[1]
    # TODO: Create / Figure out a generic function to check for inconsistent system
    if A.shape[0] != nstates or A.shape[1] != nstates:
        raise ControlDimension("Inconsistent system dimensions.")

    elif (Q.shape[0] != nstates or Q.shape[1] != nstates or
          R.shape[0] != ninputs or R.shape[1] != ninputs or
          N.shape[0] != nstates or N.shape[1] != ninputs):
        raise ControlDimension("Incorrect weighting matrix dimensions.")

    # Compute the G matrix required by SB02MD
    A_b, B_b, Q_b, R_b, L_b, ipiv, oufact, G = sb02mt(nstates, ninputs, B, R, A, Q, N, jobl='N')

    # Call the SLICOT function
    X, rcond, w, S, U, A_inv = sb02md(nstates, A_b, G, Q_b, 'C')

    # Now compute the return value
    # We assume that R is positive definite and, hence, invertible
    K = np.linalg.solve(R, np.dot(B.T, X) + N.T)
    S = X
    E = w[0:nstates]

    return K, S, E


def ctrb(A, B):
    """Controllabilty matrix

    Parameters
    ----------
    A, B: array_like or string
        Dynamics and input matrix of the system

    Returns
    -------
    C: matrix
        Controllability matrix

    Examples
    --------
    >>> C = ctrb(A, B)

    """

    # Convert input parameters to matrices (if they aren't already)
    amat = np.mat(A)
    bmat = np.mat(B)
    n = np.shape(amat)[0]
    # Construct the controllability matrix
    C = bmat
    for i in range(1, n):
        C = np.hstack((C, amat ** i * bmat))
    return C


def obsv(A, C):
    """Observability matrix

    Parameters
    ----------
    A, C: array_like or string
        Dynamics and output matrix of the system

    Returns
    -------
    O: matrix
        Observability matrix

    Examples
    --------
    >>> O = obsv(A, C)

   """

    # Convert input parameters to matrices (if they aren't already)
    amat = np.mat(A)
    cmat = np.mat(C)
    n = np.shape(amat)[0]

    # Construct the controllability matrix
    O = cmat
    for i in range(1, n):
        O = np.vstack((O, cmat * amat ** i))
    return O


def gram(sys, gramian_type):
    """Gramian (controllability or observability)

    Parameters
    ----------
    sys: StateSpace
        State-space system to compute Gramian for
    gramian_type: String
        Type of desired computation.
        `gramian_type` is either 'c' (controllability) or 'o' (observability).
        To compute the Cholesky factors of gramians use 'cf' (controllability)
        or 'of' (observability)

    Returns
    -------
    gram: array
        Gramian of system

    Raises
    ------
    ValueError
        * if system is not instance of StateSpace class
        * if `gramian_type` is not 'c', 'o', 'cf' or 'of'
        * if system is unstable (sys.A has eigenvalues not in left half plane)

    ImportError
        if slycot routine sb03md cannot be found
        if slycot routine sb03od cannot be found

    Examples
    --------
    >>> Wc = gram(sys, 'c')
    >>> Wo = gram(sys, 'o')
    >>> Rc = gram(sys, 'cf')  #, where Wc = Rc' * Rc
    >>> Ro = gram(sys, 'of')  #, where Wo = Ro' * Ro

    """

    # Check for ss system object
    if not isinstance(sys, statesp.StateSpace):
        raise ValueError("System must be StateSpace!")
    if gramian_type not in ['c', 'o', 'cf', 'of']:
        raise ValueError("That type is not supported!")

    # TODO: Check for continuous or discrete, only continuous supported right now
    # if isCont():
    #    dico = 'C'
    # elif isDisc():
    #    dico = 'D'
    # else:
    dico = 'C'

    # TODO: Check system is stable, perhaps a utility in ctrlutil.py
    # or a method of the StateSpace class?
    if np.any(np.linalg.eigvals(sys.A).real >= 0.0):
        raise ValueError("The system is unstable.")

    if gramian_type == 'c' or gramian_type == 'o':
        # Compute Gramian by the Slycot routine sb03md
        # make sure Slycot is installed
        try:
            from slycot import sb03md
        except ImportError:
            raise ControlSlycot("Can't find slycot module 'sb03md'.")
        if gramian_type == 'c':
            tra = 'T'
            C = -np.dot(sys.B, sys.B.transpose())
        elif gramian_type == 'o':
            tra = 'N'
            C = -np.dot(sys.C.transpose(), sys.C)
        n = sys.states
        U = np.zeros((n, n))
        A = np.array(sys.A)  # convert to NumPy array for slycot
        X, scale, sep, ferr, w = sb03md(n, C, A, U, dico, job='X', fact='N', trana=tra)
        gram = X
        return gram

    elif gramian_type == 'cf' or gramian_type == 'of':
        # Compute cholesky factored gramian from slycot routine sb03od
        try:
            from slycot import sb03od
        except ImportError:
            raise ControlSlycot("Can't find slycot module 'sb03od'.")
        tra = 'N'
        n = sys.states
        Q = np.zeros((n, n))
        A = np.array(sys.A)  # convert to NumPy array for slycot
        if gramian_type == 'cf':
            m = sys.B.shape[1]
            B = np.zeros_like(A)
            B[0:m, 0:n] = sys.B.transpose()
            X, scale, w = sb03od(n, m, A.transpose(), Q, B, dico, fact='N', trans=tra)
        elif gramian_type == 'of':
            m = sys.C.shape[0]
            C = np.zeros_like(A)
            C[0:n, 0:m] = sys.C.transpose()
            X, scale, w = sb03od(n, m, A, Q, C.transpose(), dico, fact='N', trans=tra)
        gram = X
        return gram
