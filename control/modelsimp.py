#! TODO: add module docstring
# modelsimp.py - tools for model simplification
#
# Author: Steve Brunton, Kevin Chen, Lauren Padilla
# Date: 30 Nov 2010
#
# This file contains routines for obtaining reduced order models
#
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
import warnings
from .exception import ControlSlycot, ControlMIMONotImplemented, \
    ControlDimension
from .namedio import isdtime, isctime
from .statesp import StateSpace
from .statefbk import gram

__all__ = ['hsvd', 'balred', 'modred', 'era', 'markov', 'minreal']


# Hankel Singular Value Decomposition
#
# The following returns the Hankel singular values, which are singular values
# of the matrix formed by multiplying the controllability and observability
# Gramians
def hsvd(sys):
    """Calculate the Hankel singular values.

    Parameters
    ----------
    sys : StateSpace
        A state space system

    Returns
    -------
    H : array
        A list of Hankel singular values

    See Also
    --------
    gram

    Notes
    -----
    The Hankel singular values are the singular values of the Hankel operator.
    In practice, we compute the square root of the eigenvalues of the matrix
    formed by taking the product of the observability and controllability
    gramians.  There are other (more efficient) methods based on solving the
    Lyapunov equation in a particular way (more details soon).

    Examples
    --------
    >>> G = ct.tf2ss([1], [1, 2])
    >>> H = ct.hsvd(G)
    >>> H[0]
    0.25

    """
    # TODO: implement for discrete time systems
    if (isdtime(sys, strict=True)):
        raise NotImplementedError("Function not implemented in discrete time")

    Wc = gram(sys, 'c')
    Wo = gram(sys, 'o')
    WoWc = Wo @ Wc
    w, v = np.linalg.eig(WoWc)

    hsv = np.sqrt(w)
    hsv = np.array(hsv)
    hsv = np.sort(hsv)
    # Return the Hankel singular values, high to low
    return hsv[::-1]


def modred(sys, ELIM, method='matchdc'):
    """
    Model reduction of `sys` by eliminating the states in `ELIM` using a given
    method.

    Parameters
    ----------
    sys: StateSpace
        Original system to reduce
    ELIM: array
        Vector of states to eliminate
    method: string
        Method of removing states in `ELIM`: either ``'truncate'`` or
        ``'matchdc'``.

    Returns
    -------
    rsys: StateSpace
        A reduced order model

    Raises
    ------
    ValueError
        Raised under the following conditions:

            * if `method` is not either ``'matchdc'`` or ``'truncate'``

            * if eigenvalues of `sys.A` are not all in left half plane
              (`sys` must be stable)

    Examples
    --------
    >>> G = ct.rss(4)
    >>> Gr = ct.modred(G, [0, 2], method='matchdc')
    >>> Gr.nstates
    2

    """

    # Check for ss system object, need a utility for this?

    # TODO: Check for continous or discrete, only continuous supported for now
    #   if isCont():
    #       dico = 'C'
    #   elif isDisc():
    #       dico = 'D'
    #   else:
    if (isctime(sys)):
        dico = 'C'
    else:
        raise NotImplementedError("Function not implemented in discrete time")

    # Check system is stable
    if np.any(np.linalg.eigvals(sys.A).real >= 0.0):
        raise ValueError("Oops, the system is unstable!")

    ELIM = np.sort(ELIM)
    # Create list of elements not to eliminate (NELIM)
    NELIM = [i for i in range(len(sys.A)) if i not in ELIM]
    # A1 is a matrix of all columns of sys.A not to eliminate
    A1 = sys.A[:, NELIM[0]].reshape(-1, 1)
    for i in NELIM[1:]:
        A1 = np.hstack((A1, sys.A[:, i].reshape(-1, 1)))
    A11 = A1[NELIM, :]
    A21 = A1[ELIM, :]
    # A2 is a matrix of all columns of sys.A to eliminate
    A2 = sys.A[:, ELIM[0]].reshape(-1, 1)
    for i in ELIM[1:]:
        A2 = np.hstack((A2, sys.A[:, i].reshape(-1, 1)))
    A12 = A2[NELIM, :]
    A22 = A2[ELIM, :]

    C1 = sys.C[:, NELIM]
    C2 = sys.C[:, ELIM]
    B1 = sys.B[NELIM, :]
    B2 = sys.B[ELIM, :]

    if method == 'matchdc':
        # if matchdc, residualize

        # Check if the matrix A22 is invertible
        if np.linalg.matrix_rank(A22) != len(ELIM):
            raise ValueError("Matrix A22 is singular to working precision.")

        # Now precompute A22\A21 and A22\B2 (A22I = inv(A22))
        # We can solve two linear systems in one pass, since the
        # coefficients matrix A22 is the same. Thus, we perform the LU
        # decomposition (cubic runtime complexity) of A22 only once!
        # The remaining back substitutions are only quadratic in runtime.
        A22I_A21_B2 = np.linalg.solve(A22, np.concatenate((A21, B2), axis=1))
        A22I_A21 = A22I_A21_B2[:, :A21.shape[1]]
        A22I_B2 = A22I_A21_B2[:, A21.shape[1]:]

        Ar = A11 - A12 @ A22I_A21
        Br = B1 - A12 @ A22I_B2
        Cr = C1 - C2 @ A22I_A21
        Dr = sys.D - C2 @ A22I_B2
    elif method == 'truncate':
        # if truncate, simply discard state x2
        Ar = A11
        Br = B1
        Cr = C1
        Dr = sys.D
    else:
        raise ValueError("Oops, method is not supported!")

    rsys = StateSpace(Ar, Br, Cr, Dr)
    return rsys


def balred(sys, orders, method='truncate', alpha=None):
    """Balanced reduced order model of sys of a given order.
    States are eliminated based on Hankel singular value.
    If sys has unstable modes, they are removed, the
    balanced realization is done on the stable part, then
    reinserted in accordance with the reference below.

    Reference: Hsu,C.S., and Hou,D., 1991,
    Reducing unstable linear control systems via real Schur transformation.
    Electronics Letters, 27, 984-986.

    Parameters
    ----------
    sys: StateSpace
        Original system to reduce
    orders: integer or array of integer
        Desired order of reduced order model (if a vector, returns a vector
        of systems)
    method: string
        Method of removing states, either ``'truncate'`` or ``'matchdc'``.
    alpha: float
        Redefines the stability boundary for eigenvalues of the system
        matrix A.  By default for continuous-time systems, alpha <= 0
        defines the stability boundary for the real part of A's eigenvalues
        and for discrete-time systems, 0 <= alpha <= 1 defines the stability
        boundary for the modulus of A's eigenvalues. See SLICOT routines
        AB09MD and AB09ND for more information.

    Returns
    -------
    rsys: StateSpace
        A reduced order model or a list of reduced order models if orders is
        a list.

    Raises
    ------
    ValueError
        If `method` is not ``'truncate'`` or ``'matchdc'``
    ImportError
        if slycot routine ab09ad, ab09md, or ab09nd is not found

    ValueError
        if there are more unstable modes than any value in orders

    Examples
    --------
    >>> G = ct.rss(4)
    >>> Gr = ct.balred(G, orders=2, method='matchdc')
    >>> Gr.nstates
    2

    """
    if method != 'truncate' and method != 'matchdc':
        raise ValueError("supported methods are 'truncate' or 'matchdc'")
    elif method == 'truncate':
        try:
            from slycot import ab09md, ab09ad
        except ImportError:
            raise ControlSlycot(
                "can't find slycot subroutine ab09md or ab09ad")
    elif method == 'matchdc':
        try:
            from slycot import ab09nd
        except ImportError:
            raise ControlSlycot("can't find slycot subroutine ab09nd")

    # Check for ss system object, need a utility for this?

    # TODO: Check for continous or discrete, only continuous supported for now
    #   if isCont():
    #       dico = 'C'
    #   elif isDisc():
    #       dico = 'D'
    #   else:
    dico = 'C'

    job = 'B'                   # balanced (B) or not (N)
    equil = 'N'                 # scale (S) or not (N)
    if alpha is None:
        if dico == 'C':
            alpha = 0.
        elif dico == 'D':
            alpha = 1.

    rsys = []                   # empty list for reduced systems

    # check if orders is a list or a scalar
    try:
        order = iter(orders)
    except TypeError:           # if orders is a scalar
        orders = [orders]

    for i in orders:
        n = np.size(sys.A, 0)
        m = np.size(sys.B, 1)
        p = np.size(sys.C, 0)
        if method == 'truncate':
            # check system stability
            if np.any(np.linalg.eigvals(sys.A).real >= 0.0):
                # unstable branch
                Nr, Ar, Br, Cr, Ns, hsv = ab09md(
                    dico, job, equil, n, m, p, sys.A, sys.B, sys.C,
                    alpha=alpha, nr=i, tol=0.0)
            else:
                # stable branch
                Nr, Ar, Br, Cr, hsv = ab09ad(
                    dico, job, equil, n, m, p, sys.A, sys.B, sys.C,
                    nr=i, tol=0.0)
            rsys.append(StateSpace(Ar, Br, Cr, sys.D))

        elif method == 'matchdc':
            Nr, Ar, Br, Cr, Dr, Ns, hsv = ab09nd(
                dico, job, equil, n, m, p, sys.A, sys.B, sys.C, sys.D,
                alpha=alpha, nr=i, tol1=0.0, tol2=0.0)
            rsys.append(StateSpace(Ar, Br, Cr, Dr))

    # if orders was a scalar, just return the single reduced model, not a list
    if len(orders) == 1:
        return rsys[0]
    # if orders was a list/vector, return a list/vector of systems
    else:
        return rsys


def minreal(sys, tol=None, verbose=True):
    '''
    Eliminates uncontrollable or unobservable states in state-space
    models or cancelling pole-zero pairs in transfer functions. The
    output sysr has minimal order and the same response
    characteristics as the original model sys.

    Parameters
    ----------
    sys: StateSpace or TransferFunction
        Original system
    tol: real
        Tolerance
    verbose: bool
        Print results if True

    Returns
    -------
    rsys: StateSpace or TransferFunction
        Cleaned model
    '''
    sysr = sys.minreal(tol)
    if verbose:
        print("{nstates} states have been removed from the model".format(
                nstates=len(sys.poles()) - len(sysr.poles())))
    return sysr


def era(YY, m, n, nin, nout, r):
    """Calculate an ERA model of order `r` based on the impulse-response data
    `YY`.

    .. note:: This function is not implemented yet.

    Parameters
    ----------
    YY: array
        `nout` x `nin` dimensional impulse-response data
    m: integer
        Number of rows in Hankel matrix
    n: integer
        Number of columns in Hankel matrix
    nin: integer
        Number of input variables
    nout: integer
        Number of output variables
    r: integer
        Order of model

    Returns
    -------
    sys: StateSpace
        A reduced order model sys=ss(Ar,Br,Cr,Dr)

    Examples
    --------
    >>> rsys = era(YY, m, n, nin, nout, r)                      # doctest: +SKIP

    """
    raise NotImplementedError('This function is not implemented yet.')


def markov(Y, U, m=None, transpose=False):
    """Calculate the first `m` Markov parameters [D CB CAB ...]
    from input `U`, output `Y`.

    This function computes the Markov parameters for a discrete time system

    .. math::

        x[k+1] &= A x[k] + B u[k] \\\\
        y[k] &= C x[k] + D u[k]

    given data for u and y.  The algorithm assumes that that C A^k B = 0 for
    k > m-2 (see [1]_).  Note that the problem is ill-posed if the length of
    the input data is less than the desired number of Markov parameters (a
    warning message is generated in this case).

    Parameters
    ----------
    Y : array_like
        Output data.  If the array is 1D, the system is assumed to be single
        input.  If the array is 2D and transpose=False, the columns of `Y`
        are taken as time points, otherwise the rows of `Y` are taken as
        time points.
    U : array_like
        Input data, arranged in the same way as `Y`.
    m : int, optional
        Number of Markov parameters to output.  Defaults to len(U).
    transpose : bool, optional
        Assume that input data is transposed relative to the standard
        :ref:`time-series-convention`. Default value is False.

    Returns
    -------
    H : ndarray
        First m Markov parameters, [D CB CAB ...]

    References
    ----------
    .. [1] J.-N. Juang, M. Phan, L. G.  Horta, and R. W. Longman,
       Identification of observer/Kalman filter Markov parameters - Theory
       and experiments. Journal of Guidance Control and Dynamics, 16(2),
       320-329, 2012. http://doi.org/10.2514/3.21006

    Notes
    -----
    Currently only works for SISO systems.

    This function does not currently comply with the Python Control Library
    :ref:`time-series-convention` for representation of time series data.
    Use `transpose=False` to make use of the standard convention (this
    will be updated in a future release).

    Examples
    --------
    >>> T = np.linspace(0, 10, 100)
    >>> U = np.ones((1, 100))
    >>> T, Y = ct.forced_response(ct.tf([1], [1, 0.5], True), T, U)
    >>> H = ct.markov(Y, U, 3, transpose=False)

    """
    # Convert input parameters to 2D arrays (if they aren't already)
    Umat = np.array(U, ndmin=2)
    Ymat = np.array(Y, ndmin=2)

    # If data is in transposed format, switch it around
    if transpose:
        Umat, Ymat = np.transpose(Umat), np.transpose(Ymat)

    # Make sure the system is a SISO system
    if Umat.shape[0] != 1 or Ymat.shape[0] != 1:
        raise ControlMIMONotImplemented

    # Make sure the number of time points match
    if Umat.shape[1] != Ymat.shape[1]:
        raise ControlDimension(
            "Input and output data are of differnent lengths")
    n = Umat.shape[1]

    # If number of desired parameters was not given, set to size of input data
    if m is None:
        m = Umat.shape[1]

    # Make sure there is enough data to compute parameters
    if m > n:
        warnings.warn("Not enough data for requested number of parameters")

    #
    # Original algorithm (with mapping to standard order)
    #
    # RMM note, 24 Dec 2020: This algorithm sets the problem up correctly
    # until the final column of the UU matrix is created, at which point it
    # makes some modifications that I don't understand. This version of the
    # algorithm does not seem to return the actual Markov parameters for a
    # system.
    #
    # # Create the matrix of (shifted) inputs
    # UU = np.transpose(Umat)
    # for i in range(1, m-1):
    #     # Shift previous column down and add a zero at the top
    #     newCol = np.vstack((0, np.reshape(UU[0:n-1, i-1], (-1, 1))))
    #     UU = np.hstack((UU, newCol))
    #
    # # Shift previous column down and add a zero at the top
    # Ulast = np.vstack((0, np.reshape(UU[0:n-1, m-2], (-1, 1))))
    #
    # # Replace the elements of the last column new values (?)
    # # Each row gets the sum of the rows above it (?)
    # for i in range(n-1, 0, -1):
    #     Ulast[i] = np.sum(Ulast[0:i-1])
    # UU = np.hstack((UU, Ulast))
    #
    # # Solve for the Markov parameters from Y = H @ UU
    # # H = [[D], [CB], [CAB], ..., [C A^{m-3} B], [???]]
    # H = np.linalg.lstsq(UU, np.transpose(Ymat))[0]
    #
    # # Markov parameters are in rows => transpose if needed
    # return H if transpose else np.transpose(H)

    #
    # New algorithm - Construct a matrix of control inputs to invert
    #
    # This algorithm sets up the following problem and solves it for
    # the Markov parameters
    #
    # [ y(0)   ]   [ u(0)    0       0                 ] [ D           ]
    # [ y(1)   ]   [ u(1)    u(0)    0                 ] [ C B         ]
    # [ y(2)   ] = [ u(2)    u(1)    u(0)              ] [ C A B       ]
    # [  :     ]   [  :      :        :          :     ] [  :          ]
    # [ y(n-1) ]   [ u(n-1)  u(n-2)  u(n-3) ... u(n-m) ] [ C A^{m-2} B ]
    #
    # Note: if the number of Markov parameters (m) is less than the size of
    # the input/output data (n), then this algorithm assumes C A^{j} B = 0
    # for j > m-2.  See equation (3) in
    #
    #   J.-N. Juang, M. Phan, L. G. Horta, and R. W. Longman, Identification
    #   of observer/Kalman filter Markov parameters - Theory and
    #   experiments. Journal of Guidance Control and Dynamics, 16(2),
    #   320-329, 2012. http://doi.org/10.2514/3.21006
    #

    # Create matrix of (shifted) inputs
    UU = Umat
    for i in range(1, m):
        # Shift previous column down and add a zero at the top
        new_row = np.hstack((0, UU[i-1, 0:-1]))
        UU = np.vstack((UU, new_row))
    UU = np.transpose(UU)

    # Invert and solve for Markov parameters
    YY = np.transpose(Ymat)
    H, _, _, _ = np.linalg.lstsq(UU, YY, rcond=None)

    # Return the first m Markov parameters
    return H if transpose else np.transpose(H)
