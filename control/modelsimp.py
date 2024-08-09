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
from .exception import ControlSlycot, ControlArgument, ControlDimension
from .iosys import isdtime, isctime
from .statesp import StateSpace
from .statefbk import gram
from .timeresp import TimeResponseData

__all__ = ['hankel_singular_values', 'balanced_reduction', 'model_reduction',
           'minimal_realization', 'eigensys_realization', 'markov', 'hsvd',
           'balred', 'modred', 'minreal', 'era']


# Hankel Singular Value Decomposition
#
# The following returns the Hankel singular values, which are singular values
# of the matrix formed by multiplying the controllability and observability
# Gramians
def hankel_singular_values(sys):
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
    np.float64(0.25)

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


def model_reduction(sys, ELIM, method='matchdc'):
    """Model reduction by state elimination.
    
    Model reduction of `sys` by eliminating the states in `ELIM` using a given
    method.

    Parameters
    ----------
    sys : StateSpace
        Original system to reduce.
    ELIM : array
        Vector of states to eliminate.
    method : string
        Method of removing states in `ELIM`: either 'truncate' or
        'matchdc'.

    Returns
    -------
    rsys : StateSpace
        A reduced order model.

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


def balanced_reduction(sys, orders, method='truncate', alpha=None):
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
    sys : StateSpace
        Original system to reduce.
    orders : integer or array of integer
        Desired order of reduced order model (if a vector, returns a vector
        of systems).
    method : string
        Method of removing states, either ``'truncate'`` or ``'matchdc'``..
    alpha : float
        Redefines the stability boundary for eigenvalues of the system
        matrix A.  By default for continuous-time systems, alpha <= 0
        defines the stability boundary for the real part of A's eigenvalues
        and for discrete-time systems, 0 <= alpha <= 1 defines the stability
        boundary for the modulus of A's eigenvalues. See SLICOT routines
        AB09MD and AB09ND for more information.

    Returns
    -------
    rsys : StateSpace
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


def minimal_realization(sys, tol=None, verbose=True):
    """ Eliminate uncontrollable or unobservable states.
    
    Eliminates uncontrollable or unobservable states in state-space
    models or cancelling pole-zero pairs in transfer functions. The
    output sysr has minimal order and the same response
    characteristics as the original model sys.

    Parameters
    ----------
    sys : StateSpace or TransferFunction
        Original system.
    tol : real
        Tolerance.
    verbose : bool
        Print results if True.

    Returns
    -------
    rsys : StateSpace or TransferFunction
        Cleaned model.

    """
    sysr = sys.minreal(tol)
    if verbose:
        print("{nstates} states have been removed from the model".format(
                nstates=len(sys.poles()) - len(sysr.poles())))
    return sysr


def _block_hankel(Y, m, n):
    """Create a block Hankel matrix from impulse response."""
    q, p, _ = Y.shape
    YY = Y.transpose(0,2,1) # transpose for reshape
    
    H = np.zeros((q*m,p*n))
    
    for r in range(m):
        # shift and add row to Hankel matrix
        new_row = YY[:,r:r+n,:]
        H[q*r:q*(r+1),:] = new_row.reshape((q,p*n))
            
    return H


def eigensys_realization(arg, r, m=None, n=None, dt=True, transpose=False):
    r"""eigensys_realization(YY, r)

    Calculate ERA model of order `r` based on impulse-response data `YY`.

    This function computes a discrete time system

    .. math::

        x[k+1] &= A x[k] + B u[k] \\\\
        y[k] &= C x[k] + D u[k]

    for a given impulse-response data (see [1]_).

    The function can be called with 2 arguments:

    * ``sysd, S = eigensys_realization(data, r)``
    * ``sysd, S = eigensys_realization(YY, r)``

    where `data` is a `TimeResponseData` object, `YY` is a 1D or 3D
    array, and r is an integer.

    Parameters
    ----------
    YY : array_like
        Impulse response from which the StateSpace model is estimated, 1D
        or 3D array.
    data : TimeResponseData
        Impulse response from which the StateSpace model is estimated.
    r : integer
        Order of model.
    m : integer, optional
        Number of rows in Hankel matrix. Default is 2*r.
    n : integer, optional
        Number of columns in Hankel matrix. Default is 2*r.
    dt : True or float, optional
        True indicates discrete time with unspecified sampling time and a
        positive float is discrete time with the specified sampling time.
        It can be used to scale the StateSpace model in order to match the
        unit-area impulse response of python-control. Default is True.
    transpose : bool, optional
        Assume that input data is transposed relative to the standard
        :ref:`time-series-convention`. For TimeResponseData this parameter 
        is ignored. Default is False.

    Returns
    -------
    sys : StateSpace
        A reduced order model sys=StateSpace(Ar,Br,Cr,Dr,dt).
    S : array
        Singular values of Hankel matrix. Can be used to choose a good r
        value.

    References
    ----------
    .. [1] Samet Oymak and Necmiye Ozay, Non-asymptotic Identification of
       LTI Systems from a Single Trajectory.
       https://arxiv.org/abs/1806.05722

    Examples
    --------
    >>> T = np.linspace(0, 10, 100)
    >>> _, YY = ct.impulse_response(ct.tf([1], [1, 0.5], True), T)
    >>> sysd, _ = ct.eigensys_realization(YY, r=1)

    >>> T = np.linspace(0, 10, 100)
    >>> response = ct.impulse_response(ct.tf([1], [1, 0.5], True), T)
    >>> sysd, _ = ct.eigensys_realization(response, r=1)
    """
    if isinstance(arg, TimeResponseData):
        YY = np.array(arg.outputs, ndmin=3)
        if arg.transpose:
            YY = np.transpose(YY)
    else:
        YY = np.array(arg, ndmin=3)
        if transpose:
            YY = np.transpose(YY)
    
    q, p, l = YY.shape

    if m is None:
        m = 2*r
    if n is None:
        n = 2*r

    if m*q < r or n*p < r:
        raise ValueError("Hankel parameters are to small")
    
    if (l-1) < m+n:
        raise ValueError("not enough data for requested number of parameters")
    
    H = _block_hankel(YY[:,:,1:], m, n+1) # Hankel matrix (q*m, p*(n+1))
    Hf = H[:,:-p] # first p*n columns of H
    Hl = H[:,p:] # last p*n columns of H
    
    U,S,Vh = np.linalg.svd(Hf, True)
    Ur =U[:,0:r]
    Vhr =Vh[0:r,:]

    # balanced realizations
    Sigma_inv = np.diag(1./np.sqrt(S[0:r]))
    Ar = Sigma_inv @ Ur.T @ Hl @ Vhr.T @ Sigma_inv
    Br = Sigma_inv @ Ur.T @ Hf[:,0:p]*dt # dt scaling for unit-area impulse
    Cr = Hf[0:q,:] @ Vhr.T @ Sigma_inv
    Dr = YY[:,:,0]

    return StateSpace(Ar,Br,Cr,Dr,dt), S


def markov(*args, m=None, transpose=False, dt=None, truncate=False):
    """markov(Y, U, [, m])
    
    Calculate the first `m` Markov parameters [D CB CAB ...] from data.

    This function computes the Markov parameters for a discrete time
    system

    .. math::

        x[k+1] &= A x[k] + B u[k] \\\\
        y[k] &= C x[k] + D u[k]

    given data for u and y.  The algorithm assumes that that C A^k B = 0
    for k > m-2 (see [1]_).  Note that the problem is ill-posed if the
    length of the input data is less than the desired number of Markov
    parameters (a warning message is generated in this case).

    The function can be called with either 1, 2 or 3 arguments:

    * ``H = markov(data)``
    * ``H = markov(data, m)``
    * ``H = markov(Y, U)``
    * ``H = markov(Y, U, m)``

    where `data` is a `TimeResponseData` object, `YY` is a 1D or 3D
    array, and r is an integer.

    Parameters
    ----------
    Y : array_like
        Output data. If the array is 1D, the system is assumed to be
        single input. If the array is 2D and transpose=False, the columns
        of `Y` are taken as time points, otherwise the rows of `Y` are
        taken as time points.
    U : array_like
        Input data, arranged in the same way as `Y`.
    data : TimeResponseData
        Response data from which the Markov parameters where estimated.
        Input and output data must be 1D or 2D array.
    m : int, optional
        Number of Markov parameters to output. Defaults to len(U).
    dt : True of float, optional
        True indicates discrete time with unspecified sampling time and a
        positive float is discrete time with the specified sampling time.
        It can be used to scale the Markov parameters in order to match
        the unit-area impulse response of python-control. Default is True
        for array_like and dt=data.time[1]-data.time[0] for
        TimeResponseData as input.
    truncate : bool, optional
        Do not use first m equation for least squares. Default is False.
    transpose : bool, optional
        Assume that input data is transposed relative to the standard
        :ref:`time-series-convention`. For TimeResponseData this parameter
        is ignored. Default is False.

    Returns
    -------
    H : ndarray
        First m Markov parameters, [D CB CAB ...].

    References
    ----------
    .. [1] J.-N. Juang, M. Phan, L. G.  Horta, and R. W. Longman,
       Identification of observer/Kalman filter Markov parameters - Theory
       and experiments. Journal of Guidance Control and Dynamics, 16(2),
       320-329, 2012. http://doi.org/10.2514/3.21006

    Examples
    --------
    >>> T = np.linspace(0, 10, 100)
    >>> U = np.ones((1, 100))
    >>> T, Y = ct.forced_response(ct.tf([1], [1, 0.5], True), T, U)
    >>> H = ct.markov(Y, U, 3, transpose=False)

    """

    # Convert input parameters to 2D arrays (if they aren't already)
    # Get the system description
    if len(args) < 1:
        raise ControlArgument("not enough input arguments")
    
    if isinstance(args[0], TimeResponseData):
        data = args[0]
        Umat = np.array(data.inputs, ndmin=2)
        Ymat = np.array(data.outputs, ndmin=2)
        if dt is None:
            dt = data.time[1] - data.time[0]
            if not np.allclose(np.diff(data.time), dt):
                raise ValueError("response time values must be equally "
                                 "spaced.")
        transpose = data.transpose
        if data.transpose and not data.issiso:
            Umat, Ymat = np.transpose(Umat), np.transpose(Ymat)
        if len(args) == 2:
            m = args[1]
        elif len(args) > 2:
            raise ControlArgument("too many positional arguments")
    else:
        if len(args) < 2:
            raise ControlArgument("not enough input arguments")
        Umat = np.array(args[1], ndmin=2)
        Ymat = np.array(args[0], ndmin=2)
        if dt is None:
            dt = True
        if transpose:
            Umat, Ymat = np.transpose(Umat), np.transpose(Ymat)
        if len(args) == 3:
            m = args[2]
        elif len(args) > 3:
            raise ControlArgument("too many positional arguments")

    # Make sure the number of time points match
    if Umat.shape[1] != Ymat.shape[1]:
        raise ControlDimension(
            "Input and output data are of differnent lengths")
    l = Umat.shape[1]

    # If number of desired parameters was not given, set to size of input data
    if m is None:
        m = l

    t = 0
    if truncate:
        t = m

    q = Ymat.shape[0] # number of outputs
    p = Umat.shape[0] # number of inputs

    # Make sure there is enough data to compute parameters
    if m*p > (l-t):
        warnings.warn("Not enough data for requested number of parameters")

    # the algorithm - Construct a matrix of control inputs to invert
    #
    # (q,l)   = (q,p*m) @ (p*m,l)
    # YY.T    = H @ UU.T
    #
    # This algorithm sets up the following problem and solves it for
    # the Markov parameters
    #
    # (l,q)   = (l,p*m) @ (p*m,q) 
    # YY      = UU @ H.T
    #
    # [ y(0)   ]   [ u(0)    0       0                 ] [ D           ]
    # [ y(1)   ]   [ u(1)    u(0)    0                 ] [ C B         ]
    # [ y(2)   ] = [ u(2)    u(1)    u(0)              ] [ C A B       ]
    # [  :     ]   [  :      :        :          :     ] [  :          ]
    # [ y(l-1) ]   [ u(l-1)  u(l-2)  u(l-3) ... u(l-m) ] [ C A^{m-2} B ]
    #
    # truncated version t=m, do not use first m equation
    #
    # [ y(t)   ]   [ u(t)    u(t-1)  u(t-2)    u(t-m)  ] [ D           ]
    # [ y(t+1) ]   [ u(t+1)  u(t)    u(t-1)    u(t-m+1)] [ C B         ]
    # [ y(t+2) ] = [ u(t+2)  u(t+1)  u(t)      u(t-m+2)] [ C B         ]
    # [  :     ]   [  :      :        :          :     ] [  :          ]
    # [ y(l-1) ]   [ u(l-1)  u(l-2)  u(l-3) ... u(l-m) ] [ C A^{m-2} B ]
    #
    # Note: This algorithm assumes C A^{j} B = 0
    # for j > m-2.  See equation (3) in
    #
    #   J.-N. Juang, M. Phan, L. G. Horta, and R. W. Longman, Identification
    #   of observer/Kalman filter Markov parameters - Theory and
    #   experiments. Journal of Guidance Control and Dynamics, 16(2),
    #   320-329, 2012. http://doi.org/10.2514/3.21006
    #

    # Set up the full problem
    # Create matrix of (shifted) inputs
    UUT = np.zeros((p*m,(l)))
    for i in range(m):
        # Shift previous column down and keep zeros at the top
        UUT[i*p:(i+1)*p,i:] = Umat[:,:l-i]

    # Truncate first t=0 or t=m time steps, transpose the problem for lsq
    YY = Ymat[:,t:].T
    UU = UUT[:,t:].T
    
    # Solve for the Markov parameters from  YY = UU @ H.T
    HT, _, _, _ = np.linalg.lstsq(UU, YY, rcond=None)
    H = HT.T/dt # scaling

    H = H.reshape(q,m,p) # output, time*input -> output, time, input
    H = H.transpose(0,2,1) # output, input, time

    # for siso return a 1D array instead of a 3D array
    if q == 1 and p == 1:
        H = np.squeeze(H)

    # Return the first m Markov parameters
    return H if not transpose else np.transpose(H)

# Function aliases
hsvd = hankel_singular_values
balred = balanced_reduction
modred = model_reduction
minreal = minimal_realization
era = eigensys_realization
