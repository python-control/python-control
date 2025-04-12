# modelsimp.py - tools for model simplification
#
# Initial authors: Steve Brunton, Kevin Chen, Lauren Padilla
# Creation date: 30 Nov 2010

"""Tools for model simplification.

This module contains routines for obtaining reduced order models for state
space systems.

"""

import warnings

# External packages and modules
import numpy as np

from .exception import ControlArgument, ControlDimension, ControlSlycot
from .iosys import isctime, isdtime
from .statefbk import gram
from .statesp import StateSpace
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
    sys : `StateSpace`
        State space system.

    Returns
    -------
    H : array
        List of Hankel singular values.

    See Also
    --------
    gram

    Notes
    -----
    The Hankel singular values are the singular values of the Hankel operator.
    In practice, we compute the square root of the eigenvalues of the matrix
    formed by taking the product of the observability and controllability
    Gramians.  There are other (more efficient) methods based on solving the
    Lyapunov equation in a particular way (more details soon).

    Examples
    --------
    >>> G = ct.tf2ss([1], [1, 2])
    >>> H = ct.hsvd(G)
    >>> H[0]
    np.float64(0.25)

    """
    # TODO: implement for discrete-time systems
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


def model_reduction(
        sys, elim_states=None, method='matchdc', elim_inputs=None,
        elim_outputs=None, keep_states=None, keep_inputs=None,
        keep_outputs=None, warn_unstable=True):
    """Model reduction by input, output, or state elimination.

    This function produces a reduced-order model of a system by eliminating
    specified inputs, outputs, and/or states from the original system.  The
    specific states, inputs, or outputs that are eliminated can be
    specified by either listing the states, inputs, or outputs to be
    eliminated or those to be kept.

    Two methods of state reduction are possible: 'truncate' removes the
    states marked for elimination, while 'matchdc' replaces the eliminated
    states with their equilibrium values (thereby keeping the input/output
    gain unchanged at zero frequency ["DC"]).

    Parameters
    ----------
    sys : `StateSpace`
        Original system to reduce.
    elim_inputs, elim_outputs, elim_states : array of int or str, optional
        Vector of inputs, outputs, or states to eliminate.  Can be specified
        either as an offset into the appropriate vector or as a signal name.
    keep_inputs, keep_outputs, keep_states : array, optional
        Vector of inputs, outputs, or states to keep.  Can be specified
        either as an offset into the appropriate vector or as a signal name.
    method : string
        Method of removing states: either 'truncate' or 'matchdc' (default).
    warn_unstable : bool, option
        If False, don't warn if system is unstable.

    Returns
    -------
    rsys : `StateSpace`
        Reduced order model.

    Raises
    ------
    ValueError
        If `method` is not either 'matchdc' or 'truncate'.
    NotImplementedError
        If the 'matchdc' method is used for a discrete-time system.

    Warns
    -----
    UserWarning
        If eigenvalues of `sys.A` are not all stable.

    Examples
    --------
    >>> G = ct.rss(4)
    >>> Gr = ct.model_reduction(G, [0, 2], method='matchdc')
    >>> Gr.nstates
    2

    See Also
    --------
    balanced_reduction, minimal_realization

    Notes
    -----
    The model_reduction function issues a warning if the system has
    unstable eigenvalues, since in those situations the stability of the
    reduced order model may be different than the stability of the full
    model.  No other checking is done, so users must to be careful not to
    render a system unobservable or unreachable.

    States, inputs, and outputs can be specified using integer offsets or
    using signal names.  Slices can also be specified, but must use the
    Python `slice` function.

    """
    if not isinstance(sys, StateSpace):
        raise TypeError("system must be a StateSpace system")

    # Check system is stable
    if warn_unstable:
        if isctime(sys) and np.any(np.linalg.eigvals(sys.A).real >= 0.0) or \
           isdtime(sys) and np.any(np.abs(np.linalg.eigvals(sys.A)) >= 1):
            warnings.warn("System is unstable; reduction may be meaningless")

    # Utility function to process keep/elim keywords
    def _process_elim_or_keep(elim, keep, labels):
        def _expand_key(key):
            if key is None:
                return []
            elif isinstance(key, str):
                return labels.index(key)
            elif isinstance(key, list):
                return [_expand_key(k) for k in key]
            elif isinstance(key, slice):
                return range(len(labels))[key]
            else:
                return key

        elim = np.atleast_1d(_expand_key(elim))
        keep = np.atleast_1d(_expand_key(keep))

        if len(elim) > 0 and len(keep) > 0:
            raise ValueError(
                "can't provide both 'keep' and 'elim' for same variables")
        elif len(keep) > 0:
            keep = np.sort(keep).tolist()
            elim = [i for i in range(len(labels)) if i not in keep]
        else:
            elim = [] if elim is None else np.sort(elim).tolist()
            keep = [i for i in range(len(labels)) if i not in elim]
        return elim, keep

    # Determine which states to keep
    elim_states, keep_states = _process_elim_or_keep(
        elim_states, keep_states, sys.state_labels)
    elim_inputs, keep_inputs = _process_elim_or_keep(
        elim_inputs, keep_inputs, sys.input_labels)
    elim_outputs, keep_outputs = _process_elim_or_keep(
        elim_outputs, keep_outputs, sys.output_labels)

    # Create submatrix of states we are keeping
    A11 = sys.A[:, keep_states][keep_states, :]     # states we are keeping
    A12 = sys.A[:, elim_states][keep_states, :]     # needed for 'matchdc'
    A21 = sys.A[:, keep_states][elim_states, :]
    A22 = sys.A[:, elim_states][elim_states, :]

    B1 = sys.B[keep_states, :]
    B2 = sys.B[elim_states, :]

    C1 = sys.C[:, keep_states]
    C2 = sys.C[:, elim_states]

    # Figure out the new state space system
    if method == 'matchdc' and A22.size > 0:
        if sys.isdtime(strict=True):
            raise NotImplementedError(
                "'matchdc' not (yet) supported for discrete-time systems")

        # if matchdc, residualize
        # Check if the matrix A22 is invertible
        if np.linalg.matrix_rank(A22) != len(elim_states):
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

    elif method == 'truncate' or A22.size == 0:
        # Get rid of unwanted states
        Ar = A11
        Br = B1
        Cr = C1
        Dr = sys.D

    else:
        raise ValueError("Oops, method is not supported!")

    # Get rid of additional inputs and outputs
    Br = Br[:, keep_inputs]
    Cr = Cr[keep_outputs, :]
    Dr = Dr[keep_outputs, :][:, keep_inputs]

    rsys = StateSpace(Ar, Br, Cr, Dr)
    return rsys


def balanced_reduction(sys, orders, method='truncate', alpha=None):
    """Balanced reduced order model of system of a given order.

    States are eliminated based on Hankel singular value.  If `sys` has
    unstable modes, they are removed, the balanced realization is done on
    the stable part, then reinserted in accordance with [1]_.

    References
    ----------
    .. [1] C. S. Hsu and D. Hou, "Reducing unstable linear control
       systems via real Schur transformation".  Electronics Letters,
       27, 984-986, 1991.

    Parameters
    ----------
    sys : `StateSpace`
        Original system to reduce.
    orders : integer or array of integer
        Desired order of reduced order model (if a vector, returns a vector
        of systems).
    method : string
        Method of removing states, either 'truncate' or 'matchdc'.
    alpha : float
        Redefines the stability boundary for eigenvalues of the system
        matrix A.  By default for continuous-time systems, alpha <= 0
        defines the stability boundary for the real part of A's eigenvalues
        and for discrete-time systems, 0 <= alpha <= 1 defines the stability
        boundary for the modulus of A's eigenvalues. See SLICOT routines
        AB09MD and AB09ND for more information.

    Returns
    -------
    rsys : `StateSpace`
        A reduced order model or a list of reduced order models if orders is
        a list.

    Raises
    ------
    ValueError
        If `method` is not 'truncate' or 'matchdc'.
    ImportError
        If slycot routine ab09ad, ab09md, or ab09nd is not found.
    ValueError
        If there are more unstable modes than any value in orders.

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
            from slycot import ab09ad, ab09md
        except ImportError:
            raise ControlSlycot(
                "can't find slycot subroutine ab09md or ab09ad")
    elif method == 'matchdc':
        try:
            from slycot import ab09nd
        except ImportError:
            raise ControlSlycot("can't find slycot subroutine ab09nd")

    # Check for ss system object, need a utility for this?

    # TODO: Check for continuous or discrete, only continuous supported for now
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
        iter(orders)
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
    """Eliminate uncontrollable or unobservable states.

    Eliminates uncontrollable or unobservable states in state-space
    models or canceling pole-zero pairs in transfer functions. The
    output `sysr` has minimal order and the same response
    characteristics as the original model `sys`.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        Original system.
    tol : real
        Tolerance.
    verbose : bool
        Print results if True.

    Returns
    -------
    rsys : `StateSpace` or `TransferFunction`
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
    YY = Y.transpose(0, 2, 1) # transpose for reshape

    H = np.zeros((q*m, p*n))

    for r in range(m):
        # shift and add row to Hankel matrix
        new_row = YY[:, r:r+n, :]
        H[q*r:q*(r+1), :] = new_row.reshape((q, p*n))

    return H


def eigensys_realization(arg, r, m=None, n=None, dt=True, transpose=False):
    r"""eigensys_realization(YY, r)

    Calculate ERA model based on impulse-response data.

    This function computes a discrete-time system

    .. math::

        x[k+1] &= A x[k] + B u[k] \\\\
        y[k] &= C x[k] + D u[k]

    of order :math:`r` for a given impulse-response data (see [1]_).

    The function can be called with 2 arguments:

    * ``sysd, S = eigensys_realization(data, r)``
    * ``sysd, S = eigensys_realization(YY, r)``

    where `data` is a `TimeResponseData` object, `YY` is a 1D or 3D
    array, and r is an integer.

    Parameters
    ----------
    YY : array_like
        Impulse response from which the `StateSpace` model is estimated, 1D
        or 3D array.
    data : `TimeResponseData`
        Impulse response from which the `StateSpace` model is estimated.
    r : integer
        Order of model.
    m : integer, optional
        Number of rows in Hankel matrix. Default is 2*r.
    n : integer, optional
        Number of columns in Hankel matrix. Default is 2*r.
    dt : True or float, optional
        True indicates discrete time with unspecified sampling time and a
        positive float is discrete time with the specified sampling time.
        It can be used to scale the `StateSpace` model in order to match the
        unit-area impulse response of python-control. Default is True.
    transpose : bool, optional
        Assume that input data is transposed relative to the standard
        :ref:`time-series-convention`. For `TimeResponseData` this parameter
        is ignored. Default is False.

    Returns
    -------
    sys : `StateSpace`
        State space model of the specified order.
    S : array
        Singular values of Hankel matrix. Can be used to choose a good `r`
        value.

    References
    ----------
    .. [1] Samet Oymak and Necmiye Ozay, Non-asymptotic Identification of
       LTI Systems from a Single Trajectory. https://arxiv.org/abs/1806.05722

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

    H = _block_hankel(YY[:, :, 1:], m, n+1) # Hankel matrix (q*m, p*(n+1))
    Hf = H[:, :-p] # first p*n columns of H
    Hl = H[:, p:] # last p*n columns of H

    U,S,Vh = np.linalg.svd(Hf, True)
    Ur =U[:, 0:r]
    Vhr =Vh[0:r, :]

    # balanced realizations
    Sigma_inv = np.diag(1./np.sqrt(S[0:r]))
    Ar = Sigma_inv @ Ur.T @ Hl @ Vhr.T @ Sigma_inv
    Br = Sigma_inv @ Ur.T @ Hf[:, 0:p]*dt # dt scaling for unit-area impulse
    Cr = Hf[0:q, :] @ Vhr.T @ Sigma_inv
    Dr = YY[:, :, 0]

    return StateSpace(Ar, Br, Cr, Dr, dt), S


def markov(*args, m=None, transpose=False, dt=None, truncate=False):
    """markov(Y, U, [, m])

    Calculate Markov parameters [D CB CAB ...] from data.

    This function computes the the first `m` Markov parameters [D CB CAB
    ...] for a discrete-time system.

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
        single input. If the array is 2D and `transpose` = False, the columns
        of `Y` are taken as time points, otherwise the rows of `Y` are
        taken as time points.
    U : array_like
        Input data, arranged in the same way as `Y`.
    data : `TimeResponseData`
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
        `TimeResponseData` as input.
    truncate : bool, optional
        Do not use first m equation for least squares. Default is False.
    transpose : bool, optional
        Assume that input data is transposed relative to the standard
        :ref:`time-series-convention`. For `TimeResponseData` this parameter
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
       320-329, 2012. https://doi.org/10.2514/3.21006

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
            "Input and output data are of different lengths")
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
    #   320-329, 2012. https://doi.org/10.2514/3.21006
    #

    # Set up the full problem
    # Create matrix of (shifted) inputs
    UUT = np.zeros((p*m, l))
    for i in range(m):
        # Shift previous column down and keep zeros at the top
        UUT[i*p:(i+1)*p, i:] = Umat[:, :l-i]

    # Truncate first t=0 or t=m time steps, transpose the problem for lsq
    YY = Ymat[:, t:].T
    UU = UUT[:, t:].T

    # Solve for the Markov parameters from  YY = UU @ H.T
    HT, _, _, _ = np.linalg.lstsq(UU, YY, rcond=None)
    H = HT.T/dt # scaling

    H = H.reshape(q, m, p) # output, time*input -> output, time, input
    H = H.transpose(0, 2, 1) # output, input, time

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
