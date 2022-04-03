# statefbk.py - tools for state feedback control
#
# Author: Richard M. Murray, Roberto Bucher
# Date: 31 May 2010
#
# This file contains routines for designing state space controllers
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

from . import statesp
from .mateqn import care, dare, _check_shape
from .statesp import StateSpace, _ssmatrix, _convert_to_statespace
from .lti import LTI
from .namedio import isdtime, isctime
from .iosys import InputOutputSystem, NonlinearIOSystem, LinearIOSystem, \
    interconnect, ss
from .exception import ControlSlycot, ControlArgument, ControlDimension, \
    ControlNotImplemented

# Make sure we have access to the right slycot routines
try:
    from slycot import sb03md57
    # wrap without the deprecation warning
    def sb03md(n, C, A, U, dico, job='X',fact='N',trana='N',ldwork=None):
        ret = sb03md57(A, U, C, dico, job, fact, trana, ldwork)
        return ret[2:]
except ImportError:
    try:
        from slycot import sb03md
    except ImportError:
        sb03md = None

try:
    from slycot import sb03od
except ImportError:
    sb03od = None


__all__ = ['ctrb', 'obsv', 'gram', 'place', 'place_varga', 'lqr', 
           'dlqr', 'acker', 'create_statefbk_iosystem']


# Pole placement
def place(A, B, p):
    """Place closed loop eigenvalues

    K = place(A, B, p)

    Parameters
    ----------
    A : 2D array_like
        Dynamics matrix
    B : 2D array_like
        Input matrix
    p : 1D array_like
        Desired eigenvalue locations

    Returns
    -------
    K : 2D array (or matrix)
        Gain such that A - B K has eigenvalues given in p

    Notes
    -----
    Algorithm
        This is a wrapper function for :func:`scipy.signal.place_poles`, which
        implements the Tits and Yang algorithm [1]_. It will handle SISO,
        MISO, and MIMO systems. If you want more control over the algorithm,
        use :func:`scipy.signal.place_poles` directly.

    Limitations
        The algorithm will not place poles at the same location more
        than rank(B) times.

    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    References
    ----------
    .. [1] A.L. Tits and Y. Yang, "Globally convergent algorithms for robust
       pole assignment by state feedback, IEEE Transactions on Automatic
       Control, Vol. 41, pp. 1432-1452, 1996.

    Examples
    --------
    >>> A = [[-1, -1], [0, 1]]
    >>> B = [[0], [1]]
    >>> K = place(A, B, [-2, -5])

    See Also
    --------
    place_varga, acker

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.
    """
    from scipy.signal import place_poles

    # Convert the system inputs to NumPy arrays
    A_mat = np.array(A)
    B_mat = np.array(B)
    if (A_mat.shape[0] != A_mat.shape[1]):
        raise ControlDimension("A must be a square matrix")

    if (A_mat.shape[0] != B_mat.shape[0]):
        err_str = "The number of rows of A must equal the number of rows in B"
        raise ControlDimension(err_str)

    # Convert desired poles to numpy array
    placed_eigs = np.atleast_1d(np.squeeze(np.asarray(p)))

    result = place_poles(A_mat, B_mat, placed_eigs, method='YT')
    K = result.gain_matrix
    return _ssmatrix(K)


def place_varga(A, B, p, dtime=False, alpha=None):
    """Place closed loop eigenvalues
    K = place_varga(A, B, p, dtime=False, alpha=None)

    Required Parameters
    ----------
    A : 2D array_like
        Dynamics matrix
    B : 2D array_like
        Input matrix
    p : 1D array_like
        Desired eigenvalue locations

    Optional Parameters
    ---------------
    dtime : bool
        False for continuous time pole placement or True for discrete time.
        The default is dtime=False.

    alpha : double scalar
        If `dtime` is false then place_varga will leave the eigenvalues with
        real part less than alpha untouched.  If `dtime` is true then
        place_varga will leave eigenvalues with modulus less than alpha
        untouched.

        By default (alpha=None), place_varga computes alpha such that all
        poles will be placed.

    Returns
    -------
    K : 2D array (or matrix)
        Gain such that A - B K has eigenvalues given in p.

    Algorithm
    ---------
    This function is a wrapper for the slycot function sb01bd, which
    implements the pole placement algorithm of Varga [1]. In contrast to the
    algorithm used by place(), the Varga algorithm can place multiple poles at
    the same location. The placement, however, may not be as robust.

    [1] Varga A. "A Schur method for pole assignment."  IEEE Trans. Automatic
        Control, Vol. AC-26, pp. 517-519, 1981.

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

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
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb01bd'")

    # Convert the system inputs to NumPy arrays
    A_mat = np.array(A)
    B_mat = np.array(B)
    if (A_mat.shape[0] != A_mat.shape[1] or A_mat.shape[0] != B_mat.shape[0]):
        raise ControlDimension("matrix dimensions are incorrect")

    # Compute the system eigenvalues and convert poles to numpy array
    system_eigs = np.linalg.eig(A_mat)[0]
    placed_eigs = np.atleast_1d(np.squeeze(np.asarray(p)))

    # Need a character parameter for SB01BD
    if dtime:
        DICO = 'D'
    else:
        DICO = 'C'

    if alpha is None:
        # SB01BD ignores eigenvalues with real part less than alpha
        # (if DICO='C') or with modulus less than alpha
        # (if DICO = 'D').
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
            alpha = -2*abs(min(system_eigs.real))
    elif dtime and alpha < 0.0:
        raise ValueError("Discrete time systems require alpha > 0")

    # Call SLICOT routine to place the eigenvalues
    A_z, w, nfp, nap, nup, F, Z = \
        sb01bd(B_mat.shape[0], B_mat.shape[1], len(placed_eigs), alpha,
               A_mat, B_mat, placed_eigs, DICO)

    # Return the gain matrix, with MATLAB gain convention
    return _ssmatrix(-F)


# Contributed by Roberto Bucher <roberto.bucher@supsi.ch>
def acker(A, B, poles):
    """Pole placement using Ackermann method

    Call:
    K = acker(A, B, poles)

    Parameters
    ----------
    A, B : 2D array_like
        State and input matrix of the system
    poles : 1D array_like
        Desired eigenvalue locations

    Returns
    -------
    K : 2D array (or matrix)
        Gains such that A - B K has given eigenvalues

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.
    """
    # Convert the inputs to matrices
    a = _ssmatrix(A)
    b = _ssmatrix(B)

    # Make sure the system is controllable
    ct = ctrb(A, B)
    if np.linalg.matrix_rank(ct) != a.shape[0]:
        raise ValueError("System not reachable; pole placement invalid")

    # Compute the desired characteristic polynomial
    p = np.real(np.poly(poles))

    # Place the poles using Ackermann's method
    # TODO: compute pmat using Horner's method (O(n) instead of O(n^2))
    n = np.size(p)
    pmat = p[n-1] * np.linalg.matrix_power(a, 0)
    for i in np.arange(1, n):
        pmat = pmat + p[n-i-1] * np.linalg.matrix_power(a, i)
    K = np.linalg.solve(ct, pmat)

    K = K[-1][:]                # Extract the last row
    return _ssmatrix(K)


def lqr(*args, **kwargs):
    """lqr(A, B, Q, R[, N])

    Linear quadratic regulator design

    The lqr() function computes the optimal state feedback controller
    u = -K x that minimizes the quadratic cost

    .. math:: J = \\int_0^\\infty (x' Q x + u' R u + 2 x' N u) dt

    The function can be called with either 3, 4, or 5 arguments:

    * ``K, S, E = lqr(sys, Q, R)``
    * ``K, S, E = lqr(sys, Q, R, N)``
    * ``K, S, E = lqr(A, B, Q, R)``
    * ``K, S, E = lqr(A, B, Q, R, N)``

    where `sys` is an `LTI` object, and `A`, `B`, `Q`, `R`, and `N` are
    2D arrays or matrices of appropriate dimension.

    Parameters
    ----------
    A, B : 2D array_like
        Dynamics and input matrices
    sys : LTI StateSpace system
        Linear system
    Q, R : 2D array
        State and input weight matrices
    N : 2D array, optional
        Cross weight matrix
    integral_action : ndarray, optional
        If this keyword is specified, the controller includes integral action
        in addition to state feedback.  The value of the `integral_action``
        keyword should be an ndarray that will be multiplied by the current to
        generate the error for the internal integrator states of the control
        law.  The number of outputs that are to be integrated must match the
        number of additional rows and columns in the ``Q`` matrix.
    method : str, optional
        Set the method used for computing the result.  Current methods are
        'slycot' and 'scipy'.  If set to None (default), try 'slycot' first
        and then 'scipy'.

    Returns
    -------
    K : 2D array (or matrix)
        State feedback gains
    S : 2D array (or matrix)
        Solution to Riccati equation
    E : 1D array
        Eigenvalues of the closed loop system

    See Also
    --------
    lqe, dlqr, dlqe

    Notes
    -----
    1. If the first argument is an LTI object, then this object will be used
       to define the dynamics and input matrices.  Furthermore, if the LTI
       object corresponds to a discrete time system, the ``dlqr()`` function
       will be called.

    2. The return type for 2D arrays depends on the default class set for
       state space operations.  See :func:`~control.use_numpy_matrix`.

    Examples
    --------
    >>> K, S, E = lqr(sys, Q, R, [N])
    >>> K, S, E = lqr(A, B, Q, R, [N])

    """
    #
    # Process the arguments and figure out what inputs we received
    #

    # If we were passed a discrete time system as the first arg, use dlqr()
    if isinstance(args[0], LTI) and isdtime(args[0], strict=True):
        # Call dlqr
        return dlqr(*args, **kwargs)

    # Get the system description
    if (len(args) < 3):
        raise ControlArgument("not enough input arguments")

    # If we were passed a state space  system, use that to get system matrices
    if isinstance(args[0], StateSpace):
        A = np.array(args[0].A, ndmin=2, dtype=float)
        B = np.array(args[0].B, ndmin=2, dtype=float)
        index = 1

    elif isinstance(args[0], LTI):
        # Don't allow other types of LTI systems
        raise ControlArgument("LTI system must be in state space form")

    else:
        # Arguments should be A and B matrices
        A = np.array(args[0], ndmin=2, dtype=float)
        B = np.array(args[1], ndmin=2, dtype=float)
        index = 2

    # Get the weighting matrices (converting to matrices, if needed)
    Q = np.array(args[index], ndmin=2, dtype=float)
    R = np.array(args[index+1], ndmin=2, dtype=float)
    if (len(args) > index + 2):
        N = np.array(args[index+2], ndmin=2, dtype=float)
    else:
        N = None

    #
    # Process keywords
    #

    # Get the method to use (if specified as a keyword)
    method = kwargs.pop('method', None)

    # See if we should augment the controller with integral feedback
    integral_action = kwargs.pop('integral_action', None)
    if integral_action is not None:
        # Figure out the size of the system
        nstates = A.shape[0]
        ninputs = B.shape[1]

        # Make sure that the integral action argument is the right type
        if not isinstance(integral_action, np.ndarray):
            raise ControlArgument("Integral action must pass an array")
        elif integral_action.shape[1] != nstates:
            raise ControlArgument(
                "Integral gain size must match system state size")

        # Process the states to be integrated
        nintegrators = integral_action.shape[0]
        C = integral_action

        # Augment the system with integrators
        A = np.block([
            [A, np.zeros((nstates, nintegrators))],
            [C, np.zeros((nintegrators, nintegrators))]
        ])
        B = np.vstack([B, np.zeros((nintegrators, ninputs))])

    if kwargs:
        raise TypeError("unrecognized keywords: ", str(kwargs))

    # Compute the result (dimension and symmetry checking done in care())
    X, L, G = care(A, B, Q, R, N, None, method=method, S_s="N")
    return G, X, L


def dlqr(*args, **kwargs):
    """dlqr(A, B, Q, R[, N])

    Discrete-time linear quadratic regulator design

    The dlqr() function computes the optimal state feedback controller
    u[n] = - K x[n] that minimizes the quadratic cost

    .. math:: J = \\sum_0^\\infty (x[n]' Q x[n] + u[n]' R u[n] + 2 x[n]' N u[n])

    The function can be called with either 3, 4, or 5 arguments:

    * ``dlqr(dsys, Q, R)``
    * ``dlqr(dsys, Q, R, N)``
    * ``dlqr(A, B, Q, R)``
    * ``dlqr(A, B, Q, R, N)``

    where `dsys` is a discrete-time :class:`StateSpace` system, and `A`, `B`,
    `Q`, `R`, and `N` are 2d arrays of appropriate dimension (`dsys.dt` must
    not be 0.)

    Parameters
    ----------
    A, B : 2D array
        Dynamics and input matrices
    dsys : LTI :class:`StateSpace`
        Discrete-time linear system
    Q, R : 2D array
        State and input weight matrices
    N : 2D array, optional
        Cross weight matrix
    integral_action : ndarray, optional
        If this keyword is specified, the controller includes integral action
        in addition to state feedback.  The value of the `integral_action``
        keyword should be an ndarray that will be multiplied by the current to
        generate the error for the internal integrator states of the control
        law.  The number of outputs that are to be integrated must match the
        number of additional rows and columns in the ``Q`` matrix.
    method : str, optional
        Set the method used for computing the result.  Current methods are
        'slycot' and 'scipy'.  If set to None (default), try 'slycot' first
        and then 'scipy'.

    Returns
    -------
    K : 2D array (or matrix)
        State feedback gains
    S : 2D array (or matrix)
        Solution to Riccati equation
    E : 1D array
        Eigenvalues of the closed loop system

    See Also
    --------
    lqr, lqe, dlqe

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    Examples
    --------
    >>> K, S, E = dlqr(dsys, Q, R, [N])
    >>> K, S, E = dlqr(A, B, Q, R, [N])
    """

    #
    # Process the arguments and figure out what inputs we received
    #

    # Get the system description
    if (len(args) < 3):
        raise ControlArgument("not enough input arguments")

    # If we were passed a continus time system as the first arg, raise error
    if isinstance(args[0], LTI) and isctime(args[0], strict=True):
        raise ControlArgument("dsys must be discrete time (dt != 0)")

    # If we were passed a state space  system, use that to get system matrices
    if isinstance(args[0], StateSpace):
        A = np.array(args[0].A, ndmin=2, dtype=float)
        B = np.array(args[0].B, ndmin=2, dtype=float)
        index = 1

    elif isinstance(args[0], LTI):
        # Don't allow other types of LTI systems
        raise ControlArgument("LTI system must be in state space form")

    else:
        # Arguments should be A and B matrices
        A = np.array(args[0], ndmin=2, dtype=float)
        B = np.array(args[1], ndmin=2, dtype=float)
        index = 2

    # Get the weighting matrices (converting to matrices, if needed)
    Q = np.array(args[index], ndmin=2, dtype=float)
    R = np.array(args[index+1], ndmin=2, dtype=float)
    if (len(args) > index + 2):
        N = np.array(args[index+2], ndmin=2, dtype=float)
    else:
        N = np.zeros((Q.shape[0], R.shape[1]))

    #
    # Process keywords
    #

    # Get the method to use (if specified as a keyword)
    method = kwargs.pop('method', None)

    # See if we should augment the controller with integral feedback
    integral_action = kwargs.pop('integral_action', None)
    if integral_action is not None:
        # Figure out the size of the system
        nstates = A.shape[0]
        ninputs = B.shape[1]

        if not isinstance(integral_action, np.ndarray):
            raise ControlArgument("Integral action must pass an array")
        elif integral_action.shape[1] != nstates:
            raise ControlArgument(
                "Integral gain size must match system state size")
        else:
            nintegrators = integral_action.shape[0]
            C = integral_action

            # Augment the system with integrators
            A = np.block([
                [A, np.zeros((nstates, nintegrators))],
                [C, np.eye(nintegrators)]
            ])
            B = np.vstack([B, np.zeros((nintegrators, ninputs))])

    if kwargs:
        raise TypeError("unrecognized keywords: ", str(kwargs))

    # Compute the result (dimension and symmetry checking done in dare())
    S, E, K = dare(A, B, Q, R, N, method=method, S_s="N")
    return _ssmatrix(K), _ssmatrix(S), E


# Function to create an I/O sytems representing a state feedback controller
def create_statefbk_iosystem(
        sys, K, integral_action=None, xd_labels='xd[{i}]', ud_labels='ud[{i}]',
        estimator=None, type='linear'):
    """Create an I/O system using a (full) state feedback controller

    This function creates an input/output system that implements a
    state feedback controller of the form

        u = ud - K_p (x - xd) - K_i integral(C x - C x_d)

    It can be called in the form

        ctrl, clsys = ct.create_statefbk_iosystem(sys, K)

    where ``sys`` is the process dynamics and ``K`` is the state (+ integral)
    feedback gain (eg, from LQR).  The function returns the controller
    ``ctrl`` and the closed loop systems ``clsys``, both as I/O systems.

    Parameters
    ----------
    sys : InputOutputSystem
        The I/O system that represents the process dynamics.  If no estimator
        is given, the output of this system should represent the full state.

    K : ndarray
        The state feedback gain.  This matrix defines the gains to be
        applied to the system.  If ``integral_action`` is None, then the
        dimensions of this array should be (sys.ninputs, sys.nstates).  If
        `integral action` is set to a matrix or a function, then additional
        columns represent the gains of the integral states of the
        controller.

    xd_labels, ud_labels : str or list of str, optional
        Set the name of the signals to use for the desired state and inputs.
        If a single string is specified, it should be a format string using
        the variable ``i`` as an index.  Otherwise, a list of strings matching
        the size of xd and ud, respectively, should be used.  Default is
        ``'xd[{i}]'`` for xd_labels and ``'xd[{i}]'`` for ud_labels.

    integral_action : None, ndarray, or func, optional
        If this keyword is specified, the controller can include integral
        action in addition to state feedback.  If ``integral_action`` is an
        ndarray, it will be multiplied by the current and desired state to
        generate the error for the internal integrator states of the control
        law.  If ``integral_action`` is a function ``h``, that function will
        be called with the signature h(t, x, u, params) to obtain the
        outputs that should be integrated.  The number of outputs that are
        to be integrated must match the number of additional columns in the
        ``K`` matrix.

    estimator : InputOutputSystem, optional
        If an estimator is provided, using the states of the estimator as
        the system inputs for the controller.

    type : 'nonlinear' or 'linear', optional
        Set the type of controller to create. The default is a linear
        controller implementing the LQR regulator. If the type is 'nonlinear',
        a :class:NonlinearIOSystem is created instead, with the gain ``K`` as
        a parameter (allowing modifications of the gain at runtime).

    Returns
    -------
    ctrl : InputOutputSystem
        Input/output system representing the controller.  This system takes
        as inputs the desired state xd, the desired input ud, and the system
        state x.  It outputs the controller action u according to the
        formula u = ud - K(x - xd).  If the keyword `integral_action` is
        specified, then an additional set of integrators is included in the
        control system (with the gain matrix K having the integral gains
        appended after the state gains).

    clsys : InputOutputSystem
        Input/output system representing the closed loop system.  This
        systems takes as inputs the desired trajectory (xd, ud) and outputs
        the system state x and the applied input u (vertically stacked).

    """
    # Make sure that we were passed an I/O system as an input
    if not isinstance(sys, InputOutputSystem):
        raise ControlArgument("Input system must be I/O system")

    # See whether we were give an estimator
    if estimator is not None:
        # Check to make sure the estimator is the right size
        if estimator.noutputs != sys.nstates:
            raise ControlArgument("Estimator output size must match state")
    elif sys.noutputs != sys.nstates:
        # If no estimator, make sure that the system has all states as outputs
        # TODO: check to make sure output map is the identity
        raise ControlArgument("System output must be the full state")
    else:
        # Use the system directly instead of an estimator
        estimator = sys

    # See whether we should implement integral action
    nintegrators = 0
    if integral_action is not None:
        if not isinstance(integral_action, np.ndarray):
            raise ControlArgument("Integral action must pass an array")
        elif integral_action.shape[1] != sys.nstates:
            raise ControlArgument(
                "Integral gain size must match system state size")
        else:
            nintegrators = integral_action.shape[0]
            C = integral_action
    else:
        # Create a C matrix with no outputs, just in case update gets called
        C = np.zeros((0, sys.nstates))

    # Check to make sure that state feedback has the right shape
    if not isinstance(K, np.ndarray) or \
       K.shape != (sys.ninputs, estimator.noutputs + nintegrators):
        raise ControlArgument(
            f'Control gain must be an array of size {sys.ninputs}'
            f'x {sys.nstates}' +
            (f'+{nintegrators}' if nintegrators > 0 else ''))

    # Figure out the labels to use
    if isinstance(xd_labels, str):
        # Gnerate the list of labels using the argument as a format string
        xd_labels = [xd_labels.format(i=i) for i in range(sys.nstates)]

    if isinstance(ud_labels, str):
        # Gnerate the list of labels using the argument as a format string
        ud_labels = [ud_labels.format(i=i) for i in range(sys.ninputs)]

    # Define the controller system
    if type == 'nonlinear':
        # Create an I/O system for the state feedback gains
        def _control_update(t, x, inputs, params):
            # Split input into desired state, nominal input, and current state
            xd_vec = inputs[0:sys.nstates]
            x_vec = inputs[-estimator.nstates:]

            # Compute the integral error in the xy coordinates
            return C @ x_vec - C @ xd_vec

        def _control_output(t, e, z, params):
            K = params.get('K')

            # Split input into desired state, nominal input, and current state
            xd_vec = z[0:sys.nstates]
            ud_vec = z[sys.nstates:sys.nstates + sys.ninputs]
            x_vec = z[-sys.nstates:]

            # Compute the control law
            u = ud_vec - K[:, 0:sys.nstates] @ (x_vec - xd_vec)
            if nintegrators > 0:
                u -= K[:, sys.nstates:] @ e

            return u

        ctrl = NonlinearIOSystem(
            _control_update, _control_output, name='control',
            inputs=xd_labels + ud_labels + estimator.output_labels,
            outputs=list(sys.input_index.keys()), params={'K': K},
            states=nintegrators)

    elif type == 'linear' or type is None:
        # Create the matrices implementing the controller
        if isctime(sys):
            # Continuous time: integrator
            A_lqr = np.zeros((C.shape[0], C.shape[0]))
        else:
            # Discrete time: summer
            A_lqr = np.eye(C.shape[0])
        B_lqr = np.hstack([-C, np.zeros((C.shape[0], sys.ninputs)), C])
        C_lqr = -K[:, sys.nstates:]
        D_lqr = np.hstack([
            K[:, 0:sys.nstates], np.eye(sys.ninputs), -K[:, 0:sys.nstates]
        ])

        ctrl = ss(
            A_lqr, B_lqr, C_lqr, D_lqr, dt=sys.dt, name='control',
            inputs=xd_labels + ud_labels + estimator.output_labels,
            outputs=list(sys.input_index.keys()), states=nintegrators)

    else:
        raise ControlArgument(f"unknown type '{type}'")

    # Define the closed loop system
    closed = interconnect(
        [sys, ctrl] if estimator == sys else [sys, ctrl, estimator],
        name=sys.name + "_" + ctrl.name,
        inplist=xd_labels + ud_labels, inputs=xd_labels + ud_labels,
        outlist=sys.output_labels + sys.input_labels,
        outputs=sys.output_labels + sys.input_labels
    )
    return ctrl, closed


def ctrb(A, B):
    """Controllabilty matrix

    Parameters
    ----------
    A, B : array_like or string
        Dynamics and input matrix of the system

    Returns
    -------
    C : 2D array (or matrix)
        Controllability matrix

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    Examples
    --------
    >>> C = ctrb(A, B)

    """

    # Convert input parameters to matrices (if they aren't already)
    amat = _ssmatrix(A)
    bmat = _ssmatrix(B)
    n = np.shape(amat)[0]

    # Construct the controllability matrix
    ctrb = np.hstack(
        [bmat] + [np.linalg.matrix_power(amat, i) @ bmat
                  for i in range(1, n)])
    return _ssmatrix(ctrb)


def obsv(A, C):
    """Observability matrix

    Parameters
    ----------
    A, C : array_like or string
        Dynamics and output matrix of the system

    Returns
    -------
    O : 2D array (or matrix)
        Observability matrix

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    Examples
    --------
    >>> O = obsv(A, C)
    """

    # Convert input parameters to matrices (if they aren't already)
    amat = _ssmatrix(A)
    cmat = _ssmatrix(C)
    n = np.shape(amat)[0]

    # Construct the observability matrix
    obsv = np.vstack([cmat] + [cmat @ np.linalg.matrix_power(amat, i)
                               for i in range(1, n)])
    return _ssmatrix(obsv)


def gram(sys, type):
    """Gramian (controllability or observability)

    Parameters
    ----------
    sys : StateSpace
        System description
    type : String
        Type of desired computation.  `type` is either 'c' (controllability)
        or 'o' (observability). To compute the Cholesky factors of Gramians
        use 'cf' (controllability) or 'of' (observability)

    Returns
    -------
    gram : 2D array (or matrix)
        Gramian of system

    Raises
    ------
    ValueError
        * if system is not instance of StateSpace class
        * if `type` is not 'c', 'o', 'cf' or 'of'
        * if system is unstable (sys.A has eigenvalues not in left half plane)

    ControlSlycot
        if slycot routine sb03md cannot be found
        if slycot routine sb03od cannot be found

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    Examples
    --------
    >>> Wc = gram(sys, 'c')
    >>> Wo = gram(sys, 'o')
    >>> Rc = gram(sys, 'cf'), where Wc = Rc' * Rc
    >>> Ro = gram(sys, 'of'), where Wo = Ro' * Ro

    """

    # Check for ss system object
    if not isinstance(sys, statesp.StateSpace):
        raise ValueError("System must be StateSpace!")
    if type not in ['c', 'o', 'cf', 'of']:
        raise ValueError("That type is not supported!")

    # TODO: Check for continuous or discrete, only continuous supported for now
        # if isCont():
        #    dico = 'C'
        # elif isDisc():
        #    dico = 'D'
        # else:
    dico = 'C'

    # TODO: Check system is stable, perhaps a utility in ctrlutil.py
    # or a method of the StateSpace class?
    if np.any(np.linalg.eigvals(sys.A).real >= 0.0):
        raise ValueError("Oops, the system is unstable!")

    if type == 'c' or type == 'o':
        # Compute Gramian by the Slycot routine sb03md
        # make sure Slycot is installed
        if sb03md is None:
            raise ControlSlycot("can't find slycot module 'sb03md'")
        if type == 'c':
            tra = 'T'
            C = -sys.B @ sys.B.T
        elif type == 'o':
            tra = 'N'
            C = -sys.C.T @ sys.C
        n = sys.nstates
        U = np.zeros((n, n))
        A = np.array(sys.A)         # convert to NumPy array for slycot
        X, scale, sep, ferr, w = sb03md(
            n, C, A, U, dico, job='X', fact='N', trana=tra)
        gram = X
        return _ssmatrix(gram)

    elif type == 'cf' or type == 'of':
        # Compute cholesky factored gramian from slycot routine sb03od
        if sb03od is None:
            raise ControlSlycot("can't find slycot module 'sb03od'")
        tra = 'N'
        n = sys.nstates
        Q = np.zeros((n, n))
        A = np.array(sys.A)         # convert to NumPy array for slycot
        if type == 'cf':
            m = sys.B.shape[1]
            B = np.zeros_like(A)
            B[0:m, 0:n] = sys.B.transpose()
            X, scale, w = sb03od(
                n, m, A.transpose(), Q, B, dico, fact='N', trans=tra)
        elif type == 'of':
            m = sys.C.shape[0]
            C = np.zeros_like(A)
            C[0:n, 0:m] = sys.C.transpose()
            X, scale, w = sb03od(
                n, m, A, Q, C.transpose(), dico, fact='N', trans=tra)
        gram = X
        return _ssmatrix(gram)
