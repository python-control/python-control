# statefbk.py - tools for state feedback control
#
# Initial authors: Richard M. Murray, Roberto Bucher
# Creation date: 31 May 2010

"""Routines for designing state space controllers."""

import warnings

import numpy as np
import scipy as sp

from . import statesp
from .config import _process_legacy_keyword
from .exception import ControlArgument, ControlSlycot
from .iosys import _process_indices, _process_labels, isctime, isdtime
from .lti import LTI
from .mateqn import care, dare
from .nlsys import NonlinearIOSystem, interconnect
from .statesp import StateSpace, _ssmatrix, ss

# Make sure we have access to the right Slycot routines
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
           'dlqr', 'acker', 'place_acker', 'create_statefbk_iosystem']


# Pole placement
def place(A, B, p):
    """Place closed loop eigenvalues.

    K = place(A, B, p)

    Parameters
    ----------
    A : 2D array_like
        Dynamics matrix.
    B : 2D array_like
        Input matrix.
    p : 1D array_like
        Desired eigenvalue locations.

    Returns
    -------
    K : 2D array
        Gain such that A - B K has eigenvalues given in p.

    Notes
    -----
    This is a wrapper function for `scipy.signal.place_poles`, which
    implements the Tits and Yang algorithm [1]_. It will handle SISO, MISO,
    and MIMO systems. If you want more control over the algorithm, use
    `scipy.signal.place_poles` directly.

    Limitations: The algorithm will not place poles at the same location
    more than rank(B) times.

    References
    ----------
    .. [1] A.L. Tits and Y. Yang, "Globally convergent algorithms for robust
       pole assignment by state feedback, IEEE Transactions on Automatic
       Control, Vol. 41, pp. 1432-1452, 1996.

    Examples
    --------
    >>> A = [[-1, -1], [0, 1]]
    >>> B = [[0], [1]]
    >>> K = ct.place(A, B, [-2, -5])

    See Also
    --------
    place_acker, place_varga

    """
    from scipy.signal import place_poles

    # Convert the system inputs to NumPy arrays
    A_mat = _ssmatrix(A, square=True, name="A")
    B_mat = _ssmatrix(B, axis=0, rows=A_mat.shape[0])

    # Convert desired poles to numpy array
    placed_eigs = np.atleast_1d(np.squeeze(np.asarray(p)))

    result = place_poles(A_mat, B_mat, placed_eigs, method='YT')
    K = result.gain_matrix
    return K


def place_varga(A, B, p, dtime=False, alpha=None):
    """Place closed loop eigenvalues using Varga method.

    K = place_varga(A, B, p, dtime=False, alpha=None)

    Parameters
    ----------
    A : 2D array_like
        Dynamics matrix.
    B : 2D array_like
        Input matrix.
    p : 1D array_like
        Desired eigenvalue locations.
    dtime : bool, optional
        False (default) for continuous-time pole placement or True
        for discrete time.
    alpha : float, optional
        If `dtime` is false then place_varga will leave the eigenvalues with
        real part less than alpha untouched.  If `dtime` is true then
        place_varga will leave eigenvalues with modulus less than alpha
        untouched.

        By default (alpha=None), place_varga computes alpha such that all
        poles will be placed.

    Returns
    -------
    K : 2D array
        Gain such that A - B K has eigenvalues given in p.

    See Also
    --------
    place, place_acker

    Notes
    -----
    This function is a wrapper for the Slycot function sb01bd, which
    implements the pole placement algorithm of Varga [1]_. In contrast
    to the algorithm used by `place`, the Varga algorithm can place
    multiple poles at the same location. The placement, however, may
    not be as robust.

    References
    ----------
    .. [1] Varga A. "A Schur method for pole assignment."  IEEE Trans.
           Automatic Control, Vol. AC-26, pp. 517-519, 1981.

    Examples
    --------
    >>> A = [[-1, -1], [0, 1]]
    >>> B = [[0], [1]]
    >>> K = ct.place_varga(A, B, [-2, -5])

    """

    # Make sure that Slycot is installed
    try:
        from slycot import sb01bd
    except ImportError:
        raise ControlSlycot("can't find slycot module sb01bd")

    # Convert the system inputs to NumPy arrays
    A_mat = _ssmatrix(A, square=True, name="A")
    B_mat = _ssmatrix(B, axis=0, rows=A_mat.shape[0])

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
            # For discrete time, Slycot only cares about modulus, so just make
            # alpha the smallest it can be.
            alpha = 0.0
        else:
            # Choosing alpha=min_eig is insufficient and can lead to an
            # error or not having all the eigenvalues placed that we wanted.
            # Evidently, what python thinks are the eigs is not precisely
            # the same as what Slycot thinks are the eigs. So we need some
            # numerical breathing room. The following is pretty heuristic,
            # but does the trick
            alpha = -2*abs(min(system_eigs.real))
    elif dtime and alpha < 0.0:
        raise ValueError("Discrete time systems require alpha > 0")

    # Call Slycot routine to place the eigenvalues
    A_z, w, nfp, nap, nup, F, Z = \
        sb01bd(B_mat.shape[0], B_mat.shape[1], len(placed_eigs), alpha,
               A_mat, B_mat, placed_eigs, DICO)

    # Return the gain matrix, with MATLAB gain convention
    return -F


# Contributed by Roberto Bucher <roberto.bucher@supsi.ch>
def place_acker(A, B, poles):
    """Pole placement using Ackermann method.

    Call:
    K = place_acker(A, B, poles)

    Parameters
    ----------
    A, B : 2D array_like
        State and input matrix of the system.
    poles : 1D array_like
        Desired eigenvalue locations.

    Returns
    -------
    K : 2D array
        Gains such that A - B K has given eigenvalues.

    See Also
    --------
    place, place_varga

    """
    # Convert the inputs to matrices
    A = _ssmatrix(A, square=True, name="A")
    B = _ssmatrix(B, axis=0, rows=A.shape[0], name="B")

    # Make sure the system is controllable
    ct = ctrb(A, B)
    if np.linalg.matrix_rank(ct) != A.shape[0]:
        raise ValueError("System not reachable; pole placement invalid")

    # Compute the desired characteristic polynomial
    p = np.real(np.poly(poles))

    # Place the poles using Ackermann's method
    # TODO: compute pmat using Horner's method (O(n) instead of O(n^2))
    n = np.size(p)
    pmat = p[n-1] * np.linalg.matrix_power(A, 0)
    for i in np.arange(1, n):
        pmat = pmat + p[n-i-1] * np.linalg.matrix_power(A, i)
    K = np.linalg.solve(ct, pmat)

    K = K[-1, :]                # Extract the last row
    return K


def lqr(*args, **kwargs):
    r"""lqr(A, B, Q, R[, N])

    Linear quadratic regulator design.

    The lqr() function computes the optimal state feedback controller
    u = -K x that minimizes the quadratic cost

    .. math:: J = \int_0^\infty (x' Q x + u' R u + 2 x' N u) dt

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
        Dynamics and input matrices.
    sys : LTI `StateSpace` system
        Linear system.
    Q, R : 2D array
        State and input weight matrices.
    N : 2D array, optional
        Cross weight matrix.
    integral_action : ndarray, optional
        If this keyword is specified, the controller includes integral
        action in addition to state feedback.  The value of the
        `integral_action` keyword should be an ndarray that will be
        multiplied by the current state to generate the error for the
        internal integrator states of the control law.  The number of
        outputs that are to be integrated must match the number of
        additional rows and columns in the `Q` matrix.
    method : str, optional
        Set the method used for computing the result.  Current methods are
        'slycot' and 'scipy'.  If set to None (default), try 'slycot'
        first and then 'scipy'.

    Returns
    -------
    K : 2D array
        State feedback gains.
    S : 2D array
        Solution to Riccati equation.
    E : 1D array
        Eigenvalues of the closed loop system.

    See Also
    --------
    lqe, dlqr, dlqe

    Notes
    -----
    If the first argument is an LTI object, then this object will be used
    to define the dynamics and input matrices.  Furthermore, if the LTI
    object corresponds to a discrete-time system, the `dlqr` function
    will be called.

    Examples
    --------
    >>> K, S, E = lqr(sys, Q, R, [N])                           # doctest: +SKIP
    >>> K, S, E = lqr(A, B, Q, R, [N])                          # doctest: +SKIP

    """
    #
    # Process the arguments and figure out what inputs we received
    #

    # If we were passed a discrete-time system as the first arg, use dlqr()
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
    X, L, G = care(A, B, Q, R, N, None, method=method, _Ss="N")
    return G, X, L


def dlqr(*args, **kwargs):
    r"""dlqr(A, B, Q, R[, N])

    Discrete-time linear quadratic regulator design.

    The dlqr() function computes the optimal state feedback controller
    u[n] = - K x[n] that minimizes the quadratic cost

    .. math:: J = \sum_0^\infty (x[n]' Q x[n] + u[n]' R u[n] + 2 x[n]' N u[n])

    The function can be called with either 3, 4, or 5 arguments:

    * ``dlqr(dsys, Q, R)``
    * ``dlqr(dsys, Q, R, N)``
    * ``dlqr(A, B, Q, R)``
    * ``dlqr(A, B, Q, R, N)``

    where `dsys` is a discrete-time `StateSpace` system, and `A`, `B`,
    `Q`, `R`, and `N` are 2d arrays of appropriate dimension (`dsys.dt` must
    not be 0.)

    Parameters
    ----------
    A, B : 2D array
        Dynamics and input matrices.
    dsys : LTI `StateSpace`
        Discrete-time linear system.
    Q, R : 2D array
        State and input weight matrices.
    N : 2D array, optional
        Cross weight matrix.
    integral_action : ndarray, optional
        If this keyword is specified, the controller includes integral
        action in addition to state feedback.  The value of the
        `integral_action` keyword should be an ndarray that will be
        multiplied by the current state to generate the error for the
        internal integrator states of the control law.  The number of
        outputs that are to be integrated must match the number of
        additional rows and columns in the `Q` matrix.
    method : str, optional
        Set the method used for computing the result.  Current methods are
        'slycot' and 'scipy'.  If set to None (default), try 'slycot'
        first and then 'scipy'.

    Returns
    -------
    K : 2D array
        State feedback gains.
    S : 2D array
        Solution to Riccati equation.
    E : 1D array
        Eigenvalues of the closed loop system.

    See Also
    --------
    lqr, lqe, dlqe

    Examples
    --------
    >>> K, S, E = dlqr(dsys, Q, R, [N])                         # doctest: +SKIP
    >>> K, S, E = dlqr(A, B, Q, R, [N])                         # doctest: +SKIP
    """

    #
    # Process the arguments and figure out what inputs we received
    #

    # Get the system description
    if (len(args) < 3):
        raise ControlArgument("not enough input arguments")

    # If we were passed a continues time system as the first arg, raise error
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
    S, E, K = dare(A, B, Q, R, N, method=method, _Ss="N")
    return K, S, E


# Function to create an I/O systems representing a state feedback controller
def create_statefbk_iosystem(
        sys, gain, feedfwd_gain=None, integral_action=None, estimator=None,
        controller_type=None, xd_labels=None, ud_labels=None, ref_labels=None,
        feedfwd_pattern='trajgen', gainsched_indices=None,
        gainsched_method='linear', control_indices=None, state_indices=None,
        name=None, inputs=None, outputs=None, states=None, params=None,
        **kwargs):
    r"""Create an I/O system using a (full) state feedback controller.

    This function creates an input/output system that implements a
    state feedback controller of the form

    .. math:: u = u_d - K_p (x - x_d) - K_i \int(C x - C x_d)

    by calling

        ctrl, clsys = ct.create_statefbk_iosystem(sys, K)

    where `sys` is the process dynamics and `K` is the state (+ integral)
    feedback gain (e.g., from LQR).  The function returns the controller
    `ctrl` and the closed loop systems `clsys`, both as I/O systems.

    A gain scheduled controller can also be created, by passing a list of
    gains and a corresponding list of values of a set of scheduling
    variables.  In this case, the controller has the form

    .. math:: u = u_d - K_p(\mu) (x - x_d) - K_i(\mu) \int(C x - C x_d)

    where :math:`\mu` represents the scheduling variable.

    Alternatively, a controller of the form

    .. math:: u = k_f r - K_p x - K_i \int(C x - r)

    can be created by calling

        ctrl, clsys = ct.create_statefbk_iosystem(
            sys, K, kf, feedfwd_pattern='refgain')

    In either form, an estimator can also be used to compute the estimated
    state from the input and output measurements.

    Parameters
    ----------
    sys : `NonlinearIOSystem`
        The I/O system that represents the process dynamics.  If no estimator
        is given, the output of this system should represent the full state.

    gain : ndarray, tuple, or I/O system
        If an array is given, it represents the state feedback gain (`K`).
        This matrix defines the gains to be applied to the system.  If
        `integral_action` is None, then the dimensions of this array
        should be (sys.ninputs, sys.nstates).  If `integral action` is
        set to a matrix or a function, then additional columns
        represent the gains of the integral states of the controller.

        If a tuple is given, then it specifies a gain schedule.  The tuple
        should be of the form ``(gains, points)`` where gains is a list of
        gains `K_j` and points is a list of values `mu_j` at which the
        gains are computed.  The `gainsched_indices` parameter should be
        used to specify the scheduling variables.

        If an I/O system is given, the error e = x - xd is passed to the
        system and the output is used as the feedback compensation term.

    feedfwd_gain : array_like, optional
        Specify the feedforward gain, `k_f`.  Used only for the reference
        gain design pattern.  If not given and if `sys` is a `StateSpace`
        (linear) system, will be computed as -1/(C (A-BK)^{-1}) B.

    feedfwd_pattern : str, optional
        If set to 'refgain', the reference gain design pattern is used to
        create the controller instead of the trajectory generation
        ('trajgen') pattern.

    integral_action : ndarray, optional
        If this keyword is specified, the controller can include integral
        action in addition to state feedback.  The value of the
        `integral_action` keyword should be an ndarray that will be
        multiplied by the current and desired state to generate the error
        for the internal integrator states of the control law.

    estimator : `NonlinearIOSystem`, optional
        If an estimator is provided, use the states of the estimator as
        the system inputs for the controller.

    gainsched_indices : int, slice, or list of int or str, optional
        If a gain scheduled controller is specified, specify the indices of
        the controller input to use for scheduling the gain. The input to
        the controller is the desired state `x_d`, the desired input `u_d`,
        and the system state `x` (or state estimate `xhat`, if an
        estimator is given). If value is an integer `q`, the first `q`
        values of the ``[x_d, u_d, x]`` vector are used.  Otherwise, the
        value should be a slice or a list of indices.  The list of indices
        can be specified as either integer offsets or as signal names. The
        default is to use the desired state `x_d`.

    gainsched_method : str, optional
        The method to use for gain scheduling.  Possible values are 'linear'
        (default), 'nearest', and 'cubic'.  More information is available in
        `scipy.interpolate.griddata`. For points outside of the convex
        hull of the scheduling points, the gain at the nearest point is
        used.

    controller_type : 'linear' or 'nonlinear', optional
        Set the type of controller to create. The default for a linear gain
        is a linear controller implementing the LQR regulator. If the type
        is 'nonlinear', a `NonlinearIOSystem` is created instead, with
        the gain `K` as a parameter (allowing modifications of the gain at
        runtime). If the gain parameter is a tuple, then a nonlinear,
        gain-scheduled controller is created.

    Returns
    -------
    ctrl : `NonlinearIOSystem`
        Input/output system representing the controller.  For the 'trajgen'
        design pattern (default), this system takes as inputs the desired
        state `x_d`, the desired input `u_d`, and either the system state
        `x` or the estimated state `xhat`.  It outputs the controller
        action `u` according to the formula u = u_d - K(x - x_d).  For
        the 'refgain' design pattern, the system takes as inputs the
        reference input `r` and the system or estimated state. If the
        keyword `integral_action` is specified, then an additional set of
        integrators is included in the control system (with the gain matrix
        `K` having the integral gains appended after the state gains).  If
        a gain scheduled controller is specified, the gain (proportional
        and integral) are evaluated using the scheduling variables
        specified by `gainsched_indices`.

    clsys : `NonlinearIOSystem`
        Input/output system representing the closed loop system.  This
        system takes as inputs the desired trajectory (x_d, u_d) and
        outputs the system state `x` and the applied input `u`
        (vertically stacked).

    Other Parameters
    ----------------
    control_indices : int, slice, or list of int or str, optional
        Specify the indices of the system inputs that should be determined
        by the state feedback controller.  If value is an integer `m`, the
        first `m` system inputs are used.  Otherwise, the value should be a
        slice or a list of indices.  The list of indices can be specified
        as either integer offsets or as system input signal names.  If not
        specified, defaults to the system inputs.

    state_indices : int, slice, or list of int or str, optional
        Specify the indices of the system (or estimator) outputs that should
        be used by the state feedback controller.  If value is an integer
        `n`, the first `n` system states are used.  Otherwise, the value
        should be a slice or a list of indices.  The list of indices can be
        specified as either integer offsets or as estimator/system output
        signal names.  If not specified, defaults to the system states.

    xd_labels, ud_labels, ref_labels : str or list of str, optional
        Set the name of the signals to use for the desired state and inputs
        or the reference inputs (for the 'refgain' design pattern).  If a
        single string is specified, it should be a format string using the
        variable `i` as an index.  Otherwise, a list of strings matching
        the size of x_d and u_d, respectively, should be used.
        Default is "xd[{i}]" for xd_labels and "ud[{i}]" for ud_labels.
        These settings can also be overridden using the `inputs` keyword.

    inputs, outputs, states : str, or list of str, optional
        List of strings that name the individual signals of the transformed
        system.  If not given, the inputs, outputs, and states are the same
        as the original system.

    name : string, optional
        System name. If unspecified, a generic name 'sys[id]' is generated
        with a unique integer id.

    params : dict, optional
        System parameter values.  By default, these will be copied from
        `sys` and `ctrl`, but can be overridden with this keyword.

    Examples
    --------
    >>> import control as ct
    >>> import numpy as np
    >>>
    >>> A = [[0, 1], [-0.5, -0.1]]
    >>> B = [[0], [1]]
    >>> C = np.eye(2)
    >>> D = np.zeros((2, 1))
    >>> sys = ct.ss(A, B, C, D)
    >>>
    >>> Q = np.eye(2)
    >>> R = np.eye(1)
    >>>
    >>> K, _, _ = ct.lqr(sys,Q,R)
    >>> ctrl, clsys = ct.create_statefbk_iosystem(sys, K)

    """
    # Make sure that we were passed an I/O system as an input
    if not isinstance(sys, NonlinearIOSystem):
        raise ControlArgument("Input system must be I/O system")

    # Process keywords
    params = sys.params if params is None else params
    controller_type = _process_legacy_keyword(
        kwargs, 'type', 'controller_type', controller_type)
    if kwargs:
        raise TypeError("unrecognized keywords: ", str(kwargs))

    # Check for consistency of positional parameters
    if feedfwd_gain is not None and feedfwd_pattern != 'refgain':
        raise ControlArgument(
            "feedfwd_gain specified but feedfwd_pattern != 'refgain'")

    # Figure out what inputs to the system to use
    control_indices = _process_indices(
        control_indices, 'control', sys.input_labels, sys.ninputs)
    sys_ninputs = len(control_indices)

    # Decide what system is going to pass the states to the controller
    if estimator is None:
        estimator = sys

    # Figure out what outputs (states) from the system/estimator to use
    state_indices = _process_indices(
        state_indices, 'state', estimator.state_labels, sys.nstates)
    sys_nstates = len(state_indices)

    # Make sure the system/estimator states are proper dimension
    if estimator.noutputs < sys_nstates:
        # If no estimator, make sure that the system has all states as outputs
        raise ControlArgument(
            ("system" if estimator == sys else "estimator") +
            " output must include the full state")
    elif estimator == sys:
        # Issue a warning if we can't verify state output
        if (isinstance(sys, NonlinearIOSystem) and
            not isinstance(sys, StateSpace) and sys.outfcn is not None) or \
           (isinstance(sys, StateSpace) and
            not (np.all(sys.C[np.ix_(state_indices, state_indices)] ==
                        np.eye(sys_nstates)) and
                 np.all(sys.D[state_indices, :] == 0))):
            warnings.warn("cannot verify system output is system state")

    # See whether we should implement integral action
    nintegrators = 0
    if integral_action is not None:
        if not isinstance(integral_action, np.ndarray):
            raise ControlArgument("Integral action must pass an array")

        C = np.atleast_2d(integral_action)
        if C.shape[1] != sys_nstates:
            raise ControlArgument(
                "Integral gain size must match system state size")
        nintegrators = C.shape[0]
    else:
        # Create a C matrix with no outputs, just in case update gets called
        C = np.zeros((0, sys_nstates))

    # Check to make sure that state feedback has the right shape
    if isinstance(gain, np.ndarray):
        K = gain
        if K.shape != (sys_ninputs, estimator.noutputs + nintegrators):
            raise ControlArgument(
                f'control gain must be an array of size {sys_ninputs}'
                f' x {sys_nstates}' +
                (f'+{nintegrators}' if nintegrators > 0 else ''))
        gainsched = False

    elif isinstance(gain, tuple):
        # Check for gain scheduled controller
        if len(gain) != 2:
            raise ControlArgument("gain must be a 2-tuple for gain scheduling")
        elif feedfwd_pattern != 'trajgen':
            raise NotImplementedError(
                "Gain scheduling is not implemented for pattern "
                f"'{feedfwd_pattern}'")
        gains, points = gain[0:2]

        # Stack gains and points if past as a list
        gains = np.stack(gains)
        points = np.stack(points)
        gainsched = True

    elif isinstance(gain, NonlinearIOSystem) and feedfwd_pattern != 'refgain':
        if controller_type not in ['iosystem', None]:
            raise ControlArgument(
                f"incompatible controller type '{controller_type}'")
        fbkctrl = gain
        controller_type = 'iosystem'
        gainsched = False

    else:
        raise ControlArgument("gain must be an array or a tuple")

    # Decide on the type of system to create
    if gainsched and controller_type == 'linear':
        raise ControlArgument(
            "controller_type 'linear' not allowed for"
            " gain scheduled controller")
    elif controller_type is None:
        controller_type = 'nonlinear' if gainsched else 'linear'
    elif controller_type not in {'linear', 'nonlinear', 'iosystem'}:
        raise ControlArgument(f"unknown controller_type '{controller_type}'")

    # Figure out the labels to use
    if feedfwd_pattern == 'trajgen':
        xd_labels = _process_labels(xd_labels, 'xd', [
            'xd[{i}]'.format(i=i) for i in range(sys_nstates)])
        ud_labels = _process_labels(ud_labels, 'ud', [
            'ud[{i}]'.format(i=i) for i in range(sys_ninputs)])

        # Create the signal and system names
        if inputs is None:
            inputs = xd_labels + ud_labels + estimator.output_labels
    elif feedfwd_pattern == 'refgain':
        ref_labels = _process_labels(ref_labels, 'r', [
            f'r[{i}]' for i in range(sys_ninputs)])
        if inputs is None:
            inputs = ref_labels + estimator.output_labels
    else:
        raise NotImplementedError(f"unknown pattern '{feedfwd_pattern}'")

    if outputs is None:
        outputs = [sys.input_labels[i] for i in control_indices]
    if states is None:
        states = nintegrators

    # Process gain scheduling variables, if present
    if gainsched:
        # Create a copy of the scheduling variable indices (default = xd)
        gainsched_indices = _process_indices(
            gainsched_indices, 'gainsched', inputs, sys_nstates)

        # If points is a 1D list, convert to 2D
        if points.ndim == 1:
            points = points.reshape(-1, 1)

        # Make sure the scheduling variable indices are the right length
        if len(gainsched_indices) != points.shape[1]:
            raise ControlArgument(
                "length of gainsched_indices must match dimension of"
                " scheduling variables")

        # Create interpolating function
        if points.shape[1] < 2:
            _interp = sp.interpolate.interp1d(
                points[:, 0], gains, axis=0, kind=gainsched_method)
            _nearest = sp.interpolate.interp1d(
                points[:, 0], gains, axis=0, kind='nearest')
        elif gainsched_method == 'nearest':
            _interp = sp.interpolate.NearestNDInterpolator(points, gains)
            def _nearest(mu):
                raise SystemError(f"could not find nearest gain at mu = {mu}")
        elif gainsched_method == 'linear':
            _interp = sp.interpolate.LinearNDInterpolator(points, gains)
            _nearest = sp.interpolate.NearestNDInterpolator(points, gains)
        elif gainsched_method == 'cubic':
            _interp = sp.interpolate.CloughTocher2DInterpolator(points, gains)
            _nearest = sp.interpolate.NearestNDInterpolator(points, gains)
        else:
            raise ControlArgument(
                f"unknown gain scheduling method '{gainsched_method}'")

        def _compute_gain(mu):
            K = _interp(mu)
            if np.isnan(K).any():
                K = _nearest(mu)
            return K

    # Define the controller system
    if controller_type == 'nonlinear' and feedfwd_pattern == 'trajgen':
        # Create an I/O system for the state feedback gains
        def _control_update(t, states, inputs, params):
            # Split input into desired state, nominal input, and current state
            xd_vec = inputs[0:sys_nstates]
            x_vec = inputs[-sys_nstates:]

            # Compute the integral error in the xy coordinates
            return C @ (x_vec - xd_vec)

        def _control_output(t, states, inputs, params):
            if gainsched:
                mu = inputs[gainsched_indices]
                K_ = _compute_gain(mu)
            else:
                K_ = params.get('K')

            # Split input into desired state, nominal input, and current state
            xd_vec = inputs[0:sys_nstates]
            ud_vec = inputs[sys_nstates:sys_nstates + sys_ninputs]
            x_vec = inputs[-sys_nstates:]

            # Compute the control law
            u = ud_vec - K_[:, 0:sys_nstates] @ (x_vec - xd_vec)
            if nintegrators > 0:
                u -= K_[:, sys_nstates:] @ states

            return u

        ctrl_params = {} if gainsched else {'K': K}
        ctrl = NonlinearIOSystem(
            _control_update, _control_output, name=name, inputs=inputs,
            outputs=outputs, states=states, params=ctrl_params)

    elif controller_type == 'iosystem' and feedfwd_pattern == 'trajgen':
        # Use the passed system to compute feedback compensation
        def _control_update(t, states, inputs, params):
            # Split input into desired state, nominal input, and current state
            xd_vec = inputs[0:sys_nstates]
            x_vec = inputs[-sys_nstates:]

            # Compute the integral error in the xy coordinates
            return fbkctrl.updfcn(t, states, (x_vec - xd_vec), params)

        def _control_output(t, states, inputs, params):
            # Split input into desired state, nominal input, and current state
            xd_vec = inputs[0:sys_nstates]
            ud_vec = inputs[sys_nstates:sys_nstates + sys_ninputs]
            x_vec = inputs[-sys_nstates:]

            # Compute the control law
            return ud_vec + fbkctrl.outfcn(t, states, (x_vec - xd_vec), params)

        # TODO: add a way to pass parameters
        ctrl = NonlinearIOSystem(
            _control_update, _control_output, name=name, inputs=inputs,
            outputs=outputs, states=fbkctrl.state_labels, dt=fbkctrl.dt)

    elif controller_type in 'linear' and feedfwd_pattern == 'trajgen':
        # Create the matrices implementing the controller
        if isctime(sys):
            # Continuous time: integrator
            A_lqr = np.zeros((C.shape[0], C.shape[0]))
        else:
            # Discrete time: summer
            A_lqr = np.eye(C.shape[0])
        B_lqr = np.hstack([-C, np.zeros((C.shape[0], sys_ninputs)), C])
        C_lqr = -K[:, sys_nstates:]
        D_lqr = np.hstack([
            K[:, 0:sys_nstates], np.eye(sys_ninputs), -K[:, 0:sys_nstates]
        ])

        ctrl = ss(
            A_lqr, B_lqr, C_lqr, D_lqr, dt=sys.dt, name=name,
            inputs=inputs, outputs=outputs, states=states)

    elif feedfwd_pattern == 'refgain':
        if controller_type not in ['linear', 'iosystem']:
            raise ControlArgument(
                "refgain design pattern only supports linear controllers")

        if feedfwd_gain is None:
            raise ControlArgument(
                "'feedfwd_gain' required for reference gain pattern")

        # Check to make sure the reference gain is valid
        Kf = np.atleast_2d(feedfwd_gain)
        if Kf.ndim != 2 or Kf.shape[0] != sys.ninputs or \
           Kf.shape[1] != sys.ninputs:
            raise ControlArgument("feedfwd_gain is not the right shape")

        # Create the matrices implementing the controller
        # [r, x]->[u]: u = k_f r - K_p x - K_i \int(C x - r)
        if isctime(sys):
            # Continuous time: integrator
            A_lqr = np.zeros((C.shape[0], C.shape[0]))
        else:
            # Discrete time: summer
            A_lqr = np.eye(C.shape[0])
        B_lqr = np.hstack([-np.eye(C.shape[0], sys_ninputs), C])
        C_lqr = -K[:, sys_nstates:]                     # integral gain (opt)
        D_lqr = np.hstack([Kf, -K])

        ctrl = ss(
            A_lqr, B_lqr, C_lqr, D_lqr, dt=sys.dt, name=name,
            inputs=inputs, outputs=outputs, states=states)

    else:
        raise ControlArgument(f"unknown controller_type '{controller_type}'")

    # Define the closed loop system
    inplist=inputs[:-sys.nstates]
    input_labels=inputs[:-sys.nstates]
    outlist=sys.output_labels + [sys.input_labels[i] for i in control_indices]
    output_labels=sys.output_labels + \
        [sys.input_labels[i] for i in control_indices]
    closed = interconnect(
        [sys, ctrl] if estimator == sys else [sys, ctrl, estimator],
        name=sys.name + "_" + ctrl.name, add_unused=True,
        inplist=inplist, inputs=input_labels,
        outlist=outlist, outputs=output_labels,
        params= ctrl.params | params
    )
    return ctrl, closed


def ctrb(A, B, t=None):
    """Controllability matrix.

    Parameters
    ----------
    A, B : array_like or string
        Dynamics and input matrix of the system.
    t : None or integer
        Maximum time horizon of the controllability matrix, max = A.shape[0].

    Returns
    -------
    C : 2D array
        Controllability matrix.

    Examples
    --------
    >>> G = ct.tf2ss([1], [1, 2, 3])
    >>> C = ct.ctrb(G.A, G.B)
    >>> np.linalg.matrix_rank(C)
    np.int64(2)

    """

    # Convert input parameters to matrices (if they aren't already)
    A = _ssmatrix(A, square=True, name="A")
    n = A.shape[0]

    B = _ssmatrix(B, axis=0, rows=n, name="B")
    m = B.shape[1]

    if t is None or t > n:
        t = n

    # Construct the controllability matrix
    ctrb = np.zeros((n, t * m))
    ctrb[:, :m] = B
    for k in range(1, t):
        ctrb[:, k * m:(k + 1) * m] = np.dot(A, ctrb[:, (k - 1) * m:k * m])

    return ctrb


def obsv(A, C, t=None):
    """Observability matrix.

    Parameters
    ----------
    A, C : array_like or string
        Dynamics and output matrix of the system.
    t : None or integer
        Maximum time horizon of the controllability matrix, max = A.shape[0].

    Returns
    -------
    O : 2D array
        Observability matrix.

    Examples
    --------
    >>> G = ct.tf2ss([1], [1, 2, 3])
    >>> C = ct.obsv(G.A, G.C)
    >>> np.linalg.matrix_rank(C)
    np.int64(2)

    """
    # Convert input parameters to matrices (if they aren't already)
    A = _ssmatrix(A, square=True, name="A")
    n = A.shape[0]

    C = _ssmatrix(C, cols=n, name="C")
    p = C.shape[0]

    if t is None or t > n:
        t = n

    # Construct the observability matrix
    obsv = np.zeros((t * p, n))
    obsv[:p, :] = C

    for k in range(1, t):
        obsv[k * p:(k + 1) * p, :] = np.dot(obsv[(k - 1) * p:k * p, :], A)

    return obsv


def gram(sys, type):
    """Gramian (controllability or observability).

    Parameters
    ----------
    sys : `StateSpace`
        System description.
    type : String
        Type of desired computation.  `type` is either 'c' (controllability)
        or 'o' (observability). To compute the Cholesky factors of Gramians
        use 'cf' (controllability) or 'of' (observability).

    Returns
    -------
    gram : 2D array
        Gramian of system.

    Raises
    ------
    ValueError
        * If system is not instance of `StateSpace` class, or
        * if `type` is not 'c', 'o', 'cf' or 'of', or
        * if system is unstable (sys.A has eigenvalues not in left half plane).

    ControlSlycot
        If slycot routine sb03md cannot be found or
        if slycot routine sb03od cannot be found.

    Examples
    --------
    >>> G = ct.rss(4)
    >>> Wc = ct.gram(G, 'c')
    >>> Wo = ct.gram(G, 'o')
    >>> Rc = ct.gram(G, 'cf')  # where Wc = Rc' * Rc
    >>> Ro = ct.gram(G, 'of')  # where Wo = Ro' * Ro

    """

    # Check for ss system object
    if not isinstance(sys, statesp.StateSpace):
        raise ValueError("System must be StateSpace!")
    if type not in ['c', 'o', 'cf', 'of']:
        raise ValueError("That type is not supported!")

    # Check if system is continuous or discrete
    if sys.isctime():
        dico = 'C'

        # TODO: Check system is stable, perhaps a utility in ctrlutil.py
        # or a method of the StateSpace class?
        if np.any(np.linalg.eigvals(sys.A).real >= 0.0):
            raise ValueError("Oops, the system is unstable!")

    else:
        assert sys.isdtime()
        dico = 'D'

        if np.any(np.abs(sys.poles()) >= 1.):
            raise ValueError("Oops, the system is unstable!")

    if type == 'c' or type == 'o':
        # Compute Gramian by the Slycot routine sb03md
        # make sure Slycot is installed
        if sb03md is None:
            raise ControlSlycot("can't find slycot module sb03md")
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
        return gram

    elif type == 'cf' or type == 'of':
        # Compute Cholesky factored Gramian from Slycot routine sb03od
        if sb03od is None:
            raise ControlSlycot("can't find slycot module sb03od")
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
        return gram


# Short versions of functions
acker = place_acker
