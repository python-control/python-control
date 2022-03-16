# stochsys.py - stochastic systems module
# RMM, 16 Mar 2022
#
# This module contains functions that are intended to be used for analysis
# and design of stochastic control systems, mainly involving Kalman
# filtering and its variants.
#

"""The :mod:`~control.stochsys` module contains functions for analyzing and
designing stochastic (control) systems, including white noise processes and
Kalman filtering.

"""

__license__ = "BSD"
__maintainer__ = "Richard Murray"
__email__ = "murray@cds.caltech.edu"

import numpy as np

from .iosys import InputOutputSystem, NonlinearIOSystem
from .lti import LTI, isctime, isdtime
from .mateqn import care, dare, _check_shape
from .statesp import StateSpace, _ssmatrix
from .exception import ControlArgument, ControlNotImplemented

__all__ = ['lqe','dlqe', 'create_estimator_iosystem']


# contributed by Sawyer B. Fuller <minster@uw.edu>
def lqe(*args, **keywords):
    """lqe(A, G, C, QN, RN, [, NN])

    Linear quadratic estimator design (Kalman filter) for continuous-time
    systems. Given the system

    .. math::

        x &= Ax + Bu + Gw \\\\
        y &= Cx + Du + v

    with unbiased process noise w and measurement noise v with covariances

    .. math::       E{ww'} = QN,    E{vv'} = RN,    E{wv'} = NN

    The lqe() function computes the observer gain matrix L such that the
    stationary (non-time-varying) Kalman filter

    .. math:: x_e = A x_e + B u + L(y - C x_e - D u)

    produces a state estimate x_e that minimizes the expected squared error
    using the sensor measurements y. The noise cross-correlation `NN` is
    set to zero when omitted.

    The function can be called with either 3, 4, 5, or 6 arguments:

    * ``L, P, E = lqe(sys, QN, RN)``
    * ``L, P, E = lqe(sys, QN, RN, NN)``
    * ``L, P, E = lqe(A, G, C, QN, RN)``
    * ``L, P, E = lqe(A, G, C, QN, RN, NN)``

    where `sys` is an `LTI` object, and `A`, `G`, `C`, `QN`, `RN`, and `NN`
    are 2D arrays or matrices of appropriate dimension.

    Parameters
    ----------
    A, G, C : 2D array_like
        Dynamics, process noise (disturbance), and output matrices
    sys : LTI (StateSpace or TransferFunction)
        Linear I/O system, with the process noise input taken as the system
        input.
    QN, RN : 2D array_like
        Process and sensor noise covariance matrices
    NN : 2D array, optional
        Cross covariance matrix.  Not currently implemented.
    method : str, optional
        Set the method used for computing the result.  Current methods are
        'slycot' and 'scipy'.  If set to None (default), try 'slycot' first
        and then 'scipy'.

    Returns
    -------
    L : 2D array (or matrix)
        Kalman estimator gain
    P : 2D array (or matrix)
        Solution to Riccati equation

        .. math::

            A P + P A^T - (P C^T + G N) R^{-1}  (C P + N^T G^T) + G Q G^T = 0

    E : 1D array
        Eigenvalues of estimator poles eig(A - L C)

    Notes
    -----
    1. If the first argument is an LTI object, then this object will be used
       to define the dynamics, noise and output matrices.  Furthermore, if
       the LTI object corresponds to a discrete time system, the ``dlqe()``
       function will be called.

    2. The return type for 2D arrays depends on the default class set for
       state space operations.  See :func:`~control.use_numpy_matrix`.

    Examples
    --------
    >>> L, P, E = lqe(A, G, C, QN, RN)
    >>> L, P, E = lqe(A, G, C, Q, RN, NN)

    See Also
    --------
    lqr, dlqe, dlqr

    """

    # TODO: incorporate cross-covariance NN, something like this,
    # which doesn't work for some reason
    # if NN is None:
    #    NN = np.zeros(QN.size(0),RN.size(1))
    # NG = G @ NN

    #
    # Process the arguments and figure out what inputs we received
    #

    # Get the method to use (if specified as a keyword)
    method = keywords.get('method', None)

    # Get the system description
    if (len(args) < 3):
        raise ControlArgument("not enough input arguments")

    # If we were passed a discrete time system as the first arg, use dlqe()
    if isinstance(args[0], LTI) and isdtime(args[0], strict=True):
        # Call dlqe
        return dlqe(*args, **keywords)

    # If we were passed a state space  system, use that to get system matrices
    if isinstance(args[0], StateSpace):
        A = np.array(args[0].A, ndmin=2, dtype=float)
        G = np.array(args[0].B, ndmin=2, dtype=float)
        C = np.array(args[0].C, ndmin=2, dtype=float)
        index = 1

    elif isinstance(args[0], LTI):
        # Don't allow other types of LTI systems
        raise ControlArgument("LTI system must be in state space form")

    else:
        # Arguments should be A and B matrices
        A = np.array(args[0], ndmin=2, dtype=float)
        G = np.array(args[1], ndmin=2, dtype=float)
        C = np.array(args[2], ndmin=2, dtype=float)
        index = 3

    # Get the weighting matrices (converting to matrices, if needed)
    QN = np.array(args[index], ndmin=2, dtype=float)
    RN = np.array(args[index+1], ndmin=2, dtype=float)

    # Get the cross-covariance matrix, if given
    if (len(args) > index + 2):
        NN = np.array(args[index+2], ndmin=2, dtype=float)
        raise ControlNotImplemented("cross-covariance not implemented")

    else:
        # For future use (not currently used below)
        NN = np.zeros((QN.shape[0], RN.shape[1]))

    # Check dimensions of G (needed before calling care())
    _check_shape("QN", QN, G.shape[1], G.shape[1])

    # Compute the result (dimension and symmetry checking done in care())
    P, E, LT = care(A.T, C.T, G @ QN @ G.T, RN, method=method,
                    B_s="C", Q_s="QN", R_s="RN", S_s="NN")
    return _ssmatrix(LT.T), _ssmatrix(P), E


# contributed by Sawyer B. Fuller <minster@uw.edu>
def dlqe(*args, **keywords):
    """dlqe(A, G, C, QN, RN, [, N])

    Linear quadratic estimator design (Kalman filter) for discrete-time
    systems. Given the system

    .. math::

        x[n+1] &= Ax[n] + Bu[n] + Gw[n] \\\\
        y[n] &= Cx[n] + Du[n] + v[n]

    with unbiased process noise w and measurement noise v with covariances

    .. math::       E{ww'} = QN,    E{vv'} = RN,    E{wv'} = NN

    The dlqe() function computes the observer gain matrix L such that the
    stationary (non-time-varying) Kalman filter

    .. math:: x_e[n+1] = A x_e[n] + B u[n] + L(y[n] - C x_e[n] - D u[n])

    produces a state estimate x_e[n] that minimizes the expected squared error
    using the sensor measurements y. The noise cross-correlation `NN` is
    set to zero when omitted.

    Parameters
    ----------
    A, G : 2D array_like
        Dynamics and noise input matrices
    QN, RN : 2D array_like
        Process and sensor noise covariance matrices
    NN : 2D array, optional
        Cross covariance matrix (not yet supported)
    method : str, optional
        Set the method used for computing the result.  Current methods are
        'slycot' and 'scipy'.  If set to None (default), try 'slycot' first
        and then 'scipy'.

    Returns
    -------
    L : 2D array (or matrix)
        Kalman estimator gain
    P : 2D array (or matrix)
        Solution to Riccati equation

        .. math::

            A P + P A^T - (P C^T + G N) R^{-1}  (C P + N^T G^T) + G Q G^T = 0

    E : 1D array
        Eigenvalues of estimator poles eig(A - L C)

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    Examples
    --------
    >>> L, P, E = dlqe(A, G, C, QN, RN)
    >>> L, P, E = dlqe(A, G, C, QN, RN, NN)

    See Also
    --------
    dlqr, lqe, lqr

    """

    #
    # Process the arguments and figure out what inputs we received
    #

    # Get the method to use (if specified as a keyword)
    method = keywords.get('method', None)

    # Get the system description
    if (len(args) < 3):
        raise ControlArgument("not enough input arguments")

    # If we were passed a continus time system as the first arg, raise error
    if isinstance(args[0], LTI) and isctime(args[0], strict=True):
        raise ControlArgument("dlqr() called with a continuous time system")

    # If we were passed a state space  system, use that to get system matrices
    if isinstance(args[0], StateSpace):
        A = np.array(args[0].A, ndmin=2, dtype=float)
        G = np.array(args[0].B, ndmin=2, dtype=float)
        C = np.array(args[0].C, ndmin=2, dtype=float)
        index = 1

    elif isinstance(args[0], LTI):
        # Don't allow other types of LTI systems
        raise ControlArgument("LTI system must be in state space form")

    else:
        # Arguments should be A and B matrices
        A = np.array(args[0], ndmin=2, dtype=float)
        G = np.array(args[1], ndmin=2, dtype=float)
        C = np.array(args[2], ndmin=2, dtype=float)
        index = 3

    # Get the weighting matrices (converting to matrices, if needed)
    QN = np.array(args[index], ndmin=2, dtype=float)
    RN = np.array(args[index+1], ndmin=2, dtype=float)

    # TODO: incorporate cross-covariance NN, something like this,
    # which doesn't work for some reason
    # if NN is None:
    #    NN = np.zeros(QN.size(0),RN.size(1))
    # NG = G @ NN
    if len(args) > index + 2:
        NN = np.array(args[index+2], ndmin=2, dtype=float)
        raise ControlNotImplemented("cross-covariance not yet implememented")

    # Check dimensions of G (needed before calling care())
    _check_shape("QN", QN, G.shape[1], G.shape[1])

    # Compute the result (dimension and symmetry checking done in dare())
    P, E, LT = dare(A.T, C.T, G @ QN @ G.T, RN, method=method,
                    B_s="C", Q_s="QN", R_s="RN", S_s="NN")
    return _ssmatrix(LT.T), _ssmatrix(P), E


# Function to create an estimator
def create_estimator_iosystem(
        sys, QN, RN, P0=None, G=None, C=None,
        state_labels='xhat[{i}]', output_labels='xhat[{i}]',
        covariance_labels='P[{i},{j}]'):
    """Create an I/O system implementing a linqear quadratic estimator

    This function creates an input/output system that implements a
    state estimator of the form

        xhat[k + 1] = A x[k] + B u[k] - L (C xhat[k] - y[k])
        P[k + 1] = A P A^T + F QN F^T - A P C^T Reps^{-1} C P A
        L = A P C^T Reps^{-1}

    where Reps = RN + C P C^T.  It can be called in the form

        estim = ct.create_estimator_iosystem(sys, QN, RN)

    where ``sys`` is the process dynamics and QN and RN are the covariance of
    the disturbance noise and sensor noise.  The function returns the
    estimator ``estim`` as I/O systems.

    Parameters
    ----------
    sys : InputOutputSystem
        The I/O system that represents the process dynamics.  If no estimator
        is given, the output of this system should represent the full state.
    QN, RN : ndarray
        Process and sensor noise covariance matrices.
    P0 : ndarray, optional
        Initial covariance matrix.  If not specified, defaults to the steady
        state covariance.
    G : ndarray, optional
        Disturbance matrix describing how the disturbances enters the
        dynamics.  Defaults to sys.B.
    C : ndarray, optional
        If the system has all full states output, define the measured values
        to be used by the estimator.  Otherwise, use the system output as the
        measured values.
    {state, covariance, output}_labels : str or list of str, optional
        Set the name of the signals to use for the internal state, covariance,
        and output (state estimate).  If a single string is specified, it
        should be a format string using the variable ``i`` as an index (or
        ``i`` and ``j`` for covariance).  Otherwise, a list of strings
        matching the size of the respective signal should be used.  Default is
        ``'xhat[{i}]'`` for state and output labels and ``'P[{i},{j}]'`` for
        covariance labels.

    Returns
    -------
    estim : InputOutputSystem
        Input/output system representing the estimator.  This system takes the
        system input and output and generates the estimated state.

    Notes
    -----
    This function can be used with the ``create_statefbk_iosystem()`` function
    to create a closed loop, output-feedback, state space controller:

        K, _, _ = ct.lqr(sys, Q, R)
        est = ct.create_estimator_iosystem(sys, QN, RN, P0)
        ctrl, clsys = ct.create_statefbk_iosystem(sys, K, estimator=est)

    """

    # Make sure that we were passed an I/O system as an input
    if not isinstance(sys, InputOutputSystem):
        raise ControlArgument("Input system must be I/O system")

    # Extract the matrices that we need for easy reference
    A, B = sys.A, sys.B

    # Set the disturbance and output matrices
    G = sys.B if G is None else G
    if C is not None:
        # Make sure that we have the full system output
        if not np.array_equal(sys.C, np.eye(sys.nstates)):
            raise ValueError("System output must be full state")

        # Make sure that the output matches the size of RN
        if C.shape[0] != RN.shape[0]:
            raise ValueError("System output is the wrong size for C")
    else:
        # Use the system outputs as the sensor outputs
        C = sys.C

    # Initialize the covariance matrix
    if P0 is None:
        # Initalize P0 to the steady state value
        _, P0, _ = lqe(A, G, C, QN, RN)

    # Figure out the labels to use
    if isinstance(state_labels, str):
        # Generate the list of labels using the argument as a format string
        state_labels = [state_labels.format(i=i) for i in range(sys.nstates)]

    if isinstance(covariance_labels, str):
        # Generate the list of labels using the argument as a format string
        covariance_labels = [
            covariance_labels.format(i=i, j=j) \
            for i in range(sys.nstates) for j in range(sys.nstates)]

    if isinstance(output_labels, str):
        # Generate the list of labels using the argument as a format string
        output_labels = [output_labels.format(i=i) for i in range(sys.nstates)]

    if isctime(sys):
        raise NotImplementedError("continuous time not yet implemented")

    else:
        # Create an I/O system for the state feedback gains
        def _estim_update(t, x, u, params):
            # See if we are estimating or predicting
            correct = params.get('correct', True)

            # Get the state of the estimator
            x = np.array(x)             # bug fix for python-control 0.9.1
            xhat = x[0:sys.nstates]
            P = x[sys.nstates:].reshape(sys.nstates, sys.nstates)

            # Extract the inputs to the estimator
            y = u[0:C.shape[0]]
            u = u[C.shape[0]:]

            # Compute the optimal again
            Reps_inv = np.linalg.inv(RN + C @ P @ C.T)
            L = A @ P @ C.T @ Reps_inv

            # Update the state estimate
            dxhat = A @ xhat + B @ u            # prediction
            if correct:
                dxhat -= L @ (C @ xhat - y)     # correction

            # Update the covariance
            dP = A @ P @ A.T + G @ QN @ G.T
            if correct:
                dP -= A @ P @ C.T @ Reps_inv @ C @ P @ A.T

            # Return the update
            return np.hstack([dxhat, dP.reshape(-1)])

        def _estim_output(t, x, u, params):
            return x[0:sys.nstates]

    # Define the estimator system
    return NonlinearIOSystem(
        _estim_update, _estim_output, states=state_labels + covariance_labels,
        inputs=sys.output_labels + sys.input_labels, outputs=output_labels,
        dt=sys.dt)
