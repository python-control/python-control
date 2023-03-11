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
import scipy as sp
from math import sqrt

from .iosys import InputOutputSystem, LinearIOSystem, NonlinearIOSystem
from .lti import LTI
from .namedio import isctime, isdtime, _process_indices
from .mateqn import care, dare, _check_shape
from .statesp import StateSpace, _ssmatrix
from .exception import ControlArgument, ControlNotImplemented

__all__ = ['lqe', 'dlqe', 'create_estimator_iosystem', 'white_noise',
           'correlation']


# contributed by Sawyer B. Fuller <minster@uw.edu>
def lqe(*args, **kwargs):
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

    # If we were passed a discrete time system as the first arg, use dlqe()
    if isinstance(args[0], LTI) and isdtime(args[0], strict=True):
        # Call dlqe
        return dlqe(*args, **kwargs)

    # Get the method to use (if specified as a keyword)
    method = kwargs.pop('method', None)
    if kwargs:
        raise TypeError("unrecognized keyword(s): ", str(kwargs))

    # Get the system description
    if (len(args) < 3):
        raise ControlArgument("not enough input arguments")

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
def dlqe(*args, **kwargs):
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
    method = kwargs.pop('method', None)
    if kwargs:
        raise TypeError("unrecognized keyword(s): ", str(kwargs))

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
#
# TODO: add `control_indices` keyword to match create_mhe_iosystem (?)
# TODO: change name to create_kalmanestimaor_iosystem (?)
# TODO: create predictor/corrector, UKF, and other variants (?)
#
def create_estimator_iosystem(
        sys, QN, RN, P0=None, G=None, C=None,
        control_indices=None, disturbance_indices=None,
        state_labels='xhat[{i}]', output_labels='xhat[{i}]',
        covariance_labels='P[{i},{j}]', sensor_labels=None):
    r"""Create an I/O system implementing a linear quadratic estimator

    This function creates an input/output system that implements a
    continuous time state estimator of the form

    .. math::

        d \hat{x}/dt &= A \hat{x} + B u - L (C \hat{x} - y) \\
           dP/dt &= A P + P A^T + F Q_N F^T - P C^T R_N^{-1} C P \\
               L &= P C^T R_N^{-1}

    or a discrete time state estimator of the form

    .. math::

        \hat{x}[k+1] &= A \hat{x}[k] + B u[k] - L (C \hat{x}[k] - y[k]) \\
               P[k+1] &= A P A^T + F Q_N F^T - A P C^T R_e^{-1} C P A \\
                    L &= A P C^T R_e^{-1}

    where :math:`R_e = R_N + C P C^T`.  It can be called in the form::

        estim = ct.create_estimator_iosystem(sys, QN, RN)

    where `sys` is the process dynamics and `QN` and `RN` are the covariance
    of the disturbance noise and sensor noise.  The function returns the
    estimator `estim` as I/O system with a parameter `correct` that can
    be used to turn off the correction term in the estimation (for forward
    predictions).

    Parameters
    ----------
    sys : LinearIOSystem
        The linear I/O system that represents the process dynamics.  If no
        estimator is given, the output of this system should represent the
        full state.
    QN, RN : ndarray
        Process and sensor noise covariance matrices.
    P0 : ndarray, optional
        Initial covariance matrix.  If not specified, defaults to the steady
        state covariance.
    G : ndarray, optional
        Disturbance matrix describing how the disturbances enters the
        dynamics.  Defaults to sys.B.
    C : ndarray, optional
        If the system has full state output, define the measured values to
        be used by the estimator.  Otherwise, use the system output as the
        measured values.

    Returns
    -------
    estim : InputOutputSystem
        Input/output system representing the estimator.  This system takes
        the system output y and input u and generates the estimated state
        xhat.

    Other Parameters
    ----------------
    control_indices : int, slice, or list of int or string, optional
        Specify the indices in the system input vector that correspond to
        the control inputs.  These inputs will be used as known control
        inputs for the estimator. If value is an integer `m`, the first `m`
        system inputs are used.  Otherwise, the value should be a slice or
        a list of indices.  The list of indices can be specified as either
        integer offsets or as system input signal names.  If not specified,
        defaults to the system inputs.
    disturbance_indices : int, list of int, or slice, optional
        Specify the indices in the system input vector that correspond to
        the unknown disturbances.  These inputs are assumed to be white
        noise with noise intensity QN.  If value is an integer `m`, the
        last `m` system inputs are used.  Otherwise, the value should be a
        slice or a list of indices.  The list of indices can be specified
        as either integer offsets or as system input signal names.  If not
        specified, the disturbances are assumed to be added to the system
        inputs.
    state_labels : str or list of str, optional
        Set the names of the internal state estimate variables.  If a
        single string is specified, it should be a format string using the
        variable `i` as an index.  Otherwise, a list of strings matching
        the number of system states should be used.  Default is "xhat[{i}]".
    covariance_labels : str or list of str, optional
        Set the name of the the covariance state variables.  If a single
        string is specified, it should be a format string using the
        variables `i` and `j` as indices.  Otherwise, a list of strings
        matching the size of the covariance matrix should be used.  Default
        is "P[{i},{j}]".
    sensor_labels : str or list of str, optional
        Set the name of the sensor signals (estimator inputs).  If
        specified, it should be a format string using the variable `i` as
        an index.  Otherwise, a list of strings matching the size of the
        measured system outputs should be used.  Default is "y[{i}]".
    output_labels : str or list of str, optional
        Set the name of the estimator outputs (state estimate).  If a
        single string is specified, it should be a format string using the
        variable `i` as an index.  Otherwise, a list of strings matching
        the size of the system state should be used.  Default is "xhat[{i}]".

    Notes
    -----
    This function can be used with the ``create_statefbk_iosystem()`` function
    to create a closed loop, output-feedback, state space controller::

        K, _, _ = ct.lqr(sys, Q, R)
        est = ct.create_estimator_iosystem(sys, QN, RN, P0)
        ctrl, clsys = ct.create_statefbk_iosystem(sys, K, estimator=est)

    The estimator can also be run on its own to process a noisy signal::

        resp = ct.input_output_response(est, T, [Y, U], [X0, P0])

    If desired, the ``correct`` parameter can be set to ``False`` to allow
    prediction with no additional sensor information::

        resp = ct.input_output_response(
           est, T, 0, [X0, P0], param={'correct': False)

    """

    # Make sure that we were passed an I/O system as an input
    if not isinstance(sys, LinearIOSystem):
        raise ControlArgument("Input system must be a linear I/O system")

    # Set the state matrix for later use
    A = sys.A

    # Set the disturbance matrices (indices take priority over G)
    ctrl_idx = _process_indices(
        control_indices, 'control', sys.input_labels, sys.ninputs)

    if disturbance_indices is None and control_indices is not None:
        # Disturbance indices are the complement of control indices
        dist_idx = [i for i in range(sys.ninputs) if i not in ctrl_idx]
        if G is not None:
            warn("'control_indices' and 'G' both specified; ignoring 'G'")
        G = sys.B[:, dist_idx]

    elif disturbance_indices is not None:
        if G is not None:
            warn("'disturbance_indices' and 'G' both specified; ignoring 'G'")

        # If passed an integer, count from the end of the input vector
        arg = -disturbance_indices if isinstance(disturbance_indices, int) \
            else disturbance_indices

        dist_idx = _process_indices(
            arg, 'disturbance', sys.input_labels, sys.ninputs)
        G = sys.B[:, dist_idx]

        # Set control indices to complement disturbance indices, if needed
        if control_indices is None:
            ctrl_idx = [i for i in range(sys.ninputs) if i not in dist_idx]

    elif G is None:
        G = sys.B

    # Set the input and direct matrices
    B = sys.B[:, ctrl_idx]
    if not np.allclose(sys.D, 0):
        raise NotImplemented("nonzero 'D' matrix not yet implemented")

    # Set the output matrices
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
        if sensor_labels is None:
            sensor_labels = sys.output_labels

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

    sensor_labels = 'y[{i}]' if sensor_labels is None else sensor_labels
    if isinstance(sensor_labels, str):
        # Generate the list of labels using the argument as a format string
        sensor_labels = [sensor_labels.format(i=i) for i in range(C.shape[0])]

    # Set the input labels based on the system input
    # TODO: allow these to be overriden
    input_labels = [sys.input_labels[i] for i in ctrl_idx]

    if isctime(sys):
        # Create an I/O system for the state feedback gains
        # Note: reshape vectors into column vectors for legacy np.matrix

        R_inv = np.linalg.inv(RN)
        Reps_inv = C.T @ R_inv @ C

        def _estim_update(t, x, u, params):
            # See if we are estimating or predicting
            correct = params.get('correct', True)

            # Get the state of the estimator
            xhat = x[0:sys.nstates].reshape(-1, 1)
            P = x[sys.nstates:].reshape(sys.nstates, sys.nstates)

            # Extract the inputs to the estimator
            y = u[0:C.shape[0]].reshape(-1, 1)
            u = u[C.shape[0]:].reshape(-1, 1)

            # Compute the optimal gain
            L = P @ C.T @ R_inv

            # Update the state estimate
            dxhat = A @ xhat + B @ u                    # prediction
            if correct:
                dxhat -= L @ (C @ xhat - y)     # correction

            # Update the covariance
            dP = A @ P + P @ A.T + G @ QN @ G.T
            if correct:
                dP -= P @ Reps_inv @ P

            # Return the update
            return np.hstack([dxhat.reshape(-1), dP.reshape(-1)])

    else:
        def _estim_update(t, x, u, params):
            # See if we are estimating or predicting
            correct = params.get('correct', True)

            # Get the state of the estimator
            xhat = x[0:sys.nstates].reshape(-1, 1)
            P = x[sys.nstates:].reshape(sys.nstates, sys.nstates)

            # Extract the inputs to the estimator
            y = u[0:C.shape[0]].reshape(-1, 1)
            u = u[C.shape[0]:].reshape(-1, 1)

            # Compute the optimal gain
            Reps_inv = np.linalg.inv(RN + C @ P @ C.T)
            L = A @ P @ C.T @ Reps_inv

            # Update the state estimate
            dxhat = A @ xhat + B @ u                    # prediction
            if correct:
                dxhat -= L @ (C @ xhat - y)     # correction

            # Update the covariance
            dP = A @ P @ A.T + G @ QN @ G.T
            if correct:
                dP -= A @ P @ C.T @ Reps_inv @ C @ P @ A.T

            # Return the update
            return np.hstack([dxhat.reshape(-1), dP.reshape(-1)])

    def _estim_output(t, x, u, params):
        return x[0:sys.nstates]

    # Define the estimator system
    return NonlinearIOSystem(
        _estim_update, _estim_output, states=state_labels + covariance_labels,
        inputs=sensor_labels + input_labels, outputs=output_labels,
        dt=sys.dt)


def white_noise(T, Q, dt=0):
    """Generate a white noise signal with specified intensity.

    This function generates a (multi-variable) white noise signal of
    specified intensity as either a sampled continous time signal or a
    discrete time signal.  A white noise signal along a 1D array
    of linearly spaced set of times T can be computing using

        V = ct.white_noise(T, Q, dt)

    where Q is a positive definite matrix providing the noise intensity.

    In continuous time, the white noise signal is scaled such that the
    integral of the covariance over a sample period is Q, thus approximating
    a white noise signal.  In discrete time, the white noise signal has
    covariance Q at each point in time (without any scaling based on the
    sample time).

    """
    # Convert input arguments to arrays
    T = np.atleast_1d(T)
    Q = np.atleast_2d(Q)

    # Check the shape of the input arguments
    if len(T.shape) != 1:
        raise ValueError("Time vector T must be 1D")
    if len(Q.shape) != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Covariance matrix Q must be square")

    # Figure out the time increment
    if dt != 0:
        # Discrete time system => white noise is not scaled
        dt = 1
    else:
        dt = T[1] - T[0]

    # Make sure data points are equally spaced
    if not np.allclose(np.diff(T), T[1] - T[0]):
        raise ValueError("Time values must be equally spaced.")

    # Generate independent white noise sources for each input
    W = np.array([
        np.random.normal(0, 1/sqrt(dt), T.size) for i in range(Q.shape[0])])

    # Return a linear combination of the noise sources
    return sp.linalg.sqrtm(Q) @ W


def correlation(T, X, Y=None, squeeze=True):
    """Compute the correlation of time signals.

    For a time series X(t) (and optionally Y(t)), the correlation() function
    computes the correlation matrix E(X'(t+tau) X(t)) or the cross-correlation
    matrix E(X'(t+tau) Y(t)]:

      tau, Rtau = correlation(T, X[, Y])

    The signal X (and Y, if present) represent a continuous time signal
    sampled at times T.  The return value provides the correlation Rtau
    between X(t+tau) and X(t) at a set of time offets tau.

    Parameters
    ----------
    T : 1D array_like
        Sample times for the signal(s).
    X : 1D or 2D array_like
        Values of the signal at each time in T.  The signal can either be
        scalar or vector values.
    Y : 1D or 2D array_like, optional
        If present, the signal with which to compute the correlation.
        Defaults to X.
    squeeze : bool, optional
        If True, squeeze Rtau to remove extra dimensions (useful if the
        signals are scalars).

    Returns
    -------

    """
    T = np.atleast_1d(T)
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y) if Y is not None else X

    # Check the shape of the input arguments
    if len(T.shape) != 1:
        raise ValueError("Time vector T must be 1D")
    if len(X.shape) != 2 or len(Y.shape) != 2:
        raise ValueError("Signals X and Y must be 2D arrays")
    if T.shape[0] != X.shape[1] or T.shape[0] != Y.shape[1]:
        raise ValueError("Signals X and Y must have same length as T")

    # Figure out the time increment
    dt = T[1] - T[0]

    # Make sure data points are equally spaced
    if not np.allclose(np.diff(T), T[1] - T[0]):
        raise ValueError("Time values must be equally spaced.")

    # Compute the correlation matrix
    R = np.array(
        [[sp.signal.correlate(X[i], Y[j])
          for i in range(X.shape[0])] for j in range(Y.shape[0])]
    ) * dt / (T[-1] - T[0])
    # From scipy.signal.correlation_lags (for use with older versions)
    # tau = sp.signal.correlation_lags(len(X[0]), len(Y[0])) * dt
    tau = np.arange(-len(Y[0]) + 1, len(X[0])) * dt

    return tau, R.squeeze() if squeeze else R
