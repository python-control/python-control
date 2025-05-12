# dde.py - Delay differential equations

"""Delay differential equations.

This module contains a minimal implementation of
a delay differential equation (DDE) solver using the
Method of Steps (MoS) approach and scipy's solve_ivp function.
The solver is designed to handle delayed 
linear time-invariant (delayLTI) systems.

"""

import numpy as np

from scipy.integrate import solve_ivp, OdeSolution
from scipy.interpolate import PchipInterpolator
from typing import List


def dde_response(
    delay_sys, T, U=0, X0=0, params=None,
    transpose=False, return_x=False, squeeze=None
):
    """Compute the output of a delay linear system given the input.

    Parameters
    ----------
    delay_sys : DelayLTI
        Delay I/O system for which forced response is computed.
    T : array_like
        An array representing the time points where the input is specified.
        The time points must be uniformly spaced.
    U : array_like or float, optional
        Input array giving input at each time T. If a scalar is passed,
        it is converted to an array with the same scalar value at each time.
        Defaults to 0.
    X0 : array_like or float, default=0.
        Initial condition of the state vector. If a scalar is passed,
        it is converted to an array with that scalar as the initial state.
    params : dict, optional
        If system is a nonlinear I/O system, set parameter values.
    transpose : bool, default=False
        If set to True, the input and output arrays will be transposed
        to match the format used in certain legacy systems or libraries.
    return_x : bool, default=None
        Used if the time response data is assigned to a tuple.  If False,
        return only the time and output vectors.  If True, also return the
        the state vector.  If None, determine the returned variables by
        `config.defaults['forced_response.return_x']`, which was True
        before version 0.9 and is False since then.
    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO) then
        the output response is returned as a 1D array (indexed by time).
        If `squeeze` is True, remove single-dimensional entries from
        the shape of the output even if the system is not SISO. If
        `squeeze` is False, keep the output as a 2D array (indexed by
        the output number and time) even if the system is SISO.
        The default behavior can be overridden by
        `config.defaults['control.squeeze_time_response']`.

    Returns
    -------
    resp : `TimeResponseData`
        Input/output response data object.  When accessed as a tuple,
        returns ``(time, outputs)`` (default) or ``(time, outputs, states)``
        if `return_x` is True.
    """
    from .timeresp import TimeResponseData, _check_convert_array
    from .delaylti import DelayLTI

    if not isinstance(delay_sys, DelayLTI):
        raise TypeError("Input must be a DelayLTI")

    n_states = delay_sys.P.A.shape[0]
    n_inputs = delay_sys.P.B1.shape[1]  # External inputs u
    n_outputs = delay_sys.P.C1.shape[0]  # External outputs y

    if U is not None:
        U = np.asarray(U)
    if T is not None:
        T = np.asarray(T)

    T = _check_convert_array(
        T, [("any",), (1, "any")], "Parameter `T`: ",
        squeeze=True, transpose=transpose
    )

    n_steps = T.shape[0]
    dt = (T[-1] - T[0]) / (n_steps - 1)
    if not np.allclose(np.diff(T), dt):
        raise ValueError("Parameter `T`: time values must be equally spaced.")

    X0 = _check_convert_array(
        X0, [(n_states,), (n_states, 1)], "Parameter `X0`: ", squeeze=True
    )

    # Test if U has correct shape and type
    legal_shapes = (
        [(n_steps,), (1, n_steps)] if n_inputs == 1 else [(n_inputs, n_steps)]
    )
    U = _check_convert_array(
        U, legal_shapes, "Parameter `U`: ", squeeze=False, transpose=transpose
    )
    xout = np.zeros((n_states, n_steps))
    xout[:, 0] = X0
    yout = np.zeros((n_outputs, n_steps))
    tout = T
    xout, yout = _solve_dde(delay_sys, T, U, X0, dt)

    return TimeResponseData(
        tout,
        yout,
        xout,
        U,
        params=params,
        issiso=delay_sys.issiso(),
        sysname=delay_sys.name,
        plot_inputs=True,
        title="Forced response for " + delay_sys.name,
        trace_types=["forced"],
        transpose=transpose,
        return_x=return_x,
        squeeze=squeeze,
    )


def _pchip_interp_u(T, U):
    """Create PCHIP interpolator functions for the
    input signal(s) U over time T.

    For time points `t < T[0]`, the interpolator returns 0.

    Parameters
    ----------
    T : array_like
        Time vector, 1D array.
    U : array_like
        Input signal(s). Can be:
        - 0D array (scalar): Assumed constant input. (Note: this path might
          not be hit if U is pre-processed by `_check_convert_array`).
        - 1D array `(n_steps,)`: Single input signal.
        - 2D array `(n_inputs, n_steps)`: Multiple input signals.

    Returns
    -------
    np.ndarray of PchipInterpolator or scalar
        If U is 1D or 2D, returns a 1D NumPy array of PchipInterpolator
        objects, one for each input signal.
        If U is a scalar (0D), returns U itself.
    """
    def negative_wrapper(interp):
        return lambda t: interp(t) if t >= T[0] else 0

    if np.ndim(U) == 1:
        # Single input signal, U.shape is (n_steps,)
        return np.array([negative_wrapper(PchipInterpolator(T, U))])
    elif np.ndim(U) == 0:
        # Scalar input, assumed constant.
        return U
    else:
        # Multiple input signals, U.shape is (n_inputs, n_steps)
        return np.array([
            negative_wrapper(PchipInterpolator(T, ui)) for ui in U
        ])


class _DDEHistory:
    """
    Stores the computed solution history for a DDE and provides a callable
    interface to retrieve the state x(t) at any requested past time t.
    The history object is callable: `history(t)` returns the state vector x(t).

    Handles three regimes:
    1. t <= t0: Uses the provided initial history function.
    2. t0 < t <= t_last_computed:
        Interpolates using dense output from solve_ivp segments.
    3. t > t_last_computed: Performs constant
       interpolation using theextrapolation using the last computed state
       (the state at `t_last_computed`).

    Attributes
    ----------
    initial_history_func : callable
        Function `f(t)` that returns the state vector for `t <= t0`.
    t0 : float
        Initial time. History before or at this time
        is given by `initial_history_func`.
    segments : list of OdeSolution
        List of `OdeSolution` objects from `scipy.integrate.solve_ivp`,
        each representing a computed segment of the solution.
    last_valid_time : float
        The time at the end of the most recently added solution segment.
    last_state : np.ndarray
        The state vector at `last_valid_time`.
    """

    def __init__(self, initial_history_func, t0):
        self.initial_history_func = initial_history_func
        self.t0: float = t0
        # Stores OdeResult objects from solve_ivp
        self.segments: List[OdeSolution] = []
        self.last_valid_time: float = t0

        initial_state = np.asarray(initial_history_func(t0))
        self.last_state = initial_state

    def add_segment(self, segment: OdeSolution):
        """
        Adds a new computed solution segment (from solve_ivp) to the history.

        Parameters
        ----------
        segment : OdeSolution
            The solution object returned by `solve_ivp` for a time segment.
        """

        self.segments.append(segment)
        self.last_valid_time = segment.t[-1]
        self.last_state = segment.y[:, -1]

    def __call__(self, t):
        """Return the state vector x(t) by looking up or
            interpolating from history.

        Parameters
        ----------
        t : float
            Time at which to retrieve the state.

        Returns
        -------
        np.ndarray
            State vector x(t).
        """
        if t <= self.t0:
            return np.asarray(self.initial_history_func(self.t0))
        elif t > self.last_valid_time:
            return self.last_state
        else:
            for segment in self.segments:
                if segment.t[0] <= t <= segment.t[-1]:
                    return segment.sol(t)
            # Fallback: should ideally not be reached
            # if t is within (t0, last_valid_time]
            # and segments cover this range.
            return np.zeros_like(self.last_state)  # Deal with first call


def _dde_wrapper(t, x, A, B1, B2, C2, D21, tau_list, u_func, history_x):
    """
    Wrapper function for DDE solver using scipy's solve_ivp.
    Computes the derivative dx/dt for the DDE system.

    The system is defined by:
        dx/dt = A @ x(t) + B1 @ u(t) + B2 @ z_delayed_vector(t)
    where:
        z_delayed_vector(t) is a vector where the k-th component is
        z_k(t - tau_list[k]) and
        z_k(t') = (C2 @ x(t') + D21 @ u(t'))_k.
        (Assuming D22 is zero for the internal feedback path).

    Parameters
    ----------
    t : float
        Current time.
    x : np.ndarray
        Current state vector, x(t).
    A, B1, B2, C2, D21 : np.ndarray
        State-space matrices of the underlying `PartitionedStateSpace`.
    tau_list : array_like
        List or array of time delays.
    u_func : array_like of callable
        Array of interpolating functions for the input u(t). `u_funci`
        gives the i-th input signal at time t.
    history_x : DdeHistory
        Callable history object to retrieve past states x(t-tau).

    Returns
    -------
    np.ndarray
        The derivative of the state vector, dx/dt.
    """
    z_delayed = []
    for i, tau in enumerate(tau_list):
        u_delayed = np.array([u_func[i](t - tau) for i in range(len(u_func))])
        z = C2 @ history_x(t - tau) + D21 @ u_delayed
        z_delayed.append(z[i])
    z_delayed = np.array(z_delayed).flatten()

    u_current = np.array([u_func[i](t) for i in range(len(u_func))])
    dxdt = A @ x + B1 @ u_current + B2 @ z_delayed
    return dxdt.flatten()


def _solve_dde(delay_sys, T, U, X0, dt):
    """
    Solving delay differential equation using Method Of Steps.

    Parameters
    ----------
    delay_sys : DelayLTI
        Delay I/O system for which forced response is computed.
    T : array_like
        An array representing the time points where the input is specified.
        The time points must be uniformly spaced.
    U : array_like or float, optional
        Input array giving input at each time in `T`.
    X0 : array_like or float, default=0.
        Initial condition.
    dt : float # Unused beyond T check
        Time step of the `T` array. Used to verify `T` is uniformly spaced.
        Not directly used as integration step by `solve_ivp`.

    Returns
    -------
    xout : array_like
        Array containing the state vector at each time step.
    yout : array_like
        Array containing the output vector at each time step.

    """
    intial_history_func = lambda t: np.zeros(X0.shape)
    t0, tf = T[0], T[-1]
    u_func = _pchip_interp_u(T, U)

    history_x = _DDEHistory(intial_history_func, t0)  # to access x(t-tau)
    current_t = 0
    current_x = np.asarray(X0).flatten()

    A, B1, B2, C1, C2 = (
        delay_sys.P.A,
        delay_sys.P.B1,
        delay_sys.P.B2,
        delay_sys.P.C1,
        delay_sys.P.C2,
    )
    D11, D12, D21 = (
        delay_sys.P.D11,
        delay_sys.P.D12,
        delay_sys.P.D21,
    )  # in control systems, D22 is always 0
    tau_list = delay_sys.tau

    solution_ts = [current_t]
    solution_xs = [current_x]

    # TODO: handle discontinuity propagation
    discontinuity_times = set(tau_list)
    while current_t < tf:
        t_stop = min(discontinuity_times) if discontinuity_times else tf
        if not np.isclose(t_stop, tf):
            discontinuity_times.remove(t_stop)
        local_t_eval = [t for t in T if current_t < t <= t_stop]

        sol_segment = solve_ivp(
            fun=_dde_wrapper,
            t_span=(current_t, t_stop),
            t_eval=local_t_eval,
            y0=current_x,
            method="LSODA",
            dense_output=True,
            args=(A, B1, B2, C2, D21, tau_list, u_func, history_x),
            rtol=1e-9,
            atol=1e-12,
        )

        # --- Update History and Store Results ---
        history_x.add_segment(sol_segment)
        segment_ts = sol_segment.t
        segment_xs = sol_segment.y

        solution_ts.extend(segment_ts)
        new_x = [segment_xs[:, i] for i in range(segment_xs.shape[1])]
        solution_xs.extend(new_x)

        current_t = sol_segment.t[-1]
        current_x = segment_xs[:, -1]

    solution_xs = np.array(solution_xs)
    solution_ts = np.array(solution_ts)

    z_delayed = []
    u_current = []
    for i, ti in enumerate(solution_ts):
        z_delayed.append([])
        for j, tau in enumerate(tau_list):
            z = C2 @ history_x(ti - tau) + D21 @ np.array(
                [u_func[i](ti - tau) for i in range(len(u_func))]
            )
            z_delayed[i].append(z[j])
        u_current.append([u_func[i](ti) for i in range(len(u_func))])

    z_delayed = np.array(z_delayed)
    u_current = np.array(u_current)

    solution_ys = C1 @ solution_xs.T + D11 @ u_current.T + D12 @ z_delayed.T
    return solution_xs.T, solution_ys
