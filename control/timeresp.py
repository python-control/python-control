# timeresp.py - time-domain simulation routines
#
# This file contains a collection of functions that calculate time
# responses for linear systems.

"""The :mod:`~control.timeresp` module contains a collection of
functions that are used to compute time-domain simulations of LTI
systems.

Arguments to time-domain simulations include a time vector, an input
vector (when needed), and an initial condition vector.  The most
general function for simulating LTI systems the
:func:`forced_response` function, which has the form::

    t, y = forced_response(sys, T, U, X0)

where `T` is a vector of times at which the response should be
evaluated, `U` is a vector of inputs (one for each time point) and
`X0` is the initial condition for the system.

See :ref:`time-series-convention` for more information on how time
series data are represented.

"""

"""Copyright (c) 2011 by California Institute of Technology
All rights reserved.

Copyright (c) 2011 by Eike Welk
Copyright (c) 2010 by SciPy Developers

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the California Institute of Technology nor
   the names of its contributors may be used to endorse or promote
   products derived from this software without specific prior
   written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

Initial Author: Eike Welk
Date: 12 May 2011
$Id$
"""

# Libraries that we make use of
import scipy as sp              # SciPy library (used all over)
import numpy as np              # NumPy library
from scipy.signal.ltisys import _default_response_times
import warnings
from .lti import LTI     # base class of StateSpace, TransferFunction
from .statesp import _convertToStateSpace, _mimo2simo, _mimo2siso
from .lti import isdtime, isctime

__all__ = ['forced_response', 'step_response', 'step_info', 'initial_response',
           'impulse_response']


# Helper function for checking array-like parameters
def _check_convert_array(in_obj, legal_shapes, err_msg_start, squeeze=False,
                         transpose=False):
    """
    Helper function for checking array-like parameters.

    * Check type and shape of ``in_obj``.
    * Convert ``in_obj`` to an array if necessary.
    * Change shape of ``in_obj`` according to parameter ``squeeze``.
    * If ``in_obj`` is a scalar (number) it is converted to an array with
      a legal shape, that is filled with the scalar value.

    The function raises an exception when it detects an error.

    Parameters
    ----------
    in_obj: array like object
        The array or matrix which is checked.

    legal_shapes: list of tuple
        A list of shapes that in_obj can legally have.
        The special value "any" means that there can be any
        number of elements in a certain dimension.

        * ``(2, 3)`` describes an array with 2 rows and 3 columns
        * ``(2, "any")`` describes an array with 2 rows and any number of
          columns

    err_msg_start: str
        String that is prepended to the error messages, when this function
        raises an exception. It should be used to identify the argument which
        is currently checked.

    squeeze: bool
        If True, all dimensions with only one element are removed from the
        array. If False the array's shape is unmodified.

        For example:
        ``array([[1,2,3]])`` is converted to ``array([1, 2, 3])``

   transpose: bool
        If True, assume that input arrays are transposed for the standard
        format.  Used to convert MATLAB-style inputs to our format.

    Returns:

    out_array: array
        The checked and converted contents of ``in_obj``.
    """
    # convert nearly everything to an array.
    out_array = np.asarray(in_obj)
    if (transpose):
        out_array = np.transpose(out_array)

    # Test element data type, elements must be numbers
    legal_kinds = set(("i", "f", "c"))  # integer, float, complex
    if out_array.dtype.kind not in legal_kinds:
        err_msg = "Wrong element data type: '{d}'. Array elements " \
                  "must be numbers.".format(d=str(out_array.dtype))
        raise TypeError(err_msg_start + err_msg)

    # If array is zero dimensional (in_obj is scalar):
    # create array with legal shape filled with the original value.
    if out_array.ndim == 0:
        for s_legal in legal_shapes:
            # search for shape that does not contain the special symbol any.
            if "any" in s_legal:
                continue
            the_val = out_array[()]
            out_array = np.empty(s_legal, 'd')
            out_array.fill(the_val)
            break

    # Test shape
    def shape_matches(s_legal, s_actual):
        """Test if two shape tuples match"""
        # Array must have required number of dimensions
        if len(s_legal) != len(s_actual):
            return False
        # All dimensions must contain required number of elements. Joker: "all"
        for n_legal, n_actual in zip(s_legal, s_actual):
            if n_legal == "any":
                continue
            if n_legal != n_actual:
                return False
        return True

    # Iterate over legal shapes, and see if any matches out_array's shape.
    for s_legal in legal_shapes:
        if shape_matches(s_legal, out_array.shape):
            break
    else:
        legal_shape_str = " or ".join([str(s) for s in legal_shapes])
        err_msg = "Wrong shape (rows, columns): {a}. Expected: {e}." \
                  .format(e=legal_shape_str, a=str(out_array.shape))
        raise ValueError(err_msg_start + err_msg)

    # Convert shape
    if squeeze:
        out_array = np.squeeze(out_array)
        # We don't want zero dimensional arrays
        if out_array.shape == tuple():
            out_array = out_array.reshape((1,))

    return out_array


# Forced response of a linear system
def forced_response(sys, T=None, U=0., X0=0., transpose=False,
                    interpolate=False, squeeze=True):
    """Simulate the output of a linear system.

    As a convenience for parameters `U`, `X0`:
    Numbers (scalars) are converted to constant arrays with the correct shape.
    The correct shape is inferred from arguments `sys` and `T`.

    For information on the **shape** of parameters `U`, `T`, `X0` and
    return values `T`, `yout`, `xout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sys: LTI (StateSpace, or TransferFunction)
        LTI system to simulate

    T: array-like, optional for discrete LTI `sys`
        Time steps at which the input is defined; values must be evenly spaced.

    U: array-like or number, optional
        Input array giving input at each time `T` (default = 0).

        If `U` is ``None`` or ``0``, a special algorithm is used. This special
        algorithm is faster than the general algorithm, which is used
        otherwise.

    X0: array-like or number, optional
        Initial condition (default = 0).

    transpose: bool, optional (default=False)
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and scipy.signal.lsim)

    interpolate: bool, optional (default=False)
        If True and system is a discrete time system, the input will
        be interpolated between the given time steps and the output
        will be given at system sampling rate.  Otherwise, only return
        the output at the times given in `T`.  No effect on continuous
        time simulations (default = False).

    squeeze: bool, optional (default=True)
        If True, remove single-dimensional entries from the shape of
        the output.  For single output systems, this converts the
        output response to a 1D array.

    Returns
    -------
    T: array
        Time values of the output.
    yout: array
        Response of the system.
    xout: array
        Time evolution of the state vector.

    See Also
    --------
    step_response, initial_response, impulse_response

    Notes
    -----
    For discrete time systems, the input/output response is computed using the
    :scipy-signal:ref:`scipy.signal.dlsim` function.

    For continuous time systems, the output is computed using the matrix
    exponential `exp(A t)` and assuming linear interpolation of the inputs
    between time points.

    Examples
    --------
    >>> T, yout, xout = forced_response(sys, T, u, X0)

    See :ref:`time-series-convention`.

    """
    if not isinstance(sys, LTI):
        raise TypeError('Parameter ``sys``: must be a ``LTI`` object. '
                        '(For example ``StateSpace`` or ``TransferFunction``)')
    sys = _convertToStateSpace(sys)
    A, B, C, D = np.asarray(sys.A), np.asarray(sys.B), np.asarray(sys.C), \
        np.asarray(sys.D)
#    d_type = A.dtype
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    n_outputs = C.shape[0]

    # Convert inputs to numpy arrays for easier shape checking
    if U is not None:
        U = np.asarray(U)
    if T is not None:
        T = np.asarray(T)

    # Set and/or check time vector in discrete time case
    if isdtime(sys, strict=True):
        if T is None:
            if U is None:
                raise ValueError('Parameters ``T`` and ``U`` can\'t both be'
                                 'zero for discrete-time simulation')
            # Set T to equally spaced samples with same length as U
            if U.ndim == 1:
                n_steps = U.shape[0]
            else:
                n_steps = U.shape[1]
            T = np.array(range(n_steps)) * (1 if sys.dt is True else sys.dt)
        else:
            # Make sure the input vector and time vector have same length
            # TODO: allow interpolation of the input vector
            if (U.ndim == 1 and U.shape[0] != T.shape[0]) or \
                    (U.ndim > 1 and U.shape[1] != T.shape[0]):
                ValueError('Pamameter ``T`` must have same elements as'
                           ' the number of columns in input array ``U``')

    # Test if T has shape (n,) or (1, n);
    # T must be array-like and values must be increasing.
    # The length of T determines the length of the input vector.
    if T is None:
        raise ValueError('Parameter ``T``: must be array-like, and contain '
                         '(strictly monotonic) increasing numbers.')
    T = _check_convert_array(T, [('any',), (1, 'any')],
                             'Parameter ``T``: ', squeeze=True,
                             transpose=transpose)
    dt = T[1] - T[0]
    if not np.allclose(T[1:] - T[:-1], dt):
        raise ValueError("Parameter ``T``: time values must be "
                         "equally spaced.")
    n_steps = T.shape[0]            # number of simulation steps

    # create X0 if not given, test if X0 has correct shape
    X0 = _check_convert_array(X0, [(n_states,), (n_states, 1)],
                              'Parameter ``X0``: ', squeeze=True)

    xout = np.zeros((n_states, n_steps))
    xout[:, 0] = X0
    yout = np.zeros((n_outputs, n_steps))

    # Separate out the discrete and continuous time cases
    if isctime(sys):
        # Solve the differential equation, copied from scipy.signal.ltisys.
        dot, squeeze, = np.dot, np.squeeze  # Faster and shorter code

        # Faster algorithm if U is zero
        if U is None or (isinstance(U, (int, float)) and U == 0):
            # Solve using matrix exponential
            expAdt = sp.linalg.expm(A * dt)
            for i in range(1, n_steps):
                xout[:, i] = dot(expAdt, xout[:, i-1])
            yout = dot(C, xout)

        # General algorithm that interpolates U in between output points
        else:
            # Test if U has correct shape and type
            legal_shapes = [(n_steps,), (1, n_steps)] if n_inputs == 1 else \
                           [(n_inputs, n_steps)]
            U = _check_convert_array(U, legal_shapes,
                                     'Parameter ``U``: ', squeeze=False,
                                     transpose=transpose)
            # convert 1D array to 2D array with only one row
            if len(U.shape) == 1:
                U = U.reshape(1, -1)  # pylint: disable=E1103

            # Algorithm: to integrate from time 0 to time dt, with linear
            # interpolation between inputs u(0) = u0 and u(dt) = u1, we solve
            #   xdot = A x + B u,        x(0) = x0
            #   udot = (u1 - u0) / dt,   u(0) = u0.
            #
            # Solution is
            #   [ x(dt) ]       [ A*dt  B*dt  0 ] [  x0   ]
            #   [ u(dt) ] = exp [  0     0    I ] [  u0   ]
            #   [u1 - u0]       [  0     0    0 ] [u1 - u0]

            M = np.block([[A * dt, B * dt, np.zeros((n_states, n_inputs))],
                         [np.zeros((n_inputs, n_states + n_inputs)),
                          np.identity(n_inputs)],
                         [np.zeros((n_inputs, n_states + 2 * n_inputs))]])
            expM = sp.linalg.expm(M)
            Ad = expM[:n_states, :n_states]
            Bd1 = expM[:n_states, n_states+n_inputs:]
            Bd0 = expM[:n_states, n_states:n_states + n_inputs] - Bd1

            for i in range(1, n_steps):
                xout[:, i] = (dot(Ad, xout[:, i-1]) + dot(Bd0, U[:, i-1]) +
                              dot(Bd1, U[:, i]))
            yout = dot(C, xout) + dot(D, U)
        tout = T

    else:
        # Discrete type system => use SciPy signal processing toolbox
        if sys.dt is not True:
            # Make sure that the time increment is a multiple of sampling time

            # First make sure that time increment is bigger than sampling time
            # (with allowance for small precision errors)
            if dt < sys.dt and not np.isclose(dt, sys.dt):
                raise ValueError("Time steps ``T`` must match sampling time")

            # Now check to make sure it is a multiple (with check against
            # sys.dt because floating point mod can have small errors
            elif not (np.isclose(dt % sys.dt, 0) or
                      np.isclose(dt % sys.dt, sys.dt)):
                raise ValueError("Time steps ``T`` must be multiples of "
                                 "sampling time")
            sys_dt = sys.dt

        else:
            sys_dt = dt         # For unspecified sampling time, use time incr

        # Discrete time simulation using signal processing toolbox
        dsys = (A, B, C, D, sys_dt)

        # Use signal processing toolbox for the discrete time simulation
        # Transpose the input to match toolbox convention
        tout, yout, xout = sp.signal.dlsim(dsys, np.transpose(U), T, X0)

        if not interpolate:
            # If dt is different from sys.dt, resample the output
            inc = int(round(dt / sys_dt))
            tout = T            # Return exact list of time steps
            yout = yout[::inc, :]
            xout = xout[::inc, :]

        # Transpose the output and state vectors to match local convention
        xout = sp.transpose(xout)
        yout = sp.transpose(yout)

    # Get rid of unneeded dimensions
    if squeeze:
        yout = np.squeeze(yout)
        xout = np.squeeze(xout)

    # See if we need to transpose the data back into MATLAB form
    if transpose:
        tout = np.transpose(tout)
        yout = np.transpose(yout)
        xout = np.transpose(xout)

    return tout, yout, xout


def _get_ss_simo(sys, input=None, output=None):
    """Return a SISO or SIMO state-space version of sys

    If input is not specified, select first input and issue warning
    """
    sys_ss = _convertToStateSpace(sys)
    if sys_ss.issiso():
        return sys_ss
    warn = False
    if input is None:
        # issue warning if input is not given
        warn = True
        input = 0
    if output is None:
        return _mimo2simo(sys_ss, input, warn_conversion=warn)
    else:
        return _mimo2siso(sys_ss, input, output, warn_conversion=warn)


def step_response(sys, T=None, X0=0., input=None, output=None,
                  transpose=False, return_x=False, squeeze=True):
    # pylint: disable=W0622
    """Step response of a linear system

    If the system has multiple inputs or outputs (MIMO), one input has
    to be selected for the simulation. Optionally, one output may be
    selected. The parameters `input` and `output` do this. All other
    inputs are set to 0, all other outputs are ignored.

    For information on the **shape** of parameters `T`, `X0` and
    return values `T`, `yout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sys: StateSpace, or TransferFunction
        LTI system to simulate

    T: array-like object, optional
        Time vector (argument is autocomputed if not given)

    X0: array-like or number, optional
        Initial condition (default = 0)

        Numbers are converted to constant arrays with the correct shape.

    input: int
        Index of the input that will be used in this simulation.

    output: int
        Index of the output that will be used in this simulation. Set to None
        to not trim outputs

    transpose: bool
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and scipy.signal.lsim)

    return_x: bool
        If True, return the state vector (default = False).

    squeeze: bool, optional (default=True)
        If True, remove single-dimensional entries from the shape of
        the output.  For single output systems, this converts the
        output response to a 1D array.

    Returns
    -------
    T: array
        Time values of the output

    yout: array
        Response of the system

    xout: array
        Individual response of each x variable

    See Also
    --------
    forced_response, initial_response, impulse_response

    Notes
    -----
    This function uses the `forced_response` function with the input set to a
    unit step.

    Examples
    --------
    >>> T, yout = step_response(sys, T, X0)

    """
    sys = _get_ss_simo(sys, input, output)
    if T is None:
        if isctime(sys):
            T = _default_response_times(sys.A, 100)
        else:
            # For discrete time, use integers
            tvec = _default_response_times(sys.A, 100)
            T = range(int(np.ceil(max(tvec))))

    U = np.ones_like(T)

    T, yout, xout = forced_response(sys, T, U, X0, transpose=transpose,
                                    squeeze=squeeze)

    if return_x:
        return T, yout, xout

    return T, yout


def step_info(sys, T=None, SettlingTimeThreshold=0.02,
              RiseTimeLimits=(0.1, 0.9)):
    '''
    Step response characteristics (Rise time, Settling Time, Peak and others).

    Parameters
    ----------
    sys: StateSpace, or TransferFunction
        LTI system to simulate

    T: array-like object, optional
        Time vector (argument is autocomputed if not given)

    SettlingTimeThreshold: float value, optional
        Defines the error to compute settling time (default = 0.02)

    RiseTimeLimits: tuple (lower_threshold, upper_theshold)
        Defines the lower and upper threshold for RiseTime computation

    Returns
    -------
    S: a dictionary containing:
        RiseTime: Time from 10% to 90% of the steady-state value.
        SettlingTime: Time to enter inside a default error of 2%
        SettlingMin: Minimum value after RiseTime
        SettlingMax: Maximum value after RiseTime
        Overshoot: Percentage of the Peak relative to steady value
        Undershoot: Percentage of undershoot
        Peak: Absolute peak value
        PeakTime: time of the Peak
        SteadyStateValue: Steady-state value


    See Also
    --------
    step, lsim, initial, impulse

    Examples
    --------
    >>> info = step_info(sys, T)
    '''
    sys = _get_ss_simo(sys)
    if T is None:
        if isctime(sys):
            T = _default_response_times(sys.A, 1000)
        else:
            # For discrete time, use integers
            tvec = _default_response_times(sys.A, 1000)
            T = range(int(np.ceil(max(tvec))))

    T, yout = step_response(sys, T)

    # Steady state value
    InfValue = yout[-1]

    # RiseTime
    tr_lower_index = (np.where(yout >= RiseTimeLimits[0] * InfValue)[0])[0]
    tr_upper_index = (np.where(yout >= RiseTimeLimits[1] * InfValue)[0])[0]
    RiseTime = T[tr_upper_index] - T[tr_lower_index]

    # SettlingTime
    sup_margin = (1. + SettlingTimeThreshold) * InfValue
    inf_margin = (1. - SettlingTimeThreshold) * InfValue
    # find Steady State looking for the first point out of specified limits
    for i in reversed(range(T.size)):
        if((yout[i] <= inf_margin) | (yout[i] >= sup_margin)):
            SettlingTime = T[i + 1]
            break

    # Peak
    PeakIndex = np.abs(yout).argmax()
    PeakValue = yout[PeakIndex]
    PeakTime = T[PeakIndex]
    SettlingMax = (yout).max()
    SettlingMin = (yout[tr_upper_index:]).min()
    # I'm really not very confident about UnderShoot:
    UnderShoot = yout.min()
    OverShoot = 100. * (yout.max() - InfValue) / (InfValue - yout[0])

    # Return as a dictionary
    S = {
        'RiseTime': RiseTime,
        'SettlingTime': SettlingTime,
        'SettlingMin': SettlingMin,
        'SettlingMax': SettlingMax,
        'Overshoot': OverShoot,
        'Undershoot': UnderShoot,
        'Peak': PeakValue,
        'PeakTime': PeakTime,
        'SteadyStateValue': InfValue
    }

    return S


def initial_response(sys, T=None, X0=0., input=0, output=None,
                     transpose=False, return_x=False, squeeze=True):
    # pylint: disable=W0622
    """Initial condition response of a linear system

    If the system has multiple outputs (MIMO), optionally, one output
    may be selected. If no selection is made for the output, all
    outputs are given.

    For information on the **shape** of parameters `T`, `X0` and
    return values `T`, `yout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sys: StateSpace, or TransferFunction
        LTI system to simulate

    T: array-like object, optional
        Time vector (argument is autocomputed if not given)

    X0: array-like object or number, optional
        Initial condition (default = 0)

        Numbers are converted to constant arrays with the correct shape.

    input: int
        Ignored, has no meaning in initial condition calculation. Parameter
        ensures compatibility with step_response and impulse_response

    output: int
        Index of the output that will be used in this simulation. Set to None
        to not trim outputs

    transpose: bool
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and scipy.signal.lsim)

    return_x: bool
        If True, return the state vector (default = False).

    squeeze: bool, optional (default=True)
        If True, remove single-dimensional entries from the shape of
        the output.  For single output systems, this converts the
        output response to a 1D array.

    Returns
    -------
    T: array
        Time values of the output
    yout: array
        Response of the system
    xout: array
        Individual response of each x variable

    See Also
    --------
    forced_response, impulse_response, step_response

    Notes
    -----
    This function uses the `forced_response` function with the input set to
    zero.

    Examples
    --------
    >>> T, yout = initial_response(sys, T, X0)
    """
    sys = _get_ss_simo(sys, input, output)

    # Create time and input vectors; checking is done in forced_response(...)
    # The initial vector X0 is created in forced_response(...) if necessary
    if T is None:
        if isctime(sys):
            T = _default_response_times(sys.A, 1000)
        else:
            # For discrete time, use integers
            tvec = _default_response_times(sys.A, 1000)
            T = range(int(np.ceil(max(tvec))))
    U = np.zeros_like(T)

    T, yout, _xout = forced_response(sys, T, U, X0, transpose=transpose,
                                     squeeze=squeeze)

    if return_x:
        return T, yout, _xout

    return T, yout


def impulse_response(sys, T=None, X0=0., input=0, output=None,
                     transpose=False, return_x=False, squeeze=True):
    # pylint: disable=W0622
    """Impulse response of a linear system

    If the system has multiple inputs or outputs (MIMO), one input has
    to be selected for the simulation. Optionally, one output may be
    selected. The parameters `input` and `output` do this. All other
    inputs are set to 0, all other outputs are ignored.

    For information on the **shape** of parameters `T`, `X0` and
    return values `T`, `yout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sys: StateSpace, TransferFunction
        LTI system to simulate

    T: array-like object, optional
        Time vector (argument is autocomputed if not given)

    X0: array-like object or number, optional
        Initial condition (default = 0)

        Numbers are converted to constant arrays with the correct shape.

    input: int
        Index of the input that will be used in this simulation.

    output: int
        Index of the output that will be used in this simulation. Set to None
        to not trim outputs

    transpose: bool
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and scipy.signal.lsim)

    return_x: bool
        If True, return the state vector (default = False).

    squeeze: bool, optional (default=True)
        If True, remove single-dimensional entries from the shape of
        the output.  For single output systems, this converts the
        output response to a 1D array.

    Returns
    -------
    T: array
        Time values of the output
    yout: array
        Response of the system
    xout: array
        Individual response of each x variable

    See Also
    --------
    forced_response, initial_response, step_response

    Notes
    -----
    This function uses the `forced_response` function to compute the time
    response. For continuous time systems, the initial condition is altered to
    account for the initial impulse.

    Examples
    --------
    >>> T, yout = impulse_response(sys, T, X0)

    """
    sys = _get_ss_simo(sys, input, output)

    # System has direct feedthrough, can't simulate impulse response
    # numerically
    if np.any(sys.D != 0) and isctime(sys):
        warnings.warn("System has direct feedthrough: ``D != 0``. The "
                      "infinite impulse at ``t=0`` does not appear in the "
                      "output.\n"
                      "Results may be meaningless!")

    # create X0 if not given, test if X0 has correct shape.
    # Must be done here because it is used for computations here.
    n_states = sys.A.shape[0]
    X0 = _check_convert_array(X0, [(n_states,), (n_states, 1)],
                              'Parameter ``X0``: \n', squeeze=True)

    # Compute T and U, no checks necessary, they will be checked in lsim
    if T is None:
        if isctime(sys):
            T = _default_response_times(sys.A, 100)
        else:
            # For discrete time, use integers
            tvec = _default_response_times(sys.A, 100)
            T = range(int(np.ceil(max(tvec))))

    U = np.zeros_like(T)

    # Compute new X0 that contains the impulse
    # We can't put the impulse into U because there is no numerical
    # representation for it (infinitesimally short, infinitely high).
    # See also: http://www.mathworks.com/support/tech-notes/1900/1901.html
    if isctime(sys):
        B = np.asarray(sys.B).squeeze()
        new_X0 = B + X0
    else:
        new_X0 = X0
        U[0] = 1.

    T, yout, _xout = forced_response(sys, T, U, new_X0, transpose=transpose,
                                     squeeze=squeeze)

    if return_x:
        return T, yout, _xout

    return T, yout
