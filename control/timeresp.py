# timeresp.py - time-domain simulation routes
"""
Time domain simulation.

This file contains a collection of functions that calculate
time responses for linear systems.

.. _time-series-convention:

Convention for Time Series
--------------------------

This is a convention for function arguments and return values that
represent time series: sequences of values that change over time. It
is used throughout the library, for example in the functions
:func:`forced_response`, :func:`step_response`, :func:`impulse_response`,
and :func:`initial_response`.

.. note::
    This convention is different from the convention used in the library
    :mod:`scipy.signal`. In Scipy's convention the meaning of rows and columns
    is interchanged.  Thus, all 2D values must be transposed when they are
    used with functions from :mod:`scipy.signal`.

Types:

    * **Arguments** can be **arrays**, **matrices**, or **nested lists**.
    * **Return values** are **arrays** (not matrices).

The time vector is either 1D, or 2D with shape (1, n)::

      T = [[t1,     t2,     t3,     ..., tn    ]]

Input, state, and output all follow the same convention. Columns are different
points in time, rows are different components. When there is only one row, a
1D object is accepted or returned, which adds convenience for SISO systems::

      U = [[u1(t1), u1(t2), u1(t3), ..., u1(tn)]
           [u2(t1), u2(t2), u2(t3), ..., u2(tn)]
           ...
           ...
           [ui(t1), ui(t2), ui(t3), ..., ui(tn)]]

      Same for X, Y

So, U[:,2] is the system's input at the third point in time; and U[1] or U[1,:]
is the sequence of values for the system's second input.

The initial conditions are either 1D, or 2D with shape (j, 1)::

     X0 = [[x1]
           [x2]
           ...
           ...
           [xj]]

As all simulation functions return *arrays*, plotting is convenient::

    t, y = step(sys)
    plot(t, y)

The output of a MIMO system can be plotted like this::

    t, y, x = lsim(sys, u, t)
    plot(t, y[0], label='y_0')
    plot(t, y[1], label='y_1')

The convention also works well with the state space form of linear systems. If
``D`` is the feedthrough *matrix* of a linear system, and ``U`` is its input
(*matrix* or *array*), then the feedthrough part of the system's response,
can be computed like this::

    ft = D * U

----------------------------------------------------------------
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
from .lti import Lti     # base class of StateSpace, TransferFunction
from .statesp import _convertToStateSpace, _mimo2simo, _mimo2siso
from .lti import isdtime, isctime


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
def forced_response(sys, T=None, U=0., X0=0., transpose=False, **keywords):
    """Simulate the output of a linear system.

    As a convenience for parameters `U`, `X0`:
    Numbers (scalars) are converted to constant arrays with the correct shape.
    The correct shape is inferred from arguments `sys` and `T`.

    For information on the **shape** of parameters `U`, `T`, `X0` and
    return values `T`, `yout`, `xout` see: :ref:`time-series-convention`

    Parameters
    ----------
    sys: Lti (StateSpace, or TransferFunction)
        LTI system to simulate

    T: array-like
        Time steps at which the input is defined, numbers must be (strictly
        monotonic) increasing.

    U: array-like or number, optional
        Input array giving input at each time `T` (default = 0).

        If `U` is ``None`` or ``0``, a special algorithm is used. This special
        algorithm is faster than the general algorithm, which is used otherwise.

    X0: array-like or number, optional
        Initial condition (default = 0).

    transpose: bool
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and scipy.signal.lsim)

    **keywords:
        Additional keyword arguments control the solution algorithm for the
        differential equations. These arguments are passed on to the function
        :func:`scipy.integrate.odeint`. See the documentation for
        :func:`scipy.integrate.odeint` for information about these
        arguments.

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

    Examples
    --------
    >>> T, yout, xout = forced_response(sys, T, u, X0)
    """
    if not isinstance(sys, Lti):
        raise TypeError('Parameter ``sys``: must be a ``Lti`` object. '
                        '(For example ``StateSpace`` or ``TransferFunction``)')
    sys = _convertToStateSpace(sys)
    A, B, C, D = np.asarray(sys.A), np.asarray(sys.B), np.asarray(sys.C), \
        np.asarray(sys.D)
#    d_type = A.dtype
    n_states = A.shape[0]
    n_inputs = B.shape[1]

    # Set and/or check time vector in discrete time case
    if isdtime(sys, strict=True):
        if T is None:
            if U is None:
                raise ValueError('Parameters ``T`` and ``U`` can\'t both be'
                                 'zero for discrete-time simulation')
            # Set T to integers with same length as U
            T = range(len(U))
        else:
            # Make sure the input vector and time vector have same length
            # TODO: allow interpolation of the input vector
            if len(U) != len(T):
                ValueError('Pamameter ``T`` must have same length as'
                           'input vector ``U``')

    # Test if T has shape (n,) or (1, n);
    # T must be array-like and values must be increasing.
    # The length of T determines the length of the input vector.
    if T is None:
        raise ValueError('Parameter ``T``: must be array-like, and contain '
                         '(strictly monotonic) increasing numbers.')
    T = _check_convert_array(T, [('any',), (1, 'any')],
                             'Parameter ``T``: ', squeeze=True,
                             transpose=transpose)
    if not all(T[1:] - T[:-1] > 0):
        raise ValueError('Parameter ``T``: time values must be '
                         '(strictly monotonic) increasing numbers.')
    n_steps = len(T)            # number of simulation steps

    # create X0 if not given, test if X0 has correct shape
    X0 = _check_convert_array(X0, [(n_states,), (n_states, 1)],
                              'Parameter ``X0``: ', squeeze=True)

    # Separate out the discrete and continuous time cases
    if isctime(sys):
        # Solve the differential equation, copied from scipy.signal.ltisys.
        dot, squeeze, = np.dot, np.squeeze  # Faster and shorter code

        # Faster algorithm if U is zero
        if U is None or (isinstance(U, (int, float)) and U == 0):
            # Function that computes the time derivative of the linear system
            def f_dot(x, _t):
                return dot(A, x)

            xout = sp.integrate.odeint(f_dot, X0, T, **keywords)
            yout = dot(C, xout.T)

        # General algorithm that interpolates U in between output points
        else:
            # Test if U has correct shape and type
            legal_shapes = [(n_steps,), (1, n_steps)] if n_inputs == 1 else \
                           [(n_inputs, n_steps)]
            U = _check_convert_array(U, legal_shapes,
                                     'Parameter ``U``: ', squeeze=False,
                                     transpose=transpose)
            # convert 1D array to D2 array with only one row
            if len(U.shape) == 1:
                U = U.reshape(1, -1)  # pylint: disable=E1103

            # Create a callable that uses linear interpolation to
            # calculate the input at any time.
            compute_u = \
                sp.interpolate.interp1d(T, U, kind='linear', copy=False,
                                        axis=-1, bounds_error=False,
                                        fill_value=0)

            # Function that computes the time derivative of the linear system
            def f_dot(x, t):
                return dot(A, x) + squeeze(dot(B, compute_u([t])))

            xout = sp.integrate.odeint(f_dot, X0, T, **keywords)
            yout = dot(C, xout.T) + dot(D, U)

        yout = squeeze(yout)
        xout = xout.T

    else:
        # Discrete time simulation using signal processing toolbox
        dsys = (A, B, C, D, sys.dt)
        tout, yout, xout = sp.signal.dlsim(dsys, U, T, X0)

    # See if we need to transpose the data back into MATLAB form
    if (transpose):
        T = np.transpose(T)
        yout = np.transpose(yout)
        xout = np.transpose(xout)

    return T, yout, xout


def step_response(sys, T=None, X0=0., input=0, output=None,
                  transpose=False, **keywords):
    # pylint: disable=W0622
    """Step response of a linear system

    If the system has multiple inputs or outputs (MIMO), one input has
    to be selected for the simulation. Optionally, one output may be
    selected. The parameters `input` and `output` do this. All other
    inputs are set to 0, all other outputs are ignored.

    For information on the **shape** of parameters `T`, `X0` and
    return values `T`, `yout` see: :ref:`time-series-convention`

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

    **keywords:
        Additional keyword arguments control the solution algorithm for the
        differential equations. These arguments are passed on to the function
        :func:`lsim`, which in turn passes them on to
        :func:`scipy.integrate.odeint`. See the documentation for
        :func:`scipy.integrate.odeint` for information about these
        arguments.

    Returns
    -------
    T: array
        Time values of the output

    yout: array
        Response of the system

    See Also
    --------
    forced_response, initial_response, impulse_response

    Examples
    --------
    >>> T, yout = step_response(sys, T, X0)
    """
    sys = _convertToStateSpace(sys)
    if output is None:
        sys = _mimo2simo(sys, input, warn_conversion=True)
    else:
        sys = _mimo2siso(sys, input, output, warn_conversion=True)
    if T is None:
        if isctime(sys):
            T = _default_response_times(sys.A, 100)
        else:
            # For discrete time, use integers
            tvec = _default_response_times(sys.A, 100)
            T = range(int(np.ceil(max(tvec))))

    U = np.ones_like(T)

    T, yout, _xout = forced_response(sys, T, U, X0,
                                     transpose=transpose, **keywords)

    return T, yout


def initial_response(sys, T=None, X0=0., input=0, output=None,
                     transpose=False, **keywords):
    # pylint: disable=W0622
    """Initial condition response of a linear system

    If the system has multiple outputs (MIMO), optionally, one output
    may be selected. If no selection is made for the output, all
    outputs are given.

    For information on the **shape** of parameters `T`, `X0` and
    return values `T`, `yout` see: :ref:`time-series-convention`

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

    **keywords:
        Additional keyword arguments control the solution algorithm for the
        differential equations. These arguments are passed on to the function
        :func:`lsim`, which in turn passes them on to
        :func:`scipy.integrate.odeint`. See the documentation for
        :func:`scipy.integrate.odeint` for information about these
        arguments.


    Returns
    -------
    T: array
        Time values of the output
    yout: array
        Response of the system

    See Also
    --------
    forced_response, impulse_response, step_response

    Examples
    --------
    >>> T, yout = initial_response(sys, T, X0)
    """
    sys = _convertToStateSpace(sys)
    if output is None:
        sys = _mimo2simo(sys, input, warn_conversion=False)
    else:
        sys = _mimo2siso(sys, input, output, warn_conversion=False)

    # Create time and input vectors; checking is done in forced_response(...)
    # The initial vector X0 is created in forced_response(...) if necessary
    if T is None:
        T = _default_response_times(sys.A, 100)
    U = np.zeros_like(T)

    T, yout, _xout = forced_response(sys, T, U, X0, transpose=transpose,
                                     **keywords)
    return T, yout


def impulse_response(sys, T=None, X0=0., input=0, output=None,
                     transpose=False, **keywords):
    # pylint: disable=W0622
    """Impulse response of a linear system

    If the system has multiple inputs or outputs (MIMO), one input has
    to be selected for the simulation. Optionally, one output may be
    selected. The parameters `input` and `output` do this. All other
    inputs are set to 0, all other outputs are ignored.

    For information on the **shape** of parameters `T`, `X0` and
    return values `T`, `yout` see: :ref:`time-series-convention`

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

    **keywords:
        Additional keyword arguments control the solution algorithm for the
        differential equations. These arguments are passed on to the function
        :func:`lsim`, which in turn passes them on to
        :func:`scipy.integrate.odeint`. See the documentation for
        :func:`scipy.integrate.odeint` for information about these
        arguments.


    Returns
    -------
    T: array
        Time values of the output
    yout: array
        Response of the system

    See Also
    --------
    ForcedReponse, initial_response, step_response

    Examples
    --------
    >>> T, yout = impulse_response(sys, T, X0)
    """
    sys = _convertToStateSpace(sys)
    if output is None:
        sys = _mimo2simo(sys, input, warn_conversion=True)
    else:
        sys = _mimo2siso(sys, input, output, warn_conversion=True)

    # System has direct feedthrough, can't simulate impulse response numerically
    if np.any(sys.D != 0) and isctime(sys):
        warnings.warn('System has direct feedthrough: ``D != 0``. The infinite '
                      'impulse at ``t=0`` does not appear in the output. \n'
                      'Results may be meaningless!')

    # create X0 if not given, test if X0 has correct shape.
    # Must be done here because it is used for computations here.
    n_states = sys.A.shape[0]
    X0 = _check_convert_array(X0, [(n_states,), (n_states, 1)],
                              'Parameter ``X0``: \n', squeeze=True)

    # Compute new X0 that contains the impulse
    # We can't put the impulse into U because there is no numerical
    # representation for it (infinitesimally short, infinitely high).
    # See also: http://www.mathworks.com/support/tech-notes/1900/1901.html
    B = np.asarray(sys.B).squeeze()
    new_X0 = B + X0

    # Compute T and U, no checks necessary, they will be checked in lsim
    if T is None:
        T = _default_response_times(sys.A, 100)
    U = np.zeros_like(T)

    T, yout, _xout = forced_response(
        sys, T, U, new_X0,
        transpose=transpose, **keywords)
    return T, yout
