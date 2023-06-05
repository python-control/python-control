"""
timeresp.py - time-domain simulation routines.

The :mod:`~control.timeresp` module contains a collection of
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

Copyright (c) 2011 by California Institute of Technology
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

Modified: Sawyer B. Fuller (minster@uw.edu) to add discrete-time
capability and better automatic time vector creation
Date: June 2020

Modified by Ilhan Polat to improve automatic time vector creation
Date: August 17, 2020

Modified by Richard Murray to add TimeResponseData class
Date: August 2021

$Id$
"""

import warnings

import numpy as np
import scipy as sp
from numpy import einsum, maximum, minimum
from scipy.linalg import eig, eigvals, matrix_balance, norm
from copy import copy

from . import config
from .exception import pandas_check
from .namedio import isctime, isdtime
from .statesp import StateSpace, _convert_to_statespace, _mimo2simo, _mimo2siso
from .xferfcn import TransferFunction

__all__ = ['forced_response', 'step_response', 'step_info',
           'initial_response', 'impulse_response', 'TimeResponseData']


class TimeResponseData:
    """A class for returning time responses.

    This class maintains and manipulates the data corresponding to the
    temporal response of an input/output system.  It is used as the return
    type for time domain simulations (step response, input/output response,
    etc).

    A time response consists of a time vector, an output vector, and
    optionally an input vector and/or state vector.  Inputs and outputs can
    be 1D (scalar input/output) or 2D (vector input/output).

    A time response can be stored for multiple input signals (called traces),
    with the output and state indexed by the trace number.  This allows for
    input/output response matrices, which is mainly useful for impulse and
    step responses for linear systems.  For multi-trace responses, the same
    time vector must be used for all traces.

    Time responses are accessed through either the raw data, stored as
    :attr:`t`, :attr:`y`, :attr:`x`, :attr:`u`, or using a set of properties
    :attr:`time`, :attr:`outputs`, :attr:`states`, :attr:`inputs`.  When
    accessing time responses via their properties, squeeze processing is
    applied so that (by default) single-input, single-output systems will have
    the output and input indices supressed.  This behavior is set using the
    ``squeeze`` keyword.

    Attributes
    ----------
    t : 1D array
        Time values of the input/output response(s).  This attribute is
        normally accessed via the :attr:`time` property.

    y : 2D or 3D array
        Output response data, indexed either by output index and time (for
        single trace responses) or output, trace, and time (for multi-trace
        responses).  These data are normally accessed via the :attr:`outputs`
        property, which performs squeeze processing.

    x : 2D or 3D array, or None
        State space data, indexed either by output number and time (for single
        trace responses) or output, trace, and time (for multi-trace
        responses).  If no state data are present, value is ``None``. These
        data are normally accessed via the :attr:`states` property, which
        performs squeeze processing.

    u : 2D or 3D array, or None
        Input signal data, indexed either by input index and time (for single
        trace responses) or input, trace, and time (for multi-trace
        responses).  If no input data are present, value is ``None``.  These
        data are normally accessed via the :attr:`inputs` property, which
        performs squeeze processing.

    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO)
        then the outputs (and inputs) are returned as a 1D array
        (indexed by time) and if a system is multi-input or
        multi-output, then the outputs are returned as a 2D array
        (indexed by output and time) or a 3D array (indexed by output,
        trace, and time).  If ``squeeze=True``, access to the output
        response will remove single-dimensional entries from the shape
        of the inputs and outputs even if the system is not SISO. If
        ``squeeze=False``, the output is returned as a 2D or 3D array
        (indexed by the output [if multi-input], trace [if multi-trace]
        and time) even if the system is SISO. The default value can be
        set using config.defaults['control.squeeze_time_response'].

    transpose : bool, optional
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and :func:`scipy.signal.lsim`).  Default
        value is False.

    issiso : bool, optional
        Set to ``True`` if the system generating the data is single-input,
        single-output.  If passed as ``None`` (default), the input data
        will be used to set the value.

    ninputs, noutputs, nstates : int
        Number of inputs, outputs, and states of the underlying system.

    input_labels, output_labels, state_labels : array of str
        Names for the input, output, and state variables.

    ntraces : int
        Number of independent traces represented in the input/output
        response.  If ntraces is 0 then the data represents a single trace
        with the trace index surpressed in the data.

    Notes
    -----
    1. For backward compatibility with earlier versions of python-control,
       this class has an ``__iter__`` method that allows it to be assigned
       to a tuple with a variable number of elements.  This allows the
       following patterns to work:

         t, y = step_response(sys)
         t, y, x = step_response(sys, return_x=True)

       When using this (legacy) interface, the state vector is not affected by
       the `squeeze` parameter.

    2. For backward compatibility with earlier version of python-control,
       this class has ``__getitem__`` and ``__len__`` methods that allow the
       return value to be indexed:

         response[0]: returns the time vector
         response[1]: returns the output vector
         response[2]: returns the state vector

       When using this (legacy) interface, the state vector is not affected by
       the `squeeze` parameter.

    3. The default settings for ``return_x``, ``squeeze`` and ``transpose``
       can be changed by calling the class instance and passing new values:

         response(tranpose=True).input

       See :meth:`TimeResponseData.__call__` for more information.

    """

    def __init__(
            self, time, outputs, states=None, inputs=None, issiso=None,
            output_labels=None, state_labels=None, input_labels=None,
            transpose=False, return_x=False, squeeze=None, multi_trace=False
    ):
        """Create an input/output time response object.

        Parameters
        ----------
        time : 1D array
            Time values of the output.  Ignored if None.

        outputs : ndarray
            Output response of the system.  This can either be a 1D array
            indexed by time (for SISO systems or MISO systems with a specified
            input), a 2D array indexed by output and time (for MIMO systems
            with no input indexing, such as initial_response or forced
            response) or trace and time (for SISO systems with multiple
            traces), or a 3D array indexed by output, trace, and time (for
            multi-trace input/output responses).

        states : array, optional
            Individual response of each state variable. This should be a 2D
            array indexed by the state index and time (for single trace
            systems) or a 3D array indexed by state, trace, and time.

        inputs : array, optional
            Inputs used to generate the output.  This can either be a 1D
            array indexed by time (for SISO systems or MISO/MIMO systems
            with a specified input), a 2D array indexed either by input and
            time (for a multi-input system) or trace and time (for a
            single-input, multi-trace response), or a 3D array indexed by
            input, trace, and time.

        sys : LTI or InputOutputSystem, optional
            System that generated the data.  If desired, the system used to
            generate the data can be stored along with the data.

        squeeze : bool, optional
            By default, if a system is single-input, single-output (SISO)
            then the inputs and outputs are returned as a 1D array (indexed
            by time) and if a system is multi-input or multi-output, then
            the inputs are returned as a 2D array (indexed by input and
            time) and the outputs are returned as either a 2D array (indexed
            by output and time) or a 3D array (indexed by output, trace, and
            time).  If squeeze=True, access to the output response will
            remove single-dimensional entries from the shape of the inputs
            and outputs even if the system is not SISO. If squeeze=False,
            keep the input as a 2D or 3D array (indexed by the input (if
            multi-input), trace (if single input) and time) and the output
            as a 3D array (indexed by the output, trace, and time) even if
            the system is SISO. The default value can be set using
            config.defaults['control.squeeze_time_response'].

        Other parameters
        ----------------
        input_labels, output_labels, state_labels: array of str, optional
            Optional labels for the inputs, outputs, and states, given as a
            list of strings matching the appropriate signal dimension.

        transpose : bool, optional
            If True, transpose all input and output arrays (for backward
            compatibility with MATLAB and :func:`scipy.signal.lsim`).
            Default value is False.

        return_x : bool, optional
            If True, return the state vector when enumerating result by
            assigning to a tuple (default = False).

        multi_trace : bool, optional
            If ``True``, then 2D input array represents multiple traces.  For
            a MIMO system, the ``input`` attribute should then be set to
            indicate which trace is being specified.  Default is ``False``.

        """
        #
        # Process and store the basic input/output elements
        #

        # Time vector
        self.t = np.atleast_1d(time)
        if self.t.ndim != 1:
            raise ValueError("Time vector must be 1D array")

        #
        # Output vector (and number of traces)
        #
        self.y = np.array(outputs)

        if self.y.ndim == 3:
            multi_trace = True
            self.noutputs = self.y.shape[0]
            self.ntraces = self.y.shape[1]

        elif multi_trace and self.y.ndim == 2:
            self.noutputs = 1
            self.ntraces = self.y.shape[0]

        elif not multi_trace and self.y.ndim == 2:
            self.noutputs = self.y.shape[0]
            self.ntraces = 0

        elif not multi_trace and self.y.ndim == 1:
            self.noutputs = 1
            self.ntraces = 0

            # Reshape the data to be 2D for consistency
            self.y = self.y.reshape(self.noutputs, -1)

        else:
            raise ValueError("Output vector is the wrong shape")

        # Check and store labels, if present
        self.output_labels = _process_labels(
            output_labels, "output", self.noutputs)

        # Make sure time dimension of output is the right length
        if self.t.shape[-1] != self.y.shape[-1]:
            raise ValueError("Output vector does not match time vector")

        #
        # State vector (optional)
        #
        # If present, the shape of the state vector should be consistent
        # with the multi-trace nature of the data.
        #
        if states is None:
            self.x = None
            self.nstates = 0
        else:
            self.x = np.array(states)
            self.nstates = self.x.shape[0]

            # Make sure the shape is OK
            if multi_trace and \
               (self.x.ndim != 3 or self.x.shape[1] != self.ntraces) or \
               not multi_trace and self.x.ndim != 2 :
                raise ValueError("State vector is the wrong shape")

            # Make sure time dimension of state is the right length
            if self.t.shape[-1] != self.x.shape[-1]:
                raise ValueError("State vector does not match time vector")

        # Check and store labels, if present
        self.state_labels = _process_labels(
            state_labels, "state", self.nstates)

        #
        # Input vector (optional)
        #
        # If present, the shape and dimensions of the input vector should be
        # consistent with the trace count computed above.
        #
        if inputs is None:
            self.u = None
            self.ninputs = 0

        else:
            self.u = np.array(inputs)

            # Make sure the shape is OK and figure out the nuumber of inputs
            if multi_trace and self.u.ndim == 3 and \
               self.u.shape[1] == self.ntraces:
                self.ninputs = self.u.shape[0]

            elif multi_trace and self.u.ndim == 2 and \
                 self.u.shape[0] == self.ntraces:
                self.ninputs = 1

            elif not multi_trace and self.u.ndim == 2 and \
                 self.ntraces == 0:
                self.ninputs = self.u.shape[0]

            elif not multi_trace and self.u.ndim == 1:
                self.ninputs = 1

                # Reshape the data to be 2D for consistency
                self.u = self.u.reshape(self.ninputs, -1)

            else:
                raise ValueError("Input vector is the wrong shape")

            # Make sure time dimension of output is the right length
            if self.t.shape[-1] != self.u.shape[-1]:
                raise ValueError("Input vector does not match time vector")

        # Check and store labels, if present
        self.input_labels = _process_labels(
            input_labels, "input", self.ninputs)

        # Figure out if the system is SISO
        if issiso is None:
            # Figure out based on the data
            if self.ninputs == 1:
                issiso = (self.noutputs == 1)
            elif self.ninputs > 1:
                issiso = False
            else:
                # Missing input data => can't resolve
                raise ValueError("Can't determine if system is SISO")
        elif issiso is True and (self.ninputs > 1 or self.noutputs > 1):
            raise ValueError("Keyword `issiso` does not match data")

        # Set the value to be used for future processing
        self.issiso = issiso

        # Keep track of whether to squeeze inputs, outputs, and states
        if not (squeeze is True or squeeze is None or squeeze is False):
            raise ValueError("Unknown squeeze value")
        self.squeeze = squeeze

        # Keep track of whether to transpose for MATLAB/scipy.signal
        self.transpose = transpose

        # Store legacy keyword values (only needed for legacy interface)
        self.return_x = return_x

    def __call__(self, **kwargs):
        """Change value of processing keywords.

        Calling the time response object will create a copy of the object and
        change the values of the keywords used to control the ``outputs``,
        ``states``, and ``inputs`` properties.

        Parameters
        ----------
        squeeze : bool, optional
            If squeeze=True, access to the output response will remove
            single-dimensional entries from the shape of the inputs, outputs,
            and states even if the system is not SISO. If squeeze=False, keep
            the input as a 2D or 3D array (indexed by the input (if
            multi-input), trace (if single input) and time) and the output and
            states as a 3D array (indexed by the output/state, trace, and
            time) even if the system is SISO.

        transpose : bool, optional
            If True, transpose all input and output arrays (for backward
            compatibility with MATLAB and :func:`scipy.signal.lsim`).
            Default value is False.

        return_x : bool, optional
            If True, return the state vector when enumerating result by
            assigning to a tuple (default = False).

        input_labels, output_labels, state_labels: array of str
            Labels for the inputs, outputs, and states, given as a
            list of strings matching the appropriate signal dimension.

        """
        # Make a copy of the object
        response = copy(self)

        # Update any keywords that we were passed
        response.transpose = kwargs.pop('transpose', self.transpose)
        response.squeeze = kwargs.pop('squeeze', self.squeeze)
        response.return_x = kwargs.pop('return_x', self.return_x)

        # Check for new labels
        input_labels = kwargs.pop('input_labels', None)
        if input_labels is not None:
            response.input_labels = _process_labels(
                input_labels, "input", response.ninputs)

        output_labels = kwargs.pop('output_labels', None)
        if output_labels is not None:
            response.output_labels = _process_labels(
                output_labels, "output", response.noutputs)

        state_labels = kwargs.pop('state_labels', None)
        if state_labels is not None:
            response.state_labels = _process_labels(
                state_labels, "state", response.nstates)

        # Make sure there were no extraneous keywords
        if kwargs:
            raise TypeError("unrecognized keywords: ", str(kwargs))

        return response

    @property
    def time(self):

        """Time vector.

        Time values of the input/output response(s).

        :type: 1D array"""
        return self.t

    # Getter for output (implements squeeze processing)
    @property
    def outputs(self):
        """Time response output vector.

        Output response of the system, indexed by either the output and time
        (if only a single input is given) or the output, trace, and time
        (for multiple traces).  See :attr:`TimeResponseData.squeeze` for a
        description of how this can be modified using the `squeeze` keyword.

        :type: 1D, 2D, or 3D array

        """
        t, y = _process_time_response(
            self.t, self.y, issiso=self.issiso,
            transpose=self.transpose, squeeze=self.squeeze)
        return y

    # Getter for states (implements squeeze processing)
    @property
    def states(self):
        """Time response state vector.

        Time evolution of the state vector, indexed indexed by either the
        state and time (if only a single trace is given) or the state, trace,
        and time (for multiple traces).  See :attr:`TimeResponseData.squeeze`
        for a description of how this can be modified using the `squeeze`
        keyword.

        :type: 2D or 3D array

        """
        if self.x is None:
            return None

        elif self.squeeze is True:
            x = self.x.squeeze()

        elif self.ninputs == 1 and self.noutputs == 1 and \
             self.ntraces == 1 and self.x.ndim == 3 and \
             self.squeeze is not False:
            # Single-input, single-output system with single trace
            x = self.x[:, 0, :]

        else:
            # Return the full set of data
            x = self.x

        # Transpose processing
        if self.transpose:
            x = np.transpose(x, np.roll(range(x.ndim), 1))

        return x

    # Getter for inputs (implements squeeze processing)
    @property
    def inputs(self):
        """Time response input vector.

        Input(s) to the system, indexed by input (optiona), trace (optional),
        and time.  If a 1D vector is passed, the input corresponds to a
        scalar-valued input.  If a 2D vector is passed, then it can either
        represent multiple single-input traces or a single multi-input trace.
        The optional ``multi_trace`` keyword should be used to disambiguate
        the two.  If a 3D vector is passed, then it represents a multi-trace,
        multi-input signal, indexed by input, trace, and time.

        See :attr:`TimeResponseData.squeeze` for a description of how the
        dimensions of the input vector can be modified using the `squeeze`
        keyword.

        :type: 1D or 2D array

        """
        if self.u is None:
            return None

        t, u = _process_time_response(
            self.t, self.u, issiso=self.issiso,
            transpose=self.transpose, squeeze=self.squeeze)
        return u

    # Getter for legacy state (implements non-standard squeeze processing)
    @property
    def _legacy_states(self):
        """Time response state vector (legacy version).

        Time evolution of the state vector, indexed indexed by either the
        state and time (if only a single trace is given) or the state,
        trace, and time (for multiple traces).

        The `legacy_states` property is not affected by the `squeeze` keyword
        and hence it will always have these dimensions.

        :type: 2D or 3D array

        """

        if self.x is None:
            return None

        elif self.ninputs == 1 and self.noutputs == 1 and \
             self.ntraces == 1 and self.x.ndim == 3:
            # Single-input, single-output system with single trace
            x = self.x[:, 0, :]

        else:
            # Return the full set of data
            x = self.x

        # Transpose processing
        if self.transpose:
            x = np.transpose(x, np.roll(range(x.ndim), 1))

        return x

    # Implement iter to allow assigning to a tuple
    def __iter__(self):
        if not self.return_x:
            return iter((self.time, self.outputs))
        return iter((self.time, self.outputs, self._legacy_states))

    # Implement (thin) getitem to allow access via legacy indexing
    def __getitem__(self, index):
        # See if we were passed a slice
        if isinstance(index, slice):
            if (index.start is None or index.start == 0) and index.stop == 2:
                return (self.time, self.outputs)

        # Otherwise assume we were passed a single index
        if index == 0:
            return self.time
        if index == 1:
            return self.outputs
        if index == 2:
            return self._legacy_states
        raise IndexError

    # Implement (thin) len to emulate legacy testing interface
    def __len__(self):
        return 3 if self.return_x else 2

    # Convert to pandas
    def to_pandas(self):
        if not pandas_check():
            raise ImportError("pandas not installed")
        import pandas

        # Create a dict for setting up the data frame
        data = {'time': self.time}
        data.update(
            {name: self.u[i] for i, name in enumerate(self.input_labels)})
        data.update(
            {name: self.y[i] for i, name in enumerate(self.output_labels)})
        data.update(
            {name: self.x[i] for i, name in enumerate(self.state_labels)})

        return pandas.DataFrame(data)


# Process signal labels
def _process_labels(labels, signal, length):
    """Process time response signal labels.

    Parameters
    ----------
    labels : list of str or dict
        Description of the labels for the signal.  This can be a list of
        strings or a dict giving the index of each signal (used in iosys).

    signal : str
        Name of the signal being processed (for error messages).

    length : int
        Number of labels required.

    Returns
    -------
    labels : list of str
        List of labels.

    """
    if labels is None or len(labels) == 0:
        return None

    # See if we got passed a dictionary (from iosys)
    if isinstance(labels, dict):
        # Form inverse dictionary
        ivd = {v: k for k, v in labels.items()}

        try:
            # Turn into a list
            labels = [ivd[n] for n in range(len(labels))]
        except KeyError:
            raise ValueError("Name dictionary for %s is incomplete" % signal)

    # Convert labels to a list
    if isinstance(labels, str):
        labels = [labels]
    else:
        labels = list(labels)

    # Make sure the signal list is the right length and type
    if len(labels) != length:
        raise ValueError("List of %s labels is the wrong length" % signal)
    elif not all([isinstance(label, str) for label in labels]):
        raise ValueError("List of %s labels must all  be strings" % signal)

    return labels


# Helper function for checking array-like parameters
def _check_convert_array(in_obj, legal_shapes, err_msg_start, squeeze=False,
                         transpose=False):

    """Helper function for checking array_like parameters.

    * Check type and shape of ``in_obj``.
    * Convert ``in_obj`` to an array if necessary.
    * Change shape of ``in_obj`` according to parameter ``squeeze``.
    * If ``in_obj`` is a scalar (number) it is converted to an array with
      a legal shape, that is filled with the scalar value.

    The function raises an exception when it detects an error.

    Parameters
    ----------
    in_obj : array like object
        The array or matrix which is checked.

    legal_shapes : list of tuple
        A list of shapes that in_obj can legally have.
        The special value "any" means that there can be any
        number of elements in a certain dimension.

        * ``(2, 3)`` describes an array with 2 rows and 3 columns
        * ``(2, "any")`` describes an array with 2 rows and any number of
          columns

    err_msg_start : str
        String that is prepended to the error messages, when this function
        raises an exception. It should be used to identify the argument which
        is currently checked.

    squeeze : bool
        If True, all dimensions with only one element are removed from the
        array. If False the array's shape is unmodified.

        For example:
        ``array([[1,2,3]])`` is converted to ``array([1, 2, 3])``

    transpose : bool, optional
        If True, assume that 2D input arrays are transposed from the standard
        format.  Used to convert MATLAB-style inputs to our format.

    Returns
    -------

    out_array : array
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
                    interpolate=False, return_x=None, squeeze=None):
    """Compute the output of a linear system given the input.

    As a convenience for parameters `U`, `X0`:
    Numbers (scalars) are converted to constant arrays with the correct shape.
    The correct shape is inferred from arguments `sys` and `T`.

    For information on the **shape** of parameters `U`, `T`, `X0` and
    return values `T`, `yout`, `xout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sys : StateSpace or TransferFunction
        LTI system to simulate

    T : array_like, optional for discrete LTI `sys`
        Time steps at which the input is defined; values must be evenly spaced.

        If None, `U` must be given and `len(U)` time steps of sys.dt are
        simulated. If sys.dt is None or True (undetermined time step), a time
        step of 1.0 is assumed.

    U : array_like or float, optional
        Input array giving input at each time `T`.
        If `U` is None or 0, `T` must be given, even for discrete
        time systems. In this case, for continuous time systems, a direct
        calculation of the matrix exponential is used, which is faster than the
        general interpolating algorithm used otherwise.

    X0 : array_like or float, default=0.
        Initial condition.

    transpose : bool, default=False
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and :func:`scipy.signal.lsim`).

    interpolate : bool, default=False
        If True and system is a discrete time system, the input will
        be interpolated between the given time steps and the output
        will be given at system sampling rate.  Otherwise, only return
        the output at the times given in `T`.  No effect on continuous
        time simulations.

    return_x : bool, default=None
        Used if the time response data is assigned to a tuple:

        * If False, return only the time and output vectors.

        * If True, also return the the state vector.

        * If None, determine the returned variables by
          config.defaults['forced_response.return_x'], which was True
          before version 0.9 and is False since then.

    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO) then
        the output response is returned as a 1D array (indexed by time).  If
        `squeeze` is True, remove single-dimensional entries from the shape of
        the output even if the system is not SISO. If `squeeze` is False, keep
        the output as a 2D array (indexed by the output number and time)
        even if the system is SISO. The default behavior can be overridden by
        config.defaults['control.squeeze_time_response'].

    Returns
    -------
    results : TimeResponseData
        Time response represented as a :class:`TimeResponseData` object
        containing the following properties:

        * time (array): Time values of the output.

        * outputs (array): Response of the system.  If the system is SISO and
          `squeeze` is not True, the array is 1D (indexed by time).  If the
          system is not SISO or `squeeze` is False, the array is 2D (indexed
          by output and time).

        * states (array): Time evolution of the state vector, represented as
          a 2D array indexed by state and time.

        * inputs (array): Input(s) to the system, indexed by input and time.

        The return value of the system can also be accessed by assigning the
        function to a tuple of length 2 (time, output) or of length 3 (time,
        output, state) if ``return_x`` is ``True``.

    See Also
    --------
    step_response, initial_response, impulse_response

    Notes
    -----
    For discrete time systems, the input/output response is computed using the
    :func:`scipy.signal.dlsim` function.

    For continuous time systems, the output is computed using the matrix
    exponential `exp(A t)` and assuming linear interpolation of the inputs
    between time points.

    Examples
    --------
    >>> G = ct.rss(4)
    >>> T = np.linspace(0, 10)
    >>> T, yout = ct.forced_response(G, T=T)

    See :ref:`time-series-convention` and
    :ref:`package-configuration-parameters`.

    """
    if not isinstance(sys, (StateSpace, TransferFunction)):
        raise TypeError('Parameter ``sys``: must be a ``StateSpace`` or'
                        ' ``TransferFunction``)')

    # If return_x was not specified, figure out the default
    if return_x is None:
        return_x = config.defaults['forced_response.return_x']

    # If return_x is used for TransferFunction, issue a warning
    if return_x and isinstance(sys, TransferFunction):
        warnings.warn(
            "return_x specified for a transfer function system. Internal "
            "conversion to state space used; results may meaningless.")

    # If we are passed a transfer function and X0 is non-zero, warn the user
    if isinstance(sys, TransferFunction) and np.any(X0 != 0):
        warnings.warn(
            "Non-zero initial condition given for transfer function system. "
            "Internal conversion to state space used; may not be consistent "
            "with given X0.")

    sys = _convert_to_statespace(sys)
    A, B, C, D = np.asarray(sys.A), np.asarray(sys.B), np.asarray(sys.C), \
        np.asarray(sys.D)
    # d_type = A.dtype
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    n_outputs = C.shape[0]

    # Convert inputs to numpy arrays for easier shape checking
    if U is not None:
        U = np.asarray(U)
    if T is not None:
        # T must be array-like
        T = np.asarray(T)

    # Set and/or check time vector in discrete time case
    if isdtime(sys):
        if T is None:
            if U is None or (U.ndim == 0 and U == 0.):
                raise ValueError('Parameters ``T`` and ``U`` can\'t both be '
                                 'zero for discrete-time simulation')
            # Set T to equally spaced samples with same length as U
            if U.ndim == 1:
                n_steps = U.shape[0]
            else:
                n_steps = U.shape[1]
            dt = 1. if sys.dt in [True, None] else sys.dt
            T = np.array(range(n_steps)) * dt
        else:
            if U.ndim == 0:
                U = np.full((n_inputs, T.shape[0]), U)
    else:
        if T is None:
            raise ValueError('Parameter ``T`` is mandatory for continuous '
                             'time systems.')

    # Test if T has shape (n,) or (1, n);
    T = _check_convert_array(T, [('any',), (1, 'any')],
                             'Parameter ``T``: ', squeeze=True,
                             transpose=transpose)

    n_steps = T.shape[0]            # number of simulation steps

    # equally spaced also implies strictly monotonic increase,
    dt = (T[-1] - T[0]) / (n_steps - 1)
    if not np.allclose(np.diff(T), dt):
        raise ValueError("Parameter ``T``: time values must be equally "
                         "spaced.")

    # create X0 if not given, test if X0 has correct shape
    X0 = _check_convert_array(X0, [(n_states,), (n_states, 1)],
                              'Parameter ``X0``: ', squeeze=True)

    # Test if U has correct shape and type
    legal_shapes = [(n_steps,), (1, n_steps)] if n_inputs == 1 else \
        [(n_inputs, n_steps)]
    U = _check_convert_array(U, legal_shapes,
                             'Parameter ``U``: ', squeeze=False,
                             transpose=transpose)

    xout = np.zeros((n_states, n_steps))
    xout[:, 0] = X0
    yout = np.zeros((n_outputs, n_steps))

    # Separate out the discrete and continuous time cases
    if isctime(sys, strict=True):
        # Solve the differential equation, copied from scipy.signal.ltisys.

        # Faster algorithm if U is zero
        # (if not None, it was converted to array above)
        if U is None or np.all(U == 0):
            # Solve using matrix exponential
            expAdt = sp.linalg.expm(A * dt)
            for i in range(1, n_steps):
                xout[:, i] = expAdt @ xout[:, i-1]
            yout = C @ xout

        # General algorithm that interpolates U in between output points
        else:
            # convert input from 1D array to 2D array with only one row
            if U.ndim == 1:
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
                xout[:, i] = (Ad @ xout[:, i-1]
                              + Bd0 @ U[:, i-1] + Bd1 @ U[:, i])
            yout = C @ xout + D @ U
        tout = T

    else:
        # Discrete type system => use SciPy signal processing toolbox

        # sp.signal.dlsim assumes T[0] == 0
        spT = T - T[0]

        if sys.dt is not True and sys.dt is not None:
            # Make sure that the time increment is a multiple of sampling time

            # First make sure that time increment is bigger than sampling time
            # (with allowance for small precision errors)
            if dt < sys.dt and not np.isclose(dt, sys.dt):
                raise ValueError("Time steps ``T`` must match sampling time")

            # Now check to make sure it is a multiple (with check against
            # sys.dt because floating point mod can have small errors
            if not (np.isclose(dt % sys.dt, 0) or
                    np.isclose(dt % sys.dt, sys.dt)):
                raise ValueError("Time steps ``T`` must be multiples of "
                                 "sampling time")
            sys_dt = sys.dt

            # sp.signal.dlsim returns not enough samples if
            # T[-1] - T[0] < sys_dt * decimation * (n_steps - 1)
            # due to rounding errors.
            # https://github.com/scipyscipy/blob/v1.6.1/scipy/signal/ltisys.py#L3462
            scipy_out_samples = int(np.floor(spT[-1] / sys_dt)) + 1
            if scipy_out_samples < n_steps:
                # parantheses: order of evaluation is important
                spT[-1] = spT[-1] * (n_steps / (spT[-1] / sys_dt + 1))

        else:
            sys_dt = dt         # For unspecified sampling time, use time incr

        # Discrete time simulation using signal processing toolbox
        dsys = (A, B, C, D, sys_dt)

        # Use signal processing toolbox for the discrete time simulation
        # Transpose the input to match toolbox convention
        tout, yout, xout = sp.signal.dlsim(dsys, np.transpose(U), spT, X0)
        tout = tout + T[0]

        if not interpolate:
            # If dt is different from sys.dt, resample the output
            inc = int(round(dt / sys_dt))
            tout = T            # Return exact list of time steps
            yout = yout[::inc, :]
            xout = xout[::inc, :]
        else:
            # Interpolate the input to get the right number of points
            U = sp.interpolate.interp1d(T, U)(tout)

        # Transpose the output and state vectors to match local convention
        xout = np.transpose(xout)
        yout = np.transpose(yout)

    return TimeResponseData(
        tout, yout, xout, U, issiso=sys.issiso(),
        output_labels=sys.output_labels, input_labels=sys.input_labels,
        state_labels=sys.state_labels,
        transpose=transpose, return_x=return_x, squeeze=squeeze)


# Process time responses in a uniform way
def _process_time_response(
        tout, yout, issiso=False, transpose=None, squeeze=None):
    """Process time response signals.

    This function processes the outputs (or inputs) of time response
    functions and processes the transpose and squeeze keywords.

    Parameters
    ----------
    T : 1D array
        Time values of the output.  Ignored if None.

    yout : ndarray
        Response of the system.  This can either be a 1D array indexed by time
        (for SISO systems), a 2D array indexed by output and time (for MIMO
        systems with no input indexing, such as initial_response or forced
        response) or a 3D array indexed by output, input, and time.

    issiso : bool, optional
        If ``True``, process data as single-input, single-output data.
        Default is ``False``.

    transpose : bool, optional
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and :func:`scipy.signal.lsim`).  Default
        value is False.

    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO) then the
        output response is returned as a 1D array (indexed by time).  If
        squeeze=True, remove single-dimensional entries from the shape of the
        output even if the system is not SISO. If squeeze=False, keep the
        output as a 3D array (indexed by the output, input, and time) even if
        the system is SISO. The default value can be set using
        config.defaults['control.squeeze_time_response'].

    Returns
    -------
    T : 1D array
        Time values of the output.

    yout : ndarray
        Response of the system.  If the system is SISO and squeeze is not
        True, the array is 1D (indexed by time).  If the system is not SISO or
        squeeze is False, the array is either 2D (indexed by output and time)
        or 3D (indexed by input, output, and time).

    """
    # If squeeze was not specified, figure out the default (might remain None)
    if squeeze is None:
        squeeze = config.defaults['control.squeeze_time_response']

    # Figure out whether and how to squeeze output data
    if squeeze is True:         # squeeze all dimensions
        yout = np.squeeze(yout)
    elif squeeze is False:      # squeeze no dimensions
        pass
    elif squeeze is None:       # squeeze signals if SISO
        if issiso:
            if yout.ndim == 3:
                yout = yout[0][0]       # remove input and output
            else:
                yout = yout[0]          # remove input
    else:
        raise ValueError("Unknown squeeze value")

    # See if we need to transpose the data back into MATLAB form
    if transpose:
        # Transpose time vector in case we are using np.matrix
        tout = np.transpose(tout)

        # For signals, put the last index (time) into the first slot
        yout = np.transpose(yout, np.roll(range(yout.ndim), 1))

    # Return time, output, and (optionally) state
    return tout, yout


def _get_ss_simo(sys, input=None, output=None, squeeze=None):
    """Return a SISO or SIMO state-space version of sys.

    This function converts the given system to a state space system in
    preparation for simulation and sets the system matrixes to match the
    desired input and output.

    If input is not specified, select first input and issue warning (legacy
    behavior that should eventually not be used).

    If the output is not specified, report on all outputs.

    """
    # If squeeze was not specified, figure out the default
    if squeeze is None:
        squeeze = config.defaults['control.squeeze_time_response']

    sys_ss = _convert_to_statespace(sys)
    if sys_ss.issiso():
        return squeeze, sys_ss
    elif squeeze is None and (input is None or output is None):
        # Don't squeeze outputs if resulting system turns out to be siso
        # Note: if we expand input to allow a tuple, need to update this check
        squeeze = False

    warn = False
    if input is None:
        # issue warning if input is not given
        warn = True
        input = 0

    if output is None:
        return squeeze, _mimo2simo(sys_ss, input, warn_conversion=warn)
    else:
        return squeeze, _mimo2siso(sys_ss, input, output, warn_conversion=warn)


def step_response(sys, T=None, X0=0., input=None, output=None, T_num=None,
                  transpose=False, return_x=False, squeeze=None):
    # pylint: disable=W0622
    """Compute the step response for a linear system.

    If the system has multiple inputs and/or multiple outputs, the step
    response is computed for each input/output pair, with all other inputs set
    to zero.  Optionally, a single input and/or single output can be selected,
    in which case all other inputs are set to 0 and all other outputs are
    ignored.

    For information on the **shape** of parameters `T`, `X0` and
    return values `T`, `yout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sys : StateSpace or TransferFunction
        LTI system to simulate

    T : array_like or float, optional
        Time vector, or simulation time duration if a number. If T is not
        provided, an attempt is made to create it automatically from the
        dynamics of sys. If sys is continuous-time, the time increment dt
        is chosen small enough to show the fastest mode, and the simulation
        time period tfinal long enough to show the slowest mode, excluding
        poles at the origin and pole-zero cancellations. If this results in
        too many time steps (>5000), dt is reduced. If sys is discrete-time,
        only tfinal is computed, and final is reduced if it requires too
        many simulation steps.

    X0 : array_like or float, optional
        Initial condition (default = 0). Numbers are converted to constant
        arrays with the correct shape.

    input : int, optional
        Only compute the step response for the listed input.  If not
        specified, the step responses for each independent input are
        computed (as separate traces).

    output : int, optional
        Only report the step response for the listed output.  If not
        specified, all outputs are reported.

    T_num : int, optional
        Number of time steps to use in simulation if T is not provided as an
        array (autocomputed if not given); ignored if sys is discrete-time.

    transpose : bool, optional
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and :func:`scipy.signal.lsim`).  Default
        value is False.

    return_x : bool, optional
        If True, return the state vector when assigning to a tuple (default =
        False).  See :func:`forced_response` for more details.

    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO) then the
        output response is returned as a 1D array (indexed by time).  If
        squeeze=True, remove single-dimensional entries from the shape of the
        output even if the system is not SISO. If squeeze=False, keep the
        output as a 3D array (indexed by the output, input, and time) even if
        the system is SISO. The default value can be set using
        config.defaults['control.squeeze_time_response'].

    Returns
    -------
    results : TimeResponseData
        Time response represented as a :class:`TimeResponseData` object
        containing the following properties:

        * time (array): Time values of the output.

        * outputs (array): Response of the system.  If the system is SISO and
          squeeze is not True, the array is 1D (indexed by time).  If the
          system is not SISO or ``squeeze`` is False, the array is 3D (indexed
          by the output, trace, and time).

        * states (array): Time evolution of the state vector, represented as
          either a 2D array indexed by state and time (if SISO) or a 3D array
          indexed by state, trace, and time.  Not affected by ``squeeze``.

        * inputs (array): Input(s) to the system, indexed in the same manner
          as ``outputs``.

        The return value of the system can also be accessed by assigning the
        function to a tuple of length 2 (time, output) or of length 3 (time,
        output, state) if ``return_x`` is ``True``.

    See Also
    --------
    forced_response, initial_response, impulse_response

    Notes
    -----
    This function uses the `forced_response` function with the input set to a
    unit step.

    Examples
    --------
    >>> G = ct.rss(4)
    >>> T, yout = ct.step_response(G)

    """
    # Create the time and input vectors
    if T is None or np.asarray(T).size == 1:
        T = _default_time_vector(sys, N=T_num, tfinal=T, is_step=True)
    U = np.ones_like(T)

    # If we are passed a transfer function and X0 is non-zero, warn the user
    if isinstance(sys, TransferFunction) and np.any(X0 != 0):
        warnings.warn(
            "Non-zero initial condition given for transfer function system. "
            "Internal conversion to state space used; may not be consistent "
            "with given X0.")

    # Convert to state space so that we can simulate
    sys = _convert_to_statespace(sys)

    # Set up arrays to handle the output
    ninputs = sys.ninputs if input is None else 1
    noutputs = sys.noutputs if output is None else 1
    yout = np.empty((noutputs, ninputs, np.asarray(T).size))
    xout = np.empty((sys.nstates, ninputs, np.asarray(T).size))
    uout = np.empty((ninputs, ninputs, np.asarray(T).size))

    # Simulate the response for each input
    for i in range(sys.ninputs):
        # If input keyword was specified, only simulate for that input
        if isinstance(input, int) and i != input:
            continue

        # Create a set of single inputs system for simulation
        squeeze, simo = _get_ss_simo(sys, i, output, squeeze=squeeze)

        response = forced_response(simo, T, U, X0, squeeze=True)
        inpidx = i if input is None else 0
        yout[:, inpidx, :] = response.y
        xout[:, inpidx, :] = response.x
        uout[:, inpidx, :] = U

    # Figure out if the system is SISO or not
    issiso = sys.issiso() or (input is not None and output is not None)

    # Select only the given input and output, if any
    input_labels = sys.input_labels if input is None \
        else sys.input_labels[input]
    output_labels = sys.output_labels if output is None \
        else sys.output_labels[output]

    return TimeResponseData(
        response.time, yout, xout, uout, issiso=issiso,
        output_labels=output_labels, input_labels=input_labels,
        state_labels=sys.state_labels,
        transpose=transpose, return_x=return_x, squeeze=squeeze)


def step_info(sysdata, T=None, T_num=None, yfinal=None,
              SettlingTimeThreshold=0.02, RiseTimeLimits=(0.1, 0.9)):
    """
    Step response characteristics (Rise time, Settling Time, Peak and others).

    Parameters
    ----------
    sysdata : StateSpace or TransferFunction or array_like
        The system data. Either LTI system to simulate (StateSpace,
        TransferFunction), or a time series of step response data.
    T : array_like or float, optional
        Time vector, or simulation time duration if a number (time vector is
        autocomputed if not given, see :func:`step_response` for more detail).
        Required, if sysdata is a time series of response data.
    T_num : int, optional
        Number of time steps to use in simulation if T is not provided as an
        array; autocomputed if not given; ignored if sysdata is a
        discrete-time system or a time series or response data.
    yfinal : scalar or array_like, optional
        Steady-state response. If not given, sysdata.dcgain() is used for
        systems to simulate and the last value of the the response data is
        used for a given time series of response data. Scalar for SISO,
        (noutputs, ninputs) array_like for MIMO systems.
    SettlingTimeThreshold : float, optional
        Defines the error to compute settling time (default = 0.02)
    RiseTimeLimits : tuple (lower_threshold, upper_theshold)
        Defines the lower and upper threshold for RiseTime computation

    Returns
    -------
    S : dict or list of list of dict
        If `sysdata` corresponds to a SISO system, S is a dictionary
        containing:

        RiseTime:
            Time from 10% to 90% of the steady-state value.
        SettlingTime:
            Time to enter inside a default error of 2%
        SettlingMin:
            Minimum value after RiseTime
        SettlingMax:
            Maximum value after RiseTime
        Overshoot:
            Percentage of the Peak relative to steady value
        Undershoot:
            Percentage of undershoot
        Peak:
            Absolute peak value
        PeakTime:
            time of the Peak
        SteadyStateValue:
            Steady-state value

        If `sysdata` corresponds to a MIMO system, `S` is a 2D list of dicts.
        To get the step response characteristics from the j-th input to the
        i-th output, access ``S[i][j]``


    See Also
    --------
    step, lsim, initial, impulse

    Examples
    --------
    >>> sys = ct.TransferFunction([-1, 1], [1, 1, 1])
    >>> S = ct.step_info(sys)
    >>> for k in S:
    ...     print(f"{k}: {S[k]:3.4}")
    ...
    RiseTime: 1.256
    SettlingTime: 9.071
    SettlingMin: 0.9011
    SettlingMax: 1.208
    Overshoot: 20.85
    Undershoot: 27.88
    Peak: 1.208
    PeakTime: 4.187
    SteadyStateValue: 1.0

    MIMO System: Simulate until a final time of 10. Get the step response
    characteristics for the second input and specify a 5% error until the
    signal is considered settled.

    >>> from math import sqrt
    >>> sys = ct.StateSpace([[-1., -1.],
    ...                   [1., 0.]],
    ...                  [[-1./sqrt(2.), 1./sqrt(2.)],
    ...                   [0, 0]],
    ...                  [[sqrt(2.), -sqrt(2.)]],
    ...                  [[0, 0]])
    >>> S = ct.step_info(sys, T=10., SettlingTimeThreshold=0.05)
    >>> for k, v in S[0][1].items():
    ...     print(f"{k}: {float(v):3.4}")
    RiseTime: 1.212
    SettlingTime: 6.061
    SettlingMin: -1.209
    SettlingMax: -0.9184
    Overshoot: 20.87
    Undershoot: 28.02
    Peak: 1.209
    PeakTime: 4.242
    SteadyStateValue: -1.0
    """
    if isinstance(sysdata, (StateSpace, TransferFunction)):
        if T is None or np.asarray(T).size == 1:
            T = _default_time_vector(sysdata, N=T_num, tfinal=T, is_step=True)
        T, Yout = step_response(sysdata, T, squeeze=False)
        if yfinal:
            InfValues = np.atleast_2d(yfinal)
        else:
            InfValues = np.atleast_2d(sysdata.dcgain())
        retsiso = sysdata.issiso()
        noutputs = sysdata.noutputs
        ninputs = sysdata.ninputs
    else:
        # Time series of response data
        errmsg = ("`sys` must be a LTI system, or time response data"
                  " with a shape following the python-control"
                  " time series data convention.")
        try:
            Yout = np.array(sysdata, dtype=float)
        except ValueError:
            raise ValueError(errmsg)
        if Yout.ndim == 1 or (Yout.ndim == 2 and Yout.shape[0] == 1):
            Yout = Yout[np.newaxis, np.newaxis, :]
            retsiso = True
        elif Yout.ndim == 3:
            retsiso = False
        else:
            raise ValueError(errmsg)
        if T is None or Yout.shape[2] != len(np.squeeze(T)):
            raise ValueError("For time response data, a matching time vector"
                             " must be given")
        T = np.squeeze(T)
        noutputs = Yout.shape[0]
        ninputs = Yout.shape[1]
        InfValues = np.atleast_2d(yfinal) if yfinal else Yout[:, :, -1]

    ret = []
    for i in range(noutputs):
        retrow = []
        for j in range(ninputs):
            yout = Yout[i, j, :]

            # Steady state value
            InfValue = InfValues[i, j]
            sgnInf = np.sign(InfValue.real)

            rise_time: float = np.NaN
            settling_time: float = np.NaN
            settling_min: float = np.NaN
            settling_max: float = np.NaN
            peak_value: float = np.Inf
            peak_time: float = np.Inf
            undershoot: float = np.NaN
            overshoot: float = np.NaN
            steady_state_value: complex = np.NaN

            if not np.isnan(InfValue) and not np.isinf(InfValue):
                # RiseTime
                tr_lower_index = np.where(
                    sgnInf * (yout - RiseTimeLimits[0] * InfValue) >= 0
                    )[0][0]
                tr_upper_index = np.where(
                    sgnInf * (yout - RiseTimeLimits[1] * InfValue) >= 0
                    )[0][0]
                rise_time = T[tr_upper_index] - T[tr_lower_index]

                # SettlingTime
                settled = np.where(
                    np.abs(yout/InfValue-1) >= SettlingTimeThreshold)[0][-1]+1
                # MIMO systems can have unsettled channels without infinite
                # InfValue
                if settled < len(T):
                    settling_time = T[settled]

                settling_min = min((yout[tr_upper_index:]).min(), InfValue)
                settling_max = max((yout[tr_upper_index:]).max(), InfValue)

                # Overshoot
                y_os = (sgnInf * yout).max()
                dy_os = np.abs(y_os) - np.abs(InfValue)
                if dy_os > 0:
                    overshoot = np.abs(100. * dy_os / InfValue)
                else:
                    overshoot = 0

                # Undershoot : InfValue and undershoot must have opposite sign
                y_us_index = (sgnInf * yout).argmin()
                y_us = yout[y_us_index]
                if (sgnInf * y_us) < 0:
                    undershoot = (-100. * y_us / InfValue)
                else:
                    undershoot = 0

                # Peak
                peak_index = np.abs(yout).argmax()
                peak_value = np.abs(yout[peak_index])
                peak_time = T[peak_index]

                # SteadyStateValue
                steady_state_value = InfValue

            retij = {
                'RiseTime': rise_time,
                'SettlingTime': settling_time,
                'SettlingMin': settling_min,
                'SettlingMax': settling_max,
                'Overshoot': overshoot,
                'Undershoot': undershoot,
                'Peak': peak_value,
                'PeakTime': peak_time,
                'SteadyStateValue': steady_state_value
                }
            retrow.append(retij)

        ret.append(retrow)

    return ret[0][0] if retsiso else ret


def initial_response(sys, T=None, X0=0., input=0, output=None, T_num=None,
                     transpose=False, return_x=False, squeeze=None):
    # pylint: disable=W0622
    """Compute the initial condition response for a linear system.

    If the system has multiple outputs (MIMO), optionally, one output
    may be selected. If no selection is made for the output, all
    outputs are given.

    For information on the **shape** of parameters `T`, `X0` and
    return values `T`, `yout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sys : StateSpace or TransferFunction
        LTI system to simulate

    T :  array_like or float, optional
        Time vector, or simulation time duration if a number (time vector is
        autocomputed if not given; see  :func:`step_response` for more detail)

    X0 : array_like or float, optional
        Initial condition (default = 0).  Numbers are converted to constant
        arrays with the correct shape.

    input : int
        Ignored, has no meaning in initial condition calculation. Parameter
        ensures compatibility with step_response and impulse_response.

    output : int
        Index of the output that will be used in this simulation. Set to None
        to not trim outputs.

    T_num : int, optional
        Number of time steps to use in simulation if T is not provided as an
        array (autocomputed if not given); ignored if sys is discrete-time.

    transpose : bool, optional
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and :func:`scipy.signal.lsim`).  Default
        value is False.

    return_x : bool, optional
        If True, return the state vector when assigning to a tuple (default =
        False).  See :func:`forced_response` for more details.

    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO) then the
        output response is returned as a 1D array (indexed by time).  If
        squeeze=True, remove single-dimensional entries from the shape of the
        output even if the system is not SISO. If squeeze=False, keep the
        output as a 2D array (indexed by the output number and time) even if
        the system is SISO. The default value can be set using
        config.defaults['control.squeeze_time_response'].

    Returns
    -------
    results : TimeResponseData
        Time response represented as a :class:`TimeResponseData` object
        containing the following properties:

        * time (array): Time values of the output.

        * outputs (array): Response of the system.  If the system is SISO and
          squeeze is not True, the array is 1D (indexed by time).  If the
          system is not SISO or ``squeeze`` is False, the array is 2D (indexed
          by the output and time).

        * states (array): Time evolution of the state vector, represented as
          either a 2D array indexed by state and time (if SISO).  Not affected
          by ``squeeze``.

        The return value of the system can also be accessed by assigning the
        function to a tuple of length 2 (time, output) or of length 3 (time,
        output, state) if ``return_x`` is ``True``.

    See Also
    --------
    forced_response, impulse_response, step_response

    Notes
    -----
    This function uses the `forced_response` function with the input set to
    zero.

    Examples
    --------
    >>> G = ct.rss(4)
    >>> T, yout = ct.initial_response(G)

    """
    squeeze, sys = _get_ss_simo(sys, input, output, squeeze=squeeze)

    # Create time and input vectors; checking is done in forced_response(...)
    # The initial vector X0 is created in forced_response(...) if necessary
    if T is None or np.asarray(T).size == 1:
        T = _default_time_vector(sys, N=T_num, tfinal=T, is_step=False)

    # Compute the forced response
    response = forced_response(sys, T, 0, X0)

    # Figure out if the system is SISO or not
    issiso = sys.issiso() or (input is not None and output is not None)

    # Select only the given output, if any
    output_labels = sys.output_labels if output is None \
        else sys.output_labels[0]

    # Store the response without an input
    return TimeResponseData(
        response.t, response.y, response.x, None, issiso=issiso,
        output_labels=output_labels, input_labels=None,
        state_labels=sys.state_labels,
        transpose=transpose, return_x=return_x, squeeze=squeeze)


def impulse_response(sys, T=None, X0=0., input=None, output=None, T_num=None,
                     transpose=False, return_x=False, squeeze=None):
    # pylint: disable=W0622
    """Compute the impulse response for a linear system.

    If the system has multiple inputs and/or multiple outputs, the impulse
    response is computed for each input/output pair, with all other inputs set
    to zero.  Optionally, a single input and/or single output can be selected,
    in which case all other inputs are set to 0 and all other outputs are
    ignored.

    For information on the **shape** of parameters `T`, `X0` and
    return values `T`, `yout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sys : StateSpace, TransferFunction
        LTI system to simulate

    T : array_like or float, optional
        Time vector, or simulation time duration if a scalar (time vector is
        autocomputed if not given; see :func:`step_response` for more detail)

    X0 : array_like or float, optional
        Initial condition (default = 0)

        Numbers are converted to constant arrays with the correct shape.

    input : int, optional
        Only compute the impulse response for the listed input.  If not
        specified, the impulse responses for each independent input are
        computed.

    output : int, optional
        Only report the step response for the listed output.  If not
        specified, all outputs are reported.

    T_num : int, optional
        Number of time steps to use in simulation if T is not provided as an
        array (autocomputed if not given); ignored if sys is discrete-time.

    transpose : bool, optional
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and :func:`scipy.signal.lsim`).  Default
        value is False.

    return_x : bool, optional
        If True, return the state vector when assigning to a tuple (default =
        False).  See :func:`forced_response` for more details.

    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO) then the
        output response is returned as a 1D array (indexed by time).  If
        squeeze=True, remove single-dimensional entries from the shape of the
        output even if the system is not SISO. If squeeze=False, keep the
        output as a 2D array (indexed by the output number and time) even if
        the system is SISO. The default value can be set using
        config.defaults['control.squeeze_time_response'].

    Returns
    -------
    results : TimeResponseData
        Impulse response represented as a :class:`TimeResponseData` object
        containing the following properties:

        * time (array): Time values of the output.

        * outputs (array): Response of the system.  If the system is SISO and
          squeeze is not True, the array is 1D (indexed by time).  If the
          system is not SISO or ``squeeze`` is False, the array is 3D (indexed
          by the output, trace, and time).

        * states (array): Time evolution of the state vector, represented as
          either a 2D array indexed by state and time (if SISO) or a 3D array
          indexed by state, trace, and time.  Not affected by ``squeeze``.

        The return value of the system can also be accessed by assigning the
        function to a tuple of length 2 (time, output) or of length 3 (time,
        output, state) if ``return_x`` is ``True``.

    See Also
    --------
    forced_response, initial_response, step_response

    Notes
    -----
    This function uses the `forced_response` function to compute the time
    response. For continuous time systems, the initial condition is altered to
    account for the initial impulse. For discrete-time aystems, the impulse is
    sized so that it has unit area.

    Examples
    --------
    >>> G = ct.rss(4)
    >>> T, yout = ct.impulse_response(G)

    """
    # Convert to state space so that we can simulate
    sys = _convert_to_statespace(sys)

    # Check to make sure there is not a direct term
    if np.any(sys.D != 0) and isctime(sys):
        warnings.warn("System has direct feedthrough: ``D != 0``. The "
                      "infinite impulse at ``t=0`` does not appear in the "
                      "output.\n"
                      "Results may be meaningless!")

    # create X0 if not given, test if X0 has correct shape.
    # Must be done here because it is used for computations below.
    n_states = sys.A.shape[0]
    X0 = _check_convert_array(X0, [(n_states,), (n_states, 1)],
                              'Parameter ``X0``: \n', squeeze=True)

    # Compute T and U, no checks necessary, will be checked in forced_response
    if T is None or np.asarray(T).size == 1:
        T = _default_time_vector(sys, N=T_num, tfinal=T, is_step=False)
    U = np.zeros_like(T)

    # Set up arrays to handle the output
    ninputs = sys.ninputs if input is None else 1
    noutputs = sys.noutputs if output is None else 1
    yout = np.empty((noutputs, ninputs, np.asarray(T).size))
    xout = np.empty((sys.nstates, ninputs, np.asarray(T).size))
    uout = np.full((ninputs, ninputs, np.asarray(T).size), None)

    # Simulate the response for each input
    for i in range(sys.ninputs):
        # If input keyword was specified, only handle that case
        if isinstance(input, int) and i != input:
            continue

        # Get the system we need to simulate
        squeeze, simo = _get_ss_simo(sys, i, output, squeeze=squeeze)

        #
        # Compute new X0 that contains the impulse
        #
        # We can't put the impulse into U because there is no numerical
        # representation for it (infinitesimally short, infinitely high).
        # See also: http://www.mathworks.com/support/tech-notes/1900/1901.html
        #
        if isctime(simo):
            B = np.asarray(simo.B).squeeze()
            new_X0 = B + X0
        else:
            new_X0 = X0
            U[0] = 1./simo.dt           # unit area impulse

        # Simulate the impulse response fo this input
        response = forced_response(simo, T, U, new_X0)

        # Store the output (and states)
        inpidx = i if input is None else 0
        yout[:, inpidx, :] = response.y
        xout[:, inpidx, :] = response.x

    # Figure out if the system is SISO or not
    issiso = sys.issiso() or (input is not None and output is not None)

    # Select only the given input and output, if any
    input_labels = sys.input_labels if input is None \
        else sys.input_labels[input]
    output_labels = sys.output_labels if output is None \
        else sys.output_labels[output]

    return TimeResponseData(
        response.time, yout, xout, uout, issiso=issiso,
        output_labels=output_labels, input_labels=input_labels,
        state_labels=sys.state_labels,
        transpose=transpose, return_x=return_x, squeeze=squeeze)


# utility function to find time period and time increment using pole locations
def _ideal_tfinal_and_dt(sys, is_step=True):
    """helper function to compute ideal simulation duration tfinal and dt, the
    time increment. Usually called by _default_time_vector, whose job it is to
    choose a realistic time vector. Considers both poles and zeros.

    For discrete-time models, dt is inherent and only tfinal is computed.

    Parameters
    ----------
    sys : StateSpace or TransferFunction
        The system whose time response is to be computed
    is_step : bool
        Scales the dc value by the magnitude of the nonzero mode since
        integrating the impulse response gives
        :math:`\\int e^{-\\lambda t} = -e^{-\\lambda t}/ \\lambda`
        Default is True.

    Returns
    -------
    tfinal : float
        The final time instance for which the simulation will be performed.
    dt : float
        The estimated sampling period for the simulation.

    Notes
    -----
    Just by evaluating the fastest mode for dt and slowest for tfinal often
    leads to unnecessary, bloated sampling (e.g., Transfer(1,[1,1001,1000]))
    since dt will be very small and tfinal will be too large though the fast
    mode hardly ever contributes. Similarly, change the numerator to [1, 2, 0]
    and the simulation would be unnecessarily long and the plot is virtually
    an L shape since the decay is so fast.

    Instead, a modal decomposition in time domain hence a truncated ZIR and ZSR
    can be used such that only the modes that have significant effect on the
    time response are taken. But the sensitivity of the eigenvalues complicate
    the matter since dlambda = <w, dA*v> with <w,v> = 1. Hence we can only work
    with simple poles with this formulation. See Golub, Van Loan Section 7.2.2
    for simple eigenvalue sensitivity about the nonunity of <w,v>. The size of
    the response is dependent on the size of the eigenshapes rather than the
    eigenvalues themselves.

    By Ilhan Polat, with modifications by Sawyer Fuller to integrate into
    python-control 2020.08.17
    """

    sqrt_eps = np.sqrt(np.spacing(1.))
    default_tfinal = 5                  # Default simulation horizon
    default_dt = 0.1
    total_cycles = 5                    # Number cycles for oscillating modes
    pts_per_cycle = 25                  # Number points divide period of osc
    log_decay_percent = np.log(1000)    # Reduction factor for real pole decays

    if sys._isstatic():
        tfinal = default_tfinal
        dt = sys.dt if isdtime(sys, strict=True) else default_dt
    elif isdtime(sys, strict=True):
        dt = sys.dt
        A = _convert_to_statespace(sys).A
        tfinal = default_tfinal
        p = eigvals(A)
        # Array Masks
        # unstable
        m_u = (np.abs(p) >= 1 + sqrt_eps)
        p_u, p = p[m_u], p[~m_u]
        if p_u.size > 0:
            m_u = (p_u.real < 0) & (np.abs(p_u.imag) < sqrt_eps)
            if np.any(~m_u):
                t_emp = np.max(
                    log_decay_percent / np.abs(np.log(p_u[~m_u]) / dt))
                tfinal = max(tfinal, t_emp)

        # zero - negligible effect on tfinal
        m_z = np.abs(p) < sqrt_eps
        p = p[~m_z]
        # Negative reals- treated as oscillary mode
        m_nr = (p.real < 0) & (np.abs(p.imag) < sqrt_eps)
        p_nr, p = p[m_nr], p[~m_nr]
        if p_nr.size > 0:
            t_emp = np.max(log_decay_percent / np.abs((np.log(p_nr)/dt).real))
            tfinal = max(tfinal, t_emp)
        # discrete integrators
        m_int = (p.real - 1 < sqrt_eps) & (np.abs(p.imag) < sqrt_eps)
        p_int, p = p[m_int], p[~m_int]
        # pure oscillatory modes
        m_w = (np.abs(np.abs(p) - 1) < sqrt_eps)
        p_w, p = p[m_w], p[~m_w]
        if p_w.size > 0:
            t_emp = total_cycles * 2 * np.pi / np.abs(np.log(p_w)/dt).min()
            tfinal = max(tfinal, t_emp)

        if p.size > 0:
            t_emp = log_decay_percent / np.abs((np.log(p)/dt).real).min()
            tfinal = max(tfinal, t_emp)

        if p_int.size > 0:
            tfinal = tfinal * 5
    else:       # cont time
        sys_ss = _convert_to_statespace(sys)
        # Improve conditioning via balancing and zeroing tiny entries
        # See <w,v> for [[1,2,0], [9,1,0.01], [1,2,10*np.pi]]
        #   before/after balance
        b, (sca, perm) = matrix_balance(sys_ss.A, separate=True)
        p, l, r = eig(b, left=True, right=True)
        # Reciprocal of inner product <w,v> for each eigval, (bound the
        #   ~infs by 1e12)
        # G = Transfer([1], [1,0,1]) gives zero sensitivity (bound by 1e-12)
        eig_sens = np.reciprocal(maximum(1e-12, einsum('ij,ij->j', l, r).real))
        eig_sens = minimum(1e12, eig_sens)
        # Tolerances
        p[np.abs(p) < np.spacing(eig_sens * norm(b, 1))] = 0.
        # Incorporate balancing to outer factors
        l[perm, :] *= np.reciprocal(sca)[:, None]
        r[perm, :] *= sca[:, None]
        w, v = sys_ss.C @ r, l.T.conj() @ sys_ss.B

        origin = False
        # Computing the "size" of the response of each simple mode
        wn = np.abs(p)
        if np.any(wn == 0.):
            origin = True

        dc = np.zeros_like(p, dtype=float)
        # well-conditioned nonzero poles, np.abs just in case
        ok = np.abs(eig_sens) <= 1/sqrt_eps
        # the averaged t->inf response of each simple eigval on each i/o
        # channel. See, A = [[-1, k], [0, -2]], response sizes are
        # k-dependent (that is R/L eigenvector dependent)
        dc[ok] = norm(v[ok, :], axis=1)*norm(w[:, ok], axis=0)*eig_sens[ok]
        dc[wn != 0.] /= wn[wn != 0] if is_step else 1.
        dc[wn == 0.] = 0.
        # double the oscillating mode magnitude for the conjugate
        dc[p.imag != 0.] *= 2

        # Now get rid of noncontributing integrators and simple modes if any
        relevance = (dc > 0.1*dc.max()) | ~ok
        psub = p[relevance]
        wnsub = wn[relevance]

        tfinal, dt = [], []
        ints = wnsub == 0.
        iw = (psub.imag != 0.) & (np.abs(psub.real) <= sqrt_eps)

        # Pure imaginary?
        if np.any(iw):
            tfinal += (total_cycles * 2 * np.pi / wnsub[iw]).tolist()
            dt += (2 * np.pi / pts_per_cycle / wnsub[iw]).tolist()
        # The rest ~ts = log(%ss value) / exp(Re(eigval)t)
        texp_mode = log_decay_percent / np.abs(psub[~iw & ~ints].real)
        tfinal += texp_mode.tolist()
        dt += minimum(
            texp_mode / 50,
            (2 * np.pi / pts_per_cycle / wnsub[~iw & ~ints])
        ).tolist()

        # All integrators?
        if len(tfinal) == 0:
            return default_tfinal*5, default_dt*5

        tfinal = np.max(tfinal)*(5 if origin else 1)
        dt = np.min(dt)

    return tfinal, dt


def _default_time_vector(sys, N=None, tfinal=None, is_step=True):
    """Returns a time vector that has a reasonable number of points.
    if system is discrete-time, N is ignored """

    N_max = 5000
    N_min_ct = 100    # min points for cont time systems
    N_min_dt = 20     # more common to see just a few samples in discrete time

    ideal_tfinal, ideal_dt = _ideal_tfinal_and_dt(sys, is_step=is_step)

    if isdtime(sys, strict=True):
        # only need to use default_tfinal if not given; N is ignored.
        if tfinal is None:
            # for discrete time, change from ideal_tfinal if N too large/small
            # [N_min, N_max]
            N = int(np.clip(np.ceil(ideal_tfinal/sys.dt)+1, N_min_dt, N_max))
            tfinal = sys.dt * (N-1)
        else:
            N = int(np.ceil(tfinal/sys.dt)) + 1
            tfinal = sys.dt * (N-1)  # make tfinal integer multiple of sys.dt
    else:
        if tfinal is None:
            # for continuous time, simulate to ideal_tfinal but limit N
            tfinal = ideal_tfinal
        if N is None:
            # [N_min, N_max]
            N = int(np.clip(np.ceil(tfinal/ideal_dt)+1, N_min_ct, N_max))

    return np.linspace(0, tfinal, N, endpoint=True)
