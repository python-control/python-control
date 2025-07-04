# timeresp.py - time-domain simulation routines.
#
# Initial author: Eike Welk
# Creation date: 12 May 2011
#
# Modified: Sawyer B. Fuller (minster@uw.edu) to add discrete-time
# capability and better automatic time vector creation
# Date: June 2020
#
# Modified by Ilhan Polat to improve automatic time vector creation
# Date: August 17, 2020
#
# Modified by Richard Murray to add TimeResponseData class
# Date: August 2021
#
# Use `git shortlog -n -s statesp.py` for full list of contributors

"""Time domain simulation routines.

This module contains a collection of functions that are used to
compute time-domain simulations of LTI systems.

Arguments to time-domain simulations include a time vector, an input
vector (when needed), and an initial condition vector.  The most
general function for simulating LTI systems the
`forced_response` function, which has the form::

    t, y = forced_response(sys, T, U, X0)

where `T` is a vector of times at which the response should be
evaluated, `U` is a vector of inputs (one for each time point) and
`X0` is the initial condition for the system.

See :ref:`time-series-convention` for more information on how time
series data are represented.

"""

import warnings
from copy import copy

import numpy as np
import scipy as sp
from numpy import einsum, maximum, minimum
from scipy.linalg import eig, eigvals, matrix_balance, norm

from . import config
from . config import _process_kwargs, _process_param
from .exception import pandas_check
from .iosys import NamedSignal, isctime, isdtime
from .timeplot import time_response_plot

__all__ = ['forced_response', 'step_response', 'step_info',
           'initial_response', 'impulse_response', 'TimeResponseData',
           'TimeResponseList']

# Dictionary of aliases for time response commands
_timeresp_aliases = {
    # param:            ([alias, ...], [legacy, ...])
    'timepts':          (['T'],        []),
    'inputs':           (['U'],        ['u']),
    'outputs':          (['Y'],        ['y']),
    'initial_state':    (['X0'],       ['x0']),
    'final_output':     (['yfinal'],   []),
    'return_states':    (['return_x'], []),
    'evaluation_times': (['t_eval'],   []),
    'timepts_num':      (['T_num'],    []),
    'input_indices':    (['input'],    []),
    'output_indices':   (['output'],   []),
}


class TimeResponseData:
    """Input/output system time response data.

    This class maintains and manipulates the data corresponding to the
    temporal response of an input/output system.  It is used as the return
    type for time domain simulations (`step_response`, `input_output_response`,
    etc).

    A time response consists of a time vector, an output vector, and
    optionally an input vector and/or state vector.  Inputs and outputs can
    be 1D (scalar input/output) or 2D (vector input/output).

    A time response can be stored for multiple input signals (called traces),
    with the output and state indexed by the trace number.  This allows for
    input/output response matrices, which is mainly useful for impulse and
    step responses for linear systems.  For multi-trace responses, the same
    time vector must be used for all traces.

    Time responses are accessed through either the raw data, stored as `t`,
    `y`, `x`, `u`, or using a set of properties `time`, `outputs`,
    `states`, `inputs`.  When accessing time responses via their
    properties, squeeze processing is applied so that (by default)
    single-input, single-output systems will have the output and input
    indices suppressed.  This behavior is set using the `squeeze` parameter.

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
        Inputs used to generate the output.  This can either be a 1D array
        indexed by time (for SISO systems or MISO/MIMO systems with a
        specified input), a 2D array indexed either by input and time (for
        a multi-input system) or trace and time (for a single-input,
        multi-trace response), or a 3D array indexed by input, trace, and
        time.
    title : str, optional
        Title of the data set (used as figure title in plotting).
    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO) then
        the inputs and outputs are returned as a 1D array (indexed by time)
        and if a system is multi-input or multi-output, then the inputs are
        returned as a 2D array (indexed by input and time) and the outputs
        are returned as either a 2D array (indexed by output and time) or a
        3D array (indexed by output, trace, and time).  If `squeeze` = True,
        access to the output response will remove single-dimensional
        entries from the shape of the inputs and outputs even if the system
        is not SISO. If squeeze=False, keep the input as a 2D or 3D array
        (indexed by the input (if multi-input), trace (if single input) and
        time) and the output as a 3D array (indexed by the output, trace,
        and time) even if the system is SISO. The default value can be set
        using `config.defaults['control.squeeze_time_response']`.

    Attributes
    ----------
    t : 1D array
        Time values of the input/output response(s).  This attribute is
        normally accessed via the `time` property.
    y : 2D or 3D array
        Output response data, indexed either by output index and time (for
        single trace responses) or output, trace, and time (for multi-trace
        responses).  These data are normally accessed via the `outputs`
        property, which performs squeeze processing.
    x : 2D or 3D array, or None
        State space data, indexed either by output number and time (for
        single trace responses) or output, trace, and time (for multi-trace
        responses).  If no state data are present, value is None. These
        data are normally accessed via the `states` property, which
        performs squeeze processing.
    u : 2D or 3D array, or None
        Input signal data, indexed either by input index and time (for single
        trace responses) or input, trace, and time (for multi-trace
        responses).  If no input data are present, value is None.  These
        data are normally accessed via the `inputs` property, which
        performs squeeze processing.
    issiso : bool, optional
        Set to True if the system generating the data is single-input,
        single-output.  If passed as None (default), the input and output
        data will be used to set the value.
    ninputs, noutputs, nstates : int
        Number of inputs, outputs, and states of the underlying system.
    params : dict, optional
        If system is a nonlinear I/O system, set parameter values.
    ntraces : int, optional
        Number of independent traces represented in the input/output
        response.  If `ntraces` is 0 (default) then the data represents a
        single trace with the trace index suppressed in the data.
    trace_labels : array of string, optional
        Labels to use for traces (set to sysname it `ntraces` is 0).
    trace_types : array of string, optional
        Type of trace.  Currently only 'step' is supported, which controls
        the way in which the signal is plotted.

    Other Parameters
    ----------------
    input_labels, output_labels, state_labels : array of str, optional
        Optional labels for the inputs, outputs, and states, given as a
        list of strings matching the appropriate signal dimension.
    sysname : str, optional
        Name of the system that created the data.
    transpose : bool, optional
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and `scipy.signal.lsim`).  Default value
        is False.
    return_x : bool, optional
        If True, return the state vector when enumerating result by
        assigning to a tuple (default = False).
    plot_inputs : bool, optional
        Whether or not to plot the inputs by default (can be overridden
        in the `~TimeResponseData.plot` method).
    multi_trace : bool, optional
        If True, then 2D input array represents multiple traces.  For
        a MIMO system, the `input` attribute should then be set to
        indicate which trace is being specified.  Default is False.
    success : bool, optional
        If False, result may not be valid (see `input_output_response`).
    message : str, optional
        Informational message if `success` is False.

    See Also
    --------
    input_output_response, forced_response, impulse_response, \
    initial_response, step_response, FrequencyResponseData

    Notes
    -----
    The responses for individual elements of the time response can be
    accessed using integers, slices, or lists of signal offsets or the
    names of the appropriate signals::

      sys = ct.rss(4, 2, 1)
      resp = ct.initial_response(sys, initial_state=[1, 1, 1, 1])
      plt.plot(resp.time, resp.outputs['y[0]'])

    In the case of multi-trace data, the responses should be indexed using
    the output signal name (or offset) and the input signal name (or
    offset)::

      sys = ct.rss(4, 2, 2, strictly_proper=True)
      resp = ct.step_response(sys)
      plt.plot(resp.time, resp.outputs[['y[0]', 'y[1]'], 'u[0]'].T)

    For backward compatibility with earlier versions of python-control,
    this class has an `__iter__` method that allows it to be assigned to
    a tuple with a variable number of elements.  This allows the following
    patterns to work::

       t, y = step_response(sys)
       t, y, x = step_response(sys, return_x=True)

    Similarly, the class has `__getitem__` and `__len__` methods that
    allow the return value to be indexed:

    * response[0]: returns the time vector
    * response[1]: returns the output vector
    * response[2]: returns the state vector

    When using this (legacy) interface, the state vector is not affected
    by the `squeeze` parameter.

    The default settings for `return_x`, `squeeze` and `transpose`
    can be changed by calling the class instance and passing new values::

         response(transpose=True).input

    See `TimeResponseData.__call__` for more information.

    """
    #
    # Class attributes
    #
    # These attributes are defined as class attributes so that they are
    # documented properly.  They are "overwritten" in __init__.
    #

    #: Squeeze processing parameter.
    #:
    #: By default, if a system is single-input, single-output (SISO)
    #: then the inputs and outputs are returned as a 1D array (indexed
    #: by time) and if a system is multi-input or multi-output, then
    #: the inputs are returned as a 2D array (indexed by input and
    #: time) and the outputs are returned as either a 2D array (indexed
    #: by output and time) or a 3D array (indexed by output, trace, and
    #: time).  If squeeze=True, access to the output response will
    #: remove single-dimensional entries from the shape of the inputs
    #: and outputs even if the system is not SISO. If squeeze=False,
    #: keep the input as a 2D or 3D array (indexed by the input (if
    #: multi-input), trace (if single input) and time) and the output
    #: as a 3D array (indexed by the output, trace, and time) even if
    #: the system is SISO. The default value can be set using
    #: config.defaults['control.squeeze_time_response'].
    #:
    #: :meta hide-value:
    squeeze = None

    def __init__(
            self, time, outputs, states=None, inputs=None, issiso=None,
            output_labels=None, state_labels=None, input_labels=None,
            title=None, transpose=False, return_x=False, squeeze=None,
            multi_trace=False, trace_labels=None, trace_types=None,
            plot_inputs=True, sysname=None, params=None, success=True,
            message=None
    ):
        """Create an input/output time response object.

        This function is used by the various time response functions, such
        as `input_output_response` and `step_response` to store the
        response of a simulation.  It can be passed to `plot_time_response`
        to plot the data, or the `~TimeResponseData.plot` method can be used.

        See `TimeResponseData` for more information on parameters.

        """
        #
        # Process and store the basic input/output elements
        #

        # Time vector
        self.t = np.atleast_1d(time)
        if self.t.ndim != 1:
            raise ValueError("Time vector must be 1D array")
        self.title = title
        self.sysname = sysname
        self.params = params

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
               not multi_trace and self.x.ndim != 2:
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
            self.plot_inputs = False

        else:
            self.u = np.array(inputs)
            self.plot_inputs = plot_inputs

            # Make sure the shape is OK and figure out the number of inputs
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

        # Check and store trace labels, if present
        self.trace_labels = _process_labels(
            trace_labels, "trace", self.ntraces)
        self.trace_types = trace_types

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

        # Information on the whether the simulation result may be incorrect
        self.success = success
        self.message = message

    def __call__(self, **kwargs):
        """Change value of processing keywords.

        Calling the time response object will create a copy of the object and
        change the values of the keywords used to control the `outputs`,
        `states`, and `inputs` properties.

        Parameters
        ----------
        squeeze : bool, optional
            If `squeeze` = True, access to the output response will remove
            single-dimensional entries from the shape of the inputs,
            outputs, and states even if the system is not SISO. If
            `squeeze` = False, keep the input as a 2D or 3D array (indexed
            by the input (if multi-input), trace (if single input) and
            time) and the output and states as a 3D array (indexed by the
            output/state, trace, and time) even if the system is SISO.

        transpose : bool, optional
            If True, transpose all input and output arrays (for backward
            compatibility with MATLAB and `scipy.signal.lsim`).
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
        (for multiple traces).  See `TimeResponseData.squeeze` for a
        description of how this can be modified using the `squeeze` keyword.

        Input and output signal names can be used to index the data in
        place of integer offsets, with the input signal names being used to
        access multi-input data.

        :type: 1D, 2D, or 3D array

        """
        # TODO: move to __init__ to avoid recomputing each time?
        y = _process_time_response(
            self.y, issiso=self.issiso,
            transpose=self.transpose, squeeze=self.squeeze)
        return NamedSignal(y, self.output_labels, self.input_labels)

    # Getter for states (implements squeeze processing)
    @property
    def states(self):
        """Time response state vector.

        Time evolution of the state vector, indexed by either the state and
        time (if only a single trace is given) or the state, trace, and
        time (for multiple traces).  See `TimeResponseData.squeeze` for a
        description of how this can be modified using the `squeeze`
        keyword.

        Input and output signal names can be used to index the data in
        place of integer offsets, with the input signal names being used to
        access multi-input data.

        :type: 2D or 3D array

        """
        # TODO: move to __init__ to avoid recomputing each time?
        x = _process_time_response(
            self.x, transpose=self.transpose,
            squeeze=self.squeeze, issiso=False)

        # Special processing for SISO case: always retain state index
        if self.issiso and self.ntraces == 1 and x.ndim == 3 and \
             self.squeeze is not False:
            # Single-input, single-output system with single trace
            x = x[:, 0, :]

        return NamedSignal(x, self.state_labels, self.input_labels)

    # Getter for inputs (implements squeeze processing)
    @property
    def inputs(self):
        """Time response input vector.

        Input(s) to the system, indexed by input (optional), trace (optional),
        and time.  If a 1D vector is passed, the input corresponds to a
        scalar-valued input.  If a 2D vector is passed, then it can either
        represent multiple single-input traces or a single multi-input trace.
        The optional `multi_trace` keyword should be used to disambiguate
        the two.  If a 3D vector is passed, then it represents a multi-trace,
        multi-input signal, indexed by input, trace, and time.

        Input and output signal names can be used to index the data in
        place of integer offsets, with the input signal names being used to
        access multi-input data.

        See `TimeResponseData.squeeze` for a description of how the
        dimensions of the input vector can be modified using the `squeeze`
        keyword.

        :type: 1D or 2D array

        """
        # TODO: move to __init__ to avoid recomputing each time?
        if self.u is None:
            return None

        u = _process_time_response(
            self.u, issiso=self.issiso,
            transpose=self.transpose, squeeze=self.squeeze)
        return NamedSignal(u, self.input_labels, self.input_labels)

    # Getter for legacy state (implements non-standard squeeze processing)
    # TODO: remove when no longer needed
    @property
    def _legacy_states(self):
        """Time response state vector (legacy version).

        Time evolution of the state vector, indexed by either the state and
        time (if only a single trace is given) or the state, trace, and
        time (for multiple traces).

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
        """Convert response data to pandas data frame.

        Creates a pandas data frame using the input, output, and state labels
        for the time response.  The column labels are given by the input and
        output (and state, when present) labels, with time labeled by 'time'
        and traces (for multi-trace responses) labeled by 'trace'.

        """
        if not pandas_check():
            raise ImportError("pandas not installed")
        import pandas

        # Create a dict for setting up the data frame
        data = {'time': np.tile(
            self.time, self.ntraces if self.ntraces > 0 else 1)}
        if self.ntraces > 0:
            data['trace'] = np.hstack([
                np.full(self.time.size, label) for label in self.trace_labels])
        if self.ninputs > 0:
            data.update(
                {name: self.u[i].reshape(-1)
                 for i, name in enumerate(self.input_labels)})
        if self.noutputs > 0:
            data.update(
                {name: self.y[i].reshape(-1)
                 for i, name in enumerate(self.output_labels)})
        if self.nstates > 0:
            data.update(
                {name: self.x[i].reshape(-1)
                 for i, name in enumerate(self.state_labels)})

        return pandas.DataFrame(data)

    # Plot data
    def plot(self, *args, **kwargs):
        """Plot the time response data objects.

        This method calls `time_response_plot`, passing all arguments
        and keywords.  See `time_response_plot` for details.

        """
        return time_response_plot(self, *args, **kwargs)


#
# Time response data list class
#
# This class is a subclass of list that adds a plot() method, enabling
# direct plotting from routines returning a list of TimeResponseData
# objects.
#

class TimeResponseList(list):
    """List of TimeResponseData objects with plotting capability.

    This class consists of a list of `TimeResponseData` objects.
    It is a subclass of the Python `list` class, with a `plot` method that
    plots the individual `TimeResponseData` objects.

    """
    def plot(self, *args, **kwargs):
        """Plot a list of time responses.

        See `time_response_plot` for details.

        """
        from .ctrlplot import ControlPlot

        lines = None
        label = kwargs.pop('label', [None] * len(self))
        for i, response in enumerate(self):
            cplt = TimeResponseData.plot(
                response, *args, label=label[i], **kwargs)
            if lines is None:
                lines = cplt.lines
            else:
                # Append the lines in the new plot to previous lines
                for row in range(cplt.lines.shape[0]):
                    for col in range(cplt.lines.shape[1]):
                        lines[row, col] += cplt.lines[row, col]
        return ControlPlot(lines, cplt.axes, cplt.figure)


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


# Helper function for checking array_like parameters
def _check_convert_array(in_obj, legal_shapes, err_msg_start, squeeze=False,
                         transpose=False):

    """Helper function for checking array_like parameters.

    * Check type and shape of `in_obj`.
    * Convert `in_obj` to an array if necessary.
    * Change shape of `in_obj` according to parameter `squeeze`.
    * If `in_obj` is a scalar (number) it is converted to an array with
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

        * (2, 3) describes an array with 2 rows and 3 columns
        * (2, 'any') describes an array with 2 rows and any number of
          columns

    err_msg_start : str
        String that is prepended to the error messages, when this function
        raises an exception. It should be used to identify the argument which
        is currently checked.

    squeeze : bool
        If True, all dimensions with only one element are removed from the
        array. If False the array's shape is unmodified.

        For example: ``array([[1, 2, 3]])`` is converted to ``array([1, 2,
        3])``.

    transpose : bool, optional
        If True, assume that 2D input arrays are transposed from the
        standard format.  Used to convert MATLAB-style inputs to our
        format.

    Returns
    -------
    out_array : array
        The checked and converted contents of `in_obj`.

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
def forced_response(
        sysdata, timepts=None, inputs=0., initial_state=0., transpose=False,
        params=None, interpolate=False, return_states=None, squeeze=None,
        **kwargs):
    """Compute the output of a linear system given the input.

    As a convenience for parameters `U`, `X0`: Numbers (scalars) are
    converted to constant arrays with the correct shape.  The correct shape
    is inferred from arguments `sys` and `T`.

    For information on the **shape** of parameters `U`, `T`, `X0` and
    return values `T`, `yout`, `xout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sysdata : I/O system or list of I/O systems
        I/O system(s) for which forced response is computed.
    timepts (or T) : array_like, optional for discrete LTI `sys`
        Time steps at which the input is defined; values must be evenly
        spaced.  If None, `inputs` must be given and ``len(inputs)`` time
        steps of `sys.dt` are simulated. If `sys.dt` is None or True
        (undetermined time step), a time step of 1.0 is assumed.
    inputs (or U) : array_like or float, optional
        Input array giving input at each time in `timepts`.  If `inputs` is
        None or 0, `timepts` must be given, even for discrete-time
        systems. In this case, for continuous-time systems, a direct
        calculation of the matrix exponential is used, which is faster than
        the general interpolating algorithm used otherwise.
    initial_state (or X0) : array_like or float, default=0.
        Initial condition.
    params : dict, optional
        If system is a nonlinear I/O system, set parameter values.
    transpose : bool, default=False
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and `scipy.signal.lsim`).
    interpolate : bool, default=False
        If True and system is a discrete-time system, the input will
        be interpolated between the given time steps and the output
        will be given at system sampling rate.  Otherwise, only return
        the output at the times given in `T`.  No effect on continuous
        time simulations.
    return_states (or return_x) : bool, default=None
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
        the output number and time) even if the system is SISO. The default
        behavior can be overridden by
        `config.defaults['control.squeeze_time_response']`.

    Returns
    -------
    resp : `TimeResponseData` or `TimeResponseList`
        Input/output response data object.  When accessed as a tuple,
        returns ``(time, outputs)`` (default) or ``(time, outputs, states)``
        if `return_x` is True.  The `~TimeResponseData.plot` method can
        be used to create a plot of the time response(s) (see
        `time_response_plot` for more information).  If `sysdata` is a list
        of systems, a `TimeResponseList` object is returned, which acts as
        a list of `TimeResponseData` objects with a `~TimeResponseList.plot`
        method that will plot responses as multiple traces.  See
        `time_response_plot` for additional information.
    resp.time : array
        Time values of the output.
    resp.outputs : array
        Response of the system.  If the system is SISO and `squeeze` is not
        True, the array is 1D (indexed by time).  If the system is not SISO or
        `squeeze` is False, the array is 2D (indexed by output and time).
    resp.states : array
        Time evolution of the state vector, represented as a 2D array
        indexed by state and time.
    resp.inputs : array
        Input(s) to the system, indexed by input and time.

    See Also
    --------
    impulse_response, initial_response, input_output_response, \
    step_response, time_response_plot

    Notes
    -----
    For discrete-time systems, the input/output response is computed
    using the `scipy.signal.dlsim` function.

    For continuous-time systems, the output is computed using the
    matrix exponential exp(A t) and assuming linear interpolation
    of the inputs between time points.

    If a nonlinear I/O system is passed to `forced_response`, the
    `input_output_response` function is called instead.  The main
    difference between `input_output_response` and `forced_response`
    is that `forced_response` is specialized (and optimized) for
    linear systems.

    (legacy) The return value of the system can also be accessed by
    assigning the function to a tuple of length 2 (time, output) or of
    length 3 (time, output, state) if `return_x` is True.

    Examples
    --------
    >>> G = ct.rss(4)
    >>> timepts = np.linspace(0, 10)
    >>> inputs = np.sin(timepts)
    >>> tout, yout = ct.forced_response(G, timepts, inputs)

    See :ref:`time-series-convention` and
    :ref:`package-configuration-parameters`.

    """
    from .nlsys import NonlinearIOSystem, input_output_response
    from .statesp import StateSpace, _convert_to_statespace
    from .xferfcn import TransferFunction

    # Process keyword arguments
    _process_kwargs(kwargs, _timeresp_aliases)
    T = _process_param('timepts', timepts, kwargs, _timeresp_aliases)
    U = _process_param('inputs', inputs, kwargs, _timeresp_aliases, sigval=0.)
    X0 = _process_param(
        'initial_state', initial_state, kwargs, _timeresp_aliases, sigval=0.)
    return_x = _process_param(
        'return_states', return_states, kwargs, _timeresp_aliases, sigval=None)

    if kwargs:
        raise TypeError("unrecognized keyword(s): ", str(kwargs))

    # If passed a list, recursively call individual responses with given T
    if isinstance(sysdata, (list, tuple)):
        responses = []
        for sys in sysdata:
            responses.append(forced_response(
                sys, T, inputs=U, initial_state=X0, transpose=transpose,
                params=params, interpolate=interpolate,
                return_states=return_x, squeeze=squeeze))
        return TimeResponseList(responses)
    else:
        sys = sysdata

    if not isinstance(sys, (StateSpace, TransferFunction)):
        if isinstance(sys, NonlinearIOSystem):
            if interpolate:
                warnings.warn(
                    "interpolation not supported for nonlinear I/O systems")
            return input_output_response(
                sys, T, U, X0, params=params, transpose=transpose,
                return_x=return_x, squeeze=squeeze)
        else:
            raise TypeError('Parameter `sys`: must be a `StateSpace` or'
                            ' `TransferFunction`)')

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
        # T must be array_like
        T = np.asarray(T)

    # Set and/or check time vector in discrete-time case
    if isdtime(sys):
        if T is None:
            if U is None or (U.ndim == 0 and U == 0.):
                raise ValueError('Parameters `T` and `U` can\'t both be '
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
            raise ValueError('Parameter `T` is mandatory for continuous '
                             'time systems.')

    # Test if T has shape (n,) or (1, n);
    T = _check_convert_array(T, [('any',), (1, 'any')],
                             'Parameter `T`: ', squeeze=True,
                             transpose=transpose)

    n_steps = T.shape[0]            # number of simulation steps

    # equally spaced also implies strictly monotonic increase,
    dt = (T[-1] - T[0]) / (n_steps - 1)
    if not np.allclose(np.diff(T), dt):
        raise ValueError("Parameter `T`: time values must be equally "
                         "spaced.")

    # create X0 if not given, test if X0 has correct shape
    X0 = _check_convert_array(X0, [(n_states,), (n_states, 1)],
                              'Parameter `X0`: ', squeeze=True)

    # Test if U has correct shape and type
    legal_shapes = [(n_steps,), (1, n_steps)] if n_inputs == 1 else \
        [(n_inputs, n_steps)]
    U = _check_convert_array(U, legal_shapes,
                             'Parameter `U`: ', squeeze=False,
                             transpose=transpose)

    xout = np.zeros((n_states, n_steps))
    xout[:, 0] = X0
    yout = np.zeros((n_outputs, n_steps))

    # Separate out the discrete and continuous-time cases
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
                raise ValueError("Time steps `T` must match sampling time")

            # Now check to make sure it is a multiple (with check against
            # sys.dt because floating point mod can have small errors
            if not (np.isclose(dt % sys.dt, 0) or
                    np.isclose(dt % sys.dt, sys.dt)):
                raise ValueError("Time steps `T` must be multiples of "
                                 "sampling time")
            sys_dt = sys.dt

            # sp.signal.dlsim returns not enough samples if
            # T[-1] - T[0] < sys_dt * decimation * (n_steps - 1)
            # due to rounding errors.
            # https://github.com/scipyscipy/blob/v1.6.1/scipy/signal/ltisys.py#L3462
            scipy_out_samples = int(np.floor(spT[-1] / sys_dt)) + 1
            if scipy_out_samples < n_steps:
                # parentheses: order of evaluation is important
                spT[-1] = spT[-1] * (n_steps / (spT[-1] / sys_dt + 1))

        else:
            sys_dt = dt         # For unspecified sampling time, use time incr

        # Discrete time simulation using signal processing toolbox
        dsys = (A, B, C, D, sys_dt)

        # Use signal processing toolbox for the discrete-time simulation
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
        tout, yout, xout, U, params=params, issiso=sys.issiso(),
        output_labels=sys.output_labels, input_labels=sys.input_labels,
        state_labels=sys.state_labels, sysname=sys.name, plot_inputs=True,
        title="Forced response for " + sys.name, trace_types=['forced'],
        transpose=transpose, return_x=return_x, squeeze=squeeze)


# Process time responses in a uniform way
def _process_time_response(
        signal, issiso=False, transpose=None, squeeze=None):
    """Process time response signals.

    This function processes the outputs (or inputs) of time response
    functions and processes the transpose and squeeze keywords.

    Parameters
    ----------
    signal : ndarray
        Data to be processed.  This can either be a 1D array indexed by
        time (for SISO systems), a 2D array indexed by output and time (for
        MIMO systems with no input indexing, such as initial_response or
        forced response) or a 3D array indexed by output, input, and time.

    issiso : bool, optional
        If True, process data as single-input, single-output data.
        Default is False.

    transpose : bool, optional
        If True, transpose data (for backward compatibility with MATLAB and
        `scipy.signal.lsim`).  Default value is False.

    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO) then
        the signals are returned as a 1D array (indexed by time).  If
        `squeeze` = True, remove single-dimensional entries from the shape
        of the signal even if the system is not SISO. If `squeeze` = False,
        keep the signal as a 3D array (indexed by the output, input, and
        time) even if the system is SISO. The default value can be set
        using `config.defaults['control.squeeze_time_response']`.

    Returns
    -------
    output : ndarray
        Processed signal.  If the system is SISO and squeeze is not True,
        the array is 1D (indexed by time).  If the system is not SISO or
        squeeze is False, the array is either 2D (indexed by output and
        time) or 3D (indexed by input, output, and time).

    """
    # If squeeze was not specified, figure out the default (might remain None)
    if squeeze is None:
        squeeze = config.defaults['control.squeeze_time_response']

    # Figure out whether and how to squeeze output data
    if squeeze is True:                 # squeeze all dimensions
        signal = np.squeeze(signal)
    elif squeeze is False:              # squeeze no dimensions
        pass
    elif squeeze is None:               # squeeze signals if SISO
        if issiso:
            if signal.ndim == 3:
                signal = signal[0][0]   # remove input and output
            else:
                signal = signal[0]      # remove input
    else:
        raise ValueError("Unknown squeeze value")

    # See if we need to transpose the data back into MATLAB form
    if transpose:
        # For signals, put the last index (time) into the first slot
        signal = np.transpose(signal, np.roll(range(signal.ndim), 1))

    # Return output
    return signal


def step_response(
        sysdata, timepts=None, initial_state=0., input_indices=None,
        output_indices=None, timepts_num=None, transpose=False,
        return_states=False, squeeze=None, params=None, **kwargs):
    # pylint: disable=W0622
    """Compute the step response for a linear system.

    If the system has multiple inputs and/or multiple outputs, the step
    response is computed for each input/output pair, with all other inputs
    set to zero.  Optionally, a single input and/or single output can be
    selected, in which case all other inputs are set to 0 and all other
    outputs are ignored.

    For information on the **shape** of parameters `T`, `X0` and
    return values `T`, `yout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sysdata : I/O system or list of I/O systems
        I/O system(s) for which step response is computed.
    timepts (or T) : array_like or float, optional
        Time vector, or simulation time duration if a number. If `T` is not
        provided, an attempt is made to create it automatically from the
        dynamics of the system. If the system continuous time, the time
        increment dt is chosen small enough to show the fastest mode, and
        the simulation time period tfinal long enough to show the slowest
        mode, excluding poles at the origin and pole-zero cancellations. If
        this results in too many time steps (>5000), dt is reduced. If the
        system is discrete time, only tfinal is computed, and final is
        reduced if it requires too many simulation steps.
    initial_state (or X0) : array_like or float, optional
        Initial condition (default = 0).  This can be used for a nonlinear
        system where the origin is not an equilibrium point.
    input_indices (or input) : int or list of int, optional
        Only compute the step response for the listed input.  If not
        specified, the step responses for each independent input are
        computed (as separate traces).
    output_indices (or output) : int, optional
        Only report the step response for the listed output.  If not
        specified, all outputs are reported.
    params : dict, optional
        If system is a nonlinear I/O system, set parameter values.
    timepts_num (or T_num) : int, optional
        Number of time steps to use in simulation if `T` is not provided as
        an array (auto-computed if not given); ignored if the system is
        discrete time.
    transpose : bool, optional
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and `scipy.signal.lsim`).  Default
        value is False.
    return_states (or return_x) : bool, optional
        If True, return the state vector when assigning to a tuple
        (default = False).  See `forced_response` for more details.
    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO) then
        the output response is returned as a 1D array (indexed by time).
        If `squeeze` = True, remove single-dimensional entries from the
        shape of the output even if the system is not SISO. If
        `squeeze` = False, keep the output as a 3D array (indexed by the
        output, input, and time) even if the system is SISO. The default
        value can be set using
        `config.defaults['control.squeeze_time_response']`.

    Returns
    -------
    results : `TimeResponseData` or `TimeResponseList`
        Time response represented as a `TimeResponseData` object or
        list of `TimeResponseData` objects.  See
        `forced_response` for additional information.

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
    from .lti import LTI
    from .statesp import _convert_to_statespace
    from .xferfcn import TransferFunction

    # Process keyword arguments
    _process_kwargs(kwargs, _timeresp_aliases)
    T = _process_param('timepts', timepts, kwargs, _timeresp_aliases)
    X0 = _process_param(
        'initial_state', initial_state, kwargs, _timeresp_aliases, sigval=0.)
    input = _process_param(
        'input_indices', input_indices, kwargs, _timeresp_aliases)
    output = _process_param(
        'output_indices', output_indices, kwargs, _timeresp_aliases)
    return_x = _process_param(
        'return_states', return_states, kwargs, _timeresp_aliases,
        sigval=False)
    T_num = _process_param(
        'timepts_num', timepts_num, kwargs, _timeresp_aliases)

    if kwargs:
        raise TypeError("unrecognized keyword(s): ", str(kwargs))

    # Create the time and input vectors
    if T is None or np.asarray(T).size == 1:
        T = _default_time_vector(sysdata, N=T_num, tfinal=T, is_step=True)
    T = np.atleast_1d(T).reshape(-1)
    if T.ndim != 1 and len(T) < 2:
        raise ValueError("invalid value of T for this type of system")

    # If passed a list, recursively call individual responses with given T
    if isinstance(sysdata, (list, tuple)):
        responses = []
        for sys in sysdata:
            responses.append(step_response(
                sys, T, initial_state=X0, input_indices=input,
                output_indices=output, timepts_num=T_num,
                transpose=transpose, return_states=return_x, squeeze=squeeze,
                params=params))
        return TimeResponseList(responses)
    else:
        sys = sysdata

    # If we are passed a transfer function and X0 is non-zero, warn the user
    if isinstance(sys, TransferFunction) and np.any(X0 != 0):
        warnings.warn(
            "Non-zero initial condition given for transfer function system. "
            "Internal conversion to state space used; may not be consistent "
            "with given X0.")

    # Convert to state space so that we can simulate
    if isinstance(sys, LTI) and sys.nstates is None:
        sys = _convert_to_statespace(sys)

    # Only single input and output are allowed for now
    if isinstance(input, (list, tuple)):
        if len(input_indices) > 1:
            raise NotImplementedError("list of input indices not allowed")
        input = input[0]
    elif isinstance(input, str):
        raise NotImplementedError("named inputs not allowed")

    if isinstance(output, (list, tuple)):
        if len(output_indices) > 1:
            raise NotImplementedError("list of output indices not allowed")
        output = output[0]
    elif isinstance(output, str):
        raise NotImplementedError("named outputs not allowed")

    # Set up arrays to handle the output
    ninputs = sys.ninputs if input is None else 1
    noutputs = sys.noutputs if output is None else 1
    yout = np.empty((noutputs, ninputs, T.size))
    xout = np.empty((sys.nstates, ninputs, T.size))
    uout = np.empty((ninputs, ninputs, T.size))

    # Simulate the response for each input
    trace_labels, trace_types = [], []
    for i in range(sys.ninputs):
        # If input keyword was specified, only simulate for that input
        if isinstance(input, int) and i != input:
            continue

        # Save a label and type for this plot
        trace_labels.append(f"From {sys.input_labels[i]}")
        trace_types.append('step')

        # Create a set of single inputs system for simulation
        U = np.zeros((sys.ninputs, T.size))
        U[i, :] = np.ones_like(T)

        response = forced_response(sys, T, U, X0, squeeze=True, params=params)
        inpidx = i if input is None else 0
        yout[:, inpidx, :] = response.y if output is None \
            else response.y[output]
        xout[:, inpidx, :] = response.x
        uout[:, inpidx, :] = U if input is None else U[i]

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
        state_labels=sys.state_labels, title="Step response for " + sys.name,
        transpose=transpose, return_x=return_x, squeeze=squeeze,
        sysname=sys.name, params=params, trace_labels=trace_labels,
        trace_types=trace_types, plot_inputs=False)


def step_info(
        sysdata, timepts=None, timepts_num=None, final_output=None,
        params=None, SettlingTimeThreshold=0.02, RiseTimeLimits=(0.1, 0.9),
        **kwargs):
    """Step response characteristics (rise time, settling time, etc).

    Parameters
    ----------
    sysdata : `StateSpace` or `TransferFunction` or array_like
        The system data. Either LTI system to simulate (`StateSpace`,
        `TransferFunction`), or a time series of step response data.
    timepts (or T) : array_like or float, optional
        Time vector, or simulation time duration if a number (time vector is
        auto-computed if not given, see `step_response` for more detail).
        Required, if sysdata is a time series of response data.
    timepts_num (or T_num) : int, optional
        Number of time steps to use in simulation if `T` is not provided as
        an array; auto-computed if not given; ignored if sysdata is a
        discrete-time system or a time series or response data.
    final_output (or yfinal) : scalar or array_like, optional
        Steady-state response. If not given, sysdata.dcgain() is used for
        systems to simulate and the last value of the the response data is
        used for a given time series of response data. Scalar for SISO,
        (noutputs, ninputs) array_like for MIMO systems.
    params : dict, optional
        If system is a nonlinear I/O system, set parameter values.
    SettlingTimeThreshold : float, optional
        Defines the error to compute settling time (default = 0.02).
    RiseTimeLimits : tuple (lower_threshold, upper_threshold)
        Defines the lower and upper threshold for RiseTime computation.

    Returns
    -------
    S : dict or list of list of dict
        If `sysdata` corresponds to a SISO system, `S` is a dictionary
        containing:

            - 'RiseTime': Time from 10% to 90% of the steady-state value.
            - 'SettlingTime': Time to enter inside a default error of 2%.
            - 'SettlingMin': Minimum value after `RiseTime`.
            - 'SettlingMax': Maximum value after `RiseTime`.
            - 'Overshoot': Percentage of the peak relative to steady value.
            - 'Undershoot': Percentage of undershoot.
            - 'Peak': Absolute peak value.
            - 'PeakTime': Time that the first peak value is obtained.
            - 'SteadyStateValue': Steady-state value.

        If `sysdata` corresponds to a MIMO system, `S` is a 2D list of dicts.
        To get the step response characteristics from the jth input to the
        ith output, access ``S[i][j]``.

    See Also
    --------
    step_response, forced_response, initial_response, impulse_response

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
    from .nlsys import NonlinearIOSystem
    from .statesp import StateSpace
    from .xferfcn import TransferFunction

    # Process keyword arguments
    _process_kwargs(kwargs, _timeresp_aliases)
    T = _process_param('timepts', timepts, kwargs, _timeresp_aliases)
    T_num = _process_param(
        'timepts_num', timepts_num, kwargs, _timeresp_aliases)
    yfinal = _process_param(
        'final_output', final_output, kwargs, _timeresp_aliases)

    if kwargs:
        raise TypeError("unrecognized keyword(s): ", str(kwargs))

    if isinstance(sysdata, (StateSpace, TransferFunction, NonlinearIOSystem)):
        T, Yout = step_response(
            sysdata, T, timepts_num=T_num, squeeze=False, params=params)
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

            rise_time: float = np.nan
            settling_time: float = np.nan
            settling_min: float = np.nan
            settling_max: float = np.nan
            peak_value: float = np.inf
            peak_time: float = np.inf
            undershoot: float = np.nan
            overshoot: float = np.nan
            steady_state_value: complex = np.nan

            if not np.isnan(InfValue) and not np.isinf(InfValue):
                # RiseTime
                tr_lower_index = np.nonzero(
                    sgnInf * (yout - RiseTimeLimits[0] * InfValue) >= 0
                    )[0][0]
                tr_upper_index = np.nonzero(
                    sgnInf * (yout - RiseTimeLimits[1] * InfValue) >= 0
                    )[0][0]
                rise_time = T[tr_upper_index] - T[tr_lower_index]

                # SettlingTime
                outside_threshold = np.nonzero(
                    np.abs(yout/InfValue - 1) >= SettlingTimeThreshold)[0]
                settled = 0 if outside_threshold.size == 0 \
                    else outside_threshold[-1] + 1
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
                'RiseTime': float(rise_time),
                'SettlingTime': float(settling_time),
                'SettlingMin': float(settling_min),
                'SettlingMax': float(settling_max),
                'Overshoot': float(overshoot),
                'Undershoot': float(undershoot),
                'Peak': float(peak_value),
                'PeakTime': float(peak_time),
                'SteadyStateValue': float(steady_state_value)
                }
            retrow.append(retij)

        ret.append(retrow)

    return ret[0][0] if retsiso else ret


def initial_response(
        sysdata, timepts=None, initial_state=0, output_indices=None,
        timepts_num=None, params=None, transpose=False, return_states=False,
        squeeze=None, **kwargs):
    # pylint: disable=W0622
    """Compute the initial condition response for a linear system.

    If the system has multiple outputs (MIMO), optionally, one output
    may be selected. If no selection is made for the output, all
    outputs are given.

    For information on the **shape** of parameters `T`, `X0` and
    return values `T`, `yout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sysdata : I/O system or list of I/O systems
        I/O system(s) for which initial response is computed.
    timepts (or T) :  array_like or float, optional
        Time vector, or simulation time duration if a number (time vector is
        auto-computed if not given; see  `step_response` for more detail).
    initial_state (or X0) : array_like or float, optional
        Initial condition (default = 0).  Numbers are converted to constant
        arrays with the correct shape.
    output_indices (or output) : int
        Index of the output that will be used in this simulation. Set
        to None to not trim outputs.
    timepts_num (or T_num) : int, optional
        Number of time steps to use in simulation if `timepts` is not
        provided as an array (auto-computed if not given); ignored if the
        system is discrete time.
    params : dict, optional
        If system is a nonlinear I/O system, set parameter values.
    transpose : bool, optional
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and `scipy.signal.lsim`).  Default
        value is False.
    return_states (or return_x) : bool, optional
        If True, return the state vector when assigning to a tuple
        (default = False).  See `forced_response` for more details.
    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO) then
        the output response is returned as a 1D array (indexed by time).
        If `squeeze` = True, remove single-dimensional entries from the
        shape of the output even if the system is not SISO. If
        `squeeze` = False, keep the output as a 2D array (indexed by the
        output number and time) even if the system is SISO. The default
        value can be set using
        `config.defaults['control.squeeze_time_response']`.

    Returns
    -------
    results : `TimeResponseData` or `TimeResponseList`
        Time response represented as a `TimeResponseData` object or
        list of `TimeResponseData` objects.  See
        `forced_response` for additional information.

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
    # Process keyword arguments
    _process_kwargs(kwargs, _timeresp_aliases)
    T = _process_param('timepts', timepts, kwargs, _timeresp_aliases)
    X0 = _process_param(
        'initial_state', initial_state, kwargs, _timeresp_aliases, sigval=0.)
    output = _process_param(
        'output_indices', output_indices, kwargs, _timeresp_aliases)
    return_x = _process_param(
        'return_states', return_states, kwargs, _timeresp_aliases,
        sigval=False)
    T_num = _process_param(
        'timepts_num', timepts_num, kwargs, _timeresp_aliases)

    if kwargs:
        raise TypeError("unrecognized keyword(s): ", str(kwargs))

    # Create the time and input vectors
    if T is None or np.asarray(T).size == 1:
        T = _default_time_vector(sysdata, N=T_num, tfinal=T, is_step=False)
    T = np.atleast_1d(T).reshape(-1)
    if T.ndim != 1 and len(T) < 2:
        raise ValueError("invalid value of T for this type of system")

    # If passed a list, recursively call individual responses with given T
    if isinstance(sysdata, (list, tuple)):
        responses = []
        for sys in sysdata:
            responses.append(initial_response(
                sys, T, initial_state=X0, output_indices=output,
                timepts_num=T_num, transpose=transpose,
                return_states=return_x, squeeze=squeeze, params=params))
        return TimeResponseList(responses)
    else:
        sys = sysdata

    # Compute the forced response
    response = forced_response(sys, T, 0, X0, params=params)

    # Figure out if the system is SISO or not
    issiso = sys.issiso() or output is not None

    # Select only the given output, if any
    yout = response.y if output is None else response.y[output]
    output_labels = sys.output_labels if output is None \
        else sys.output_labels[output]

    # Store the response without an input
    return TimeResponseData(
        response.t, yout, response.x, None, params=params, issiso=issiso,
        output_labels=output_labels, input_labels=None,
        state_labels=sys.state_labels, sysname=sys.name,
        title="Initial response for " + sys.name, trace_types=['initial'],
        transpose=transpose, return_x=return_x, squeeze=squeeze)


def impulse_response(
        sysdata, timepts=None, input_indices=None, output_indices=None,
        timepts_num=None, transpose=False, return_states=False, squeeze=None,
        **kwargs):
    # pylint: disable=W0622
    """Compute the impulse response for a linear system.

    If the system has multiple inputs and/or multiple outputs, the impulse
    response is computed for each input/output pair, with all other inputs
    set to zero.  Optionally, a single input and/or single output can be
    selected, in which case all other inputs are set to 0 and all other
    outputs are ignored.

    For information on the **shape** of parameters `T`, `X0` and
    return values `T`, `yout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sysdata : I/O system or list of I/O systems
        I/O system(s) for which impulse response is computed.
    timepts (or T) : array_like or float, optional
        Time vector, or simulation time duration if a scalar (time vector is
        auto-computed if not given; see `step_response` for more detail).
    input_indices (or input) : int, optional
        Only compute the impulse response for the listed input.  If not
        specified, the impulse responses for each independent input are
        computed.
    output_indices (or output) : int, optional
        Only report the step response for the listed output.  If not
        specified, all outputs are reported.
    timepts_num (or T_num) : int, optional
        Number of time steps to use in simulation if `T` is not provided as
        an array (auto-computed if not given); ignored if the system is
        discrete time.
    transpose : bool, optional
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and `scipy.signal.lsim`).  Default
        value is False.
    return_states (or return_x) : bool, optional
        If True, return the state vector when assigning to a tuple
        (default = False).  See `forced_response` for more details.
    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO) then
        the output response is returned as a 1D array (indexed by time).
        If `squeeze` = True, remove single-dimensional entries from the
        shape of the output even if the system is not SISO. If
        `squeeze` = False, keep the output as a 2D array (indexed by the
        output number and time) even if the system is SISO. The default
        value can be set using
        `config.defaults['control.squeeze_time_response']`.

    Returns
    -------
    results : `TimeResponseData` or `TimeResponseList`
        Time response represented as a `TimeResponseData` object or
        list of `TimeResponseData` objects.  See
        `forced_response` for additional information.

    See Also
    --------
    forced_response, initial_response, step_response

    Notes
    -----
    This function uses the `forced_response` function to compute the time
    response. For continuous-time systems, the initial condition is altered
    to account for the initial impulse. For discrete-time systems, the
    impulse is sized so that it has unit area.  The impulse response for
    nonlinear systems is not implemented.

    Examples
    --------
    >>> G = ct.rss(4)
    >>> T, yout = ct.impulse_response(G)

    """
    from .lti import LTI
    from .statesp import _convert_to_statespace

    # Process keyword arguments
    _process_kwargs(kwargs, _timeresp_aliases)
    T = _process_param('timepts', timepts, kwargs, _timeresp_aliases)
    input = _process_param(
        'input_indices', input_indices, kwargs, _timeresp_aliases)
    output = _process_param(
        'output_indices', output_indices, kwargs, _timeresp_aliases)
    return_x = _process_param(
        'return_states', return_states, kwargs, _timeresp_aliases,
        sigval=False)
    T_num = _process_param(
        'timepts_num', timepts_num, kwargs, _timeresp_aliases)

    if kwargs:
        raise TypeError("unrecognized keyword(s): ", str(kwargs))

    # Create the time and input vectors
    if T is None or np.asarray(T).size == 1:
        T = _default_time_vector(sysdata, N=T_num, tfinal=T, is_step=False)
    T = np.atleast_1d(T).reshape(-1)
    if T.ndim != 1 and len(T) < 2:
        raise ValueError("invalid value of T for this type of system")

    # If passed a list, recursively call individual responses with given T
    if isinstance(sysdata, (list, tuple)):
        responses = []
        for sys in sysdata:
            responses.append(impulse_response(
                sys, T, input=input, output=output, T_num=T_num,
                transpose=transpose, return_x=return_x, squeeze=squeeze))
        return TimeResponseList(responses)
    else:
        sys = sysdata

    # Make sure we have an LTI system
    if not isinstance(sys, LTI):
        raise ValueError("system must be LTI system for impulse response")

    # Convert to state space so that we can simulate
    if sys.nstates is None:
        sys = _convert_to_statespace(sys)

    # Check to make sure there is not a direct term
    if np.any(sys.D != 0) and isctime(sys):
        warnings.warn("System has direct feedthrough: `D != 0`. The "
                      "infinite impulse at `t=0` does not appear in the "
                      "output.\n"
                      "Results may be meaningless!")

    # Only single input and output are allowed for now
    if isinstance(input, (list, tuple)):
        if len(input_indices) > 1:
            raise NotImplementedError("list of input indices not allowed")
        input = input[0]
    elif isinstance(input, str):
        raise NotImplementedError("named inputs not allowed")

    if isinstance(output, (list, tuple)):
        if len(output_indices) > 1:
            raise NotImplementedError("list of output indices not allowed")
        output = output[0]
    elif isinstance(output, str):
        raise NotImplementedError("named outputs not allowed")

    # Set up arrays to handle the output
    ninputs = sys.ninputs if input is None else 1
    noutputs = sys.noutputs if output is None else 1
    yout = np.empty((noutputs, ninputs, np.asarray(T).size))
    xout = np.empty((sys.nstates, ninputs, np.asarray(T).size))
    uout = np.full((ninputs, ninputs, np.asarray(T).size), None)

    # Simulate the response for each input
    trace_labels, trace_types = [], []
    for i in range(sys.ninputs):
        # If input keyword was specified, only handle that case
        if isinstance(input, int) and i != input:
            continue

        # Save a label for this plot
        trace_labels.append(f"From {sys.input_labels[i]}")
        trace_types.append('impulse')

        #
        # Compute new X0 that contains the impulse
        #
        # We can't put the impulse into U because there is no numerical
        # representation for it (infinitesimally short, infinitely high).
        # See also: https://www.mathworks.com/support/tech-notes/1900/1901.html
        #
        if isctime(sys):
            X0 = sys.B[:, i]
            U = np.zeros((sys.ninputs, T.size))
        else:
            X0 = 0
            U = np.zeros((sys.ninputs, T.size))
            U[i, 0] = 1./sys.dt         # unit area impulse

        # Simulate the impulse response for this input
        response = forced_response(sys, T, U, X0)

        # Store the output (and states)
        inpidx = i if input is None else 0
        yout[:, inpidx, :] = response.y if output is None \
            else response.y[output]
        xout[:, inpidx, :] = response.x
        uout[:, inpidx, :] = U if input is None else U[i]

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
        state_labels=sys.state_labels, trace_labels=trace_labels,
        trace_types=trace_types, title="Impulse response for " + sys.name,
        sysname=sys.name, plot_inputs=False, transpose=transpose,
        return_x=return_x, squeeze=squeeze)


# utility function to find time period and time increment using pole locations
def _ideal_tfinal_and_dt(sys, is_step=True):
    """Helper function to compute ideal simulation duration tfinal and dt,
    the time increment. Usually called by _default_time_vector, whose job
    it is to choose a realistic time vector. Considers both poles and zeros.

    For discrete-time models, dt is inherent and only tfinal is computed.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
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

    Instead, a modal decomposition in time domain hence a truncated ZIR and
    ZSR can be used such that only the modes that have significant effect
    on the time response are taken. But the sensitivity of the eigenvalues
    complicate the matter since dlambda = <w, dA*v> with <w,v> = 1. Hence
    we can only work with simple poles with this formulation. See Golub,
    Van Loan Section 7.2.2 for simple eigenvalue sensitivity about the
    nonunity of <w,v>. The size of the response is dependent on the size of
    the eigenshapes rather than the eigenvalues themselves.

    By Ilhan Polat, with modifications by Sawyer Fuller to integrate into
    python-control 2020.08.17

    """
    from .statesp import _convert_to_statespace

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
        # Negative reals- treated as oscillatory mode
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


def _default_time_vector(sysdata, N=None, tfinal=None, is_step=True):
    """Returns a time vector that has a reasonable number of points.
    if system is discrete time, N is ignored """
    from .lti import LTI

    if isinstance(sysdata, (list, tuple)):
        tfinal_max = N_max = 0
        for sys in sysdata:
            timevec = _default_time_vector(
                sys, N=N, tfinal=tfinal, is_step=is_step)
            tfinal_max = max(tfinal_max, timevec[-1])
            N_max = max(N_max, timevec.size)
        return np.linspace(0, tfinal_max, N_max, endpoint=True)
    else:
        sys = sysdata

    # For non-LTI system, need tfinal
    if not isinstance(sys, LTI):
        if tfinal is None:
            raise ValueError(
                "can't automatically compute T for non-LTI system")
        elif isinstance(tfinal, (int, float, np.number)):
            if N is None:
                return np.linspace(0, tfinal)
            else:
                return np.linspace(0, tfinal, N)
        else:
            return tfinal       # Assume we got passed something appropriate

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
