# nlsys.py - input/output system module
# RMM, 28 April 2019
#
# Additional features to add:
#   * Allow constant inputs for MIMO input_output_response (w/out ones)
#   * Add unit tests (and example?) for time-varying systems
#   * Allow time vector for discrete-time simulations to be multiples of dt
#   * Check the way initial outputs for discrete-time systems are handled

"""This module contains the `NonlinearIOSystem` class that
represents (possibly nonlinear) input/output systems.  The
`NonlinearIOSystem` class is a general class that defines any
continuous- or discrete-time dynamical system.  Input/output systems
can be simulated and also used to compute operating points and
linearizations.

"""

from warnings import warn

import numpy as np
import scipy as sp

from . import config
from .config import _process_param, _process_kwargs
from .iosys import InputOutputSystem, _parse_spec, _process_iosys_keywords, \
    common_timebase, iosys_repr, isctime, isdtime
from .timeresp import TimeResponseData, TimeResponseList, \
    _check_convert_array, _process_time_response, _timeresp_aliases

__all__ = ['NonlinearIOSystem', 'InterconnectedSystem', 'nlsys',
           'input_output_response', 'find_eqpt', 'linearize',
           'interconnect', 'connection_table', 'OperatingPoint',
           'find_operating_point']


class NonlinearIOSystem(InputOutputSystem):
    """Nonlinear input/output system model.

    Creates an `InputOutputSystem` for a nonlinear system
    by specifying a state update function and an output function.  The new
    system can be a continuous or discrete-time system. Nonlinear I/O
    systems are usually created with the `nlsys` factory
    function.

    Parameters
    ----------
    updfcn : callable
        Function returning the state update function

            ``updfcn(t, x, u, params) -> array``

        where `t` is a float representing the current time, `x` is a 1-D
        array with shape (nstates,), `u` is a 1-D array with shape
        (ninputs,), and `params` is a dict containing the values of
        parameters used by the function.

    outfcn : callable
        Function returning the output at the given state

            `outfcn(t, x, u, params) -> array`

        where the arguments are the same as for `updfcn`.

    inputs, outputs, states : int, list of str or None, optional
        Description of the system inputs, outputs, and states.  See
        `control.nlsys` for more details.

    params : dict, optional
        Parameter values for the systems.  Passed to the evaluation functions
        for the system as default values, overriding internal defaults.

    dt : timebase, optional
        The timebase for the system, used to specify whether the system is
        operating in continuous or discrete time.  It can have the
        following values:

        * `dt` = 0: continuous-time system (default)
        * `dt` > 0: discrete-time system with sampling period `dt`
        * `dt` = True: discrete time with unspecified sampling period
        * `dt` = None: no timebase specified

    Attributes
    ----------
    ninputs, noutputs, nstates : int
        Number of input, output and state variables.
    shape : tuple
        2-tuple of I/O system dimension, (noutputs, ninputs).
    input_labels, output_labels, state_labels : list of str
        Names for the input, output, and state variables.
    name : string, optional
        System name.

    See Also
    --------
    nlsys, InputOutputSystem

    Notes
    -----
    The `InputOutputSystem` class (and its subclasses) makes use of two
    special methods for implementing much of the work of the class:

    * _rhs(t, x, u): compute the right hand side of the differential or
      difference equation for the system.  If not specified, the system
      has no state.

    * _out(t, x, u): compute the output for the current state of the system.
      The default is to return the entire system state.

    """
    def __init__(self, updfcn, outfcn=None, params=None, **kwargs):
        """Create a nonlinear I/O system given update and output functions."""
        # Process keyword arguments
        name, inputs, outputs, states, dt = _process_iosys_keywords(kwargs)

        # Initialize the rest of the structure
        super().__init__(
            inputs=inputs, outputs=outputs, states=states, dt=dt, name=name,
            **kwargs
        )
        self.params = {} if params is None else params.copy()

        # Store the update and output functions
        self.updfcn = updfcn
        self.outfcn = outfcn

        # Check to make sure arguments are consistent
        if updfcn is None:
            if self.nstates is None:
                self.nstates = 0
                self.updfcn = lambda t, x, u, params: np.zeros(0)
            else:
                raise ValueError(
                    "states specified but no update function given.")

        if outfcn is None:
            if self.noutputs == 0:
                self.outfcn = lambda t, x, u, params: np.zeros(0)
            elif self.noutputs is None and self.nstates is not None:
                self.noutputs = self.nstates
                if len(self.output_index) == 0:
                    # Use state names for outputs
                    self.output_index = self.state_index
            elif self.noutputs is not None and self.noutputs == self.nstates:
                # Number of outputs = number of states => all is OK
                pass
            elif self.noutputs is not None and self.noutputs != 0:
                raise ValueError("outputs specified but no output function "
                                 "(and nstates not known).")

        # Initialize current parameters to default parameters
        self._current_params = {} if params is None else params.copy()

    def __str__(self):
        out = f"{InputOutputSystem.__str__(self)}"
        if len(self.params) > 0:
            out += f"\nParameters: {[p for p in self.params.keys()]}"
        out += "\n\n" + \
            f"Update: {self.updfcn}\n" + \
            f"Output: {self.outfcn}"
        return out

    # Return the value of a static nonlinear system
    def __call__(sys, u, params=None, squeeze=None):
        """Evaluate a (static) nonlinearity at a given input value.

        If a nonlinear I/O system has no internal state, then evaluating
        the system at an input `u` gives the output ``y = F(u)``,
        determined by the output function.

        Parameters
        ----------
        params : dict, optional
            Parameter values for the system. Passed to the evaluation function
            for the system as default values, overriding internal defaults.
        squeeze : bool, optional
            If True and if the system has a single output, return the
            system output as a 1D array rather than a 2D array.  If
            False, return the system output as a 2D array even if the
            system is SISO.  Default value set by
            `config.defaults['control.squeeze_time_response']`.

        """
        # Make sure the call makes sense
        if sys.nstates != 0:
            raise TypeError(
                "function evaluation is only supported for static "
                "input/output systems")

        # If we received any parameters, update them before calling _out()
        if params is not None:
            sys._update_params(params)

        # Evaluate the function on the argument
        out = sys._out(0, np.array((0,)), np.asarray(u))
        out = _process_time_response(
            out, issiso=sys.issiso(), squeeze=squeeze)
        return out

    def __mul__(self, other):
        """Multiply two input/output systems (series interconnection)"""
        # Convert 'other' to an I/O system if needed
        other = _convert_to_iosystem(other)
        if not isinstance(other, InputOutputSystem):
            return NotImplemented

        # Make sure systems can be interconnected
        if other.noutputs != self.ninputs:
            raise ValueError(
                "can't multiply systems with incompatible inputs and outputs")

        # Make sure timebase are compatible
        common_timebase(other.dt, self.dt)

        # Create a new system to handle the composition
        inplist = [(0, i) for i in range(other.ninputs)]
        outlist = [(1, i) for i in range(self.noutputs)]
        newsys = InterconnectedSystem(
            (other, self), inplist=inplist, outlist=outlist)

        # Set up the connection map manually
        newsys.set_connect_map(np.block(
            [[np.zeros((other.ninputs, other.noutputs)),
              np.zeros((other.ninputs, self.noutputs))],
             [np.eye(self.ninputs, other.noutputs),
              np.zeros((self.ninputs, self.noutputs))]]
        ))

        # Return the newly created InterconnectedSystem
        return newsys

    def __rmul__(self, other):
        """Pre-multiply an input/output systems by a scalar/matrix"""
        # Convert other to an I/O system if needed
        other = _convert_to_iosystem(other)
        if not isinstance(other, InputOutputSystem):
            return NotImplemented

        # Make sure systems can be interconnected
        if self.noutputs != other.ninputs:
            raise ValueError("Can't multiply systems with incompatible "
                             "inputs and outputs")

        # Make sure timebase are compatible
        common_timebase(self.dt, other.dt)

        # Create a new system to handle the composition
        inplist = [(0, i) for i in range(self.ninputs)]
        outlist = [(1, i) for i in range(other.noutputs)]
        newsys = InterconnectedSystem(
            (self, other), inplist=inplist, outlist=outlist)

        # Set up the connection map manually
        newsys.set_connect_map(np.block(
            [[np.zeros((self.ninputs, self.noutputs)),
              np.zeros((self.ninputs, other.noutputs))],
             [np.eye(self.ninputs, self.noutputs),
              np.zeros((other.ninputs, other.noutputs))]]
        ))

        # Return the newly created InterconnectedSystem
        return newsys

    def __add__(self, other):
        """Add two input/output systems (parallel interconnection)"""
        # Convert other to an I/O system if needed
        other = _convert_to_iosystem(other)
        if not isinstance(other, InputOutputSystem):
            return NotImplemented

        # Make sure number of input and outputs match
        if self.ninputs != other.ninputs or self.noutputs != other.noutputs:
            raise ValueError("Can't add systems with incompatible numbers of "
                             "inputs or outputs")

        # Create a new system to handle the composition
        inplist = [[(0, i), (1, i)] for i in range(self.ninputs)]
        outlist = [[(0, i), (1, i)] for i in range(self.noutputs)]
        newsys = InterconnectedSystem(
            (self, other), inplist=inplist, outlist=outlist)

        # Return the newly created InterconnectedSystem
        return newsys

    def __radd__(self, other):
        """Parallel addition of input/output system to a compatible object."""
        # Convert other to an I/O system if needed
        other = _convert_to_iosystem(other)
        if not isinstance(other, InputOutputSystem):
            return NotImplemented

        # Make sure number of input and outputs match
        if self.ninputs != other.ninputs or self.noutputs != other.noutputs:
            raise ValueError("can't add systems with incompatible numbers of "
                             "inputs or outputs")

        # Create a new system to handle the composition
        inplist = [[(0, i), (1, i)] for i in range(other.ninputs)]
        outlist = [[(0, i), (1, i)] for i in range(other.noutputs)]
        newsys = InterconnectedSystem(
            (other, self), inplist=inplist, outlist=outlist)

        # Return the newly created InterconnectedSystem
        return newsys

    def __sub__(self, other):
        """Subtract two input/output systems (parallel interconnection)"""
        # Convert other to an I/O system if needed
        other = _convert_to_iosystem(other)
        if not isinstance(other, InputOutputSystem):
            return NotImplemented

        # Make sure number of input and outputs match
        if self.ninputs != other.ninputs or self.noutputs != other.noutputs:
            raise ValueError(
                "can't subtract systems with incompatible numbers of "
                "inputs or outputs")
        ninputs = self.ninputs
        noutputs = self.noutputs

        # Create a new system to handle the composition
        inplist = [[(0, i), (1, i)] for i in range(ninputs)]
        outlist = [[(0, i), (1, i, -1)] for i in range(noutputs)]
        newsys = InterconnectedSystem(
            (self, other), inplist=inplist, outlist=outlist)

        # Return the newly created InterconnectedSystem
        return newsys

    def __rsub__(self, other):
        """Parallel subtraction of I/O system to a compatible object."""
        # Convert other to an I/O system if needed
        other = _convert_to_iosystem(other)
        if not isinstance(other, InputOutputSystem):
            return NotImplemented
        return other - self

    def __neg__(self):
        """Negate an input/output system (rescale)"""
        if self.ninputs is None or self.noutputs is None:
            raise ValueError("Can't determine number of inputs or outputs")

        # Create a new system to hold the negation
        inplist = [(0, i) for i in range(self.ninputs)]
        outlist = [(0, i, -1) for i in range(self.noutputs)]
        newsys = InterconnectedSystem(
            (self,), dt=self.dt, inplist=inplist, outlist=outlist)

        # Return the newly created system
        return newsys

    def __truediv__(self, other):
        """Division of input/output system (by scalar or array)"""
        if not isinstance(other, InputOutputSystem):
            return self * (1/other)
        else:
            return NotImplemented

    # Determine if a system is static (memoryless)
    def _isstatic(self):
        return self.nstates == 0

    def _update_params(self, params):
        # Update the current parameter values
        self._current_params = self.params.copy()
        if params:
            self._current_params.update(params)

    def _rhs(self, t, x, u):
        """Evaluate right hand side of a differential or difference equation.

        Private function used to compute the right hand side of an
        input/output system model. Intended for fast evaluation; for a more
        user-friendly interface you may want to use `dynamics`.

        """
        return np.asarray(
            self.updfcn(t, x, u, self._current_params)).reshape(-1)

    def dynamics(self, t, x, u, params=None):
        """Dynamics of a differential or difference equation.

        Given time `t`, input `u` and state `x`, returns the value of the
        right hand side of the dynamical system. If the system is a
        continuous-time system, returns the time derivative::

            dx/dt = updfcn(t, x, u[, params])

        where `updfcn` is the system's (possibly nonlinear) update function.
        If the system is discrete time, returns the next value of `x`::

            x[t+dt] = updfcn(t, x[t], u[t][, params])

        where `t` is a scalar.

        The inputs `x` and `u` must be of the correct length.  The `params`
        argument is an optional dictionary of parameter values.

        Parameters
        ----------
        t : float
            Time at which to evaluate.
        x : array_like
            Current state.
        u : array_like
            Current input.
        params : dict, optional
            System parameter values.

        Returns
        -------
        dx/dt or x[t+dt] : ndarray

        """
        self._update_params(params)
        return self._rhs(
            t, np.asarray(x).reshape(-1), np.asarray(u).reshape(-1))

    def _out(self, t, x, u):
        """Evaluate the output of a system at a given state, input, and time

        Private function used to compute the output of of an input/output
        system model given the state, input, parameters. Intended for fast
        evaluation; for a more user-friendly interface you may want to use
        `output`.

        """
        #
        # To allow lazy evaluation of the system size, we allow for the
        # possibility that noutputs is left unspecified when the system
        # is created => we have to check for that case here (and return
        # the system state or a portion of it).
        #
        if self.outfcn is None:
            return x if self.noutputs is None else x[:self.noutputs]
        else:
            return np.asarray(
                self.outfcn(t, x, u, self._current_params)).reshape(-1)

    def output(self, t, x, u, params=None):
        """Compute the output of the system.

        Given time `t`, input `u` and state `x`, returns the output of the
        system::

            y = outfcn(t, x, u[, params])

        The inputs `x` and `u` must be of the correct length.

        Parameters
        ----------
        t : float
            The time at which to evaluate.
        x : array_like
            Current state.
        u : array_like
            Current input.
        params : dict, optional
            System parameter values.

        Returns
        -------
        y : ndarray

        """
        self._update_params(params)
        return self._out(
            t, np.asarray(x).reshape(-1), np.asarray(u).reshape(-1))

    def feedback(self, other=1, sign=-1, params=None):
        """Feedback interconnection between two I/O systems.

        Parameters
        ----------
        other : `InputOutputSystem`
            System in the feedback path.

        sign : float, optional
            Gain to use in feedback path.  Defaults to -1.

        params : dict, optional
            Parameter values for the overall system.  Passed to the
            evaluation functions for the system as default values,
            overriding defaults for the individual systems.

        Returns
        -------
        `NonlinearIOSystem`

        """
        # Convert sys2 to an I/O system if needed
        other = _convert_to_iosystem(other)

        # Make sure systems can be interconnected
        if self.noutputs != other.ninputs or other.noutputs != self.ninputs:
            raise ValueError("Can't connect systems with incompatible "
                             "inputs and outputs")

        # Make sure timebases are compatible
        dt = common_timebase(self.dt, other.dt)

        inplist = [(0, i) for i in range(self.ninputs)]
        outlist = [(0, i) for i in range(self.noutputs)]

        # Return the series interconnection between the systems
        newsys = InterconnectedSystem(
            (self, other), inplist=inplist, outlist=outlist,
            params=params, dt=dt)

        #  Set up the connection map manually
        newsys.set_connect_map(np.block(
            [[np.zeros((self.ninputs, self.noutputs)),
              sign * np.eye(self.ninputs, other.noutputs)],
             [np.eye(other.ninputs, self.noutputs),
              np.zeros((other.ninputs, other.noutputs))]]
        ))

        # Return the newly created system
        return newsys

    def linearize(self, x0, u0=None, t=0, params=None, eps=1e-6,
                  copy_names=False, **kwargs):
        """Linearize an input/output system at a given state and input.

        Return the linearization of an input/output system at a given
        operating point (or state and input value) as a `StateSpace` system.
        See `linearize` for complete documentation.

        """
        #
        # Default method: if the linearization is not defined by the
        # subclass, perform a numerical linearization use the `_rhs()` and
        # `_out()` member functions.
        #
        from .statesp import StateSpace

        # Allow first argument to be an operating point
        if isinstance(x0, OperatingPoint):
            u0 = x0.inputs if u0 is None else u0
            x0 = x0.states
        elif u0 is None:
            u0 = 0

        # Process nominal states and inputs
        x0, nstates = _process_vector_argument(x0, "x0", self.nstates)
        u0, ninputs = _process_vector_argument(u0, "u0", self.ninputs)

        # Update the current parameters (prior to calling _out())
        self._update_params(params)

        # Compute number of outputs by evaluating the output function
        noutputs = _find_size(self.noutputs, self._out(t, x0, u0), "outputs")

        # Compute the nominal value of the update law and output
        F0 = self._rhs(t, x0, u0)
        H0 = self._out(t, x0, u0)

        # Create empty matrices that we can fill up with linearizations
        A = np.zeros((nstates, nstates))        # Dynamics matrix
        B = np.zeros((nstates, ninputs))        # Input matrix
        C = np.zeros((noutputs, nstates))       # Output matrix
        D = np.zeros((noutputs, ninputs))       # Direct term

        # Perturb each of the state variables and compute linearization
        for i in range(nstates):
            dx = np.zeros((nstates,))
            dx[i] = eps
            A[:, i] = (self._rhs(t, x0 + dx, u0) - F0) / eps
            C[:, i] = (self._out(t, x0 + dx, u0) - H0) / eps

        # Perturb each of the input variables and compute linearization
        for i in range(ninputs):
            du = np.zeros((ninputs,))
            du[i] = eps
            B[:, i] = (self._rhs(t, x0, u0 + du) - F0) / eps
            D[:, i] = (self._out(t, x0, u0 + du) - H0) / eps

        # Create the state space system
        linsys = StateSpace(A, B, C, D, self.dt, remove_useless_states=False)

        # Set the system name, inputs, outputs, and states
        if copy_names:
            linsys._copy_names(self, prefix_suffix_name='linearized')

        # re-init to include desired signal names if names were provided
        return StateSpace(linsys, **kwargs)


class InterconnectedSystem(NonlinearIOSystem):
    """Interconnection of a set of input/output systems.

    This class is used to implement a system that is an interconnection of
    input/output systems.  The system consists of a collection of subsystems
    whose inputs and outputs are connected via a connection map.  The overall
    system inputs and outputs are subsets of the subsystem inputs and outputs.

    The `interconnect` factory function should be used to create an
    interconnected I/O system since it performs additional argument
    processing and checking.

    Parameters
    ----------
    syslist : list of `NonlinearIOSystem`
        List of state space systems to interconnect.
    connections : list of connections
        Description of the internal connections between the subsystem.  See
        `interconnect` for details.
    inplist, outlist : list of input and output connections
        Description of the inputs and outputs for the overall system.  See
        `interconnect` for details.
    inputs, outputs, states : int, list of str or None, optional
        Description of the system inputs, outputs, and states.  See
        `control.nlsys` for more details.
    params : dict, optional
        Parameter values for the systems.  Passed to the evaluation functions
        for the system as default values, overriding internal defaults.
    connection_type : str
        Type of connection: 'explicit' (or None) for explicitly listed
        set of connections, 'implicit' for connections made via signal names.

    Attributes
    ----------
    ninputs, noutputs, nstates : int
        Number of input, output and state variables.
    shape : tuple
        2-tuple of I/O system dimension, (noutputs, ninputs).
    name : string, optional
        System name.
    connect_map : 2D array
        Mapping of subsystem outputs to subsystem inputs.
    input_map : 2D array
        Mapping of system inputs to subsystem inputs.
    output_map : 2D array
        Mapping of (stacked) subsystem outputs and inputs to system outputs.
    input_labels, output_labels, state_labels : list of str
        Names for the input, output, and state variables.
    input_offset, output_offset, state_offset : list of int
        Offset to the subsystem inputs, outputs, and states in the overall
        system input, output, and state arrays.
    syslist_index : dict
        Index of the subsystem with key given by the name of the subsystem.

    See Also
    --------
    interconnect, NonlinearIOSystem, LinearICSystem

    """
    def __init__(self, syslist, connections=None, inplist=None, outlist=None,
                 params=None, warn_duplicate=None, connection_type=None,
                 **kwargs):
        """Create an I/O system from a list of systems + connection info."""
        from .statesp import _convert_to_statespace
        from .xferfcn import TransferFunction

        self.connection_type = connection_type # explicit, implicit, or None

        # Convert input and output names to lists if they aren't already
        if inplist is not None and not isinstance(inplist, list):
            inplist = [inplist]
        if outlist is not None and not isinstance(outlist, list):
            outlist = [outlist]

        # Check if dt argument was given; if not, pull from systems
        dt = kwargs.pop('dt', None)

        # Process keyword arguments (except dt)
        name, inputs, outputs, states, _ = _process_iosys_keywords(kwargs)

        # Initialize the system list and index
        self.syslist = list(syslist) # ensure modifications can be made
        self.syslist_index = {}

        # Initialize the input, output, and state counts, indices
        nstates, self.state_offset = 0, []
        ninputs, self.input_offset = 0, []
        noutputs, self.output_offset = 0, []

        # Keep track of system objects and names we have already seen
        sysobj_name_dct = {}
        sysname_count_dct = {}

        # Go through the system list and keep track of counts, offsets
        for sysidx, sys in enumerate(self.syslist):
            # Convert transfer functions to state space
            if isinstance(sys, TransferFunction):
                sys = _convert_to_statespace(sys)
                self.syslist[sysidx] = sys

            # Make sure time bases are consistent
            dt = common_timebase(dt, sys.dt)

            # Make sure number of inputs, outputs, states is given
            if sys.ninputs is None or sys.noutputs is None:
                raise TypeError("system '%s' must define number of inputs, "
                                "outputs, states in order to be connected" %
                                sys.name)
            elif sys.nstates is None:
                raise TypeError("can't interconnect systems with no state")

            # Keep track of the offsets into the states, inputs, outputs
            self.input_offset.append(ninputs)
            self.output_offset.append(noutputs)
            self.state_offset.append(nstates)

            # Keep track of the total number of states, inputs, outputs
            nstates += sys.nstates
            ninputs += sys.ninputs
            noutputs += sys.noutputs

            # Check for duplicate systems or duplicate names
            # Duplicates are renamed sysname_1, sysname_2, etc.
            if sys in sysobj_name_dct:
                # Make a copy of the object using a new name
                if warn_duplicate is None and sys._generic_name_check():
                    # Make a copy w/out warning, using generic format
                    sys = sys.copy(use_prefix_suffix=False)
                    warn_flag = False
                else:
                    sys = sys.copy()
                    warn_flag = warn_duplicate

                # Warn the user about the new object
                if warn_flag is not False:
                    warn("duplicate object found in system list; "
                         "created copy: %s" % str(sys.name), stacklevel=2)

            # Check to see if the system name shows up more than once
            if sys.name is not None and sys.name in sysname_count_dct:
                count = sysname_count_dct[sys.name]
                sysname_count_dct[sys.name] += 1
                sysname = sys.name + "_" + str(count)
                sysobj_name_dct[sys] = sysname
                self.syslist_index[sysname] = sysidx

                if warn_duplicate is not False:
                    warn("duplicate name found in system list; "
                         "renamed to {}".format(sysname), stacklevel=2)

            else:
                sysname_count_dct[sys.name] = 1
                sysobj_name_dct[sys] = sys.name
                self.syslist_index[sys.name] = sysidx

        if states is None:
            states = []
            state_name_delim = config.defaults['iosys.state_name_delim']
            for sys, sysname in sysobj_name_dct.items():
                states += [sysname + state_name_delim +
                           statename for statename in sys.state_index.keys()]

        # Make sure we the state list is the right length (internal check)
        if isinstance(states, list) and len(states) != nstates:
            raise RuntimeError(
                f"construction of state labels failed; found: "
                f"{len(states)} labels; expecting {nstates}")

        # Figure out what the inputs and outputs are
        if inputs is None and inplist is not None:
            inputs = len(inplist)

        if outputs is None and outlist is not None:
            outputs = len(outlist)

        if params is None:
            params = {}
            for sys in self.syslist:
                params = params | sys.params

        # Create updfcn and outfcn
        def updfcn(t, x, u, params):
            self._update_params(params)
            return self._rhs(t, x, u)
        def outfcn(t, x, u, params):
            self._update_params(params)
            return self._out(t, x, u)

        # Initialize NonlinearIOSystem object
        super().__init__(
            updfcn, outfcn, inputs=inputs, outputs=outputs,
            states=states, dt=dt, name=name, params=params, **kwargs)

        # Convert the list of interconnections to a connection map (matrix)
        self.connect_map = np.zeros((ninputs, noutputs))
        for connection in connections or []:
            input_indices = self._parse_input_spec(connection[0])
            for output_spec in connection[1:]:
                output_indices, gain = self._parse_output_spec(output_spec)
                if len(output_indices) != len(input_indices):
                    raise ValueError(
                        f"inconsistent number of signals in connecting"
                        f" '{output_spec}' to '{connection[0]}'")

                for input_index, output_index in zip(
                        input_indices, output_indices):
                    if self.connect_map[input_index, output_index] != 0:
                        warn("multiple connections given for input %d" %
                             input_index + "; combining with previous entries")
                    self.connect_map[input_index, output_index] += gain

        # Convert the input list to a matrix: maps system to subsystems
        self.input_map = np.zeros((ninputs, self.ninputs))
        for index, inpspec in enumerate(inplist or []):
            if isinstance(inpspec, (int, str, tuple)):
                inpspec = [inpspec]
            if not isinstance(inpspec, list):
                raise ValueError("specifications in inplist must be of type "
                                 "int, str, tuple or list")
            for spec in inpspec:
                ulist_indices = self._parse_input_spec(spec)
                for j, ulist_index in enumerate(ulist_indices):
                    if self.input_map[ulist_index, index] != 0:
                        warn("multiple connections given for input %d" %
                             index + "; combining with previous entries.")
                    self.input_map[ulist_index, index + j] += 1

        # Convert the output list to a matrix: maps subsystems to system
        self.output_map = np.zeros((self.noutputs, noutputs + ninputs))
        for index, outspec in enumerate(outlist or []):
            if isinstance(outspec, (int, str, tuple)):
                outspec = [outspec]
            if not isinstance(outspec, list):
                raise ValueError("specifications in outlist must be of type "
                                 "int, str, tuple or list")
            for spec in outspec:
                ylist_indices, gain = self._parse_output_spec(spec)
                for j, ylist_index in enumerate(ylist_indices):
                    if self.output_map[index, ylist_index] != 0:
                        warn("multiple connections given for output %d" %
                             index + "; combining with previous entries")
                    self.output_map[index + j, ylist_index] += gain

    def __str__(self):
        import textwrap
        out = InputOutputSystem.__str__(self)

        out += f"\n\nSubsystems ({len(self.syslist)}):\n"
        for sys in self.syslist:
            out += "\n".join(textwrap.wrap(
                iosys_repr(sys, format='info'), width=78,
                initial_indent=" * ", subsequent_indent="    ")) + "\n"

        # Build a list of input, output, and inpout signals
        input_list, output_list, inpout_list = [], [], []
        for sys in self.syslist:
            input_list += [sys.name + "." + lbl for lbl in sys.input_labels]
            output_list += [sys.name + "." + lbl for lbl in sys.output_labels]
        inpout_list = input_list + output_list

        # Define a utility function to generate the signal
        def cxn_string(signal, gain, first):
            if gain == 1:
                return (" + " if not first else "") + f"{signal}"
            elif gain == -1:
                return (" - " if not first else "-") + f"{signal}"
            elif gain > 0:
                return (" + " if not first else "") + f"{gain} * {signal}"
            elif gain < 0:
                return (" - " if not first else "-") + \
                    f"{abs(gain)} * {signal}"

        out += "\nConnections:\n"
        for i in range(len(input_list)):
            first = True
            cxn = f"{input_list[i]} <- "
            if np.any(self.connect_map[i]):
                for j in range(len(output_list)):
                    if self.connect_map[i, j]:
                        cxn += cxn_string(
                            output_list[j], self.connect_map[i,j], first)
                        first = False
            if np.any(self.input_map[i]):
                for j in range(len(self.input_labels)):
                    if self.input_map[i, j]:
                        cxn += cxn_string(
                            self.input_labels[j], self.input_map[i, j], first)
                        first = False
            out += "\n".join(textwrap.wrap(
                cxn, width=78, initial_indent=" * ",
                subsequent_indent="     ")) + "\n"

        out += "\nOutputs:"
        for i in range(len(self.output_labels)):
            first = True
            cxn = f"{self.output_labels[i]} <- "
            if np.any(self.output_map[i]):
                for j in range(len(inpout_list)):
                    if self.output_map[i, j]:
                        cxn += cxn_string(
                            output_list[j], self.output_map[i, j], first)
                        first = False
                out += "\n" + "\n".join(textwrap.wrap(
                    cxn, width=78, initial_indent=" * ",
                    subsequent_indent="     "))

        return out

    def _update_params(self, params):
        for sys in self.syslist:
            local = sys.params.copy()   # start with system parameters
            local.update(self.params)   # update with global params
            if params:
                local.update(params)    # update with locally passed parameters
            sys._update_params(local)

    def _rhs(self, t, x, u):
        # Make sure state and input are vectors
        x = np.array(x, ndmin=1)
        u = np.array(u, ndmin=1)

        # Compute the input and output vectors
        ulist, ylist = self._compute_static_io(t, x, u)

        # Go through each system and update the right hand side for that system
        xdot = np.zeros((self.nstates,))        # Array to hold results
        state_index, input_index = 0, 0         # Start at the beginning
        for sys in self.syslist:
            # Update the right hand side for this subsystem
            if sys.nstates != 0:
                xdot[state_index:state_index + sys.nstates] = sys._rhs(
                    t, x[state_index:state_index + sys.nstates],
                    ulist[input_index:input_index + sys.ninputs])

            # Update the state and input index counters
            state_index += sys.nstates
            input_index += sys.ninputs

        return xdot

    def _out(self, t, x, u):
        # Make sure state and input are vectors
        x = np.array(x, ndmin=1)
        u = np.array(u, ndmin=1)

        # Compute the input and output vectors
        ulist, ylist = self._compute_static_io(t, x, u)

        # Make the full set of subsystem outputs to system output
        return self.output_map @ ylist

    # Find steady state (static) inputs and outputs
    def _compute_static_io(self, t, x, u):
        # Figure out the total number of inputs and outputs
        (ninputs, noutputs) = self.connect_map.shape

        #
        # Get the outputs and inputs at the current system state
        #

        # Initialize the lists used to keep track of internal signals
        ulist = np.dot(self.input_map, u)
        ylist = np.zeros((noutputs + ninputs,))

        # To allow for feedthrough terms, iterate multiple times to allow
        # feedthrough elements to propagate.  For n systems, we could need to
        # cycle through n+1 times before reaching steady state
        # TODO (later): see if there is a more efficient way to compute
        cycle_count = len(self.syslist) + 1
        while cycle_count > 0:
            state_index, input_index, output_index = 0, 0, 0
            for sys in self.syslist:
                # Compute outputs for each system from current state
                ysys = sys._out(
                    t, x[state_index:state_index + sys.nstates],
                    ulist[input_index:input_index + sys.ninputs])

                # Store the outputs at the start of ylist
                ylist[output_index:output_index + sys.noutputs] = \
                    ysys.reshape((-1,))

                # Store the input in the second part of ylist
                ylist[noutputs + input_index:
                      noutputs + input_index + sys.ninputs] = \
                          ulist[input_index:input_index + sys.ninputs]

                # Increment the index pointers
                state_index += sys.nstates
                input_index += sys.ninputs
                output_index += sys.noutputs

            # Compute inputs based on connection map
            new_ulist = self.connect_map @ ylist[:noutputs] \
                + np.dot(self.input_map, u)

            # Check to see if any of the inputs changed
            if (ulist == new_ulist).all():
                break
            else:
                ulist = new_ulist

            # Decrease the cycle counter
            cycle_count -= 1

        # Make sure that we stopped before detecting an algebraic loop
        if cycle_count == 0:
            raise RuntimeError("algebraic loop detected")

        return ulist, ylist

    def _parse_input_spec(self, spec):
        """Parse an input specification and returns the indices."""
        # Parse the signal that we received
        subsys_index, input_indices, gain = _parse_spec(
            self.syslist, spec, 'input')
        if gain != 1:
            raise ValueError("gain not allowed in spec '%s'" % str(spec))

        # Return the indices into the input vector list (ylist)
        return [self.input_offset[subsys_index] + i for i in input_indices]

    def _parse_output_spec(self, spec):
        """Parse an output specification and returns the indices and gain."""
        # Parse the rest of the spec with standard signal parsing routine
        try:
            # Start by looking in the set of subsystem outputs
            subsys_index, output_indices, gain = \
                _parse_spec(self.syslist, spec, 'output')
            output_offset = self.output_offset[subsys_index]

        except ValueError:
            # Try looking in the set of subsystem *inputs*
            subsys_index, output_indices, gain = _parse_spec(
                self.syslist, spec, 'input or output', dictname='input_index')

            # Return the index into the input vector list (ylist)
            output_offset = sum(sys.noutputs for sys in self.syslist) + \
                self.input_offset[subsys_index]

        return [output_offset + i for i in output_indices], gain

    def _find_system(self, name):
        return self.syslist_index.get(name, None)

    def set_connect_map(self, connect_map):
        """Set the connection map for an interconnected I/O system.

        Parameters
        ----------
        connect_map : 2D array
             Specify the matrix that will be used to multiply the vector of
             subsystem outputs to obtain the vector of subsystem inputs.

        """
        # Make sure the connection map is the right size
        if connect_map.shape != self.connect_map.shape:
            ValueError("Connection map is not the right shape")
        self.connect_map = connect_map

    def set_input_map(self, input_map):
        """Set the input map for an interconnected I/O system.

        Parameters
        ----------
        input_map : 2D array
             Specify the matrix that will be used to multiply the vector of
             system inputs to obtain the vector of subsystem inputs.  These
             values are added to the inputs specified in the connection map.

        """
        # Figure out the number of internal inputs
        ninputs = sum(sys.ninputs for sys in self.syslist)

        # Make sure the input map is the right size
        if input_map.shape[0] != ninputs:
            ValueError("Input map is not the right shape")
        self.input_map = input_map
        self.ninputs = input_map.shape[1]

    def set_output_map(self, output_map):
        """Set the output map for an interconnected I/O system.

        Parameters
        ----------
        output_map : 2D array
             Specify the matrix that will be used to multiply the vector of
             subsystem outputs concatenated with subsystem inputs to obtain
             the vector of system outputs.

        """
        # Figure out the number of internal inputs and outputs
        ninputs = sum(sys.ninputs for sys in self.syslist)
        noutputs = sum(sys.noutputs for sys in self.syslist)

        # Make sure the output map is the right size
        if output_map.shape[1] == noutputs:
            # For backward compatibility, add zeros to the end of the array
            output_map = np.concatenate(
                (output_map,
                 np.zeros((output_map.shape[0], ninputs))),
                axis=1)

        if output_map.shape[1] != noutputs + ninputs:
            ValueError("Output map is not the right shape")
        self.output_map = output_map
        self.noutputs = output_map.shape[0]

    def unused_signals(self):
        """Find unused subsystem inputs and outputs.

        Returns
        -------
        unused_inputs : dict
          A mapping from tuple of indices (isys, isig) to string
          '{sys}.{sig}', for all unused subsystem inputs.

        unused_outputs : dict
          A mapping from tuple of indices (osys, osig) to string
          '{sys}.{sig}', for all unused subsystem outputs.

        """
        used_sysinp_via_inp = np.nonzero(self.input_map)[0]
        used_sysout_via_out = np.nonzero(self.output_map)[1]
        used_sysinp_via_con, used_sysout_via_con = np.nonzero(self.connect_map)

        used_sysinp = set(used_sysinp_via_inp) | set(used_sysinp_via_con)
        used_sysout = set(used_sysout_via_out) | set(used_sysout_via_con)

        nsubsysinp = sum(sys.ninputs for sys in self.syslist)
        nsubsysout = sum(sys.noutputs for sys in self.syslist)

        unused_sysinp = sorted(set(range(nsubsysinp)) - used_sysinp)
        unused_sysout = sorted(set(range(nsubsysout)) - used_sysout)

        inputs = [(isys, isig, f'{sys.name}.{sig}')
                  for isys, sys in enumerate(self.syslist)
                  for sig, isig in sys.input_index.items()]

        outputs = [(isys, isig, f'{sys.name}.{sig}')
                   for isys, sys in enumerate(self.syslist)
                   for sig, isig in sys.output_index.items()]

        return ({inputs[i][:2]: inputs[i][2] for i in unused_sysinp},
                {outputs[i][:2]: outputs[i][2] for i in unused_sysout})

    def connection_table(self, show_names=False, column_width=32):
        """Table of connections inside an interconnected system.

        Intended primarily for `InterconnectedSystem`'s that have been
        connected implicitly using signal names.

        Parameters
        ----------
        show_names : bool, optional
            Instead of printing out the system number, print out the name
            of each system. Default is False because system name is not
            usually specified when performing implicit interconnection
            using `interconnect`.
        column_width : int, optional
            Character width of printed columns.

        Examples
        --------
        >>> P = ct.ss(1,1,1,0, inputs='u', outputs='y', name='P')
        >>> C = ct.tf(10, [.1, 1], inputs='e', outputs='u', name='C')
        >>> L = ct.interconnect([C, P], inputs='e', outputs='y')
        >>> L.connection_table(show_names=True) # doctest: +SKIP
        signal    | source                        | destination
        --------------------------------------------------------------------
        e         | input                         | C
        u         | C                             | P
        y         | P                             | output

        """

        print('signal'.ljust(10) + '| source'.ljust(column_width) + \
            '| destination')
        print('-'*(10 + column_width * 2))

        # TODO: update this method for explicitly-connected systems
        if not self.connection_type == 'implicit':
            warn('connection_table only gives useful output for implicitly-'\
                'connected systems')

        # collect signal labels
        signal_labels = []
        for sys in self.syslist:
            signal_labels += sys.input_labels + sys.output_labels
        signal_labels = set(signal_labels)

        for signal_label in signal_labels:
            print(signal_label.ljust(10), end='')
            sources = '| '
            dests = '| '

            #  overall interconnected system inputs and outputs
            if self.find_input(signal_label) is not None:
                sources += 'input'
            if self.find_output(signal_label) is not None:
                dests += 'output'

            # internal connections
            for idx, sys in enumerate(self.syslist):
                loc = sys.find_output(signal_label)
                if loc is not None:
                    if not sources.endswith(' '):
                        sources += ', '
                    sources += sys.name if show_names else 'system ' + str(idx)
                loc = sys.find_input(signal_label)
                if loc is not None:
                    if not dests.endswith(' '):
                        dests += ', '
                    dests += sys.name if show_names else 'system ' + str(idx)
            if len(sources) >= column_width:
                sources = sources[:column_width - 3] + '.. '
            print(sources.ljust(column_width), end='')
            if len(dests) > column_width:
                dests = dests[:column_width - 3] + '.. '
            print(dests.ljust(column_width), end='\n')

    def _find_inputs_by_basename(self, basename):
        """Find all subsystem inputs matching basename

        Returns
        -------
        Mapping from (isys, isig) to '{sys}.{sig}'

        """
        return {(isys, isig): f'{sys.name}.{basename}'
                for isys, sys in enumerate(self.syslist)
                for sig, isig in sys.input_index.items()
                if sig == (basename)}

    def _find_outputs_by_basename(self, basename):
        """Find all subsystem outputs matching basename

        Returns
        -------
        Mapping from (isys, isig) to '{sys}.{sig}'

        """
        return {(isys, isig): f'{sys.name}.{basename}'
                for isys, sys in enumerate(self.syslist)
                for sig, isig in sys.output_index.items()
                if sig == (basename)}

    # TODO: change to internal function?  (not sure users need to see this)
    def check_unused_signals(
            self, ignore_inputs=None, ignore_outputs=None, print_warning=True):
        """Check for unused subsystem inputs and outputs.

        Check to see if there are any unused signals and return a list of
        unused input and output signal descriptions.  If `warning` is
        True and any unused inputs or outputs are found, emit a warning.

        Parameters
        ----------
        ignore_inputs : list of input-spec
          Subsystem inputs known to be unused.  input-spec can be any of:
            'sig', 'sys.sig', (isys, isig), ('sys', isig)

          If the 'sig' form is used, all subsystem inputs with that
          name are considered ignored.

        ignore_outputs : list of output-spec
          Subsystem outputs known to be unused.  output-spec can be any of:
            'sig', 'sys.sig', (isys, isig), ('sys', isig)

          If the 'sig' form is used, all subsystem outputs with that
          name are considered ignored.

        print_warning : bool, optional
            If True, print a warning listing any unused signals.

        Returns
        -------
        dropped_inputs : list of tuples
            A list of the dropped input signals, with each element of the
            list in the form of (isys, isig).

        dropped_outputs : list of tuples
            A list of the dropped output signals, with each element of the
            list in the form of (osys, osig).

        """

        if ignore_inputs is None:
            ignore_inputs = []

        if ignore_outputs is None:
            ignore_outputs = []

        unused_inputs, unused_outputs = self.unused_signals()

        # (isys, isig) -> signal-spec
        ignore_input_map = {}
        for ignore_input in ignore_inputs:
            if isinstance(ignore_input, str) and '.' not in ignore_input:
                ignore_idxs = self._find_inputs_by_basename(ignore_input)
                if not ignore_idxs:
                    raise ValueError("Couldn't find ignored input "
                                     f"{ignore_input} in subsystems")
                ignore_input_map.update(ignore_idxs)
            else:
                isys, isigs = _parse_spec(
                    self.syslist, ignore_input, 'input')[:2]
                for isig in isigs:
                    ignore_input_map[(isys, isig)] = ignore_input

        # (osys, osig) -> signal-spec
        ignore_output_map = {}
        for ignore_output in ignore_outputs:
            if isinstance(ignore_output, str) and '.' not in ignore_output:
                ignore_found = self._find_outputs_by_basename(ignore_output)
                if not ignore_found:
                    raise ValueError("Couldn't find ignored output "
                                     f"{ignore_output} in subsystems")
                ignore_output_map.update(ignore_found)
            else:
                osys, osigs = _parse_spec(
                    self.syslist, ignore_output, 'output')[:2]
                for osig in osigs:
                    ignore_output_map[(osys, osig)] = ignore_output

        dropped_inputs = set(unused_inputs) - set(ignore_input_map)
        dropped_outputs = set(unused_outputs) - set(ignore_output_map)

        used_ignored_inputs = set(ignore_input_map) - set(unused_inputs)
        used_ignored_outputs = set(ignore_output_map) - set(unused_outputs)

        if print_warning and dropped_inputs:
            msg = ('Unused input(s) in InterconnectedSystem: '
                   + '; '.join(f'{inp}={unused_inputs[inp]}'
                               for inp in dropped_inputs))
            warn(msg)

        if print_warning and dropped_outputs:
            msg = ('Unused output(s) in InterconnectedSystem: '
                   + '; '.join(f'{out} : {unused_outputs[out]}'
                               for out in dropped_outputs))
            warn(msg)

        if print_warning and used_ignored_inputs:
            msg = ('Input(s) specified as ignored is (are) used: '
                   + '; '.join(f'{inp} : {ignore_input_map[inp]}'
                               for inp in used_ignored_inputs))
            warn(msg)

        if print_warning and used_ignored_outputs:
            msg = ('Output(s) specified as ignored is (are) used: '
                   + '; '.join(f'{out}={ignore_output_map[out]}'
                               for out in used_ignored_outputs))
            warn(msg)

        return dropped_inputs, dropped_outputs


def nlsys(updfcn, outfcn=None, **kwargs):
    """Create a nonlinear input/output system.

    Creates an `InputOutputSystem` for a nonlinear system by specifying a
    state update function and an output function.  The new system can be a
    continuous or discrete-time system.

    Parameters
    ----------
    updfcn : callable (or `StateSpace`)
        Function returning the state update function

            ``updfcn(t, x, u, params) -> array``

        where `x` is a 1-D array with shape (nstates,), `u` is a 1-D array
        with shape (ninputs,), `t` is a float representing the current
        time, and `params` is a dict containing the values of parameters
        used by the function.

        If a `StateSpace` system is passed as the update function,
        then a nonlinear I/O system is created that implements the linear
        dynamics of the state space system.

    outfcn : callable
        Function returning the output at the given state

            ``outfcn(t, x, u, params) -> array``

        where the arguments are the same as for `updfcn`.

    inputs : int, list of str or None, optional
        Description of the system inputs.  This can be given as an integer
        count or as a list of strings that name the individual signals.
        If an integer count is specified, the names of the signal will be
        of the form 's[i]' (where 's' is one of 'u', 'y', or 'x').  If
        this parameter is not given or given as None, the relevant
        quantity will be determined when possible based on other
        information provided to functions using the system.

    outputs : int, list of str or None, optional
        Description of the system outputs.  Same format as `inputs`.

    states : int, list of str, or None, optional
        Description of the system states.  Same format as `inputs`.

    dt : timebase, optional
        The timebase for the system, used to specify whether the system is
        operating in continuous or discrete time.  It can have the
        following values:

        * `dt` = 0: continuous-time system (default)
        * `dt` > 0: discrete-time system with sampling period `dt`
        * `dt` = True: discrete time with unspecified sampling period
        * `dt` = None: no timebase specified

    name : string, optional
        System name (used for specifying signals). If unspecified, a
        generic name 'sys[id]' is generated with a unique integer id.

    params : dict, optional
        Parameter values for the system.  Passed to the evaluation functions
        for the system as default values, overriding internal defaults.

    Returns
    -------
    sys : `NonlinearIOSystem`
        Nonlinear input/output system.

    Other Parameters
    ----------------
    input_prefix, output_prefix, state_prefix : string, optional
        Set the prefix for input, output, and state signals.  Defaults =
        'u', 'y', 'x'.

    See Also
    --------
    ss, tf

    Examples
    --------
    >>> def kincar_update(t, x, u, params):
    ...     l = params['l']              # wheelbase
    ...     return np.array([
    ...         np.cos(x[2]) * u[0],     # x velocity
    ...         np.sin(x[2]) * u[0],     # y velocity
    ...         np.tan(u[1]) * u[0] / l  # angular velocity
    ...     ])
    >>>
    >>> def kincar_output(t, x, u, params):
    ...     return x[0:2]  # x, y position
    >>>
    >>> kincar = ct.nlsys(
    ...     kincar_update, kincar_output, states=3, inputs=2, outputs=2,
    ...     params={'l': 1})
    >>>
    >>> timepts = np.linspace(0, 10)
    >>> response = ct.input_output_response(
    ...     kincar, timepts, [10, 0.05 * np.sin(timepts)])

    """
    from .iosys import _extended_system_name
    from .statesp import StateSpace

    if isinstance(updfcn, StateSpace):
        sys_ss = updfcn
        kwargs['inputs'] = kwargs.get('inputs', sys_ss.input_labels)
        kwargs['outputs'] = kwargs.get('outputs', sys_ss.output_labels)
        kwargs['states'] = kwargs.get('states', sys_ss.state_labels)
        kwargs['name'] = kwargs.get('name', _extended_system_name(
            sys_ss.name, prefix_suffix_name='converted'))

        sys_nl = NonlinearIOSystem(
            lambda t, x, u, params:
                sys_ss.A @ np.atleast_1d(x) + sys_ss.B @ np.atleast_1d(u),
            lambda t, x, u, params:
                sys_ss.C @ np.atleast_1d(x) + sys_ss.D @ np.atleast_1d(u),
            **kwargs)

        if sys_nl.nstates != sys_ss.nstates or sys_nl.shape != sys_ss.shape:
            raise ValueError(
                "new input, output, or state specification "
                "doesn't match system size")

        return sys_nl
    else:
        return NonlinearIOSystem(updfcn, outfcn, **kwargs)


def input_output_response(
        sys, timepts=None, inputs=0., initial_state=0., params=None,
        ignore_errors=False, transpose=False, return_states=False,
        squeeze=None, solve_ivp_kwargs=None, evaluation_times='T', **kwargs):
    """Compute the output response of a system to a given input.

    Simulate a dynamical system with a given input and return its output
    and state values.

    Parameters
    ----------
    sys : `NonlinearIOSystem` or list of `NonlinearIOSystem`
        I/O system(s) for which input/output response is simulated.
    timepts (or T) : array_like
        Time steps at which the input is defined; values must be evenly spaced.
    inputs (or U) : array_like, list, or number, optional
        Input array giving input at each time in `timepts` (default =
        0). If a list is specified, each element in the list will be
        treated as a portion of the input and broadcast (if necessary) to
        match the time vector.
    initial_state (or X0) : array_like, list, or number, optional
        Initial condition (default = 0).  If a list is given, each element
        in the list will be flattened and stacked into the initial
        condition.  If a smaller number of elements are given that the
        number of states in the system, the initial condition will be padded
        with zeros.
    evaluation_times (or t_eval) : array-list, optional
        List of times at which the time response should be computed.
        Defaults to `timepts`.
    return_states (or return_x) : bool, optional
        If True, return the state vector when assigning to a tuple.  See
        `forced_response` for more details.  If True, return the values of
        the state at each time Default is False.
    params : dict, optional
        Parameter values for the system.  Passed to the evaluation functions
        for the system as default values, overriding internal defaults.
    squeeze : bool, optional
        If True and if the system has a single output, return the system
        output as a 1D array rather than a 2D array.  If False, return the
        system output as a 2D array even if the system is SISO.  Default
        value set by `config.defaults['control.squeeze_time_response']`.

    Returns
    -------
    response : `TimeResponseData`
        Time response data object representing the input/output response.
        When accessed as a tuple, returns ``(time, outputs)`` or ``(time,
        outputs, states`` if `return_x` is True.  If the input/output system
        signals are named, these names will be used as labels for the time
        response.  If `sys` is a list of systems, returns a `TimeResponseList`
        object.  Results can be plotted using the `~TimeResponseData.plot`
        method.  See `TimeResponseData` for more detailed information.
    response.time : array
        Time values of the output.
    response.outputs : array
        Response of the system.  If the system is SISO and `squeeze` is not
        True, the array is 1D (indexed by time).  If the system is not SISO
        or `squeeze` is False, the array is 2D (indexed by output and time).
    response.states : array
        Time evolution of the state vector, represented as a 2D array
        indexed by state and time.
    response.inputs : array
        Input(s) to the system, indexed by input and time.
    response.params : dict
        Parameters values used for the simulation.

    Other Parameters
    ----------------
    ignore_errors : bool, optional
        If False (default), errors during computation of the trajectory
        will raise a `RuntimeError` exception.  If True, do not raise
        an exception and instead set `response.success` to False and
        place an error message in `response.message`.
    solve_ivp_method : str, optional
        Set the method used by `scipy.integrate.solve_ivp`.  Defaults
        to 'RK45'.
    solve_ivp_kwargs : dict, optional
        Pass additional keywords to `scipy.integrate.solve_ivp`.
    transpose : bool, default=False
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and `scipy.signal.lsim`).

    Raises
    ------
    TypeError
        If the system is not an input/output system.
    ValueError
        If time step does not match sampling time (for discrete-time systems).

    Notes
    -----
    If a smaller number of initial conditions are given than the number of
    states in the system, the initial conditions will be padded with zeros.
    This is often useful for interconnected control systems where the
    process dynamics are the first system and all other components start
    with zero initial condition since this can be specified as [xsys_0, 0].
    A warning is issued if the initial conditions are padded and and the
    final listed initial state is not zero.

    If discontinuous inputs are given, the underlying SciPy numerical
    integration algorithms can sometimes produce erroneous results due to
    the default tolerances that are used.  The `solve_ivp_method` and
    `solve_ivp_keywords` parameters can be used to tune the ODE solver and
    produce better results. In particular, using 'LSODA' as the
    `solve_ivp_method`, setting the `rtol` parameter to a smaller value
    (e.g. using ``solve_ivp_kwargs={'rtol': 1e-4}``), or setting the
    maximum step size to a smaller value (e.g. ``solve_ivp_kwargs=
    {'max_step': 0.01}``) can provide more accurate results.

    """
    #
    # Process keyword arguments
    #
    _process_kwargs(kwargs, _timeresp_aliases)
    T = _process_param('timepts', timepts, kwargs, _timeresp_aliases)
    U = _process_param('inputs', inputs, kwargs, _timeresp_aliases, sigval=0.)
    X0 = _process_param(
        'initial_state', initial_state, kwargs, _timeresp_aliases, sigval=0.)
    return_x = _process_param(
        'return_states', return_states, kwargs, _timeresp_aliases,
        sigval=False)
    # TODO: replace default value of evaluation_times with None?
    t_eval = _process_param(
        'evaluation_times', evaluation_times, kwargs, _timeresp_aliases,
        sigval='T')

    # Figure out the method to be used
    solve_ivp_kwargs = solve_ivp_kwargs.copy() if solve_ivp_kwargs else {}
    if kwargs.get('solve_ivp_method', None):
        if kwargs.get('method', None):
            raise ValueError("ivp_method specified more than once")
        solve_ivp_kwargs['method'] = kwargs.pop('solve_ivp_method')
    elif kwargs.get('method', None):
        # Allow method as an alternative to solve_ivp_method
        solve_ivp_kwargs['method'] = kwargs.pop('method')

    # Set the default method to 'RK45'
    if solve_ivp_kwargs.get('method', None) is None:
        solve_ivp_kwargs['method'] = 'RK45'

    # Make sure there were no extraneous keywords
    if kwargs:
        raise TypeError("unrecognized keyword(s): ", str(kwargs))

    # If passed a list, recursively call individual responses with given T
    if isinstance(sys, (list, tuple)):
        sysdata, responses = sys, []
        for sys in sysdata:
            responses.append(input_output_response(
                sys, timepts=T, inputs=U, initial_state=X0, params=params,
                transpose=transpose, return_states=return_x, squeeze=squeeze,
                evaluation_times=t_eval, solve_ivp_kwargs=solve_ivp_kwargs,
                **kwargs))
        return TimeResponseList(responses)

    # Sanity checking on the input
    if not isinstance(sys, NonlinearIOSystem):
        raise TypeError("System of type ", type(sys), " not valid")

    # Compute the time interval and number of steps
    T0, Tf = T[0], T[-1]
    ntimepts = len(T)

    # Figure out simulation times (t_eval)
    if solve_ivp_kwargs.get('t_eval'):
        if t_eval == 'T':
            # Override the default with the solve_ivp keyword
            t_eval = solve_ivp_kwargs.pop('t_eval')
        else:
            raise ValueError("t_eval specified more than once")
    if isinstance(t_eval, str) and t_eval == 'T':
        # Use the input time points as the output time points
        t_eval = T

    #
    # Process input argument
    #
    # The input argument is interpreted very flexibly, allowing the
    # use of lists and/or tuples of mixed scalar and vector elements.
    #
    # Much of the processing here is similar to the processing in
    # _process_vector_argument, but applied to a time series.

    # If we were passed a list of inputs, concatenate them (w/ broadcast)
    if isinstance(U, (tuple, list)) and len(U) != ntimepts:
        U_elements = []
        for i, u in enumerate(U):
            u = np.array(u)     # convert everything to an array
            # Process this input
            if u.ndim == 0 or (u.ndim == 1 and u.shape[0] != T.shape[0]):
                # Broadcast array to the length of the time input
                u = np.outer(u, np.ones_like(T))

            elif (u.ndim == 1 and u.shape[0] == T.shape[0]) or \
                 (u.ndim == 2 and u.shape[1] == T.shape[0]):
                # No processing necessary; just stack
                pass

            else:
                raise ValueError(f"Input element {i} has inconsistent shape")

            # Append this input to our list
            U_elements.append(u)

        # Save the newly created input vector
        U = np.vstack(U_elements)

    # Figure out the number of inputs
    if sys.ninputs is None:
        if isinstance(U, np.ndarray):
            ninputs = U.shape[0] if U.size > 1 else U.size
        else:
            ninputs = 1
    else:
        ninputs = sys.ninputs

    # Make sure the input has the right shape
    if ninputs is None or ninputs == 1:
        legal_shapes = [(ntimepts,), (1, ntimepts)]
    else:
        legal_shapes = [(ninputs, ntimepts)]

    U = _check_convert_array(
        U, legal_shapes, 'Parameter `U`: ', squeeze=False)

    # Always store the input as a 2D array
    U = U.reshape(-1, ntimepts)
    ninputs = U.shape[0]

    # Process initial states
    X0, nstates = _process_vector_argument(X0, "X0", sys.nstates)

    # Update the parameter values (prior to evaluating outfcn)
    sys._update_params(params)

    # Figure out the number of outputs
    if sys.outfcn is None:
        noutputs = nstates if sys.noutputs is None else sys.noutputs
    else:
        noutputs = np.shape(sys._out(T[0], X0, U[:, 0]))[0]

    if sys.noutputs is not None and sys.noutputs != noutputs:
        raise RuntimeError(
            f"inconsistent size of outputs; system specified {sys.noutputs}, "
            f"output function returned {noutputs}")

    #
    # Define a function to evaluate the input at an arbitrary time
    #
    # This is equivalent to the function
    #
    #   ufun = sp.interpolate.interp1d(T, U, fill_value='extrapolate')
    #
    # but has a lot less overhead => simulation runs much faster
    def ufun(t):
        # Find the value of the index using linear interpolation
        # Use clip to allow for extrapolation if t is out of range
        idx = np.clip(np.searchsorted(T, t, side='left'), 1, len(T)-1)
        dt = (t - T[idx-1]) / (T[idx] - T[idx-1])
        return U[..., idx-1] * (1. - dt) + U[..., idx] * dt

    # Check to make sure see if this is a static function
    if sys.nstates == 0:
        # Make sure the user gave a time vector for evaluation (or 'T')
        if t_eval is None:
            # User overrode t_eval with None, but didn't give us the times...
            warn("t_eval set to None, but no dynamics; using T instead")
            t_eval = T

        # Allocate space for the inputs and outputs
        u = np.zeros((ninputs, len(t_eval)))
        y = np.zeros((noutputs, len(t_eval)))

        # Compute the input and output at each point in time
        for i, t in enumerate(t_eval):
            u[:, i] = ufun(t)
            y[:, i] = sys._out(t, [], u[:, i])

        return TimeResponseData(
            t_eval, y, None, u, issiso=sys.issiso(),
            output_labels=sys.output_labels, input_labels=sys.input_labels,
            title="Input/output response for " + sys.name, sysname=sys.name,
            transpose=transpose, return_x=return_x, squeeze=squeeze)

    # Create a lambda function for the right hand side
    def ivp_rhs(t, x):
        return sys._rhs(t, x, ufun(t))

    # Perform the simulation
    if isctime(sys):
        if not hasattr(sp.integrate, 'solve_ivp'):
            raise NameError("scipy.integrate.solve_ivp not found; "
                            "use SciPy 1.0 or greater")
        soln = sp.integrate.solve_ivp(
            ivp_rhs, (T0, Tf), X0, t_eval=t_eval,
            vectorized=False, **solve_ivp_kwargs)
        if not soln.success:
            message = "solve_ivp failed: " + soln.message
            if not ignore_errors:
                raise RuntimeError(message)
        else:
            message = None

        # Compute inputs and outputs for each time point
        u = np.zeros((ninputs, len(soln.t)))
        y = np.zeros((noutputs, len(soln.t)))
        for i, t in enumerate(soln.t):
            u[:, i] = ufun(t)
            y[:, i] = sys._out(t, soln.y[:, i], u[:, i])

    elif isdtime(sys):
        # If t_eval was not specified, use the sampling time
        if t_eval is None:
            t_eval = np.arange(T[0], T[1] + sys.dt, sys.dt)

        # Make sure the time vector is uniformly spaced
        dt = t_eval[1] - t_eval[0]
        if not np.allclose(t_eval[1:] - t_eval[:-1], dt):
            raise ValueError("parameter `t_eval`: time values must be "
                             "equally spaced")

        # Make sure the sample time matches the given time
        if sys.dt is not True:
            # Make sure that the time increment is a multiple of sampling time

            # TODO: add back functionality for undersampling
            # TODO: this test is brittle if dt =  sys.dt
            # First make sure that time increment is bigger than sampling time
            # if dt < sys.dt:
            #     raise ValueError("Time steps `T` must match sampling time")

            # Check to make sure sampling time matches time increments
            if not np.isclose(dt, sys.dt):
                raise ValueError("Time steps `T` must be equal to "
                                 "sampling time")

        # Compute the solution
        soln = sp.optimize.OptimizeResult()
        soln.t = t_eval                 # Store the time vector directly
        x = np.array(X0)                # State vector (store as floats)
        soln.y = []                     # Solution, following scipy convention
        u, y = [], []                   # System input, output
        for t in t_eval:
            # Store the current input, state, and output
            soln.y.append(x)
            u.append(ufun(t))
            y.append(sys._out(t, x, u[-1]))

            # Update the state for the next iteration
            x = sys._rhs(t, x, u[-1])

        # Convert output to numpy arrays
        soln.y = np.transpose(np.array(soln.y))
        y = np.transpose(np.array(y))
        u = np.transpose(np.array(u))

        # Mark solution as successful
        soln.success, message = True, None      # No way to fail

    else:                       # Neither ctime or dtime??
        raise TypeError("Can't determine system type")

    return TimeResponseData(
        soln.t, y, soln.y, u, params=params, issiso=sys.issiso(),
        output_labels=sys.output_labels, input_labels=sys.input_labels,
        state_labels=sys.state_labels, sysname=sys.name,
        title="Input/output response for " + sys.name,
        transpose=transpose, return_x=return_x, squeeze=squeeze,
        success=soln.success, message=message)


class OperatingPoint():
    """Operating point of nonlinear I/O system.

    The OperatingPoint class stores the operating point of a nonlinear
    system, consisting of the state and input vectors for the system.  The
    main use for this class is as the return object for the
    `find_operating_point` function and as an input to the
    `linearize` function.

    Parameters
    ----------
    states : array
        State vector at the operating point.
    inputs : array
        Input vector at the operating point.
    outputs : array, optional
        Output vector at the operating point.
    result : `scipy.optimize.OptimizeResult`, optional
        Result from the `scipy.optimize.root` function, if available.
    return_outputs, return_result : bool, optional
        If set to True, then when accessed a tuple the output values
        and/or result of the root finding function will be returned.

    Notes
    -----
    In addition to accessing the elements of the operating point as
    attributes, if accessed as a list then the object will return ``(x0,
    u0[, y0, res])``, where `y0` and `res` are returned depending on the
    `return_outputs` and `return_result` parameters.

    """
    def __init__(
            self, states, inputs, outputs=None, result=None,
            return_outputs=False, return_result=False):
        self.states = states
        self.inputs = inputs

        if outputs is None and return_outputs and not return_result:
            raise ValueError("return_outputs specified but no outputs value")
        self.outputs = outputs
        self.return_outputs = return_outputs

        if result is None and return_result:
            raise ValueError("return_result specified but no result value")
        self.result = result
        self.return_result = return_result

    # Implement iter to allow assigning to a tuple
    def __iter__(self):
        if self.return_outputs and self.return_result:
            return iter((self.states, self.inputs, self.outputs, self.result))
        elif self.return_outputs:
            return iter((self.states, self.inputs, self.outputs))
        elif self.return_result:
            return iter((self.states, self.inputs, self.result))
        else:
            return iter((self.states, self.inputs))

    # Implement (thin) getitem to allow access via legacy indexing
    def __getitem__(self, index):
        return list(self.__iter__())[index]

    # Implement (thin) len to emulate legacy return value
    def __len__(self):
        return len(list(self.__iter__()))


def find_operating_point(
        sys, initial_state=0., inputs=None, outputs=None, t=0, params=None,
        input_indices=None, output_indices=None, state_indices=None,
        deriv_indices=None, derivs=None, root_method=None, root_kwargs=None,
        return_outputs=None, return_result=None, **kwargs):
    """Find an operating point for an input/output system.

    An operating point for a nonlinear system is a state and input around
    which a nonlinear system operates.  This point is most commonly an
    equilibrium point for the system, but in some cases a non-equilibrium
    operating point can be used.

    This function attempts to find an operating point given a specification
    for the desired inputs, outputs, states, or state updates of the system.

    In its simplest form, `find_operating_point` finds an equilibrium point
    given either the desired input or desired output::

        xeq, ueq = find_operating_point(sys, x0, u0)
        xeq, ueq = find_operating_point(sys, x0, u0, y0)

    The first form finds an equilibrium point for a given input u0 based on
    an initial guess x0.  The second form fixes the desired output values
    and uses x0 and u0 as an initial guess to find the equilibrium point.
    If no equilibrium point can be found, the function returns the
    operating point that minimizes the state update (state derivative for
    continuous-time systems, state difference for discrete-time systems).

    More complex operating points can be found by specifying which states,
    inputs, or outputs should be used in computing the operating point, as
    well as desired values of the states, inputs, outputs, or state
    updates.

    Parameters
    ----------
    sys : `NonlinearIOSystem`
        I/O system for which the operating point is sought.
    initial_state (or x0) : list of initial state values
        Initial guess for the value of the state near the operating point.
    inputs (or u0) : list of input values, optional
        If `y0` is not specified, sets the value of the input.  If `y0` is
        given, provides an initial guess for the value of the input.  Can
        be omitted if the system does not have any inputs.
    outputs (or y0) : list of output values, optional
        If specified, sets the desired values of the outputs at the
        operating point.
    t : float, optional
        Evaluation time, for time-varying systems.
    params : dict, optional
        Parameter values for the system.  Passed to the evaluation functions
        for the system as default values, overriding internal defaults.
    input_indices (or iu) : list of input indices, optional
        If specified, only the inputs with the given indices will be fixed at
        the specified values in solving for an operating point.  All other
        inputs will be varied.  Input indices can be listed in any order.
    output_indices (or iy) : list of output indices, optional
        If specified, only the outputs with the given indices will be fixed
        at the specified values in solving for an operating point.  All other
        outputs will be varied.  Output indices can be listed in any order.
    state_indices (or ix) : list of state indices, optional
        If specified, states with the given indices will be fixed at the
        specified values in solving for an operating point.  All other
        states will be varied.  State indices can be listed in any order.
    derivs (or dx0) : list of update values, optional
        If specified, the value of update map must match the listed value
        instead of the default value for an equilibrium point.
    deriv_indices (or idx) : list of state indices, optional
        If specified, state updates with the given indices will have their
        update maps fixed at the values given in `dx0`.  All other update
        values will be ignored in solving for an operating point.  State
        indices can be listed in any order.  By default, all updates will be
        fixed at `dx0` in searching for an operating point.
    root_method : str, optional
        Method to find the operating point.  If specified, this parameter
        is passed to the `scipy.optimize.root` function.
    root_kwargs : dict, optional
        Additional keyword arguments to pass `scipy.optimize.root`.
    return_outputs : bool, optional
        If True, return the value of outputs at the operating point.
    return_result : bool, optional
        If True, return the `result` option from the
        `scipy.optimize.root` function used to compute the
        operating point.

    Returns
    -------
    op_point : `OperatingPoint`
        The solution represented as an `OperatingPoint` object.  The main
        attributes are `states` and `inputs`, which represent the state and
        input arrays at the operating point.  If accessed as a tuple, returns
        `states`, `inputs`, and optionally `outputs` and `result` based on the
        `return_outputs` and `return_result` parameters.  See `OperatingPoint`
        for a description of other attributes.
    op_point.states : array
        State vector at the operating point.
    op_point.inputs : array
        Input vector at the operating point.
    op_point.outputs : array, optional
        Output vector at the operating point.

    Notes
    -----
    For continuous-time systems, equilibrium points are defined as points
    for which the right hand side of the differential equation is zero:
    :math:`f(t, x_e, u_e) = 0`. For discrete-time systems, equilibrium
    points are defined as points for which the right hand side of the
    difference equation returns the current state: :math:`f(t, x_e, u_e) =
    x_e`.

    Operating points are found using the `scipy.optimize.root`
    function, which will attempt to find states and inputs that satisfy the
    specified constraints.  If no solution is found and `return_result` is
    False, the returned state and input for the operating point will be
    None.  If `return_result` is True, then the return values from
    `scipy.optimize.root` will be returned (but may not be valid).
    If `root_method` is set to 'lm', then the least squares solution (in
    the free variables) will be returned.

    """
    from scipy.optimize import root

    # Process keyword arguments
    aliases = {
        'initial_state': (['x0', 'X0'], []),
        'inputs': (['u0'], []),
        'outputs': (['y0'], []),
        'derivs': (['dx0'], []),
        'input_indices': (['iu'], []),
        'output_indices': (['iy'], []),
        'state_indices': (['ix'], []),
        'deriv_indices': (['idx'], []),
        'return_outputs': ([], ['return_y']),
    }
    _process_kwargs(kwargs, aliases)
    x0 = _process_param(
        'initial_state', initial_state, kwargs, aliases, sigval=0.)
    u0 = _process_param('inputs', inputs, kwargs, aliases)
    y0 = _process_param('outputs', outputs, kwargs, aliases)
    dx0 = _process_param('derivs', derivs, kwargs, aliases)
    iu = _process_param('input_indices', input_indices, kwargs, aliases)
    iy = _process_param('output_indices', output_indices, kwargs, aliases)
    ix = _process_param('state_indices', state_indices, kwargs, aliases)
    idx = _process_param('deriv_indices', deriv_indices, kwargs, aliases)
    return_outputs = _process_param(
        'return_outputs', return_outputs, kwargs, aliases)
    if kwargs:
        raise TypeError("unrecognized keyword(s): " + str(kwargs))

    # Process arguments for the root function
    root_kwargs = dict() if root_kwargs is None else root_kwargs
    if root_method:
        root_kwargs['method'] = root_method

    # Figure out the number of states, inputs, and outputs
    x0, nstates = _process_vector_argument(x0, "initial_states", sys.nstates)
    u0, ninputs = _process_vector_argument(u0, "inputs", sys.ninputs)
    y0, noutputs = _process_vector_argument(y0, "outputs", sys.noutputs)

    # Make sure the input arguments match the sizes of the system
    if len(x0) != nstates or \
       (u0 is not None and len(u0) != ninputs) or \
       (y0 is not None and len(y0) != noutputs) or \
       (dx0 is not None and len(dx0) != nstates):
        raise ValueError("length of input arguments does not match system")

    # Update the parameter values
    sys._update_params(params)

    # Decide what variables to minimize
    if all([x is None for x in (iu, iy, ix, idx)]):
        # Special cases: either inputs or outputs are constrained
        if y0 is None:
            # Take u0 as fixed and minimize over x
            if sys.isdtime(strict=True):
                def state_rhs(z): return sys._rhs(t, z, u0) - z
            else:
                def state_rhs(z): return sys._rhs(t, z, u0)

            result = root(state_rhs, x0, **root_kwargs)
            z = (result.x, u0, sys._out(t, result.x, u0))

        else:
            # Take y0 as fixed and minimize over x and u
            if sys.isdtime(strict=True):
                def rootfun(z):
                    x, u = np.split(z, [nstates])
                    return np.concatenate(
                        (sys._rhs(t, x, u) - x, sys._out(t, x, u) - y0),
                        axis=0)
            else:
                def rootfun(z):
                    x, u = np.split(z, [nstates])
                    return np.concatenate(
                        (sys._rhs(t, x, u), sys._out(t, x, u) - y0), axis=0)

            # Find roots with (x, u) as free variables
            z0 = np.concatenate((x0, u0), axis=0)
            result = root(rootfun, z0, **root_kwargs)
            x, u = np.split(result.x, [nstates])
            z = (x, u, sys._out(t, x, u))

    else:
        # General case: figure out what variables to constrain
        # Verify the indices we are using are all in range
        if iu is not None:
            iu = np.unique(iu)
            if any([not isinstance(x, int) for x in iu]) or \
               (len(iu) > 0 and (min(iu) < 0 or max(iu) >= ninputs)):
                assert ValueError("One or more input indices is invalid")
        else:
            iu = []

        if iy is not None:
            iy = np.unique(iy)
            if any([not isinstance(x, int) for x in iy]) or \
               min(iy) < 0 or max(iy) >= noutputs:
                assert ValueError("One or more output indices is invalid")
        else:
            iy = list(range(noutputs))

        if ix is not None:
            ix = np.unique(ix)
            if any([not isinstance(x, int) for x in ix]) or \
               min(ix) < 0 or max(ix) >= nstates:
                assert ValueError("One or more state indices is invalid")
        else:
            ix = []

        if idx is not None:
            idx = np.unique(idx)
            if any([not isinstance(x, int) for x in idx]) or \
               min(idx) < 0 or max(idx) >= nstates:
                assert ValueError("One or more deriv indices is invalid")
        else:
            idx = list(range(nstates))

        # Construct the index lists for mapping variables and constraints
        #
        # The mechanism by which we implement the root finding function is
        # to map the subset of variables we are searching over into the
        # inputs and states, and then return a function that represents the
        # equations we are trying to solve.
        #
        # To do this, we need to carry out the following operations:
        #
        # 1. Given the current values of the free variables (z), map them into
        #    the portions of the state and input vectors that are not fixed.
        #
        # 2. Compute the update and output maps for the input/output system
        #    and extract the subset of equations that should be equal to zero.
        #
        # We perform these functions by computing four sets of index lists:
        #
        # * state_vars: indices of states that are allowed to vary
        # * input_vars: indices of inputs that are allowed to vary
        # * deriv_vars: indices of derivatives that must be constrained
        # * output_vars: indices of outputs that must be constrained
        #
        # This index lists can all be precomputed based on the `iu`, `iy`,
        # `ix`, and `idx` lists that were passed as arguments to
        # `find_operating_point` and were processed above.

        # Get the states and inputs that were not listed as fixed
        state_vars = (range(nstates) if not len(ix)
                      else np.delete(np.array(range(nstates)), ix))
        input_vars = (range(ninputs) if not len(iu)
                      else np.delete(np.array(range(ninputs)), iu))

        # Set the outputs and derivs that will serve as constraints
        output_vars = np.array(iy)
        deriv_vars = np.array(idx)

        # Verify that the number of degrees of freedom all add up correctly
        num_freedoms = len(state_vars) + len(input_vars)
        num_constraints = len(output_vars) + len(deriv_vars)
        if num_constraints != num_freedoms:
            warn("number of constraints (%d) does not match number of degrees "
                 "of freedom (%d); results may be meaningless" %
                 (num_constraints, num_freedoms))

        # Make copies of the state and input variables to avoid overwriting
        # and convert to floats (in case ints were used for initial conditions)
        x = np.array(x0, dtype=float)
        u = np.array(u0, dtype=float)
        dx0 = np.array(dx0, dtype=float) if dx0 is not None \
            else np.zeros(x.shape)

        # Keep track of the number of states in the set of free variables
        nstate_vars = len(state_vars)

        def rootfun(z):
            # Map the vector of values into the states and inputs
            x[state_vars] = z[:nstate_vars]
            u[input_vars] = z[nstate_vars:]

            # Compute the update and output maps
            dx = sys._rhs(t, x, u) - dx0
            if sys.isdtime(strict=True):
                dx -= x

            # If no y0 is given, don't evaluate the output function
            if y0 is None:
                return dx[deriv_vars]
            else:
                dy = sys._out(t, x, u) - y0

                # Map the results into the constrained variables
                return np.concatenate((dx[deriv_vars], dy[output_vars]), axis=0)

        # Set the initial condition for the root finding algorithm
        z0 = np.concatenate((x[state_vars], u[input_vars]), axis=0)

        # Finally, call the root finding function
        result = root(rootfun, z0, **root_kwargs)

        # Extract out the results and insert into x and u
        x[state_vars] = result.x[:nstate_vars]
        u[input_vars] = result.x[nstate_vars:]
        z = (x, u, sys._out(t, x, u))

    # Return the result based on what the user wants and what we found
    if return_result or result.success:
        return OperatingPoint(
            z[0], z[1], z[2], result, return_outputs, return_result)
    else:
        # Something went wrong, don't return anything
        return OperatingPoint(
            None, None, None, result, return_outputs, return_result)

    # TODO: remove code when ready
    if not return_outputs:
        z = z[0:2]              # Strip y from result if not desired
    if return_result:
        # Return whatever we got, along with the result dictionary
        return z + (result,)
    elif result.success:
        # Return the result of the optimization
        return z
    else:
        # Something went wrong, don't return anything
        return (None, None, None) if return_outputs else (None, None)


# Linearize an input/output system
def linearize(sys, xeq, ueq=None, t=0, params=None, **kw):
    """Linearize an input/output system at a given state and input.

    Compute the linearization of an I/O system at an operating point (state
    and input) and returns a `StateSpace` object.  The
    operating point need not be an equilibrium point.

    Parameters
    ----------
    sys : `InputOutputSystem`
        The system to be linearized.
    xeq : array or `OperatingPoint`
        The state or operating point at which the linearization will be
        evaluated (does not need to be an equilibrium state).
    ueq : array, optional
        The input at which the linearization will be evaluated (does not need
        to correspond to an equilibrium state).  Can be omitted if `xeq` is
        an `OperatingPoint`.  Defaults to 0.
    t : float, optional
        The time at which the linearization will be computed (for time-varying
        systems).
    params : dict, optional
        Parameter values for the systems.  Passed to the evaluation functions
        for the system as default values, overriding internal defaults.
    name : string, optional
        Set the name of the linearized system.  If not specified and
        if `copy_names` is False, a generic name 'sys[id]' is generated
        with a unique integer id.  If `copy_names` is True, the new system
        name is determined by adding the prefix and suffix strings in
        `config.defaults['iosys.linearized_system_name_prefix']` and
        `config.defaults['iosys.linearized_system_name_suffix']`, with the
        default being to add the suffix '$linearized'.
    copy_names : bool, Optional
        If True, Copy the names of the input signals, output signals, and
        states to the linearized system.

    Returns
    -------
    ss_sys : `StateSpace`
        The linearization of the system, as a `StateSpace`
        object.

    Other Parameters
    ----------------
    inputs : int, list of str or None, optional
        Description of the system inputs.  If not specified, the original
        system inputs are used.  See `InputOutputSystem` for more
        information.
    outputs : int, list of str or None, optional
        Description of the system outputs.  Same format as `inputs`.
    states : int, list of str, or None, optional
        Description of the system states.  Same format as `inputs`.

    """
    if not isinstance(sys, InputOutputSystem):
        raise TypeError("Can only linearize InputOutputSystem types")
    return sys.linearize(xeq, ueq, t=t, params=params, **kw)


def _find_size(sysval, vecval, name="system component"):
    """Utility function to find the size of a system parameter

    If both parameters are not None, they must be consistent.
    """
    if hasattr(vecval, '__len__'):
        if sysval is not None and sysval != len(vecval):
            raise ValueError(
                f"inconsistent information to determine size of {name}; "
                f"expected {sysval} values, received {len(vecval)}")
        return len(vecval)
    # None or 0, which is a valid value for "a (sysval, ) vector of zeros".
    if not vecval:
        return 0 if sysval is None else sysval
    elif sysval == 1:
        # (1, scalar) is also a valid combination from legacy code
        return 1
    raise ValueError(f"can't determine size of {name}")


# Function to create an interconnected system
def interconnect(
        syslist, connections=None, inplist=None, outlist=None, params=None,
        check_unused=True, add_unused=False, ignore_inputs=None,
        ignore_outputs=None, warn_duplicate=None, debug=False, **kwargs):
    """Interconnect a set of input/output systems.

    This function creates a new system that is an interconnection of a set of
    input/output systems.  If all of the input systems are linear I/O systems
    (type `StateSpace`) then the resulting system will be
    a linear interconnected I/O system (type `LinearICSystem`)
    with the appropriate inputs, outputs, and states.  Otherwise, an
    interconnected I/O system (type `InterconnectedSystem`)
    will be created.

    Parameters
    ----------
    syslist : list of `NonlinearIOSystem`
        The list of (state-based) input/output systems to be connected.

    connections : list of connections, optional
        Description of the internal connections between the subsystems::

            [connection1, connection2, ...]

        Each connection is itself a list that describes an input to one of
        the subsystems.  The entries are of the form::

            [input-spec, output-spec1, output-spec2, ...]

        The input-spec can be in a number of different forms.  The lowest
        level representation is a tuple of the form ``(subsys_i, inp_j)``
        where `subsys_i` is the index into `syslist` and `inp_j` is the
        index into the input vector for the subsystem.  If the signal index
        is omitted, then all subsystem inputs are used.  If systems and
        signals are given names, then the forms 'sys.sig' or ('sys', 'sig')
        are also recognized.  Finally, for multivariable systems the signal
        index can be given as a list, for example '(subsys_i, [inp_j1, ...,
        inp_jn])'; or as a slice, for example, 'sys.sig[i:j]'; or as a base
        name 'sys.sig' (which matches 'sys.sig[i]').

        Similarly, each output-spec should describe an output signal from
        one of the subsystems.  The lowest level representation is a tuple
        of the form ``(subsys_i, out_j, gain)``.  The input will be
        constructed by summing the listed outputs after multiplying by the
        gain term.  If the gain term is omitted, it is assumed to be 1.  If
        the subsystem index 'subsys_i' is omitted, then all outputs of the
        subsystem are used.  If systems and signals are given names, then
        the form 'sys.sig', ('sys', 'sig') or ('sys', 'sig', gain) are also
        recognized, and the special form '-sys.sig' can be used to specify
        a signal with gain -1.  Lists, slices, and base names can also be
        used, as long as the number of elements for each output spec
        matches the input spec.

        If omitted, the `interconnect` function will attempt to create the
        interconnection map by connecting all signals with the same base
        names (ignoring the system name).  Specifically, for each input
        signal name in the list of systems, if that signal name corresponds
        to the output signal in any of the systems, it will be connected to
        that input (with a summation across all signals if the output name
        occurs in more than one system).

        The `connections` keyword can also be set to False, which will leave
        the connection map empty and it can be specified instead using the
        low-level `InterconnectedSystem.set_connect_map`
        method.

    inplist : list of input connections, optional
        List of connections for how the inputs for the overall system are
        mapped to the subsystem inputs.  The entries for a connection are
        of the form::

            [input-spec1, input-spec2, ...]

        Each system input is added to the input for the listed subsystem.
        If the system input connects to a subsystem with a single input, a
        single input specification can be given (without the inner list).

        If omitted the `input` parameter will be used to identify the list
        of input signals to the overall system.

    outlist : list of output connections, optional
        List of connections for how the outputs from the subsystems are
        mapped to overall system outputs.  The entries for a connection are
        of the form::

            [output-spec1, output-spec2, ...]

        If an output connection contains more than one signal specification,
        then those signals are added together (multiplying by the any gain
        term) to form the system output.

        If omitted, the output map can be specified using the
        `InterconnectedSystem.set_output_map` method.

    inputs : int, list of str or None, optional
        Description of the system inputs.  This can be given as an integer
        count or as a list of strings that name the individual signals.  If
        an integer count is specified, the names of the signal will be of
        the form 's[i]' (where 's' is one of 'u', 'y', or 'x').  If this
        parameter is not given or given as None, the relevant quantity will
        be determined when possible based on other information provided to
        functions using the system.

    outputs : int, list of str or None, optional
        Description of the system outputs.  Same format as `inputs`.

    states : int, list of str, or None, optional
        Description of the system states.  Same format as `inputs`. The
        default is None, in which case the states will be given names of
        the form '<subsys_name><delim><state_name>', for each subsys in
        syslist and each state_name of each subsys, where <delim> is the
        value of `config.defaults['iosys.state_name_delim']`.

    params : dict, optional
        Parameter values for the systems.  Passed to the evaluation functions
        for the system as default values, overriding internal defaults.  If
        not specified, defaults to parameters from subsystems.

    dt : timebase, optional
        The timebase for the system, used to specify whether the system is
        operating in continuous or discrete-time.  It can have the following
        values:

        * `dt` = 0: continuous-time system (default)
        * `dt` > 0`: discrete-time system with sampling period `dt`
        * `dt` = True: discrete time with unspecified sampling period
        * `dt` = None: no timebase specified

    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.

    Returns
    -------
    sys : `InterconnectedSystem`
        `NonlinearIOSystem` consisting of the interconnected subsystems.

    Other Parameters
    ----------------
    input_prefix, output_prefix, state_prefix : string, optional
        Set the prefix for input, output, and state signals.  Defaults =
        'u', 'y', 'x'.

    check_unused : bool, optional
        If True, check for unused sub-system signals.  This check is
        not done if connections is False, and neither input nor output
        mappings are specified.

    add_unused : bool, optional
        If True, subsystem signals that are not connected to other components
        are added as inputs and outputs of the interconnected system.

    ignore_inputs : list of input-spec, optional
        A list of sub-system inputs known not to be connected.  This is
        *only* used in checking for unused signals, and does not
        disable use of the input.

        Besides the usual input-spec forms (see `connections`), an
        input-spec can be just the signal base name, in which case all
        signals from all sub-systems with that base name are
        considered ignored.

    ignore_outputs : list of output-spec, optional
        A list of sub-system outputs known not to be connected.  This
        is *only* used in checking for unused signals, and does not
        disable use of the output.

        Besides the usual output-spec forms (see `connections`), an
        output-spec can be just the signal base name, in which all
        outputs from all sub-systems with that base name are
        considered ignored.

    warn_duplicate : None, True, or False, optional
        Control how warnings are generated if duplicate objects or names are
        detected.  In None (default), then warnings are generated for
        systems that have non-generic names.  If False, warnings are not
        generated and if True then warnings are always generated.

    debug : bool, default=False
        Print out information about how signals are being processed that
        may be useful in understanding why something is not working.

    Examples
    --------
    >>> P = ct.rss(2, 2, 2, strictly_proper=True, name='P')
    >>> C = ct.rss(2, 2, 2, name='C')
    >>> T = ct.interconnect(
    ...     [P, C],
    ...     connections=[
    ...         ['P.u[0]', 'C.y[0]'], ['P.u[1]', 'C.y[1]'],
    ...         ['C.u[0]', '-P.y[0]'], ['C.u[1]', '-P.y[1]']],
    ...     inplist=['C.u[0]', 'C.u[1]'],
    ...     outlist=['P.y[0]', 'P.y[1]'],
    ... )

    This expression can be simplified using either slice notation or
    just signal basenames:

    >>> T = ct.interconnect(
    ...     [P, C], connections=[['P.u[:]', 'C.y[:]'], ['C.u', '-P.y']],
    ...     inplist='C.u', outlist='P.y[:]')

    or further simplified by omitting the input and output signal
    specifications (since all inputs and outputs are used):

    >>> T = ct.interconnect(
    ...     [P, C], connections=[['P', 'C'], ['C', '-P']],
    ...     inplist=['C'], outlist=['P'])

    A feedback system can also be constructed using the
    `summing_junction` function and the ability to
    automatically interconnect signals with the same names:

    >>> P = ct.tf(1, [1, 0], inputs='u', outputs='y')
    >>> C = ct.tf(10, [1, 1], inputs='e', outputs='u')
    >>> sumblk = ct.summing_junction(inputs=['r', '-y'], output='e')
    >>> T = ct.interconnect([P, C, sumblk], inputs='r', outputs='y')

    Notes
    -----
    If a system is duplicated in the list of systems to be connected,
    a warning is generated and a copy of the system is created with the
    name of the new system determined by adding the prefix and suffix
    strings in `config.defaults['iosys.duplicate_system_name_prefix']`
    and `config.defaults['iosys.duplicate_system_name_suffix']`, with the
    default being to add the suffix '$copy' to the system name.

    In addition to explicit lists of system signals, it is possible to
    lists vectors of signals, using one of the following forms::

      (subsys, [i1, ..., iN], gain)   # signals with indices i1, ..., in
      'sysname.signal[i:j]'           # range of signal names, i through j-1
      'sysname.signal[:]'             # all signals with given prefix

    While in many Python functions tuples can be used in place of lists,
    for the interconnect() function the only use of tuples should be in the
    specification of an input- or output-signal via the tuple notation
    ``(subsys_i, signal_j, gain)`` (where `gain` is optional).  If you get an
    unexpected error message about a specification being of the wrong type
    or not being found, check to make sure you are not using a tuple where
    you should be using a list.

    In addition to its use for general nonlinear I/O systems, the
    `interconnect` function allows linear systems to be
    interconnected using named signals (compared with the
    legacy `connect` function, which uses signal indices) and to be
    treated as both a `StateSpace` system as well as an
    `InputOutputSystem`.

    The `input` and `output` keywords can be used instead of `inputs` and
    `outputs`, for more natural naming of SISO systems.

    """
    from .statesp import LinearICSystem, StateSpace

    dt = kwargs.pop('dt', None)         # bypass normal 'dt' processing
    name, inputs, outputs, states, _ = _process_iosys_keywords(kwargs)
    connection_type = None # explicit, implicit, or None

    if not check_unused and (ignore_inputs or ignore_outputs):
        raise ValueError('check_unused is False, but either '
                         + 'ignore_inputs or ignore_outputs non-empty')

    if connections is False and not any((inplist, outlist, inputs, outputs)):
        # user has disabled auto-connect, and supplied neither input
        # nor output mappings; assume they know what they're doing
        check_unused = False

    # If connections was not specified, assume implicit interconnection.
    # set up default connection list
    if connections is None:
        connection_type = 'implicit'
        # For each system input, look for outputs with the same name
        connections = []
        for input_sys in syslist:
            for input_name in input_sys.input_labels:
                connect = [input_sys.name + "." + input_name]
                for output_sys in syslist:
                    if input_name in output_sys.output_labels:
                        connect.append(output_sys.name + "." + input_name)
                if len(connect) > 1:
                    connections.append(connect)

    elif connections is False:
        check_unused = False
        # Use an empty connections list
        connections = []

    else:
        connection_type = 'explicit'
        if isinstance(connections, list) and \
                all([isinstance(cnxn, (str, tuple)) for cnxn in connections]):
            # Special case where there is a single connection
            connections = [connections]

    # If inplist/outlist is not present, try using inputs/outputs instead
    inplist_none, outlist_none = False, False
    if inplist is None:
        inplist = inputs or []
        inplist_none = True     # use to rewrite inputs below
    if outlist is None:
        outlist = outputs or []
        outlist_none = True     # use to rewrite outputs below

    # Define a local debugging function
    dprint = lambda s: None if not debug else print(s)

    #
    # Pre-process connecton list
    #
    # Support for various "vector" forms of specifications is handled here,
    # by expanding any specifications that refer to more than one signal.
    # This includes signal lists such as ('sysname', ['sig1', 'sig2', ...])
    # as well as slice-based specifications such as 'sysname.signal[i:j]'.
    #
    dprint("Pre-processing connections:")
    new_connections = []
    for connection in connections:
        dprint(f"  parsing {connection=}")
        if not isinstance(connection, list):
            raise ValueError(
                f"invalid connection {connection}: should be a list")
        # Parse and expand the input specification
        input_spec = _parse_spec(syslist, connection[0], 'input')
        input_spec_list = [input_spec]

        # Parse and expand the output specifications
        output_specs_list = [[]] * len(input_spec_list)
        for spec in connection[1:]:
            output_spec = _parse_spec(syslist, spec, 'output')
            output_specs_list[0].append(output_spec)

        # Create the new connection entry
        for input_spec, output_specs in zip(input_spec_list, output_specs_list):
            new_connection = [input_spec] + output_specs
            dprint(f"    adding {new_connection=}")
            new_connections.append(new_connection)
    connections = new_connections

    #
    # Pre-process input connections list
    #
    # Similar to the connections list, we now handle "vector" forms of
    # specifications in the inplist parameter.  This needs to be handled
    # here because the InterconnectedSystem constructor assumes that the
    # number of elements in `inplist` will match the number of inputs for
    # the interconnected system.
    #
    # If inplist_none is True then inplist is a copy of inputs and so we
    # also have to be careful that if we encounter any multivariable
    # signals, we need to update the input list.
    #
    dprint(f"Pre-processing input connections: {inplist}")
    if not isinstance(inplist, list):
        dprint("  converting inplist to list")
        inplist = [inplist]
    new_inplist, new_inputs = [], [] if inplist_none else inputs

    # Go through the list of inputs and process each one
    for iinp, connection in enumerate(inplist):
        # Check for system name or signal names without a system name
        if isinstance(connection, str) and len(connection.split('.')) == 1:
            # Create an empty connections list to store matching connections
            new_connections = []

            # Get the signal/system name
            sname = connection[1:] if connection[0] == '-' else connection
            gain = -1 if connection[0] == '-' else 1

            # Look for the signal name as a system input
            found_system, found_signal = False, False
            for isys, sys in enumerate(syslist):
                # Look for matching signals (returns None if no matches
                indices = sys._find_signals(sname, sys.input_index)

                # See what types of matches we found
                if sname == sys.name:
                    # System name matches => use all inputs
                    for isig in range(sys.ninputs):
                        dprint(f"  adding input {(isys, isig, gain)}")
                        new_inplist.append((isys, isig, gain))
                    found_system = True
                elif indices:
                    # Signal name matches => store new connections
                    new_connection = []
                    for isig in indices:
                        dprint(f"  collecting input {(isys, isig, gain)}")
                        new_connection.append((isys, isig, gain))

                    if len(new_connections) == 0:
                        # First time we have seen this signal => initialize
                        for cnx in new_connection:
                            new_connections.append([cnx])
                        if inplist_none:
                            # See if we need to rewrite the inputs
                            if len(new_connection) != 1:
                                new_inputs += [
                                    sys.input_labels[i] for i in indices]
                            else:
                                new_inputs.append(inputs[iinp])
                    else:
                        # Additional signal match found =. add to the list
                        for i, cnx in enumerate(new_connection):
                            new_connections[i].append(cnx)
                    found_signal = True

            if found_system and found_signal:
                raise ValueError(
                    f"signal '{sname}' is both signal and system name")
            elif found_signal:
                dprint(f"  adding inputs {new_connections}")
                new_inplist += new_connections
            elif not found_system:
                raise ValueError("could not find signal %s" % sname)
        else:
            if isinstance(connection, list):
                # Passed a list => create input map
                dprint("  detected input list")
                signal_list = []
                for spec in connection:
                    isys, indices, gain = _parse_spec(syslist, spec, 'input')
                    for isig in indices:
                        signal_list.append((isys, isig, gain))
                        dprint(f"    adding input {(isys, isig, gain)} to list")
                new_inplist.append(signal_list)
            else:
                # Passed a single signal name => add individual input(s)
                isys, indices, gain = _parse_spec(syslist, connection, 'input')
                for isig in indices:
                    new_inplist.append((isys, isig, gain))
                    dprint(f"  adding input {(isys, isig, gain)}")
    inplist, inputs = new_inplist, new_inputs
    dprint(f"  {inplist=}\n  {inputs=}")

    #
    # Pre-process output list
    #
    # This is similar to the processing of the input list, but we need to
    # additionally take into account the fact that you can list subsystem
    # inputs as system outputs.
    #
    dprint(f"Pre-processing output connections: {outlist}")
    if not isinstance(outlist, list):
        dprint("  converting outlist to list")
        outlist = [outlist]
    new_outlist, new_outputs = [], [] if outlist_none else outputs
    for iout, connection in enumerate(outlist):
        # Create an empty connection list
        new_connections = []

        # Check for system name or signal names without a system name
        if isinstance(connection, str) and len(connection.split('.')) == 1:
            # Get the signal/system name
            sname = connection[1:] if connection[0] == '-' else connection
            gain = -1 if connection[0] == '-' else 1

            # Look for the signal name as a system output
            found_system, found_signal = False, False
            for osys, sys in enumerate(syslist):
                indices = sys._find_signals(sname, sys.output_index)
                if sname == sys.name:
                    # Use all outputs
                    for osig in range(sys.noutputs):
                        dprint(f"  adding output {(osys, osig, gain)}")
                        new_outlist.append((osys, osig, gain))
                    found_system = True
                elif indices:
                    new_connection = []
                    for osig in indices:
                        dprint(f"  collecting output {(osys, osig, gain)}")
                        new_connection.append((osys, osig, gain))
                    if len(new_connections) == 0:
                        for cnx in new_connection:
                            new_connections.append([cnx])
                        if outlist_none:
                            # See if we need to rewrite the outputs
                            if len(new_connection) != 1:
                                new_outputs += [
                                    sys.output_labels[i] for i in indices]
                            else:
                                new_outputs.append(outputs[iout])
                    else:
                        # Additional signal match found =. add to the list
                        for i, cnx in enumerate(new_connection):
                            new_connections[i].append(cnx)
                    found_signal = True

            if found_system and found_signal:
                raise ValueError(
                    f"signal '{sname}' is both signal and system name")
            elif found_signal:
                dprint(f"  adding outputs {new_connections}")
                new_outlist += new_connections
            elif not found_system:
                raise ValueError("could not find signal %s" % sname)
        else:
            # Utility function to find named output or input signal
            def _find_output_or_input_signal(spec):
                signal_list = []
                try:
                    # First trying looking in the output signals
                    osys, indices, gain = _parse_spec(syslist, spec, 'output')
                    for osig in indices:
                        dprint(f"  adding output {(osys, osig, gain)}")
                        signal_list.append((osys, osig, gain))
                except ValueError:
                    # If not, see if we can find it in inputs
                    isys, indices, gain = _parse_spec(
                        syslist, spec, 'input or output',
                        dictname='input_index')
                    for isig in indices:
                        # Use string form to allow searching input list
                        dprint(f"  adding input {(isys, isig, gain)}")
                        signal_list.append(
                            (syslist[isys].name,
                             syslist[isys].input_labels[isig], gain))
                return signal_list

            if isinstance(connection, list):
                # Passed a list => create input map
                dprint("  detected output list")
                signal_list = []
                for spec in connection:
                    signal_list += _find_output_or_input_signal(spec)
                new_outlist.append(signal_list)
            else:
                new_outlist += _find_output_or_input_signal(connection)

    outlist, outputs = new_outlist, new_outputs
    dprint(f"  {outlist=}\n  {outputs=}")

    # Make sure inputs and outputs match inplist outlist, if specified
    if inputs and (
            isinstance(inputs, (list, tuple)) and len(inputs) != len(inplist)
            or isinstance(inputs, int) and inputs != len(inplist)):
        raise ValueError("`inputs` incompatible with `inplist`")
    if outputs and (
            isinstance(outputs, (list, tuple)) and len(outputs) != len(outlist)
            or isinstance(outputs, int) and outputs != len(outlist)):
        raise ValueError("`outputs` incompatible with `outlist`")

    newsys = InterconnectedSystem(
        syslist, connections=connections, inplist=inplist,
        outlist=outlist, inputs=inputs, outputs=outputs, states=states,
        params=params, dt=dt, name=name, warn_duplicate=warn_duplicate,
        connection_type=connection_type, **kwargs)

    # See if we should add any signals
    if add_unused:
        # Get all unused signals
        dropped_inputs, dropped_outputs = newsys.check_unused_signals(
            ignore_inputs, ignore_outputs, print_warning=False)

        # Add on any unused signals that we aren't ignoring
        for isys, isig in dropped_inputs:
            inplist.append((isys, isig))
            inputs.append(newsys.syslist[isys].input_labels[isig])
        for osys, osig in dropped_outputs:
            outlist.append((osys, osig))
            outputs.append(newsys.syslist[osys].output_labels[osig])

        # Rebuild the system with new inputs/outputs
        newsys = InterconnectedSystem(
            syslist, connections=connections, inplist=inplist,
            outlist=outlist, inputs=inputs, outputs=outputs, states=states,
            params=params, dt=dt, name=name, warn_duplicate=warn_duplicate,
            connection_type=connection_type, **kwargs)

    # check for implicitly dropped signals
    if check_unused:
        newsys.check_unused_signals(ignore_inputs, ignore_outputs)

    # If all subsystems are linear systems, maintain linear structure
    if all([isinstance(sys, StateSpace) for sys in newsys.syslist]):
        newsys = LinearICSystem(newsys, None, connection_type=connection_type)

    return newsys


def _process_vector_argument(arg, name, size):
    """Utility function to process vector elements (states, inputs)

    Process state and input arguments to turn them into lists of the
    appropriate length.

    Parameters
    ----------
    arg : array_like
        Value of the parameter passed to the function.  Can be a list,
        tuple, ndarray, scalar, or None.
    name : string
        Name of the argument being processed.  Used in errors/warnings.
    size : int or None
        Size of the element.  If None, size is determined by arg.

    Returns
    -------
    val : array or None
        Value of the element, zero-padded to proper length.
    nelem : int or None
        Number of elements in the returned value.

    Warns
    -----
    UserWarning : "{name} too short; padding with zeros"
        If argument is too short and last value in arg is not 0.

    """
    # Allow and expand list
    if isinstance(arg, (tuple, list)):
        val_list = []
        for i, v in enumerate(arg):
            v = np.array(v).reshape(-1)             # convert to 1D array
            val_list += v.tolist()                  # add elements to list
        val = np.array(val_list)
    elif np.isscalar(arg) and size is not None:     # extend scalars
        val = np.ones((size, )) * arg
    elif np.isscalar(arg) and size is None:         # single scalar
        val = np.array([arg])
    elif isinstance(arg, np.ndarray):
        val = arg.reshape(-1)                       # convert to 1D array
    else:
        val = arg                                   # return what we were given

    if size is not None and isinstance(val, np.ndarray) and val.size < size:
        # If needed, extend the size of the vector to match desired size
        if val[-1] != 0:
            warn(f"{name} too short; padding with zeros")
        val = np.hstack([val, np.zeros(size - val.size)])

    nelem = _find_size(size, val, name)                 # determine size
    return val, nelem


# Utility function to create an I/O system (from number or array)
def _convert_to_iosystem(sys):
    # If we were given an I/O system, do nothing
    if isinstance(sys, InputOutputSystem):
        return sys

    # Convert sys1 to an I/O system if needed
    if isinstance(sys, (int, float, np.number)):
        return NonlinearIOSystem(
            None, lambda t, x, u, params: sys * u,
            outputs=1, inputs=1, dt=None)

    elif isinstance(sys, np.ndarray):
        sys = np.atleast_2d(sys)
        return NonlinearIOSystem(
            None, lambda t, x, u, params: sys @ u,
            outputs=sys.shape[0], inputs=sys.shape[1], dt=None)

def connection_table(sys, show_names=False, column_width=32):
    """Print table of connections inside interconnected system.

    Intended primarily for `InterconnectedSystem`'s that have been
    connected implicitly using signal names.

    Parameters
    ----------
    sys : `InterconnectedSystem`
        Interconnected system object.
    show_names : bool, optional
        Instead of printing out the system number, print out the name of
        each system. Default is False because system name is not usually
        specified when performing implicit interconnection using
        `interconnect`.
    column_width : int, optional
        Character width of printed columns.

    Examples
    --------
    >>> P = ct.ss(1,1,1,0, inputs='u', outputs='y', name='P')
    >>> C = ct.tf(10, [.1, 1], inputs='e', outputs='u', name='C')
    >>> L = ct.interconnect([C, P], inputs='e', outputs='y')
    >>> L.connection_table(show_names=True) # doctest: +SKIP
    signal    | source                  | destination
    --------------------------------------------------------------
    e         | input                   | C
    u         | C                       | P
    y         | P                       | output

    """
    assert isinstance(sys, InterconnectedSystem), "system must be"\
        "an InterconnectedSystem."

    sys.connection_table(show_names=show_names, column_width=column_width)


# Short versions of function call
find_eqpt = find_operating_point
