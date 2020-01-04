# iosys.py - input/output system module
#
# RMM, 28 April 2019
#
# Additional features to add
#   * Improve support for signal names, specially in operator overloads
#       - Figure out how to handle "nested" names (icsys.sys[1].x[1])
#       - Use this to implement signal names for operators?
#   * Allow constant inputs for MIMO input_output_response (w/out ones)
#   * Add support for constants/matrices as part of operators (1 + P)
#   * Add unit tests (and example?) for time-varying systems
#   * Allow time vector for discrete time simulations to be multiples of dt
#   * Check the way initial outputs for discrete time systems are handled
#   * Rename 'connections' as 'conlist' to match 'inplist' and 'outlist'?
#   * Allow signal summation in InterconnectedSystem diagrams (via new output?)
#

"""The :mod:`~control.iosys` module contains the
:class:`~control.InputOutputSystem` class that represents (possibly nonlinear)
input/output systems.  The :class:`~control.InputOutputSystem` class is a
general class that defines any continuous or discrete time dynamical system.
Input/output systems can be simulated and also used to compute equilibrium
points and linearizations.

"""

__author__ = "Richard Murray"
__copyright__ = "Copyright 2019, California Institute of Technology"
__credits__ = ["Richard Murray"]
__license__ = "BSD"
__maintainer__ = "Richard Murray"
__email__ = "murray@cds.caltech.edu"

import numpy as np
import scipy as sp
import copy
from warnings import warn

from .statesp import StateSpace, tf2ss
from .timeresp import _check_convert_array
from .lti import isctime, isdtime, _find_timebase

__all__ = ['InputOutputSystem', 'LinearIOSystem', 'NonlinearIOSystem',
           'InterconnectedSystem', 'input_output_response', 'find_eqpt',
           'linearize', 'ss2io', 'tf2io']


class InputOutputSystem(object):
    """A class for representing input/output systems.

    The InputOutputSystem class allows (possibly nonlinear) input/output
    systems to be represented in Python.  It is intended as a parent
    class for a set of subclasses that are used to implement specific
    structures and operations for different types of input/output
    dynamical systems.

    Parameters
    ----------
    inputs : int, list of str, or None
        Description of the system inputs.  This can be given as an integer
        count or as a list of strings that name the individual signals.  If an
        integer count is specified, the names of the signal will be of the
        form `s[i]` (where `s` is one of `u`, `y`, or `x`).  If this parameter
        is not given or given as `None`, the relevant quantity will be
        determined when possible based on other information provided to
        functions using the system.
    outputs : int, list of str, or None
        Description of the system outputs.  Same format as `inputs`.
    states : int, list of str, or None
        Description of the system states.  Same format as `inputs`.
    dt : None, True or float, optional
        System timebase.  None (default) indicates continuous time, True
        indicates discrete time with undefined sampling time, positive number
        is discrete time with specified sampling time.
    params : dict, optional
        Parameter values for the systems.  Passed to the evaluation functions
        for the system as default values, overriding internal defaults.
    name : string, optional
        System name (used for specifying signals)

    Attributes
    ----------
    ninputs, noutputs, nstates : int
        Number of input, output and state variables
    input_index, output_index, state_index : dict
        Dictionary of signal names for the inputs, outputs and states and the
        index of the corresponding array
    dt : None, True or float
        System timebase.  None (default) indicates continuous time, True
        indicates discrete time with undefined sampling time, positive number
        is discrete time with specified sampling time.
    params : dict, optional
        Parameter values for the systems.  Passed to the evaluation functions
        for the system as default values, overriding internal defaults.
    name : string, optional
        System name (used for specifying signals)

    Notes
    -----
    The `InputOuputSystem` class (and its subclasses) makes use of two special
    methods for implementing much of the work of the class:

    * _rhs(t, x, u): compute the right hand side of the differential or
      difference equation for the system.  This must be specified by the
      subclass for the system.

    * _out(t, x, u): compute the output for the current state of the system.
      The default is to return the entire system state.

    """
    def __init__(self, inputs=None, outputs=None, states=None, params={},
                 dt=None, name=None):
        """Create an input/output system.

        The InputOutputSystem contructor is used to create an input/output
        object with the core information required for all input/output
        systems.  Instances of this class are normally created by one of the
        input/output subclasses: :class:`~control.LinearIOSystem`,
        :class:`~control.NonlinearIOSystem`,
        :class:`~control.InterconnectedSystem`.

        Parameters
        ----------
        inputs : int, list of str, or None
            Description of the system inputs.  This can be given as an integer
            count or as a list of strings that name the individual signals.
            If an integer count is specified, the names of the signal will be
            of the form `s[i]` (where `s` is one of `u`, `y`, or `x`).  If
            this parameter is not given or given as `None`, the relevant
            quantity will be determined when possible based on other
            information provided to functions using the system.
        outputs : int, list of str, or None
            Description of the system outputs.  Same format as `inputs`.
        states : int, list of str, or None
            Description of the system states.  Same format as `inputs`.
        dt : None, True or float, optional
            System timebase.  None (default) indicates continuous
            time, True indicates discrete time with undefined sampling
            time, positive number is discrete time with specified
            sampling time.
        params : dict, optional
            Parameter values for the systems.  Passed to the evaluation
            functions for the system as default values, overriding internal
            defaults.
        name : string, optional
            System name (used for specifying signals)

        Returns
        -------
        InputOutputSystem
            Input/output system object

        """
        # Store the input arguments
        self.params = params.copy()     # default parameters
        self.dt = dt                    # timebase
        self.name = name                # system name

        # Parse and store the number of inputs, outputs, and states
        self.set_inputs(inputs)
        self.set_outputs(outputs)
        self.set_states(states)

    def __repr__(self):
        return self.name if self.name is not None else str(type(self))

    def __str__(self):
        """String representation of an input/output system"""
        str = "System: " + (self.name if self.name else "(None)") + "\n"
        str += "Inputs (%s): " % self.ninputs
        for key in self.input_index: str += key + ", "
        str += "\nOutputs (%s): " % self.noutputs
        for key in self.output_index: str += key + ", "
        str += "\nStates (%s): " % self.nstates
        for key in self.state_index: str += key + ", "
        return str

    def __mul__(sys2, sys1):
        """Multiply two input/output systems (series interconnection)"""

        if isinstance(sys1, (int, float, np.number)):
            # TODO: Scale the output
            raise NotImplemented("Scalar multiplication not yet implemented")
        elif isinstance(sys1, np.ndarray):
            # TODO: Post-multiply by a matrix
            raise NotImplemented("Matrix multiplication not yet implemented")
        elif isinstance(sys1, StateSpace) and isinstance(sys2, StateSpace):
            # Special case: maintain linear systems structure
            new_ss_sys = StateSpace.__mul__(sys2, sys1)
            # TODO: set input and output names
            new_io_sys = LinearIOSystem(new_ss_sys)

            return new_io_sys
        elif not isinstance(sys1, InputOutputSystem):
            raise ValueError("Unknown I/O system object ", sys1)

        # Make sure systems can be interconnected
        if sys1.noutputs != sys2.ninputs:
            raise ValueError("Can't multiply systems with incompatible "
                             "inputs and outputs")

        # Make sure timebase are compatible
        dt = _find_timebase(sys1, sys2)
        if dt is False:
            raise ValueError("System timebases are not compabile")

        # Return the series interconnection between the systems
        newsys = InterconnectedSystem((sys1, sys2))

        #  Set up the connecton map
        newsys.set_connect_map(np.block(
            [[np.zeros((sys1.ninputs, sys1.noutputs)),
              np.zeros((sys1.ninputs, sys2.noutputs))],
             [np.eye(sys2.ninputs, sys1.noutputs),
              np.zeros((sys2.ninputs, sys2.noutputs))]]
        ))

        # Set up the input map
        newsys.set_input_map(np.concatenate(
            (np.eye(sys1.ninputs), np.zeros((sys2.ninputs, sys1.ninputs))),
            axis=0))
        # TODO: set up input names

        # Set up the output map
        newsys.set_output_map(np.concatenate(
            (np.zeros((sys2.noutputs, sys1.noutputs)), np.eye(sys2.noutputs)),
            axis=1))
        # TODO: set up output names

        # Return the newly created system
        return newsys

    def __rmul__(sys1, sys2):
        """Pre-multiply an input/output systems by a scalar/matrix"""
        if isinstance(sys2, (int, float, np.number)):
            # TODO: Scale the output
            raise NotImplemented("Scalar multiplication not yet implemented")
        elif isinstance(sys2, np.ndarray):
            # TODO: Post-multiply by a matrix
            raise NotImplemented("Matrix multiplication not yet implemented")
        elif isinstance(sys1, StateSpace) and isinstance(sys2, StateSpace):
            # Special case: maintain linear systems structure
            new_ss_sys = StateSpace.__rmul__(sys1, sys2)
            # TODO: set input and output names
            new_io_sys = LinearIOSystem(new_ss_sys)

            return new_io_sys
        elif not isinstance(sys2, InputOutputSystem):
            raise ValueError("Unknown I/O system object ", sys1)
        else:
            # Both systetms are InputOutputSystems => use __mul__
            return InputOutputSystem.__mul__(sys2, sys1)

    def __add__(sys1, sys2):
        """Add two input/output systems (parallel interconnection)"""
        # TODO: Allow addition of scalars and matrices
        if not isinstance(sys2, InputOutputSystem):
            raise ValueError("Unknown I/O system object ", sys2)
        elif isinstance(sys1, StateSpace) and isinstance(sys2, StateSpace):
            # Special case: maintain linear systems structure
            new_ss_sys = StateSpace.__add__(sys1, sys2)
            # TODO: set input and output names
            new_io_sys = LinearIOSystem(new_ss_sys)

            return new_io_sys

        # Make sure number of input and outputs match
        if sys1.ninputs != sys2.ninputs or sys1.noutputs != sys2.noutputs:
            raise ValueError("Can't add systems with different numbers of "
                             "inputs or outputs.")
        ninputs = sys1.ninputs
        noutputs = sys1.noutputs

        # Create a new system to handle the composition
        newsys = InterconnectedSystem((sys1, sys2))

        # Set up the input map
        newsys.set_input_map(np.concatenate(
            (np.eye(ninputs), np.eye(ninputs)), axis=0))
        # TODO: set up input names

        # Set up the output map
        newsys.set_output_map(np.concatenate(
            (np.eye(noutputs), np.eye(noutputs)), axis=1))
        # TODO: set up output names

        # Return the newly created system
        return newsys

    # TODO: add __radd__ to allow postaddition by scalars and matrices

    def __neg__(sys):
        """Negate an input/output systems (rescale)"""
        if isinstance(sys, StateSpace):
            # Special case: maintain linear systems structure
            new_ss_sys = StateSpace.__neg__(sys)
            # TODO: set input and output names
            new_io_sys = LinearIOSystem(new_ss_sys)

            return new_io_sys
        if sys.ninputs is None or sys.noutputs is None:
            raise ValueError("Can't determine number of inputs or outputs")

        # Create a new system to hold the negation
        newsys = InterconnectedSystem((sys,), dt=sys.dt)

        # Set up the input map (identity)
        newsys.set_input_map(np.eye(sys.ninputs))
        # TODO: set up input names

        # Set up the output map (negate the output)
        newsys.set_output_map(-np.eye(sys.noutputs))
        # TODO: set up output names

        # Return the newly created system
        return newsys

    # Utility function to parse a list of signals
    def _process_signal_list(self, signals, prefix='s'):
        if signals is None:
            # No information provided; try and make it up later
            return None, {}

        elif isinstance(signals, int):
            # Number of signals given; make up the names
            return signals, {'%s[%d]' % (prefix, i): i for i in range(signals)}

        elif isinstance(signals, str):
            # Single string given => single signal with given name
            return 1, {signals: 0}

        elif all(isinstance(s, str) for s in signals):
            # Use the list of strings as the signal names
            return len(signals), {signals[i]: i for i in range(len(signals))}

        else:
            raise TypeError("Can't parse signal list %s" % str(signals))

    # Find a signal by name
    def _find_signal(self, name, sigdict): return sigdict.get(name, None)

    # Update parameters used for _rhs, _out (used by subclasses)
    def _update_params(self, params, warning=False):
        if (warning):
            warn("Parameters passed to InputOutputSystem ignored.")

    def _rhs(self, t, x, u):
        """Evaluate right hand side of a differential or difference equation.

        Private function used to compute the right hand side of an
        input/output system model.

        """
        NotImplemented("Evaluation not implemented for system of type ",
                       type(self))

    def _out(self, t, x, u, params={}):
        """Evaluate the output of a system at a given state, input, and time

        Private function used to compute the output of of an input/output
        system model given the state, input, parameters, and time.

        """
        # If no output function was defined in subclass, return state
        return x

    def set_inputs(self, inputs, prefix='u'):
        """Set the number/names of the system inputs.

        Parameters
        ----------
        inputs : int, list of str, or None
            Description of the system inputs.  This can be given as an integer
            count or as a list of strings that name the individual signals.
            If an integer count is specified, the names of the signal will be
            of the form `u[i]` (where the prefix `u` can be changed using the
            optional prefix parameter).
        prefix : string, optional
            If `inputs` is an integer, create the names of the states using
            the given prefix (default = 'u').  The names of the input will be
            of the form `prefix[i]`.

        """
        self.ninputs, self.input_index = \
            self._process_signal_list(inputs, prefix=prefix)

    def set_outputs(self, outputs, prefix='y'):
        """Set the number/names of the system outputs.

        Parameters
        ----------
        outputs : int, list of str, or None
            Description of the system outputs.  This can be given as an integer
            count or as a list of strings that name the individual signals.
            If an integer count is specified, the names of the signal will be
            of the form `u[i]` (where the prefix `u` can be changed using the
            optional prefix parameter).
        prefix : string, optional
            If `outputs` is an integer, create the names of the states using
            the given prefix (default = 'y').  The names of the input will be
            of the form `prefix[i]`.

        """
        self.noutputs, self.output_index = \
            self._process_signal_list(outputs, prefix=prefix)

    def set_states(self, states, prefix='x'):
        """Set the number/names of the system states.

        Parameters
        ----------
        states : int, list of str, or None
            Description of the system states.  This can be given as an integer
            count or as a list of strings that name the individual signals.
            If an integer count is specified, the names of the signal will be
            of the form `u[i]` (where the prefix `u` can be changed using the
            optional prefix parameter).
        prefix : string, optional
            If `states` is an integer, create the names of the states using
            the given prefix (default = 'x').  The names of the input will be
            of the form `prefix[i]`.

        """
        self.nstates, self.state_index = \
            self._process_signal_list(states, prefix=prefix)

    def find_input(self, name):
        """Find the index for an input given its name (`None` if not found)"""
        return self.input_index.get(name, None)

    def find_output(self, name):
        """Find the index for an output given its name (`None` if not found)"""
        return self.output_index.get(name, None)

    def find_state(self, name):
        """Find the index for a state given its name (`None` if not found)"""
        return self.state_index.get(name, None)

    def feedback(self, other=1, sign=-1, params={}):
        """Feedback interconnection between two input/output systems

        Parameters
        ----------
        sys1: InputOutputSystem
            The primary process.
        sys2: InputOutputSystem
            The feedback process (often a feedback controller).
        sign: scalar, optional
            The sign of feedback.  `sign` = -1 indicates negative feedback,
            and `sign` = 1 indicates positive feedback.  `sign` is an optional
            argument; it assumes a value of -1 if not specified.

        Returns
        -------
        out: InputOutputSystem

        Raises
        ------
        ValueError
            if the inputs, outputs, or timebases of the systems are
            incompatible.

        """
        # TODO: add conversion to I/O system when needed
        if not isinstance(other, InputOutputSystem):
            raise TypeError("Feedback around I/O system must be I/O system.")
        elif isinstance(self, StateSpace) and isinstance(other, StateSpace):
            # Special case: maintain linear systems structure
            new_ss_sys = StateSpace.feedback(self, other, sign=sign)
            # TODO: set input and output names
            new_io_sys = LinearIOSystem(new_ss_sys)

            return new_io_sys

        # Make sure systems can be interconnected
        if self.noutputs != other.ninputs or other.noutputs != self.ninputs:
            raise ValueError("Can't connect systems with incompatible "
                             "inputs and outputs")

        # Make sure timebases are compatible
        dt = _find_timebase(self, other)
        if dt is False:
            raise ValueError("System timebases are not compabile")

        # Return the series interconnection between the systems
        newsys = InterconnectedSystem((self, other), params=params, dt=dt)

        #  Set up the connecton map
        newsys.set_connect_map(np.block(
            [[np.zeros((self.ninputs, self.noutputs)),
              sign * np.eye(self.ninputs, other.noutputs)],
             [np.eye(other.ninputs, self.noutputs),
              np.zeros((other.ninputs, other.noutputs))]]
        ))

        # Set up the input map
        newsys.set_input_map(np.concatenate(
            (np.eye(self.ninputs), np.zeros((other.ninputs, self.ninputs))),
            axis=0))
        # TODO: set up input names

        # Set up the output map
        newsys.set_output_map(np.concatenate(
            (np.eye(self.noutputs), np.zeros((self.noutputs, other.noutputs))),
            axis=1))
        # TODO: set up output names

        # Return the newly created system
        return newsys

    def linearize(self, x0, u0, t=0, params={}, eps=1e-6):
        """Linearize an input/output system at a given state and input.

        Return the linearization of an input/output system at a given state
        and input value as a StateSpace system.  See
        :func:`~control.linearize` for complete documentation.

        """
        #
        # If the linearization is not defined by the subclass, perform a
        # numerical linearization use the `_rhs()` and `_out()` member
        # functions.
        #

        # Figure out dimensions if they were not specified.
        nstates = _find_size(self.nstates, x0)
        ninputs = _find_size(self.ninputs, u0)

        # Convert x0, u0 to arrays, if needed
        if np.isscalar(x0): x0 = np.ones((nstates,)) * x0
        if np.isscalar(u0): u0 = np.ones((ninputs,)) * u0

        # Compute number of outputs by evaluating the output function
        noutputs = _find_size(self.noutputs, self._out(t, x0, u0))

        # Update the current parameters
        self._update_params(params)

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
        linsys = StateSpace(A, B, C, D, self.dt, remove_useless=False)
        return LinearIOSystem(linsys)

    def copy(self):
        """Make a copy of an input/output system."""
        return copy.copy(self)


class LinearIOSystem(InputOutputSystem, StateSpace):
    """Input/output representation of a linear (state space) system.

    This class is used to implementat a system that is a linear state
    space system (defined by the StateSpace system object).

    """
    def __init__(self, linsys, inputs=None, outputs=None, states=None,
                 name=None):
        """Create an I/O system from a state space linear system.

        Converts a :class:`~control.StateSpace` system into an
        :class:`~control.InputOutputSystem` with the same inputs, outputs, and
        states.  The new system can be a continuous or discrete time system

        Parameters
        ----------
        linsys : StateSpace
            LTI StateSpace system to be converted
        inputs : int, list of str or None, optional
            Description of the system inputs.  This can be given as an integer
            count or as a list of strings that name the individual signals.
            If an integer count is specified, the names of the signal will be
            of the form `s[i]` (where `s` is one of `u`, `y`, or `x`).  If
            this parameter is not given or given as `None`, the relevant
            quantity will be determined when possible based on other
            information provided to functions using the system.
        outputs : int, list of str or None, optional
            Description of the system outputs.  Same format as `inputs`.
        states : int, list of str, or None, optional
            Description of the system states.  Same format as `inputs`.
        dt : None, True or float, optional
            System timebase.  None (default) indicates continuous
            time, True indicates discrete time with undefined sampling
            time, positive number is discrete time with specified
            sampling time.
        params : dict, optional
            Parameter values for the systems.  Passed to the evaluation
            functions for the system as default values, overriding internal
            defaults.
        name : string, optional
            System name (used for specifying signals)

        Returns
        -------
        iosys : LinearIOSystem
            Linear system represented as an input/output system

        """
        if not isinstance(linsys, StateSpace):
            raise TypeError("Linear I/O system must be a state space object")

        # Create the I/O system object
        super(LinearIOSystem, self).__init__(
            inputs=linsys.inputs, outputs=linsys.outputs,
            states=linsys.states, params={}, dt=linsys.dt, name=name)

        # Initalize additional state space variables
        StateSpace.__init__(self, linsys, remove_useless=False)

        # Process input, output, state lists, if given
        # Make sure they match the size of the linear system
        ninputs, self.input_index = self._process_signal_list(
            inputs if inputs is not None else linsys.inputs, prefix='u')
        if ninputs is not None and linsys.inputs != ninputs:
            raise ValueError("Wrong number/type of inputs given.")
        noutputs, self.output_index = self._process_signal_list(
            outputs if outputs is not None else linsys.outputs, prefix='y')
        if noutputs is not None and linsys.outputs != noutputs:
            raise ValueError("Wrong number/type of outputs given.")
        nstates, self.state_index = self._process_signal_list(
            states if states is not None else linsys.states, prefix='x')
        if nstates is not None and linsys.states != nstates:
            raise ValueError("Wrong number/type of states given.")

    def _update_params(self, params={}, warning=True):
        # Parameters not supported; issue a warning
        if params and warning:
            warn("Parameters passed to LinearIOSystems are ignored.")

    def _rhs(self, t, x, u):
        # Convert input to column vector and then change output to 1D array
        xdot = np.dot(self.A, np.reshape(x, (-1, 1))) \
            + np.dot(self.B, np.reshape(u, (-1, 1)))
        return np.array(xdot).reshape((-1,))

    def _out(self, t, x, u):
        y = self.C * np.reshape(x, (-1, 1)) + self.D * np.reshape(u, (-1, 1))
        return np.array(y).reshape((self.noutputs,))


class NonlinearIOSystem(InputOutputSystem):
    """Nonlinear I/O system.

    This class is used to implement a system that is a nonlinear state
    space system (defined by and update function and an output function).

    """
    def __init__(self, updfcn, outfcn=None, inputs=None, outputs=None,
                 states=None, params={}, dt=None, name=None):
        """Create a nonlinear I/O system given update and output functions.

        Creates an `InputOutputSystem` for a nonlinear system by specifying a
        state update function and an output function.  The new system can be a
        continuous or discrete time system (Note: discrete-time systems not
        yet supported by most function.)

        Parameters
        ----------
        updfcn : callable
            Function returning the state update function

                `updfcn(t, x, u[, param]) -> array`

            where `x` is a 1-D array with shape (nstates,), `u` is a 1-D array
            with shape (ninputs,), `t` is a float representing the currrent
            time, and `param` is an optional dict containing the values of
            parameters used by the function.

        outfcn : callable
            Function returning the output at the given state

                `outfcn(t, x, u[, param]) -> array`

            where the arguments are the same as for `upfcn`.

        inputs : int, list of str or None, optional
            Description of the system inputs.  This can be given as an integer
            count or as a list of strings that name the individual signals.
            If an integer count is specified, the names of the signal will be
            of the form `s[i]` (where `s` is one of `u`, `y`, or `x`).  If
            this parameter is not given or given as `None`, the relevant
            quantity will be determined when possible based on other
            information provided to functions using the system.

        outputs : int, list of str or None, optional
            Description of the system outputs.  Same format as `inputs`.

        states : int, list of str, or None, optional
            Description of the system states.  Same format as `inputs`.

        params : dict, optional
            Parameter values for the systems.  Passed to the evaluation
            functions for the system as default values, overriding internal
            defaults.

        dt : timebase, optional
            The timebase for the system, used to specify whether the system is
            operating in continuous or discrete time.  It can have the
            following values:

            * dt = None       No timebase specified
            * dt = 0          Continuous time system
            * dt > 0          Discrete time system with sampling time dt
            * dt = True       Discrete time with unspecified sampling time

        name : string, optional
            System name (used for specifying signals).

        Returns
        -------
        iosys : NonlinearIOSystem
            Nonlinear system represented as an input/output system.

        """
        # Store the update and output functions
        self.updfcn = updfcn
        self.outfcn = outfcn

        # Initialize the rest of the structure
        super(NonlinearIOSystem, self).__init__(
            inputs=inputs, outputs=outputs, states=states,
            params=params, dt=dt, name=name
        )

        # Check to make sure arguments are consistent
        if updfcn is None:
            if self.nstates is None:
                self.nstates = 0
            else:
                raise ValueError("States specified but no update function "
                                 "given.")
        if outfcn is None:
            # No output function specified => outputs = states
            if self.noutputs is None and self.nstates is not None:
                self.noutputs = self.nstates
            elif self.noutputs is not None and self.noutputs == self.nstates:
                # Number of outputs = number of states => all is OK
                pass
            elif self.noutputs is not None and self.noutputs != 0:
                raise ValueError("Outputs specified but no output function "
                                 "(and nstates not known).")

        # Initialize current parameters to default parameters
        self._current_params = params.copy()

    def _update_params(self, params, warning=False):
        # Update the current parameter values
        self._current_params = self.params.copy()
        self._current_params.update(params)

    def _rhs(self, t, x, u):
        xdot = self.updfcn(t, x, u, self._current_params) \
            if self.updfcn is not None else []
        return np.array(xdot).reshape((-1,))

    def _out(self, t, x, u):
        y = self.outfcn(t, x, u, self._current_params) \
            if self.outfcn is not None else x
        return np.array(y).reshape((-1,))


class InterconnectedSystem(InputOutputSystem):
    """Interconnection of a set of input/output systems.

    This class is used to implement a system that is an interconnection of
    input/output systems.  The sys consists of a collection of subsystems
    whose inputs and outputs are connected via a connection map.  The overall
    system inputs and outputs are subsets of the subsystem inputs and outputs.

    """
    def __init__(self, syslist, connections=[], inplist=[], outlist=[],
                 inputs=None, outputs=None, states=None,
                 params={}, dt=None, name=None):
        """Create an I/O system from a list of systems + connection info.

        The InterconnectedSystem class is used to represent an input/output
        system that consists of an interconnection between a set of subystems.
        The outputs of each subsystem can be summed together to to provide
        inputs to other subsystems.  The overall system inputs and outputs can
        be any subset of subsystem inputs and outputs.

        Parameters
        ----------
        syslist : array_like of InputOutputSystems
            The list of input/output systems to be connected

        connections : tuple of connection specifications, optional
            Description of the internal connections between the subsystems.
            Each element of the tuple describes an input to one of the
            subsystems.  The entries are are of the form:

                (input-spec, output-spec1, output-spec2, ...)

            The input-spec should be a tuple of the form `(subsys_i, inp_j)`
            where `subsys_i` is the index into `syslist` and `inp_j` is the
            index into the input vector for the subsystem.  If `subsys_i` has
            a single input, then the subsystem index `subsys_i` can be listed
            as the input-spec.  If systems and signals are given names, then
            the form 'sys.sig' or ('sys', 'sig') are also recognized.

            Each output-spec should be a tuple of the form `(subsys_i, out_j,
            gain)`.  The input will be constructed by summing the listed
            outputs after multiplying by the gain term.  If the gain term is
            omitted, it is assumed to be 1.  If the system has a single
            output, then the subsystem index `subsys_i` can be listed as the
            input-spec.  If systems and signals are given names, then the form
            'sys.sig', ('sys', 'sig') or ('sys', 'sig', gain) are also
            recognized, and the special form '-sys.sig' can be used to specify
            a signal with gain -1.

            If omitted, the connection map (matrix) can be specified using the
            :func:`~control.InterconnectedSystem.set_connect_map` method.

        inplist : tuple of input specifications, optional
            List of specifications for how the inputs for the overall system
            are mapped to the subsystem inputs.  The input specification is
            the same as the form defined in the connection specification.
            Each system input is added to the input for the listed subsystem.

            If omitted, the input map can be specified using the
            `set_input_map` method.

        outlist : tuple of output specifications, optional
            List of specifications for how the outputs for the subsystems are
            mapped to overall system outputs.  The output specification is the
            same as the form defined in the connection specification
            (including the optional gain term).  Numbered outputs must be
            chosen from the list of subsystem outputs, but named outputs can
            also be contained in the list of subsystem inputs.

            If omitted, the output map can be specified using the
            `set_output_map` method.

        params : dict, optional
            Parameter values for the systems.  Passed to the evaluation
            functions for the system as default values, overriding internal
            defaults.

        dt : timebase, optional
            The timebase for the system, used to specify whether the system is
            operating in continuous or discrete time.  It can have the
            following values:

            * dt = None       No timebase specified
            * dt = 0          Continuous time system
            * dt > 0          Discrete time system with sampling time dt
            * dt = True       Discrete time with unspecified sampling time

        name : string, optional
            System name (used for specifying signals).

        """
        # Convert input and output names to lists if they aren't already
        if not isinstance(inplist, (list, tuple)): inplist = [inplist]
        if not isinstance(outlist, (list, tuple)): outlist = [outlist]

        # Check to make sure all systems are consistent
        self.syslist = syslist
        self.syslist_index = {}
        dt = None
        nstates = 0; self.state_offset = []
        ninputs = 0; self.input_offset = []
        noutputs = 0; self.output_offset = []
        system_count = 0
        for sys in syslist:
            # Make sure time bases are consistent
            # TODO: Use lti._find_timebase() instead?
            if dt is None and sys.dt is not None:
                # Timebase was not specified; set to match this system
                dt = sys.dt
            elif dt != sys.dt:
                raise TypeError("System timebases are not compatible")

            # Make sure number of inputs, outputs, states is given
            if sys.ninputs is None or sys.noutputs is None or \
               sys.nstates is None:
                raise TypeError("System '%s' must define number of inputs, "
                                "outputs, states in order to be connected" %
                                sys.name)

            # Keep track of the offsets into the states, inputs, outputs
            self.input_offset.append(ninputs)
            self.output_offset.append(noutputs)
            self.state_offset.append(nstates)

            # Keep track of the total number of states, inputs, outputs
            nstates += sys.nstates
            ninputs += sys.ninputs
            noutputs += sys.noutputs

            # Store the index to the system for later retrieval
            # TODO: look for duplicated system names
            self.syslist_index[sys.name] = system_count
            system_count += 1

        # Check for duplicate systems or duplicate names
        sysobj_list = []
        sysname_list = []
        for sys in syslist:
            if sys in sysobj_list:
                warn("Duplicate object found in system list: %s" % str(sys))
            elif sys.name is not None and sys.name in sysname_list:
                warn("Duplicate name found in system list: %s" % sys.name)
            sysobj_list.append(sys)
            sysname_list.append(sys.name)

        # Create the I/O system
        super(InterconnectedSystem, self).__init__(
            inputs=len(inplist), outputs=len(outlist),
            states=nstates, params=params, dt=dt)

        # If input or output list was specified, update it
        nsignals, self.input_index = \
            self._process_signal_list(inputs, prefix='u')
        if nsignals is not None and len(inplist) != nsignals:
            raise ValueError("Wrong number/type of inputs given.")
        nsignals, self.output_index = \
            self._process_signal_list(outputs, prefix='y')
        if nsignals is not None and len(outlist) != nsignals:
            raise ValueError("Wrong number/type of outputs given.")

        # Convert the list of interconnections to a connection map (matrix)
        self.connect_map = np.zeros((ninputs, noutputs))
        for connection in connections:
            input_index = self._parse_input_spec(connection[0])
            for output_spec in connection[1:]:
                output_index, gain = self._parse_output_spec(output_spec)
                self.connect_map[input_index, output_index] = gain

        # Convert the input list to a matrix: maps system to subsystems
        self.input_map = np.zeros((ninputs, self.ninputs))
        for index, inpspec in enumerate(inplist):
            if isinstance(inpspec, (int, str, tuple)): inpspec = [inpspec]
            for spec in inpspec:
                self.input_map[self._parse_input_spec(spec), index] = 1

        # Convert the output list to a matrix: maps subsystems to system
        self.output_map = np.zeros((self.noutputs, noutputs + ninputs))
        for index in range(len(outlist)):
            ylist_index, gain = self._parse_output_spec(outlist[index])
            self.output_map[index, ylist_index] = gain

        # Save the parameters for the system
        self.params = params.copy()

    def __add__(self, sys):
        # TODO: implement special processing to maintain flat structure
        return super(InterconnectedSystem, self).__add__(sys)

    def __radd__(self, sys):
        # TODO: implement special processing to maintain flat structure
        return super(InterconnectedSystem, self).__radd__(sys)

    def __mul__(self, sys):
        # TODO: implement special processing to maintain flat structure
        return super(InterconnectedSystem, self).__mul__(sys)

    def __rmul__(self, sys):
        # TODO: implement special processing to maintain flat structure
        return super(InterconnectedSystem, self).__rmul__(sys)

    def __neg__(self):
        # TODO: implement special processing to maintain flat structure
        return super(InterconnectedSystem, self).__neg__()

    def _update_params(self, params, warning=False):
        for sys in self.syslist:
            local = sys.params.copy()   # start with system parameters
            local.update(self.params)   # update with global params
            local.update(params)        # update with locally passed parameters
            sys._update_params(local, warning=warning)

    def _rhs(self, t, x, u):
        # Make sure state and input are vectors
        x = np.array(x, ndmin=1)
        u = np.array(u, ndmin=1)

        # Compute the input and output vectors
        ulist, ylist = self._compute_static_io(t, x, u)

        # Go through each system and update the right hand side for that system
        xdot = np.zeros((self.nstates,))        # Array to hold results
        state_index = 0; input_index = 0        # Start at the beginning
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
        return np.dot(self.output_map, ylist)

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
            state_index = 0; input_index = 0; output_index = 0
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
            new_ulist = np.dot(self.connect_map, ylist[:noutputs]) \
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
            raise RuntimeError("Algebraic loop detected.")

        return ulist, ylist

    def _parse_input_spec(self, spec):
        """Parse an input specification and returns the index

        This function parses a specification of an input of an interconnected
        system component and returns the index of that input in the internal
        input vector.  Input specifications are of one of the following forms:

            i               first input for the ith system
            (i,)            first input for the ith system
            (i, j)          jth input for the ith system
            'sys.sig'       signal 'sig' in subsys 'sys'
            ('sys', 'sig')  signal 'sig' in subsys 'sys'

        The function returns an index into the input vector array and
        the gain to use for that input.

        """
        # Parse the signal that we received
        subsys_index, input_index = self._parse_signal(spec, 'input')

        # Return the index into the input vector list (ylist)
        return self.input_offset[subsys_index] + input_index

    def _parse_output_spec(self, spec):
        """Parse an output specification and returns the index and gain

        This function parses a specification of an output of an
        interconnected system component and returns the index of that
        output in the internal output vector (ylist).  Output specifications
        are of one of the following forms:

            i                       first output for the ith system
            (i,)                    first output for the ith system
            (i, j)                  jth output for the ith system
            (i, j, gain)            jth output for the ith system with gain
            'sys.sig'               signal 'sig' in subsys 'sys'
            '-sys.sig'              signal 'sig' in subsys 'sys' with gain -1
            ('sys', 'sig', gain)    signal 'sig' in subsys 'sys' with gain

        If the gain is not specified, it is taken to be 1.  Numbered outputs
        must be chosen from the list of subsystem outputs, but named outputs
        can also be contained in the list of subsystem inputs.

        The function returns an index into the output vector array and
        the gain to use for that output.

        """
        gain = 1                # Default gain

        # Check for special forms of the input
        if isinstance(spec, tuple) and len(spec) == 3:
            gain = spec[2]
            spec = spec[:2]
        elif isinstance(spec, str) and spec[0] == '-':
            gain = -1
            spec = spec[1:]

        # Parse the rest of the spec with standard signal parsing routine
        try:
            # Start by looking in the set of subsystem outputs
            subsys_index, output_index = self._parse_signal(spec, 'output')

            # Return the index into the input vector list (ylist)
            return self.output_offset[subsys_index] + output_index, gain

        except ValueError:
            # Try looking in the set of subsystem *inputs*
            subsys_index, input_index = self._parse_signal(
                spec, 'input or output', dictname='input_index')

            # Return the index into the input vector list (ylist)
            noutputs = sum(sys.noutputs for sys in self.syslist)
            return noutputs + \
                self.input_offset[subsys_index] + input_index, gain

    def _parse_signal(self, spec, signame='input', dictname=None):
        """Parse a signal specification, returning system and signal index.

        Signal specifications are of one of the following forms:

            i               system_index = i, signal_index = 0
            (i,)            system_index = i, signal_index = 0
            (i, j)          system_index = i, signal_index = j
            'sys.sig'       signal 'sig' in subsys 'sys'
            ('sys', 'sig')  signal 'sig' in subsys 'sys'
            ('sys', j)      signal_index j in subsys 'sys'

        The function returns an index into the input vector array and
        the gain to use for that input.
        """
        import re

        # Process cases where we are given indices as integers
        if isinstance(spec, int):
            return spec, 0

        elif isinstance(spec, tuple) and len(spec) == 1 \
             and isinstance(spec[0], int):
            return spec[0], 0

        elif isinstance(spec, tuple) and len(spec) == 2 \
             and all([isinstance(index, int) for index in spec]):
            return spec

        # Figure out the name of the dictionary to use
        if dictname is None: dictname = signame + '_index'

        if isinstance(spec, str):
            # If we got a dotted string, break up into pieces
            namelist = re.split(r'\.', spec)

            # For now, only allow signal level of system name
            # TODO: expand to allow nested signal names
            if len(namelist) != 2:
                raise ValueError("Couldn't parse %s signal reference '%s'."
                                 % (signame, spec))

            system_index = self._find_system(namelist[0])
            if system_index is None:
                raise ValueError("Couldn't find system '%s'." % namelist[0])

            signal_index = self.syslist[system_index]._find_signal(
                namelist[1], getattr(self.syslist[system_index], dictname))
            if signal_index is None:
                raise ValueError("Couldn't find %s signal '%s.%s'." %
                                 (signame, namelist[0], namelist[1]))

            return system_index, signal_index

        # Handle the ('sys', 'sig'), (i, j), and mixed cases
        elif isinstance(spec, tuple) and len(spec) == 2 and \
             isinstance(spec[0], (str, int)) and \
             isinstance(spec[1], (str, int)):
            if isinstance(spec[0], int):
                system_index = spec[0]
                if system_index < 0 or system_index > len(self.syslist):
                    system_index = None
            else:
                system_index = self._find_system(spec[0])
            if system_index is None:
                raise ValueError("Couldn't find system %s." % spec[0])

            if isinstance(spec[1], int):
                signal_index = spec[1]
                # TODO (later): check against max length of appropriate list?
                if signal_index < 0:
                    system_index = None
            else:
                signal_index = self.syslist[system_index]._find_signal(
                    spec[1], getattr(self.syslist[system_index], dictname))
            if signal_index is None:
                raise ValueError("Couldn't find signal %s.%s." % tuple(spec))

            return system_index, signal_index

        else:
            raise ValueError("Couldn't parse signal reference %s." % str(spec))

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
             subsystem outputs to obtain the vector of system outputs.
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


def input_output_response(sys, T, U=0., X0=0, params={}, method='RK45',
                          return_x=False, squeeze=True):

    """Compute the output response of a system to a given input.

    Simulate a dynamical system with a given input and return its output
    and state values.

    Parameters
    ----------
    sys: InputOutputSystem
        Input/output system to simulate.
    T: array-like
        Time steps at which the input is defined; values must be evenly spaced.
    U: array-like or number, optional
        Input array giving input at each time `T` (default = 0).
    X0: array-like or number, optional
        Initial condition (default = 0).
    return_x : bool, optional
        If True, return the values of the state at each time (default = False).
    squeeze : bool, optional
        If True (default), squeeze unused dimensions out of the output
        response.  In particular, for a single output system, return a
        vector of shape (nsteps) instead of (nsteps, 1).

    Returns
    -------
    T : array
        Time values of the output.
    yout : array
        Response of the system.
    xout : array
        Time evolution of the state vector (if return_x=True)

    Raises
    ------
    TypeError
        If the system is not an input/output system.
    ValueError
        If time step does not match sampling time (for discrete time systems)

    """
    # Sanity checking on the input
    if not isinstance(sys, InputOutputSystem):
        raise TypeError("System of type ", type(sys), " not valid")

    # Compute the time interval and number of steps
    T0, Tf = T[0], T[-1]
    n_steps = len(T)

    # Check and convert the input, if needed
    # TODO: improve MIMO ninputs check (choose from U)
    if sys.ninputs is None or sys.ninputs == 1:
        legal_shapes = [(n_steps,), (1, n_steps)]
    else:
        legal_shapes = [(sys.ninputs, n_steps)]
    U = _check_convert_array(U, legal_shapes,
                             'Parameter ``U``: ', squeeze=False)

    # Check to make sure this is not a static function
    nstates = _find_size(sys.nstates, X0)
    if nstates == 0:
        # No states => map input to output
        u = U[0] if len(U.shape) == 1 else U[:, 0]
        y = np.zeros((np.shape(sys._out(T[0], X0, u))[0], len(T)))
        for i in range(len(T)):
            u = U[i] if len(U.shape) == 1 else U[:, i]
            y[:, i] = sys._out(T[i], [], u)
        if (squeeze): y = np.squeeze(y)
        if return_x:
            return T, y, []
        else:
            return T, y

    # create X0 if not given, test if X0 has correct shape
    X0 = _check_convert_array(X0, [(nstates,), (nstates, 1)],
                              'Parameter ``X0``: ', squeeze=True)

    # Update the parameter values
    sys._update_params(params)

    # Create a lambda function for the right hand side
    u = sp.interpolate.interp1d(T, U, fill_value="extrapolate")
    def ivp_rhs(t, x): return sys._rhs(t, x, u(t))

    # Perform the simulation
    if isctime(sys):
        if not hasattr(sp.integrate, 'solve_ivp'):
            raise NameError("scipy.integrate.solve_ivp not found; "
                            "use SciPy 1.0 or greater")
        soln = sp.integrate.solve_ivp(ivp_rhs, (T0, Tf), X0, t_eval=T,
                                      method=method, vectorized=False)

        # Compute the output associated with the state (and use sys.out to
        # figure out the number of outputs just in case it wasn't specified)
        u = U[0] if len(U.shape) == 1 else U[:, 0]
        y = np.zeros((np.shape(sys._out(T[0], X0, u))[0], len(T)))
        for i in range(len(T)):
            u = U[i] if len(U.shape) == 1 else U[:, i]
            y[:, i] = sys._out(T[i], soln.y[:, i], u)

    elif isdtime(sys):
        # Make sure the time vector is uniformly spaced
        dt = T[1] - T[0]
        if not np.allclose(T[1:] - T[:-1], dt):
            raise ValueError("Parameter ``T``: time values must be "
                             "equally spaced.")

        # Make sure the sample time matches the given time
        if (sys.dt is not True):
            # Make sure that the time increment is a multiple of sampling time

            # TODO: add back functionality for undersampling
            # TODO: this test is brittle if dt =  sys.dt
            # First make sure that time increment is bigger than sampling time
            # if dt < sys.dt:
            #     raise ValueError("Time steps ``T`` must match sampling time")

            # Check to make sure sampling time matches time increments
            if not np.isclose(dt, sys.dt):
                raise ValueError("Time steps ``T`` must be equal to "
                                 "sampling time")

        # Compute the solution
        soln = sp.optimize.OptimizeResult()
        soln.t = T                      # Store the time vector directly
        x = [float(x0) for x0 in X0]    # State vector (store as floats)
        soln.y = []                     # Solution, following scipy convention
        y = []                          # System output
        for i in range(len(T)):
            # Store the current state and output
            soln.y.append(x)
            y.append(sys._out(T[i], x, u(T[i])))

            # Update the state for the next iteration
            x = sys._rhs(T[i], x, u(T[i]))

        # Convert output to numpy arrays
        soln.y = np.transpose(np.array(soln.y))
        y = np.transpose(np.array(y))

        # Mark solution as successful
        soln.success = True     # No way to fail

    else:                       # Neither ctime or dtime??
        raise TypeError("Can't determine system type")

    # Get rid of extra dimensions in the output, of desired
    if (squeeze): y = np.squeeze(y)

    if return_x:
        return soln.t, y, soln.y
    else:
        return soln.t, y


def find_eqpt(sys, x0, u0=[], y0=None, t=0, params={},
              iu=None, iy=None, ix=None, idx=None, dx0=None,
              return_y=False, return_result=False, **kw):
    """Find the equilibrium point for an input/output system.

    Returns the value of an equlibrium point given the initial state and
    either input value or desired output value for the equilibrium point.

    Parameters
    ----------
    x0 : list of initial state values
        Initial guess for the value of the state near the equilibrium point.
    u0 : list of input values, optional
        If `y0` is not specified, sets the equilibrium value of the input.  If
        `y0` is given, provides an initial guess for the value of the input.
        Can be omitted if the system does not have any inputs.
    y0 : list of output values, optional
        If specified, sets the desired values of the outputs at the
        equilibrium point.
    t : float, optional
        Evaluation time, for time-varying systems
    params : dict, optional
        Parameter values for the system.  Passed to the evaluation functions
        for the system as default values, overriding internal defaults.
    iu : list of input indices, optional
        If specified, only the inputs with the given indices will be fixed at
        the specified values in solving for an equilibrium point.  All other
        inputs will be varied.  Input indices can be listed in any order.
    iy : list of output indices, optional
        If specified, only the outputs with the given indices will be fixed at
        the specified values in solving for an equilibrium point.  All other
        outputs will be varied.  Output indices can be listed in any order.
    ix : list of state indices, optional
        If specified, states with the given indices will be fixed at the
        specified values in solving for an equilibrium point.  All other
        states will be varied.  State indices can be listed in any order.
    dx0 : list of update values, optional
        If specified, the value of update map must match the listed value
        instead of the default value of 0.
    idx : list of state indices, optional
        If specified, state updates with the given indices will have their
        update maps fixed at the values given in `dx0`.  All other update
        values will be ignored in solving for an equilibrium point.  State
        indices can be listed in any order.  By default, all updates will be
        fixed at `dx0` in searching for an equilibrium point.
    return_y : bool, optional
        If True, return the value of output at the equilibrium point.
    return_result : bool, optional
        If True, return the `result` option from the scipy root function used
        to compute the equilibrium point.

    Returns
    -------
    xeq : array of states
        Value of the states at the equilibrium point, or `None` if no
        equilibrium point was found and `return_result` was False.
    ueq : array of input values
        Value of the inputs at the equilibrium point, or `None` if no
        equilibrium point was found and `return_result` was False.
    yeq : array of output values, optional
        If `return_y` is True, returns the value of the outputs at the
        equilibrium point, or `None` if no equilibrium point was found and
        `return_result` was False.
    result : scipy root() result object, optional
        If `return_result` is True, returns the `result` from the scipy root
        function.

    """
    from scipy.optimize import root

    # Figure out the number of states, inputs, and outputs
    nstates = _find_size(sys.nstates, x0)
    ninputs = _find_size(sys.ninputs, u0)
    noutputs = _find_size(sys.noutputs, y0)

    # Convert x0, u0, y0 to arrays, if needed
    if np.isscalar(x0): x0 = np.ones((nstates,)) * x0
    if np.isscalar(u0): u0 = np.ones((ninputs,)) * u0
    if np.isscalar(y0): y0 = np.ones((ninputs,)) * y0

    # Discrete-time not yet supported
    if isdtime(sys, strict=True):
        raise NotImplementedError(
            "Discrete time systems are not yet supported.")

    # Make sure the input arguments match the sizes of the system
    if len(x0) != nstates or \
       (u0 is not None and len(u0) != ninputs) or \
       (y0 is not None and len(y0) != noutputs) or \
       (dx0 is not None and len(dx0) != nstates):
        raise ValueError("Length of input arguments does not match system.")

    # Update the parameter values
    sys._update_params(params)

    # Decide what variables to minimize
    if all([x is None for x in (iu, iy, ix, idx)]):
        # Special cases: either inputs or outputs are constrained
        if y0 is None:
            # Take u0 as fixed and minimize over x
            # TODO: update to allow discrete time systems
            def ode_rhs(z): return sys._rhs(t, z, u0)
            result = root(ode_rhs, x0, **kw)
            z = (result.x, u0, sys._out(t, result.x, u0))
        else:
            # Take y0 as fixed and minimize over x and u
            def rootfun(z):
                # Split z into x and u
                x, u = np.split(z, [nstates])
                # TODO: update to allow discrete time systems
                return np.concatenate(
                    (sys._rhs(t, x, u), sys._out(t, x, u) - y0), axis=0)
            z0 = np.concatenate((x0, u0), axis=0)   # Put variables together
            result = root(rootfun, z0, **kw)        # Find the eq point
            x, u = np.split(result.x, [nstates])    # Split result back in two
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
        # The mechanism by which we implement the root finding function is to
        # map the subset of variables we are searching over into the inputs
        # and states, and then return a function that represents the equations
        # we are trying to solve.
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
        # `ix`, and `idx` lists that were passed as arguments to `find_eqpt`
        # and were processed above.

        # Get the states and inputs that were not listed as fixed
        state_vars = np.delete(np.array(range(nstates)), ix)
        input_vars = np.delete(np.array(range(ninputs)), iu)

        # Set the outputs and derivs that will serve as constraints
        output_vars = np.array(iy)
        deriv_vars = np.array(idx)

        # Verify that the number of degrees of freedom all add up correctly
        num_freedoms = len(state_vars) + len(input_vars)
        num_constraints = len(output_vars) + len(deriv_vars)
        if num_constraints != num_freedoms:
            warn("Number of constraints (%d) does not match number of degrees "
                 "of freedom (%d).  Results may be meaningless." %
                 (num_constraints, num_freedoms))

        # Make copies of the state and input variables to avoid overwriting
        # and convert to floats (in case ints were used for initial conditions)
        x = np.array(x0, dtype=float)
        u = np.array(u0, dtype=float)
        dx0 = np.array(dx0, dtype=float) if dx0 is not None \
            else np.zeros(x.shape)

        # Keep track of the number of states in the set of free variables
        nstate_vars = len(state_vars)
        dtime = isdtime(sys, strict=True)

        def rootfun(z):
            # Map the vector of values into the states and inputs
            x[state_vars] = z[:nstate_vars]
            u[input_vars] = z[nstate_vars:]

            # Compute the update and output maps
            dx = sys._rhs(t, x, u) - dx0
            if dtime: dx -= x           # TODO: check
            dy = sys._out(t, x, u) - y0

            # Map the results into the constrained variables
            return np.concatenate((dx[deriv_vars], dy[output_vars]), axis=0)

        # Set the initial condition for the root finding algorithm
        z0 = np.concatenate((x[state_vars], u[input_vars]), axis=0)

        # Finally, call the root finding function
        result = root(rootfun, z0, **kw)

        # Extract out the results and insert into x and u
        x[state_vars] = result.x[:nstate_vars]
        u[input_vars] = result.x[nstate_vars:]
        z = (x, u, sys._out(t, x, u))

    # Return the result based on what the user wants and what we found
    if not return_y: z = z[0:2]     # Strip y from result if not desired
    if return_result:
        # Return whatever we got, along with the result dictionary
        return z + (result,)
    elif result.success:
        # Return the result of the optimization
        return z
    else:
        # Something went wrong, don't return anything
        return (None, None, None) if return_y else (None, None)


# Linearize an input/output system
def linearize(sys, xeq, ueq=[], t=0, params={}, **kw):
    """Linearize an input/output system at a given state and input.

    This function computes the linearization of an input/output system at a
    given state and input value and returns a :class:`control.StateSpace`
    object.  The eavaluation point need not be an equilibrium point.

    Parameters
    ----------
    sys : InputOutputSystem
        The system to be linearized
    xeq : array
        The state at which the linearization will be evaluated (does not need
        to be an equlibrium state).
    ueq : array
        The input at which the linearization will be evaluated (does not need
        to correspond to an equlibrium state).
    t : float, optional
        The time at which the linearization will be computed (for time-varying
        systems).
    params : dict, optional
        Parameter values for the systems.  Passed to the evaluation functions
        for the system as default values, overriding internal defaults.

    Returns
    -------
    ss_sys : LinearIOSystem
        The linearization of the system, as a :class:`~control.LinearIOSystem`
        object (which is also a :class:`~control.StateSpace` object.

    """
    if not isinstance(sys, InputOutputSystem):
        raise TypeError("Can only linearize InputOutputSystem types")
    return sys.linearize(xeq, ueq, t=t, params=params, **kw)


# Utility function to find the size of a system parameter
def _find_size(sysval, vecval):
    if sysval is not None:
        return sysval
    elif hasattr(vecval, '__len__'):
        return len(vecval)
    elif vecval is None:
        return 0
    else:
        raise ValueError("Can't determine size of system component.")


# Convert a state space system into an input/output system (wrapper)
def ss2io(*args, **kw): return LinearIOSystem(*args, **kw)
ss2io.__doc__ = LinearIOSystem.__init__.__doc__


# Convert a transfer function into an input/output system (wrapper)
def tf2io(*args, **kw):
    """Convert a transfer function into an I/O system"""
    # TODO: add remaining documentation
    # Convert the system to a state space system
    linsys = tf2ss(*args)

    # Now convert the state space system to an I/O system
    return LinearIOSystem(linsys, **kw)
