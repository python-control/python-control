# iosys.py - I/O system class and helper functions
# RMM, 13 Mar 2022

"""I/O system class and helper functions.

This module implements the `InputOutputSystem` class, which is used as a
parent class for `LTI`, `StateSpace`, `TransferFunction`,
`NonlinearIOSystem`, class:`FrequencyResponseData`, `InterconnectedSystem`
and other similar classes that allow naming of signals.

"""

import re
from copy import deepcopy

import numpy as np

from . import config
from .exception import ControlIndexError

__all__ = ['InputOutputSystem', 'NamedSignal', 'issiso', 'timebase',
           'common_timebase', 'isdtime', 'isctime', 'iosys_repr']

# Define module default parameter values
_iosys_defaults = {
    'iosys.state_name_delim': '_',
    'iosys.duplicate_system_name_prefix': '',
    'iosys.duplicate_system_name_suffix': '$copy',
    'iosys.linearized_system_name_prefix': '',
    'iosys.linearized_system_name_suffix': '$linearized',
    'iosys.sampled_system_name_prefix': '',
    'iosys.sampled_system_name_suffix': '$sampled',
    'iosys.indexed_system_name_prefix': '',
    'iosys.indexed_system_name_suffix': '$indexed',
    'iosys.converted_system_name_prefix': '',
    'iosys.converted_system_name_suffix': '$converted',
    'iosys.repr_format': 'eval',
    'iosys.repr_show_count': True,
}


# Named signal class
class NamedSignal(np.ndarray):
    """Named signal with label-based access.

    This class modifies the `numpy.ndarray` class and allows signals to
    be accessed using the signal name in addition to indices and slices.

    Signals can be either a 2D array, index by signal and time, or a 3D
    array, indexed by signal, trace, and time.

    Attributes
    ----------
    signal_labels : list of str
        Label names for each of the signal elements in the signal.
    trace_labels : list of str, optional
        Label names for each of the traces in the signal (if multi-trace).

    Examples
    --------
    >>> sys = ct.rss(
    ...     states=['p1', 'p2', 'p3'], inputs=['u1', 'u2'], outputs=['y'])
    >>> resp = ct.step_response(sys)
    >>> resp.states['p1', 'u1']  # Step response from u1 to p1
    NamedSignal(...)

    """
    def __new__(cls, input_array, signal_labels=None, trace_labels=None):
        # See https://numpy.org/doc/stable/user/basics.subclassing.html
        obj = np.asarray(input_array).view(cls)     # Cast to our class type
        obj.signal_labels = signal_labels           # Save signal labels
        obj.trace_labels = trace_labels             # Save trace labels
        obj.data_shape = input_array.shape          # Save data shape
        return obj                                  # Return new object

    def __array_finalize__(self, obj):
        # See https://numpy.org/doc/stable/user/basics.subclassing.html
        if obj is None:
            return
        self.signal_labels = getattr(obj, 'signal_labels', None)
        self.trace_labels = getattr(obj, 'trace_labels', None)
        self.data_shape = getattr(obj, 'data_shape', None)

    def _parse_key(self, key, labels=None, level=0):
        if labels is None:
            labels = self.signal_labels
        try:
            if isinstance(key, str):
                key = labels.index(item := key)
                if level == 0 and len(self.data_shape) < 2:
                    # This is the only signal => use it
                    return ()
            elif isinstance(key, list):
                keylist = []
                for item in key:        # use for loop to save item for error
                    keylist.append(
                        self._parse_key(item, labels=labels, level=level+1))
                if level == 0 and key != keylist and len(self.data_shape) < 2:
                    raise ControlIndexError
                key = keylist
            elif isinstance(key, tuple) and len(key) > 0:
                keylist = []
                keylist.append(
                    self._parse_key(
                        item := key[0], labels=self.signal_labels,
                        level=level+1))
                if len(key) > 1:
                    keylist.append(
                        self._parse_key(
                            item := key[1], labels=self.trace_labels,
                            level=level+1))
                if level == 0 and key[:len(keylist)] != tuple(keylist) \
                   and len(keylist) > len(self.data_shape) - 1:
                    raise ControlIndexError
                for i in range(2, len(key)):
                    keylist.append(key[i])      # pass on remaining elements
                key = tuple(keylist)
        except ValueError:
            raise ValueError(f"unknown signal name '{item}'")
        except ControlIndexError:
            raise ControlIndexError(
                "signal name(s) not valid for squeezed data")

        return key

    def __getitem__(self, key):
        return super().__getitem__(self._parse_key(key))

    def __repr__(self):
        out = "NamedSignal(\n"
        out += repr(np.array(self))             # NamedSignal -> array
        if self.signal_labels is not None:
            out += f",\nsignal_labels={self.signal_labels}"
        if self.trace_labels is not None:
            out += f",\ntrace_labels={self.trace_labels}"
        out += ")"
        return out


class InputOutputSystem():
    """Base class for input/output systems.

    The `InputOutputSystem` class allows (possibly nonlinear) input/output
    systems to be represented in Python.  It is used as a parent class for
    a set of subclasses that are used to implement specific structures and
    operations for different types of input/output dynamical systems.

    The timebase for the system, `dt`, is used to specify whether the system
    is operating in continuous or discrete time. It can have the following
    values:

      * `dt` = None: No timebase specified
      * `dt` = 0: Continuous time system
      * `dt` > 0: Discrete time system with sampling time dt
      * `dt` = True: Discrete time system with unspecified sampling time

    Parameters
    ----------
    inputs : int, list of str, or None
        Description of the system inputs.  This can be given as an integer
        count or a list of strings that name the individual signals.  If an
        integer count is specified, the names of the signal will be of the
        form 's[i]' (where 's' is given by the `input_prefix` parameter and
        has default value 'u').  If this parameter is not given or given as
        None, the relevant quantity will be determined when possible
        based on other information provided to functions using the system.
    outputs : int, list of str, or None
        Description of the system outputs.  Same format as `inputs`, with
        the prefix given by `output_prefix` (defaults to 'y').
    states : int, list of str, or None
        Description of the system states.  Same format as `inputs`, with
        the prefix given by `state_prefix` (defaults to 'x').
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous time, True
        indicates discrete time with unspecified sampling time, positive
        number is discrete time with specified sampling time, None indicates
        unspecified timebase (either continuous or discrete time).
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.
    params : dict, optional
        Parameter values for the system.  Passed to the evaluation functions
        for the system as default values, overriding internal defaults.

    Attributes
    ----------
    ninputs, noutputs, nstates : int
        Number of input, output, and state variables.
    input_index, output_index, state_index : dict
        Dictionary of signal names for the inputs, outputs, and states and
        the index of the corresponding array.
    input_labels, output_labels, state_labels : list of str
        List of signal names for inputs, outputs, and states.
    shape : tuple
        2-tuple of I/O system dimension, (noutputs, ninputs).

    Other Parameters
    ----------------
    input_prefix : string, optional
        Set the prefix for input signals.  Default = 'u'.
    output_prefix : string, optional
        Set the prefix for output signals.  Default = 'y'.
    state_prefix : string, optional
        Set the prefix for state signals.  Default = 'x'.
    repr_format : str
        String representation format.  See `control.iosys_repr`.

    """
    # Allow ndarray * IOSystem to give IOSystem._rmul_() priority
    # https://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    __array_priority__ = 20

    def __init__(
            self, name=None, inputs=None, outputs=None, states=None,
            input_prefix='u', output_prefix='y', state_prefix='x', **kwargs):

        # system name
        self.name = self._name_or_default(name)

        # Parse and store the number of inputs and outputs
        self.set_inputs(inputs, prefix=input_prefix)
        self.set_outputs(outputs, prefix=output_prefix)
        self.set_states(states, prefix=state_prefix)

        # Process timebase: if not given use default, but allow None as value
        self.dt = _process_dt_keyword(kwargs)

        self._repr_format = kwargs.pop('repr_format', None)

        # Make sure there were no other keywords
        if kwargs:
            raise TypeError("unrecognized keywords: ", str(kwargs))

    # Keep track of the keywords that we recognize
    _kwargs_list = [
        'name', 'inputs', 'outputs', 'states', 'input_prefix',
        'output_prefix', 'state_prefix', 'dt']

    #
    # Functions to manipulate the system name
    #
    _idCounter = 0              # Counter for creating generic system name

    # Return system name
    def _name_or_default(self, name=None, prefix_suffix_name=None):
        if name is None:
            name = "sys[{}]".format(InputOutputSystem._idCounter)
            InputOutputSystem._idCounter += 1
        elif re.match(r".*\..*", name):
            raise ValueError(f"invalid system name '{name}' ('.' not allowed)")

        prefix = "" if prefix_suffix_name is None else config.defaults[
            'iosys.' + prefix_suffix_name + '_system_name_prefix']
        suffix = "" if prefix_suffix_name is None else config.defaults[
            'iosys.' + prefix_suffix_name + '_system_name_suffix']
        return prefix + name + suffix

    # Check if system name is generic
    def _generic_name_check(self):
        return re.match(r'^sys\[\d*\]$', self.name) is not None

    #
    # Class attributes
    #
    # These attributes are defined as class attributes so that they are
    # documented properly.  They are "overwritten" in __init__.
    #

    #: Number of system inputs.
    #:
    #: :meta hide-value:
    ninputs = None

    #: Number of system outputs.
    #:
    #: :meta hide-value:
    noutputs = None

    #: Number of system states.
    #:
    #: :meta hide-value:
    nstates = None

    #: System timebase.
    #:
    #: :meta hide-value:
    dt = None

    #
    # System representation
    #

    def __str__(self):
        """String representation of an input/output object"""
        out = f"<{self.__class__.__name__}>: {self.name}"
        out += f"\nInputs ({self.ninputs}): {self.input_labels}"
        out += f"\nOutputs ({self.noutputs}): {self.output_labels}"
        if self.nstates is not None:
            out += f"\nStates ({self.nstates}): {self.state_labels}"
        out += self._dt_repr(separator="\n", space=" ")
        return out

    def __repr__(self):
        return iosys_repr(self, format=self.repr_format)

    def _repr_info_(self, html=False):
        out = f"<{self.__class__.__name__} {self.name}: " + \
            f"{list(self.input_labels)} -> {list(self.output_labels)}"
        out += self._dt_repr(separator=", ", space="") + ">"

        if html:
            # Replace symbols that might be interpreted by HTML processing
            # TODO: replace -> with right arrow (later)
            escape_chars = {
                '$': r'\$',
                '<': '&lt;',
                '>': '&gt;',
            }
            return "".join([c if c not in escape_chars else
                            escape_chars[c] for c in out])
        else:
            return out

    def _repr_eval_(self):
        # Defaults to _repr_info_; override in subclasses
        return self._repr_info_()

    def _repr_latex_(self):
        # Defaults to using __repr__; override in subclasses
        return None

    def _repr_html_(self):
        # Defaults to using __repr__; override in subclasses
        return None

    def _repr_markdown_(self):
        return self._repr_html_()

    @property
    def repr_format(self):
        """String representation format.

        Format used in creating the representation for the system:

          * 'info' : <IOSystemType sysname: [inputs] -> [outputs]>
          * 'eval' : system specific, loadable representation
          * 'latex' : HTML/LaTeX representation of the object

        The default representation for an input/output is set to 'eval'.
        This value can be changed for an individual system by setting the
        `repr_format` parameter when the system is created or by setting
        the `repr_format` property after system creation.  Set
        `config.defaults['iosys.repr_format']` to change for all I/O systems
        or use the `repr_format` parameter/attribute for a single system.

        """
        return self._repr_format if self._repr_format is not None \
            else config.defaults['iosys.repr_format']

    @repr_format.setter
    def repr_format(self, value):
        self._repr_format = value

    def _label_repr(self, show_count=None):
        show_count = config._get_param(
            'iosys', 'repr_show_count', show_count, True)
        out, count = "", 0

        # Include the system name if not generic
        if not self._generic_name_check():
            name_spec = f"name='{self.name}'"
            count += len(name_spec)
            out += name_spec

        # Include the state, output, and input names if not generic
        for sig_name, sig_default, sig_labels in zip(
                ['states', 'outputs', 'inputs'],
                ['x', 'y', 'u'],        # TODO: replace with defaults
                [self.state_labels, self.output_labels, self.input_labels]):
            if sig_name == 'states' and self.nstates is None:
                continue

            # Check if the signal labels are generic
            if any([re.match(r'^' + sig_default + r'\[\d*\]$', label) is None
                    for label in sig_labels]):
                spec = f"{sig_name}={sig_labels}"
            elif show_count:
                spec = f"{sig_name}={len(sig_labels)}"
            else:
                spec = ""

            # Append the specification string to the output, with wrapping
            if count == 0:
                count = len(spec)       # no system name => suppress comma
            elif count + len(spec) > 72:
                # TODO: check to make sure a single line is enough (minor)
                out += ",\n"
                count = len(spec)
            elif len(spec) > 0:
                out += ", "
                count += len(spec) + 2
            out += spec

        return out

    def _dt_repr(self, separator="\n", space=""):
        if config.defaults['control.default_dt'] != self.dt:
            return "{separator}dt{space}={space}{dt}".format(
                separator=separator, space=space,
                dt='None' if self.dt is None else self.dt)
        else:
            return ""

    # Find a list of signals by name, index, or pattern
    def _find_signals(self, name_list, sigdict):
        if not isinstance(name_list, (list, tuple)):
            name_list = [name_list]

        index_list = []
        for name in name_list:
            # Look for signal ranges (slice-like or base name)
            ms = re.match(r'([\w$]+)\[([\d]*):([\d]*)\]$', name)  # slice
            mb = re.match(r'([\w$]+)$', name)                     # base
            if ms:
                base = ms.group(1)
                start = None if ms.group(2) == '' else int(ms.group(2))
                stop = None if ms.group(3) == '' else int(ms.group(3))
                for var in sigdict:
                    # Find variables that match
                    msig = re.match(r'([\w$]+)\[([\d]+)\]$', var)
                    if msig and msig.group(1) == base and \
                       (start is None or int(msig.group(2)) >= start) and \
                       (stop is None or int(msig.group(2)) < stop):
                            index_list.append(sigdict.get(var))
            elif mb and sigdict.get(name, None) is None:
                # Try to use name as a base name
                for var in sigdict:
                    msig = re.match(name + r'\[([\d]+)\]$', var)
                    if msig:
                        index_list.append(sigdict.get(var))
            else:
                index_list.append(sigdict.get(name, None))

        return None if len(index_list) == 0 or \
            any([idx is None for idx in index_list]) else index_list

    def _copy_names(self, sys, prefix="", suffix="", prefix_suffix_name=None):
        """copy the signal and system name of sys. Name is given as a keyword
        in case a specific name (e.g. append 'linearized') is desired. """
        # Figure out the system name and assign it
        self.name = _extended_system_name(
            sys.name, prefix, suffix, prefix_suffix_name)

        # Name the inputs, outputs, and states
        self.input_index = sys.input_index.copy()
        self.output_index = sys.output_index.copy()
        if self.nstates and sys.nstates:
            # only copy state names for state space systems
            self.state_index = sys.state_index.copy()

    def copy(self, name=None, use_prefix_suffix=True):
        """Make a copy of an input/output system.

        A copy of the system is made, with a new name.  The `name` keyword
        can be used to specify a specific name for the system.  If no name
        is given and `use_prefix_suffix` is True, the name is constructed
        by prepending `config.defaults['iosys.duplicate_system_name_prefix']`
        and appending `config.defaults['iosys.duplicate_system_name_suffix']`.
        Otherwise, a generic system name of the form 'sys[<id>]' is used,
        where '<id>' is based on an internal counter.

        Parameters
        ----------
        name : str, optional
            Name of the newly created system.

        use_prefix_suffix : bool, optional
            If True and `name` is None, set the name of the new system
            to the name of the original system with prefix
            `config.defaults['duplicate_system_name_prefix']` and
            suffix `config.defaults['duplicate_system_name_suffix']`.

        Returns
        -------
        `InputOutputSystem`

        """
        # Create a copy of the system
        newsys = deepcopy(self)

        # Update the system name
        if name is None and use_prefix_suffix:
            # Get the default prefix and suffix to use
            newsys.name = self._name_or_default(
                self.name, prefix_suffix_name='duplicate')
        else:
            newsys.name = self._name_or_default(name)

        return newsys

    def set_inputs(self, inputs, prefix='u'):
        """Set the number/names of the system inputs.

        Parameters
        ----------
        inputs : int, list of str, or None
            Description of the system inputs.  This can be given as an integer
            count or as a list of strings that name the individual signals.
            If an integer count is specified, the names of the signal will be
            of the form 'u[i]' (where the prefix 'u' can be changed using the
            optional prefix parameter).
        prefix : string, optional
            If `inputs` is an integer, create the names of the states using
            the given prefix (default = 'u').  The names of the input will be
            of the form 'prefix[i]'.

        """
        self.ninputs, self.input_index = \
            _process_signal_list(inputs, prefix=prefix)

    def find_input(self, name):
        """Find the index for an input given its name (None if not found).

        Parameters
        ----------
        name : str
            Signal name for the desired input.

        Returns
        -------
        int
            Index of the named input.

        """
        return self.input_index.get(name, None)

    def find_inputs(self, name_list):
        """Return list of indices matching input spec (None if not found).

        Parameters
        ----------
        name_list : str or list of str
            List of signal specifications for the desired inputs.  A
            signal can be described by its name or by a slice-like
            description of the form 'start:end` where 'start' and
            'end' are signal names.  If either is omitted, it is taken
            as the first or last signal, respectively.

        Returns
        -------
        list of int
            List of indices for the specified inputs.

        """
        return self._find_signals(name_list, self.input_index)

    # Property for getting and setting list of input signals
    input_labels = property(
        lambda self: list(self.input_index.keys()),     # getter
        set_inputs,                                     # setter
        doc="List of labels for the input signals.")

    def set_outputs(self, outputs, prefix='y'):
        """Set the number/names of the system outputs.

        Parameters
        ----------
        outputs : int, list of str, or None
            Description of the system outputs.  This can be given as an
            integer count or as a list of strings that name the individual
            signals.  If an integer count is specified, the names of the
            signal will be of the form 'y[i]' (where the prefix 'y' can be
            changed using the optional prefix parameter).
        prefix : string, optional
            If `outputs` is an integer, create the names of the states using
            the given prefix (default = 'y').  The names of the input will be
            of the form 'prefix[i]'.

        """
        self.noutputs, self.output_index = \
            _process_signal_list(outputs, prefix=prefix)

    def find_output(self, name):
        """Find the index for a output given its name (None if not found).

        Parameters
        ----------
        name : str
            Signal name for the desired output.

        Returns
        -------
        int
            Index of the named output.

        """
        return self.output_index.get(name, None)

    def find_outputs(self, name_list):
        """Return list of indices matching output spec (None if not found).

        Parameters
        ----------
        name_list : str or list of str
            List of signal specifications for the desired outputs.  A
            signal can be described by its name or by a slice-like
            description of the form 'start:end` where 'start' and
            'end' are signal names.  If either is omitted, it is taken
            as the first or last signal, respectively.

        Returns
        -------
        list of int
            List of indices for the specified outputs.

        """
        return self._find_signals(name_list, self.output_index)

    # Property for getting and setting list of output signals
    output_labels = property(
        lambda self: list(self.output_index.keys()),     # getter
        set_outputs,                                     # setter
        doc="List of labels for the output signals.")

    def set_states(self, states, prefix='x'):
        """Set the number/names of the system states.

        Parameters
        ----------
        states : int, list of str, or None
            Description of the system states.  This can be given as an integer
            count or as a list of strings that name the individual signals.
            If an integer count is specified, the names of the signal will be
            of the form 'x[i]' (where the prefix 'x' can be changed using the
            optional prefix parameter).
        prefix : string, optional
            If `states` is an integer, create the names of the states using
            the given prefix (default = 'x').  The names of the input will be
            of the form 'prefix[i]'.

        """
        self.nstates, self.state_index = \
            _process_signal_list(states, prefix=prefix, allow_dot=True)

    def find_state(self, name):
        """Find the index for a state given its name (None if not found).

        Parameters
        ----------
        name : str
            Signal name for the desired state.

        Returns
        -------
        int
            Index of the named state.

        """
        return self.state_index.get(name, None)

    def find_states(self, name_list):
        """Return list of indices matching state spec (None if not found).

        Parameters
        ----------
        name_list : str or list of str
            List of signal specifications for the desired states.  A
            signal can be described by its name or by a slice-like
            description of the form 'start:end` where 'start' and
            'end' are signal names.  If either is omitted, it is taken
            as the first or last signal, respectively.

        Returns
        -------
        list of int
            List of indices for the specified states..

        """
        return self._find_signals(name_list, self.state_index)

    # Property for getting and setting list of state signals
    state_labels = property(
        lambda self: list(self.state_index.keys()),     # getter
        set_states,                                     # setter
        doc="List of labels for the state signals.")

    @property
    def shape(self):
        """2-tuple of I/O system dimension, (noutputs, ninputs)."""
        return (self.noutputs, self.ninputs)

    # TODO: add dict as a means to selective change names?  [GH #1019]
    def update_names(self, **kwargs):
        """update_names([name, inputs, outputs, states])

        Update signal and system names for an I/O system.

        Parameters
        ----------
        name : str, optional
            New system name.
        inputs : list of str, int, or None, optional
            List of strings that name the individual input signals.  If
            given as an integer or None, signal names default to the form
            'u[i]'.  See `InputOutputSystem` for more information.
        outputs : list of str, int, or None, optional
            Description of output signals; defaults to 'y[i]'.
        states : int, list of str, int, or None, optional
            Description of system states; defaults to 'x[i]'.
        input_prefix : string, optional
            Set the prefix for input signals.  Default = 'u'.
        output_prefix : string, optional
            Set the prefix for output signals.  Default = 'y'.
        state_prefix : string, optional
            Set the prefix for state signals.  Default = 'x'.

        """
        self.name = kwargs.pop('name', self.name)
        if 'inputs' in kwargs:
            ninputs, input_index = _process_signal_list(
                kwargs.pop('inputs'), prefix=kwargs.pop('input_prefix', 'u'))
            if self.ninputs and self.ninputs != ninputs:
                raise ValueError("number of inputs does not match system size")
            self.input_index = input_index
        if 'outputs' in kwargs:
            noutputs, output_index = _process_signal_list(
                kwargs.pop('outputs'), prefix=kwargs.pop('output_prefix', 'y'))
            if self.noutputs and self.noutputs != noutputs:
                raise ValueError("number of outputs does not match system size")
            self.output_index = output_index
        if 'states' in kwargs:
            nstates, state_index = _process_signal_list(
                kwargs.pop('states'), prefix=kwargs.pop('state_prefix', 'x'))
            if self.nstates != nstates:
                raise ValueError("number of states does not match system size")
            self.state_index = state_index

        # Make sure we processed all of the arguments
        if kwargs:
            raise TypeError("unrecognized keywords: ", str(kwargs))


    def isctime(self, strict=False):
        """
        Check to see if a system is a continuous-time system.

        Parameters
        ----------
        strict : bool, optional
            If strict is True, make sure that timebase is not None.  Default
            is False.

        """
        # If no timebase is given, answer depends on strict flag
        if self.dt is None:
            return True if not strict else False
        return self.dt == 0

    def isdtime(self, strict=False):
        """
        Check to see if a system is a discrete-time system.

        Parameters
        ----------
        strict : bool, optional
            If strict is True, make sure that timebase is not None.  Default
            is False.
        """

        # If no timebase is given, answer depends on strict flag
        if self.dt == None:
            return True if not strict else False

        # Look for dt > 0 (also works if dt = True)
        return self.dt > 0

    def issiso(self):
        """Check to see if a system is single input, single output."""
        return self.ninputs == 1 and self.noutputs == 1


# Test to see if a system is SISO
def issiso(sys, strict=False):
    """
    Check to see if a system is single input, single output.

    Parameters
    ----------
    sys : I/O or LTI system
        System to be checked.
    strict : bool (default = False)
        If strict is True, do not treat scalars as SISO.
    """
    if isinstance(sys, (int, float, complex, np.number)) and not strict:
        return True
    elif not isinstance(sys, InputOutputSystem):
        raise ValueError("Object is not an I/O or LTI system")

    # Done with the tricky stuff...
    return sys.issiso()

# Return the timebase (with conversion if unspecified)
def timebase(sys, strict=True):
    """Return the timebase for a system.

    dt = timebase(sys)

    returns the timebase for a system 'sys'.  If the strict option is
    set to True, `dt` = True will be returned as 1.

    Parameters
    ----------
    sys : `InputOutputSystem` or float
        System whose timebase is to be determined.
    strict : bool, optional
        Whether to implement strict checking.  If set to True (default),
        a float will always be returned (`dt` = True will be returned as 1).

    Returns
    -------
    dt : timebase
        Timebase for the system (0 = continuous time, None = unspecified).

    """
    # System needs to be either a constant or an I/O or LTI system
    if isinstance(sys, (int, float, complex, np.number)):
        return None
    elif not isinstance(sys, InputOutputSystem):
        raise ValueError("Timebase not defined")

    # Return the sample time, with conversion to float if strict is false
    if sys.dt == None:
        return None
    elif strict:
        return float(sys.dt)

    return sys.dt

def common_timebase(dt1, dt2):
    """
    Find the common timebase when interconnecting systems.

    Parameters
    ----------
    dt1, dt2 : `InputOutputSystem` or float
        Number or system with a 'dt' attribute (e.g. `TransferFunction`
        or `StateSpace` system).

    Returns
    -------
    dt : number
        The common timebase of dt1 and dt2, as specified in
        :ref:`conventions-ref`.

    Raises
    ------
    ValueError
        When no compatible time base can be found.

    """
    # explanation:
    # if either dt is None, they are compatible with anything
    # if either dt is True (discrete with unspecified time base),
    #   use the timebase of the other, if it is also discrete
    # otherwise both dt's must be equal
    if hasattr(dt1, 'dt'):
        dt1 = dt1.dt
    if hasattr(dt2, 'dt'):
        dt2 = dt2.dt

    if dt1 is None:
        return dt2
    elif dt2 is None:
        return dt1
    elif dt1 is True:
        if dt2 > 0:
            return dt2
        else:
            raise ValueError("Systems have incompatible timebases")
    elif dt2 is True:
        if dt1 > 0:
            return dt1
        else:
            raise ValueError("Systems have incompatible timebases")
    elif np.isclose(dt1, dt2):
        return dt1
    else:
        raise ValueError("Systems have incompatible timebases")

# Check to see if a system is a discrete-time system
def isdtime(sys=None, strict=False, dt=None):
    """
    Check to see if a system is a discrete-time system.

    Parameters
    ----------
    sys : I/O system, optional
        System to be checked.
    dt : None or number, optional
        Timebase to be checked.
    strict : bool, default=False
        If strict is True, make sure that timebase is not None.
    """

    # See if we were passed a timebase instead of a system
    if sys is None:
        if dt is None:
            return True if not strict else False
        else:
            return dt > 0
    elif dt is not None:
        raise TypeError("passing both system and timebase not allowed")

    # Check timebase of the system
    if isinstance(sys, (int, float, complex, np.number)):
        # Constants OK as long as strict checking is off
        return True if not strict else False
    else:
        return sys.isdtime(strict)


# Check to see if a system is a continuous-time system
def isctime(sys=None, dt=None, strict=False):
    """
    Check to see if a system is a continuous-time system.

    Parameters
    ----------
    sys : I/O system, optional
        System to be checked.
    dt : None or number, optional
        Timebase to be checked.
    strict : bool (default = False)
        If strict is True, make sure that timebase is not None.
    """

    # See if we were passed a timebase instead of a system
    if sys is None:
        if dt is None:
            return True if not strict else False
        else:
            return dt == 0
    elif dt is not None:
        raise TypeError("passing both system and timebase not allowed")

    # Check timebase of the system
    if isinstance(sys, (int, float, complex, np.number)):
        # Constants OK as long as strict checking is off
        return True if not strict else False
    else:
        return sys.isctime(strict)


def iosys_repr(sys, format=None):
    """Return representation of an I/O system.

    Parameters
    ----------
    sys : `InputOutputSystem`
        System for which the representation is generated.
    format : str
        Format to use in creating the representation:

          * 'info' : <IOSystemType sysname: [inputs] -> [outputs]>
          * 'eval' : system specific, loadable representation
          * 'latex' : HTML/LaTeX representation of the object

    Returns
    -------
    str
        String representing the input/output system.

    Notes
    -----
    By default, the representation for an input/output is set to 'eval'.
    Set `config.defaults['iosys.repr_format']` to change for all I/O systems
    or use the `repr_format` parameter for a single system.

    Jupyter will automatically use the 'latex' representation for I/O
    systems, when available.

    """
    format = config.defaults['iosys.repr_format'] if format is None else format
    match format:
        case 'info':
            return sys._repr_info_()
        case 'eval':
            return sys._repr_eval_()
        case 'latex':
            return sys._repr_html_()
        case _:
            raise ValueError(f"format '{format}' unknown")


# Utility function to parse iosys keywords
def _process_iosys_keywords(
        keywords={}, defaults={}, static=False, end=False):
    """Process iosys specification.

    This function processes the standard keywords used in initializing an
    I/O system.  It first looks in the `keywords` dictionary to see if a
    value is specified.  If not, the `defaults` dictionary is used.  The
    `defaults` dictionary can also be set to an `InputOutputSystem` object,
    which is useful for copy constructors that change system/signal names.

    If `end` is True, then generate an error if there are any remaining
    keywords.

    """
    # If default is a system, redefine as a dictionary
    if isinstance(defaults, InputOutputSystem):
        sys = defaults
        defaults = {
            'name': sys.name, 'inputs': sys.input_labels,
            'outputs': sys.output_labels, 'dt': sys.dt}

        if sys.nstates is not None:
            defaults['states'] = sys.state_labels
    else:
        sys = None

    # Sort out singular versus plural signal names
    for singular in ['input', 'output', 'state']:
        kw = singular + 's'
        if singular in keywords and kw in keywords:
            raise TypeError(f"conflicting keywords '{singular}' and '{kw}'")

        if singular in keywords:
            keywords[kw] = keywords.pop(singular)

    # Utility function to get keyword with defaults, processing
    def pop_with_default(kw, defval=None, return_list=True):
        val = keywords.pop(kw, None)
        if val is None:
            val = defaults.get(kw, defval)
        if return_list and isinstance(val, str):
            val = [val]         # make sure to return a list
        return val

    # Process system and signal names
    name = pop_with_default('name', return_list=False)
    inputs = pop_with_default('inputs')
    outputs = pop_with_default('outputs')
    states = pop_with_default('states')

    # If we were given a system, make sure sizes match list lengths
    if sys:
        if isinstance(inputs, list) and sys.ninputs != len(inputs):
            raise ValueError("wrong number of input labels given")
        if isinstance(outputs, list) and sys.noutputs != len(outputs):
            raise ValueError("wrong number of output labels given")
        if sys.nstates is not None and \
           isinstance(states, list) and sys.nstates != len(states):
            raise ValueError("wrong number of state labels given")

    # Process timebase: if not given use default, but allow None as value
    dt = _process_dt_keyword(keywords, defaults, static=static)

    # If desired, make sure we processed all keywords
    if end and keywords:
        raise TypeError("unrecognized keywords: ", str(keywords))

    # Return the processed keywords
    return name, inputs, outputs, states, dt

#
# Parse 'dt' for I/O system
#
# The 'dt' keyword is used to set the timebase for a system.  Its
# processing is a bit unusual: if it is not specified at all, then the
# value is pulled from config.defaults['control.default_dt'].  But
# since 'None' is an allowed value, we can't just use the default if
# dt is None.  Instead, we have to look to see if it was listed as a
# variable keyword.
#
# In addition, if a system is static and dt is not specified, we set dt =
# None to allow static systems to be combined with either discrete-time or
# continuous-time systems.
#
# TODO: update all 'dt' processing to call this function, so that
# everything is done consistently.
#
def _process_dt_keyword(keywords, defaults={}, static=False):
    if static and 'dt' not in keywords and 'dt' not in defaults:
        dt = None
    elif 'dt' in keywords:
        dt = keywords.pop('dt')
    elif 'dt' in defaults:
        dt = defaults.pop('dt')
    else:
        dt = config.defaults['control.default_dt']

    # Make sure that the value for dt is valid
    if dt is not None and not isinstance(dt, (bool, int, float)) or \
       isinstance(dt, (bool, int, float)) and dt < 0:
        raise ValueError(f"invalid timebase, dt = {dt}")

    return dt


# Utility function to parse a list of signals
def _process_signal_list(signals, prefix='s', allow_dot=False):
    if signals is None:
        # No information provided; try and make it up later
        return None, {}

    elif isinstance(signals, (int, np.integer)):
        # Number of signals given; make up the names
        return signals, {'%s[%d]' % (prefix, i): i for i in range(signals)}

    elif isinstance(signals, str):
        # Single string given => single signal with given name
        if not allow_dot and re.match(r".*\..*", signals):
            raise ValueError(
                f"invalid signal name '{signals}' ('.' not allowed)")
        return 1, {signals: 0}

    elif all(isinstance(s, str) for s in signals):
        # Use the list of strings as the signal names
        for signal in signals:
            if not allow_dot and re.match(r".*\..*", signal):
                raise ValueError(
                    f"invalid signal name '{signal}' ('.' not allowed)")
        return len(signals), {signals[i]: i for i in range(len(signals))}

    else:
        raise TypeError("Can't parse signal list %s" % str(signals))


#
# Utility functions to process signal indices
#
# Signal indices can be specified in one of four ways:
#
# 1. As a positive integer 'm', in which case we return a list
#    corresponding to the first 'm' elements of a range of a given length
#
# 2. As a negative integer '-m', in which case we return a list
#    corresponding to the last 'm' elements of a range of a given length
#
# 3. As a slice, in which case we return the a list corresponding to the
#    indices specified by the slice of a range of a given length
#
# 4. As a list of ints or strings specifying specific indices.  Strings are
#    compared to a list of labels to determine the index.
#
def _process_indices(arg, name, labels, length):
    # Default is to return indices up to a certain length
    arg = length if arg is None else arg

    if isinstance(arg, int):
        # Return the start or end of the list of possible indices
        return list(range(arg)) if arg > 0 else list(range(length))[arg:]

    elif isinstance(arg, slice):
        # Return the indices referenced by the slice
        return list(range(length))[arg]

    elif isinstance(arg, list):
        # Make sure the length is OK
        if len(arg) > length:
            raise ValueError(
                f"{name}_indices list is too long; max length = {length}")

        # Return the list, replacing strings with corresponding indices
        arg=arg.copy()
        for i, idx in enumerate(arg):
            if isinstance(idx, str):
                arg[i] = labels.index(arg[i])
        return arg

    raise ValueError(f"invalid argument for {name}_indices")

#
# Process control and disturbance indices
#
# For systems with inputs and disturbances, the control_indices and
# disturbance_indices keywords are used to specify which is which.  If only
# one is given, the other is assumed to be the remaining indices in the
# system input.  If neither is given, the disturbance inputs are assumed to
# be the same as the control inputs.
#
def _process_control_disturbance_indices(
        sys, control_indices, disturbance_indices):

    if control_indices is None and disturbance_indices is None:
        # Disturbances enter in the same place as the controls
        dist_idx = ctrl_idx = list(range(sys.ninputs))

    elif control_indices is not None:
        # Process the control indices
        ctrl_idx = _process_indices(
            control_indices, 'control', sys.input_labels, sys.ninputs)

        # Disturbance indices are the complement of control indices
        dist_idx = [i for i in range(sys.ninputs) if i not in ctrl_idx]

    else:  # disturbance_indices is not None
        # If passed an integer, count from the end of the input vector
        arg = -disturbance_indices if isinstance(disturbance_indices, int) \
            else disturbance_indices

        dist_idx = _process_indices(
            arg, 'disturbance', sys.input_labels, sys.ninputs)

        # Set control indices to complement disturbance indices
        ctrl_idx = [i for i in range(sys.ninputs) if i not in dist_idx]

    return ctrl_idx, dist_idx


# Process labels
def _process_labels(labels, name, default):
    if isinstance(labels, str):
        labels = [labels.format(i=i) for i in range(len(default))]

    if labels is None:
        labels = default
    elif isinstance(labels, list):
        if len(labels) != len(default):
            raise ValueError(
                f"incorrect length of {name}_labels: {len(labels)}"
                f" instead of {len(default)}")
    else:
        raise ValueError(f"{name}_labels should be a string or a list")

    return labels

#
# Utility function for parsing input/output specifications
#
# This function can be used to convert various forms of signal
# specifications used in the interconnect() function and the
# InterconnectedSystem class into a list of signals.  Signal specifications
# are of one of the following forms (where 'n' is the number of signals in
# the named dictionary):
#
#   i                    system_index = i, signal_list = [0, ..., n]
#   (i,)                 system_index = i, signal_list = [0, ..., n]
#   (i, j)               system_index = i, signal_list = [j]
#   (i, [j1, ..., jn])   system_index = i, signal_list = [j1, ..., jn]
#   'sys'                system_index = i, signal_list = [0, ..., n]
#   'sys.sig'            signal 'sig' in subsys 'sys'
#   ('sys', 'sig')       signal 'sig' in subsys 'sys'
#   'sys.sig[...]'       signals 'sig[...]' (slice) in subsys 'sys'
#   ('sys', j)           signal_index j in subsys 'sys'
#   ('sys', 'sig[...]')  signals 'sig[...]' (slice) in subsys 'sys'
#
# This function returns the subsystem index, a list of indices for the
# system signals, and the gain to use for that set of signals.
#

def _parse_spec(syslist, spec, signame, dictname=None):
    """Parse a signal specification, returning system and signal index."""

    # Parse the signal spec into a system, signal, and gain spec
    if isinstance(spec, int):
        system_spec, signal_spec, gain = spec, None, None
    elif isinstance(spec, str):
        # If we got a dotted string, break up into pieces
        namelist = re.split(r'\.', spec)
        system_spec, gain = namelist[0], None
        signal_spec = None if len(namelist) < 2 else namelist[1]
        if len(namelist) > 2:
            # TODO: expand to allow nested signal names
            raise ValueError(f"couldn't parse signal reference '{spec}'")
    elif isinstance(spec, tuple) and len(spec) <= 3:
        system_spec = spec[0]
        signal_spec = None if len(spec) < 2 else spec[1]
        gain = None if len(spec) < 3 else spec[2]
    else:
        raise ValueError(f"unrecognized signal spec format '{spec}'")

    # Determine the gain
    check_sign = lambda spec: isinstance(spec, str) and spec[0] == '-'
    if (check_sign(system_spec) and gain is not None) or \
       (check_sign(signal_spec) and gain is not None) or \
       (check_sign(system_spec) and check_sign(signal_spec)):
        # Gain is specified multiple times
        raise ValueError(f"gain specified multiple times '{spec}'")
    elif check_sign(system_spec):
        gain = -1
        system_spec = system_spec[1:]
    elif check_sign(signal_spec):
        gain = -1
        signal_spec = signal_spec[1:]
    elif gain is None:
        gain = 1

    # Figure out the subsystem index
    if isinstance(system_spec, int):
        system_index = system_spec
    elif isinstance(system_spec, str):
        syslist_index = {sys.name: i for i, sys in enumerate(syslist)}
        system_index = syslist_index.get(system_spec, None)
        if system_index is None:
            raise ValueError(f"couldn't find system '{system_spec}'")
    else:
        raise ValueError(f"unknown system spec '{system_spec}'")

    # Make sure the system index is valid
    if system_index < 0 or system_index >= len(syslist):
        ValueError(f"system index '{system_index}' is out of range")

    # Figure out the name of the dictionary to use for signal names
    dictname = signame + '_index' if dictname is None else dictname
    signal_dict = getattr(syslist[system_index], dictname)
    nsignals = len(signal_dict)

    # Figure out the signal indices
    if signal_spec is None:
        # No indices given => use the entire range of signals
        signal_indices = list(range(nsignals))
    elif isinstance(signal_spec, int):
        # Single index given
        signal_indices = [signal_spec]
    elif isinstance(signal_spec, list) and \
         all([isinstance(index, int) for index in signal_spec]):
        # Simple list of integer indices
        signal_indices = signal_spec
    else:
        signal_indices = syslist[system_index]._find_signals(
            signal_spec, signal_dict)
        if signal_indices is None:
            raise ValueError(f"couldn't find {signame} signal '{spec}'")

    # Make sure the signal indices are valid
    for index in signal_indices:
        if index < 0 or index >= nsignals:
            ValueError(f"signal index '{index}' is out of range")

    return system_index, signal_indices, gain


#
# Utility function for processing subsystem indices
#
# This function processes an index specification (int, list, or slice) and
# returns a index specification that can be used to create a subsystem
#
def _process_subsys_index(idx, sys_labels, slice_to_list=False):
    if not isinstance(idx, (slice, list, int)):
        raise TypeError("system indices must be integers, slices, or lists")

    # Convert singleton lists to integers for proper slicing (below)
    if isinstance(idx, (list, tuple)) and len(idx) == 1:
        idx = idx[0]

    # Convert int to slice so that numpy doesn't drop dimension
    if isinstance(idx, int):
        idx = slice(idx, idx+1, 1)

    # Get label names (taking care of possibility that we were passed a list)
    labels = [sys_labels[i] for i in idx] if isinstance(idx, list) \
        else sys_labels[idx]

    if slice_to_list and isinstance(idx, slice):
        idx = range(len(sys_labels))[idx]

    return idx, labels


# Create an extended system name
def _extended_system_name(name, prefix="", suffix="", prefix_suffix_name=None):
    if prefix == "" and prefix_suffix_name is not None:
        prefix = config.defaults[
            'iosys.' + prefix_suffix_name + '_system_name_prefix']
    if suffix == "" and prefix_suffix_name is not None:
        suffix = config.defaults[
            'iosys.' + prefix_suffix_name + '_system_name_suffix']
    return prefix + name + suffix
