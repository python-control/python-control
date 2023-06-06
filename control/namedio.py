# namedio.py - named I/O system class and helper functions
# RMM, 13 Mar 2022
#
# This file implements the NamedIOSystem class, which is used as a parent
# class for FrequencyResponseData, InputOutputSystem, LTI, TimeResponseData,
# and other similar classes to allow naming of signals.

import numpy as np
from copy import deepcopy
from warnings import warn
from . import config

__all__ = ['issiso', 'timebase', 'common_timebase', 'timebaseEqual',
           'isdtime', 'isctime']

# Define module default parameter values
_namedio_defaults = {
    'namedio.state_name_delim': '_',
    'namedio.duplicate_system_name_prefix': '',
    'namedio.duplicate_system_name_suffix': '$copy',
    'namedio.linearized_system_name_prefix': '',
    'namedio.linearized_system_name_suffix': '$linearized',
    'namedio.sampled_system_name_prefix': '',
    'namedio.sampled_system_name_suffix': '$sampled',
    'namedio.converted_system_name_prefix': '',
    'namedio.converted_system_name_suffix': '$converted',
}


class NamedIOSystem(object):
    def __init__(
            self, name=None, inputs=None, outputs=None, states=None, **kwargs):

        # system name
        self.name = self._name_or_default(name)

        # Parse and store the number of inputs and outputs
        self.set_inputs(inputs)
        self.set_outputs(outputs)
        self.set_states(states)

        # Process timebase: if not given use default, but allow None as value
        self.dt = _process_dt_keyword(kwargs)

        # Make sure there were no other keywords
        if kwargs:
            raise TypeError("unrecognized keywords: ", str(kwargs))

    #
    # Functions to manipulate the system name
    #
    _idCounter = 0              # Counter for creating generic system name

    # Return system name
    def _name_or_default(self, name=None, prefix_suffix_name=None):
        if name is None:
            name = "sys[{}]".format(NamedIOSystem._idCounter)
            NamedIOSystem._idCounter += 1
        prefix = "" if prefix_suffix_name is None else config.defaults[
            'namedio.' + prefix_suffix_name + '_system_name_prefix']
        suffix = "" if prefix_suffix_name is None else config.defaults[
            'namedio.' + prefix_suffix_name + '_system_name_suffix']
        return prefix + name + suffix

    # Check if system name is generic
    def _generic_name_check(self):
        import re
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

    def __repr__(self):
        return f'<{self.__class__.__name__}:{self.name}:' + \
            f'{list(self.input_labels)}->{list(self.output_labels)}>'

    def __str__(self):
        """String representation of an input/output object"""
        str = f"<{self.__class__.__name__}>: {self.name}\n"
        str += f"Inputs ({self.ninputs}): {self.input_labels}\n"
        str += f"Outputs ({self.noutputs}): {self.output_labels}\n"
        if self.nstates is not None:
            str += f"States ({self.nstates}): {self.state_labels}"
        return str

    # Find a signal by name
    def _find_signal(self, name, sigdict):
        return sigdict.get(name, None)

    def _copy_names(self, sys, prefix="", suffix="", prefix_suffix_name=None):
        """copy the signal and system name of sys. Name is given as a keyword
        in case a specific name (e.g. append 'linearized') is desired. """
        # Figure out the system name and assign it
        if prefix == "" and prefix_suffix_name is not None:
            prefix = config.defaults[
                'namedio.' + prefix_suffix_name + '_system_name_prefix']
        if suffix == "" and prefix_suffix_name is not None:
            suffix = config.defaults[
                'namedio.' + prefix_suffix_name + '_system_name_suffix']
        self.name = prefix + sys.name + suffix

        # Name the inputs, outputs, and states
        self.input_index = sys.input_index.copy()
        self.output_index = sys.output_index.copy()
        if self.nstates and sys.nstates:
            # only copy state names for state space systems
            self.state_index = sys.state_index.copy()

    def copy(self, name=None, use_prefix_suffix=True):
        """Make a copy of an input/output system

        A copy of the system is made, with a new name.  The `name` keyword
        can be used to specify a specific name for the system.  If no name
        is given and `use_prefix_suffix` is True, the name is constructed
        by prepending config.defaults['namedio.duplicate_system_name_prefix']
        and appending config.defaults['namedio.duplicate_system_name_suffix'].
        Otherwise, a generic system name of the form `sys[<id>]` is used,
        where `<id>` is based on an internal counter.

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
            of the form `u[i]` (where the prefix `u` can be changed using the
            optional prefix parameter).
        prefix : string, optional
            If `inputs` is an integer, create the names of the states using
            the given prefix (default = 'u').  The names of the input will be
            of the form `prefix[i]`.

        """
        self.ninputs, self.input_index = \
            _process_signal_list(inputs, prefix=prefix)

    def find_input(self, name):
        """Find the index for an input given its name (`None` if not found)"""
        return self.input_index.get(name, None)

    # Property for getting and setting list of input signals
    input_labels = property(
        lambda self: list(self.input_index.keys()),     # getter
        set_inputs)                                     # setter

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
            _process_signal_list(outputs, prefix=prefix)

    def find_output(self, name):
        """Find the index for an output given its name (`None` if not found)"""
        return self.output_index.get(name, None)

    # Property for getting and setting list of output signals
    output_labels = property(
        lambda self: list(self.output_index.keys()),     # getter
        set_outputs)                                     # setter

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
            _process_signal_list(states, prefix=prefix)

    def find_state(self, name):
        """Find the index for a state given its name (`None` if not found)"""
        return self.state_index.get(name, None)

    # Property for getting and setting list of state signals
    state_labels = property(
        lambda self: list(self.state_index.keys()),     # getter
        set_states)                                     # setter

    def isctime(self, strict=False):
        """
        Check to see if a system is a continuous-time system

        Parameters
        ----------
        sys : Named I/O system
            System to be checked
        strict: bool, optional
            If strict is True, make sure that timebase is not None.  Default
            is False.
        """
        # If no timebase is given, answer depends on strict flag
        if self.dt is None:
            return True if not strict else False
        return self.dt == 0

    def isdtime(self, strict=False):
        """
        Check to see if a system is a discrete-time system

        Parameters
        ----------
        strict: bool, optional
            If strict is True, make sure that timebase is not None.  Default
            is False.
        """

        # If no timebase is given, answer depends on strict flag
        if self.dt == None:
            return True if not strict else False

        # Look for dt > 0 (also works if dt = True)
        return self.dt > 0

    def issiso(self):
        """Check to see if a system is single input, single output"""
        return self.ninputs == 1 and self.noutputs == 1

    def _isstatic(self):
        """Check to see if a system is a static system (no states)"""
        return self.nstates == 0


# Test to see if a system is SISO
def issiso(sys, strict=False):
    """
    Check to see if a system is single input, single output

    Parameters
    ----------
    sys : I/O or LTI system
        System to be checked
    strict: bool (default = False)
        If strict is True, do not treat scalars as SISO
    """
    if isinstance(sys, (int, float, complex, np.number)) and not strict:
        return True
    elif not isinstance(sys, NamedIOSystem):
        raise ValueError("Object is not an I/O or LTI system")

    # Done with the tricky stuff...
    return sys.issiso()

# Return the timebase (with conversion if unspecified)
def timebase(sys, strict=True):
    """Return the timebase for a system

    dt = timebase(sys)

    returns the timebase for a system 'sys'.  If the strict option is
    set to False, dt = True will be returned as 1.
    """
    # System needs to be either a constant or an I/O or LTI system
    if isinstance(sys, (int, float, complex, np.number)):
        return None
    elif not isinstance(sys, NamedIOSystem):
        raise ValueError("Timebase not defined")

    # Return the sample time, with converstion to float if strict is false
    if (sys.dt == None):
        return None
    elif (strict):
        return float(sys.dt)

    return sys.dt

def common_timebase(dt1, dt2):
    """
    Find the common timebase when interconnecting systems

    Parameters
    ----------
    dt1, dt2: number or system with a 'dt' attribute (e.g. TransferFunction
        or StateSpace system)

    Returns
    -------
    dt: number
        The common timebase of dt1 and dt2, as specified in
        :ref:`conventions-ref`.

    Raises
    ------
    ValueError
        when no compatible time base can be found
    """
    # explanation:
    # if either dt is None, they are compatible with anything
    # if either dt is True (discrete with unspecified time base),
    #   use the timebase of the other, if it is also discrete
    # otherwise both dts must be equal
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

# Check to see if two timebases are equal
def timebaseEqual(sys1, sys2):
    """
    Check to see if two systems have the same timebase

    timebaseEqual(sys1, sys2)

    returns True if the timebases for the two systems are compatible.  By
    default, systems with timebase 'None' are compatible with either
    discrete or continuous timebase systems.  If two systems have a discrete
    timebase (dt > 0) then their timebases must be equal.
    """
    warn("timebaseEqual will be deprecated in a future release of "
         "python-control; use :func:`common_timebase` instead",
         PendingDeprecationWarning)

    if (type(sys1.dt) == bool or type(sys2.dt) == bool):
        # Make sure both are unspecified discrete timebases
        return type(sys1.dt) == type(sys2.dt) and sys1.dt == sys2.dt
    elif (sys1.dt is None or sys2.dt is None):
        # One or the other is unspecified => the other can be anything
        return True
    else:
        return sys1.dt == sys2.dt


# Check to see if a system is a discrete time system
def isdtime(sys, strict=False):
    """
    Check to see if a system is a discrete time system

    Parameters
    ----------
    sys : I/O or LTI system
        System to be checked
    strict: bool (default = False)
        If strict is True, make sure that timebase is not None
    """

    # Check to see if this is a constant
    if isinstance(sys, (int, float, complex, np.number)):
        # OK as long as strict checking is off
        return True if not strict else False

    # Check for a transfer function or state-space object
    if isinstance(sys, NamedIOSystem):
        return sys.isdtime(strict)

    # Check to see if object has a dt object
    if hasattr(sys, 'dt'):
        # If no timebase is given, answer depends on strict flag
        if sys.dt == None:
            return True if not strict else False

        # Look for dt > 0 (also works if dt = True)
        return sys.dt > 0

    # Got passed something we don't recognize
    return False

# Check to see if a system is a continuous time system
def isctime(sys, strict=False):
    """
    Check to see if a system is a continuous-time system

    Parameters
    ----------
    sys : I/O or LTI system
        System to be checked
    strict: bool (default = False)
        If strict is True, make sure that timebase is not None
    """

    # Check to see if this is a constant
    if isinstance(sys, (int, float, complex, np.number)):
        # OK as long as strict checking is off
        return True if not strict else False

    # Check for a transfer function or state space object
    if isinstance(sys, NamedIOSystem):
        return sys.isctime(strict)

    # Check to see if object has a dt object
    if hasattr(sys, 'dt'):
        # If no timebase is given, answer depends on strict flag
        if sys.dt is None:
            return True if not strict else False
        return sys.dt == 0

    # Got passed something we don't recognize
    return False


# Utility function to parse nameio keywords
def _process_namedio_keywords(
        keywords={}, defaults={}, static=False, end=False):
    """Process namedio specification

    This function processes the standard keywords used in initializing a named
    I/O system.  It first looks in the `keyword` dictionary to see if a value
    is specified.  If not, the `default` dictionary is used.  The `default`
    dictionary can also be set to a NamedIOSystem object, which is useful for
    copy constructors that change system and signal names.

    If `end` is True, then generate an error if there are any remaining
    keywords.

    """
    # If default is a system, redefine as a dictionary
    if isinstance(defaults, NamedIOSystem):
        sys = defaults
        defaults = {
            'name': sys.name, 'inputs': sys.input_labels,
            'outputs': sys.output_labels, 'dt': sys.dt}

        if sys.nstates is not None:
            defaults['states'] = sys.state_labels

    elif not isinstance(defaults, dict):
        raise TypeError("default must be dict or sys")

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
            raise ValueError("Wrong number of input labels given.")
        if isinstance(outputs, list) and sys.noutputs != len(outputs):
            raise ValueError("Wrong number of output labels given.")
        if sys.nstates is not None and \
           isinstance(states, list) and sys.nstates != len(states):
            raise ValueError("Wrong number of state labels given.")

    # Process timebase: if not given use default, but allow None as value
    dt = _process_dt_keyword(keywords, defaults, static=static)

    # If desired, make sure we processed all keywords
    if end and keywords:
        raise TypeError("unrecognized keywords: ", str(keywords))

    # Return the processed keywords
    return name, inputs, outputs, states, dt

#
# Parse 'dt' in for named I/O system
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
def _process_signal_list(signals, prefix='s'):
    if signals is None:
        # No information provided; try and make it up later
        return None, {}

    elif isinstance(signals, (int, np.integer)):
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
